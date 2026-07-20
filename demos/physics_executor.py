"""Drives a demos.scripted_expert waypoint sequence through REAL contact physics
(mj_step, not just kinematic IK), so demonstrations reflect actual grasp
dynamics rather than a piece teleporting/welding to the gripper.

Chosen over a kinematic weld per project decision: dataset should reflect real
grasp-failure modes (piece slipping, knight tipping -- see piece_geometry.py's
fragility note) since there's no PPO/contact-physics phase downstream that
would otherwise be the first place this matters.

Each ScriptedExpert waypoint (a *kinematic* IK solution) is treated as a
setpoint for the scene's existing position actuators (see build_scene.py's
attached UR5e/2F85: <position> actuators, kp=2000/500 arm, kp≈100 gripper --
confirmed via actuator_biastype/biasprm before writing this). Between
waypoints, the setpoint is linearly ramped over PHASE_DURATIONS seconds and
physics is stepped at the model's native timestep, so the logged trajectory is
a dense, physically-simulated rollout, not the sparse 8-16 IK waypoints
themselves.

Grasp/placement are verified by reading back actual body state after physics
settles -- NOT assumed from the commanded waypoints succeeding -- since the
whole point of real contact physics is that a commanded grasp can fail.
"""

import os
import sys
from dataclasses import dataclass, field

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(HERE, "..", "assets")
sys.path.insert(0, ASSETS_DIR)
sys.path.insert(0, HERE)
from board_geometry import BOARD_TOP_Z, square_center  # noqa: E402
from piece_geometry import LIFT_CLEARANCE, PIECE_DIMENSIONS  # noqa: E402
from scripted_expert import SCENE_XML, ScriptedExpert, square_to_idx  # noqa: E402

CONTROL_DT = 0.05  # 20Hz control/logging rate
PHASE_DURATION = {
    "approach_source": 1.0,
    "descend_to_grasp": 0.8,
    "close_gripper": 0.6,
    "lift": 0.8,
    "transit": 1.0,
    "descend_to_place": 0.8,
    "open_gripper": 0.6,
    "retract": 0.8,
}
SETTLE_TIME = 0.3  # extra hold time after close/open for contact forces to stabilize

# every joint before the first piece freejoint is arm(6)+gripper(8) -- computed
# once against the compiled model rather than hardcoded, see build_scene.py's
# joint ordering (attach() puts arm+gripper joints first, pieces added after).
GRIPPER_CTRL_MAX = 255.0

PLACEMENT_XY_TOL = 0.012  # 12mm: piece center must land within this of the target square center
LIFT_XY_TOL = 0.015  # piece must have stayed within 15mm (xy) of its start -- else it slipped/was left behind
LIFT_Z_TOL = 0.02  # piece must have actually risen close to the commanded lift height


@dataclass
class TickLog:
    t: float
    phase: str
    qpos: np.ndarray  # (n_arm_gripper,) actual joint positions
    ctrl: np.ndarray  # (7,) commanded setpoint
    ee_pos: np.ndarray  # (3,) pinch site position


@dataclass
class MoveResult:
    success: bool
    ticks: list[TickLog] = field(default_factory=list)
    failure_reason: str | None = None
    grasp_xy_drift: float | None = None
    placement_error: float | None = None


class PhysicsExecutor:
    def __init__(self, scene_xml: str = SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.expert = ScriptedExpert(scene_xml)

        first_free = next(j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        self.n_arm_gripper = self.model.jnt_qposadr[first_free]  # == dofadr too (all 1-dof hinges)

        self.pinch_site_id = self.model.site("/pinch").id
        self.home_arm_gripper_qpos = self.model.key("home").qpos[: self.n_arm_gripper].copy()

        pad_names = ["/left_pad1", "/left_pad2", "/right_pad1", "/right_pad2"]
        self.pad_geom_ids = {self.model.geom(n).id for n in pad_names}

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def reset_arm(self) -> None:
        """Resets only the arm+gripper state, leaving piece positions as tracked
        by BoardState (a real robot doesn't teleport pieces, only itself)."""
        self.data.qpos[: self.n_arm_gripper] = self.home_arm_gripper_qpos
        self.data.qvel[: self.n_arm_gripper] = 0
        self.data.ctrl[:6] = self.home_arm_gripper_qpos[:6]
        self.data.ctrl[6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_piece_pose(self, body_name: str, xy: tuple[float, float], z: float = BOARD_TOP_Z) -> None:
        body = self.model.body(body_name)
        joint = self.model.joint(body.jntadr[0])
        qadr, dadr = joint.qposadr[0], joint.dofadr[0]
        self.data.qpos[qadr : qadr + 3] = [xy[0], xy[1], z]
        self.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[dadr : dadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def piece_xyz(self, body_name: str) -> np.ndarray:
        body = self.model.body(body_name)
        joint = self.model.joint(body.jntadr[0])
        return self.data.qpos[joint.qposadr[0] : joint.qposadr[0] + 3].copy()

    def _piece_geom_ids(self, body_name: str) -> set[int]:
        body = self.model.body(body_name)
        return set(range(body.geomadr[0], body.geomadr[0] + body.geomnum[0]))

    def _gripper_contacts_piece(self, body_name: str) -> bool:
        piece_geoms = self._piece_geom_ids(body_name)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if pair & self.pad_geom_ids and pair & piece_geoms:
                return True
        return False

    def _ramp(self, arm_target: np.ndarray, gripper_target: float, duration: float, phase: str, log: list[TickLog]) -> None:
        """Straight-line-in-joint-space ramp. Only safe when the two endpoints'
        end-effector orientation is identical AND the path between them doesn't
        need to stay task-space-straight (e.g. the no-op close/open-gripper
        holds). For any actual end-effector translation, use _move_ee_to
        instead -- see its docstring for why."""
        n_ticks = max(1, round(duration / CONTROL_DT))
        n_sub = max(1, round(CONTROL_DT / self.model.opt.timestep))
        arm_start = self.data.ctrl[:6].copy()
        gripper_start = self.data.ctrl[6]
        for tick in range(1, n_ticks + 1):
            alpha = tick / n_ticks
            self.data.ctrl[:6] = arm_start + alpha * (arm_target - arm_start)
            self.data.ctrl[6] = gripper_start + alpha * (gripper_target - gripper_start)
            for _ in range(n_sub):
                mujoco.mj_step(self.model, self.data)
            log.append(
                TickLog(
                    t=self.data.time,
                    phase=phase,
                    qpos=self.data.qpos[: self.n_arm_gripper].copy(),
                    ctrl=self.data.ctrl.copy(),
                    ee_pos=self.data.site_xpos[self.pinch_site_id].copy(),
                )
            )

    def _move_ee_to(
        self, target_pos, gripper_target: float, duration: float, phase: str, log: list[TickLog], n_waypoints: int = 8
    ) -> bool:
        """Moves the pinch site to target_pos along a straight Cartesian line,
        re-solving IK at each intermediate point (orientation pinned to
        DOWN_QUAT throughout) rather than linearly interpolating the two
        endpoints' joint angles.

        This matters because linear joint-space interpolation between two IK
        solutions has no guarantee of holding end-effector ORIENTATION constant
        along the way -- with 6 DOF and IK's damped-least-squares solving each
        endpoint independently, the intermediate path can transiently tilt the
        gripper off vertical even though both endpoints are perfectly
        DOWN_QUAT. Confirmed by direct observation: a grasped pawn held with
        the arm perfectly still stayed gripped indefinitely, but the same
        grasp popped the piece out partway through a joint-space lift ramp --
        including a 3x-slower, eased version of the same ramp, ruling out
        speed/jerk as the cause. Re-solving IK at each sub-point keeps the
        gripper exactly vertical the whole way, which resolved it.

        Returns False (and leaves `log` populated with what was executed so
        far) if any intermediate IK solve fails to converge.
        """
        start_pos = self.expert.configuration.data.site_xpos[self.pinch_site_id].copy()
        target_pos = np.array(target_pos, dtype=float)
        sub_duration = duration / n_waypoints
        for i in range(1, n_waypoints + 1):
            alpha = i / n_waypoints
            waypoint_pos = start_pos + alpha * (target_pos - start_pos)
            wp, ok, pos_err, ori_err = self.expert.solve_to(tuple(waypoint_pos), gripper_target, phase)
            if not ok:
                return False
            self._ramp(wp.qpos[:6], gripper_target, sub_duration, phase, log)
        return True

    def _close_gripper_adaptive(self, body_name: str, log: list[TickLog], ctrl_step: float = 5.0, margin: float = 15.0) -> float | None:
        """Closes in small increments, stopping shortly after first contact
        instead of driving straight to GRIPPER_CTRL_MAX.

        Commanding max close (255) for the whole grasp+lift+transit+place
        sequence keeps the position servo pushing for a fully-closed setpoint
        it can never reach once it touches the piece -- observed directly: the
        driver joint kept creeping closed (0.53 -> 0.78 rad) throughout lift
        even with the arm held perfectly still in an isolated test, and with
        dense IK-tracked (non-wobbling) motion. On a smooth cylindrical piece
        that continued clamping eventually extrudes it out from between the
        pads under any dynamic load, rather than holding it -- a real grasp
        instability, not a trajectory artifact. Stopping at first-contact +
        a small margin avoids the over-squeeze regime entirely.

        Returns the ctrl value settled on, or None if max ctrl was reached
        with no contact ever detected (missed the piece).
        """
        n_sub = max(1, round(CONTROL_DT / self.model.opt.timestep))
        ctrl = self.data.ctrl[6]
        first_contact_ctrl = None
        while ctrl < GRIPPER_CTRL_MAX:
            ctrl = min(GRIPPER_CTRL_MAX, ctrl + ctrl_step)
            self.data.ctrl[6] = ctrl
            for _ in range(n_sub):
                mujoco.mj_step(self.model, self.data)
            log.append(
                TickLog(
                    t=self.data.time,
                    phase="close_gripper",
                    qpos=self.data.qpos[: self.n_arm_gripper].copy(),
                    ctrl=self.data.ctrl.copy(),
                    ee_pos=self.data.site_xpos[self.pinch_site_id].copy(),
                )
            )
            if self._gripper_contacts_piece(body_name):
                first_contact_ctrl = ctrl
                break

        if first_contact_ctrl is None:
            return None

        target = min(GRIPPER_CTRL_MAX, first_contact_ctrl + margin)
        self._ramp(self.data.ctrl[:6].copy(), target, 0.3, "close_gripper", log)
        return target

    def _hold(self, duration: float, phase: str, log: list[TickLog]) -> None:
        n_ticks = max(1, round(duration / CONTROL_DT))
        n_sub = max(1, round(CONTROL_DT / self.model.opt.timestep))
        for _ in range(n_ticks):
            for _ in range(n_sub):
                mujoco.mj_step(self.model, self.data)
            log.append(
                TickLog(
                    t=self.data.time,
                    phase=phase,
                    qpos=self.data.qpos[: self.n_arm_gripper].copy(),
                    ctrl=self.data.ctrl.copy(),
                    ee_pos=self.data.site_xpos[self.pinch_site_id].copy(),
                )
            )

    def execute_pick_and_place(
        self, body_name: str, source_xy: tuple[float, float], target_xy: tuple[float, float], piece_kind: str
    ) -> MoveResult:
        self.reset_arm()
        self.expert.reset()
        grasp_z = PIECE_DIMENSIONS[piece_kind]["grasp_z"]
        sx, sy = source_xy
        tx, ty = target_xy

        ik_sequence = [
            ("approach_source", (sx, sy, BOARD_TOP_Z + LIFT_CLEARANCE), 0.0),
            ("descend_to_grasp", (sx, sy, BOARD_TOP_Z + grasp_z), 0.0),
        ]
        log: list[TickLog] = []

        for phase, pos, gripper in ik_sequence:
            if not self._move_ee_to(pos, gripper, PHASE_DURATION[phase], phase, log):
                return MoveResult(False, log, f"ik_failed:{phase}")

        pre_grasp_xy = self.piece_xyz(body_name)[:2].copy()
        closed_ctrl = self._close_gripper_adaptive(body_name, log)
        self._hold(SETTLE_TIME, "close_gripper_settle", log)

        if closed_ctrl is None or not self._gripper_contacts_piece(body_name):
            return MoveResult(False, log, f"grasp_failed:no_contact_after_close on {body_name}")

        for phase, pos in [
            ("lift", (sx, sy, BOARD_TOP_Z + LIFT_CLEARANCE)),
            ("transit", (tx, ty, BOARD_TOP_Z + LIFT_CLEARANCE)),
            ("descend_to_place", (tx, ty, BOARD_TOP_Z + grasp_z)),
        ]:
            if not self._move_ee_to(pos, closed_ctrl, PHASE_DURATION[phase], phase, log):
                return MoveResult(False, log, f"ik_failed:{phase}")

            if phase == "lift":
                lifted_xyz = self.piece_xyz(body_name)
                xy_drift = float(np.linalg.norm(lifted_xyz[:2] - pre_grasp_xy))
                z_err = abs(lifted_xyz[2] - (BOARD_TOP_Z + LIFT_CLEARANCE))
                if xy_drift > LIFT_XY_TOL or z_err > LIFT_Z_TOL:
                    return MoveResult(
                        False, log, f"grasp_lost_during_lift xy_drift={xy_drift:.4f} z_err={z_err:.4f}",
                        grasp_xy_drift=xy_drift,
                    )

        self._ramp(self.data.ctrl[:6].copy(), 0.0, PHASE_DURATION["open_gripper"], "open_gripper", log)
        self._hold(SETTLE_TIME, "open_gripper_settle", log)

        final_xy = self.piece_xyz(body_name)[:2]
        placement_error = float(np.linalg.norm(final_xy - np.array([tx, ty])))
        if placement_error > PLACEMENT_XY_TOL:
            return MoveResult(False, log, f"placement_error={placement_error:.4f}", placement_error=placement_error)

        self._move_ee_to((tx, ty, BOARD_TOP_Z + LIFT_CLEARANCE), 0.0, PHASE_DURATION["retract"], "retract", log)

        return MoveResult(True, log, placement_error=placement_error)


def _smoke_test() -> None:
    """Same hand-picked set as scripted_expert.py's kinematic smoke test, plus a
    knight (already flagged as the fragile grasp case in piece_geometry.py) --
    now executed under real contact physics rather than pure IK, to check
    whether the flagged fragility actually manifests as a grasp failure."""
    from board_state import BoardState

    board = BoardState()
    executor = PhysicsExecutor()

    moves = [
        ("e2", "e4", None),
        ("d1", "d6", None),  # stress test, not a legal opening move
        ("b1", "c3", None),  # knight -- the flagged fragile grasp
    ]

    for source_sq, target_sq, _ in moves:
        body = board.body_at(source_sq)
        kind = board.body_kind[body]
        sx, sy, _ = square_center(*square_to_idx(source_sq))
        tx, ty, _ = square_center(*square_to_idx(target_sq))
        result = executor.execute_pick_and_place(body, (sx, sy), (tx, ty), kind)
        status = "OK" if result.success else f"FAILED ({result.failure_reason})"
        print(f"{source_sq}->{target_sq} ({kind}, body={body}): {status}, {len(result.ticks)} ticks")
        if result.placement_error is not None:
            print(f"  placement_error={result.placement_error * 1000:.2f}mm")
        if result.success:
            board.apply_move(source_sq, target_sq)
            executor.set_piece_pose(body, (tx, ty))


if __name__ == "__main__":
    _smoke_test()
