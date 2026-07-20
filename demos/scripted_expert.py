"""Engine move -> mink IK -> joint trajectory demo generator.

For a White move (source_square, target_square, piece_kind[, capture info]),
solves a joint-space waypoint sequence via mink IK: approach -> descend to
grasp height -> close gripper -> lift -> transit -> descend to place height
-> open gripper -> retract. Captures are handled by first relocating the
captured piece to an off-board graveyard slot (see board_geometry.graveyard_slot),
then executing the capturing piece's own move into the now-vacated square.

Per-piece-type grasp heights come from piece_geometry.PIECE_DIMENSIONS, since a
single fixed grasp height fails for most pieces (a pawn is 50mm tall, a king
114mm) -- see that module for the derivation from build_scene.py's geometry.

Only White's moves are generated here. Black's moves are applied procedurally
elsewhere (direct scene-state edit, no IK/gripper) -- see project memory.

This module solves the KINEMATIC path only (joint-space waypoints); it does
not simulate contact-based grasping physics. Whether the piece is physically
attached to the gripper during transport (weld/kinematic-follow vs. real
contact+friction) is a decision for the full-trajectory-generation step, not
this core solver -- flagged in the module docstring for demos/generate_dataset.py.

Run: python demos/scripted_expert.py   # smoke-tests a small hand-picked move set
"""

import os
import sys

import mujoco
import mink
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(HERE, "..", "assets")
sys.path.insert(0, ASSETS_DIR)
from board_geometry import BOARD_TOP_Z, FILES, RANKS, graveyard_slot, square_center  # noqa: E402
from piece_geometry import LIFT_CLEARANCE, PIECE_DIMENSIONS  # noqa: E402

SCENE_XML = os.path.join(ASSETS_DIR, "ur5e_chess_scene.xml")

# gripper pointing straight down: 180deg rotation about world x (w,x,y,z)
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

POS_TOL = 0.005  # 5mm
ORI_TOL = 0.05  # rad
MAX_ITERS = 300
DT = 0.01

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0


def square_to_idx(square: str) -> tuple[int, int]:
    return FILES.index(square[0]), RANKS.index(square[1])


class Waypoint:
    def __init__(self, qpos: np.ndarray, gripper: float, phase: str):
        self.qpos = qpos
        self.gripper = gripper
        self.phase = phase

    def __repr__(self) -> str:
        return f"Waypoint(phase={self.phase!r}, gripper={self.gripper})"


class ScriptedExpert:
    def __init__(self, scene_xml: str = SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.configuration = mink.Configuration(self.model)
        self.configuration.update_from_keyframe("home")
        self.task = mink.FrameTask(
            frame_name="/pinch",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.limits = [mink.ConfigurationLimit(self.model)]

    def reset(self) -> None:
        self.configuration.update_from_keyframe("home")

    def solve_to(self, pos, gripper: float, phase: str):
        """Warm-starts from the current configuration, so consecutive waypoints
        produce a continuous joint path rather than jumping between independent
        solutions -- important for a demo trajectory that's meant to look like
        one continuous arm motion, not teleporting between poses.
        """
        target = mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=DOWN_QUAT), np.array(pos))
        self.task.set_target(target)
        q_before = self.configuration.q.copy()

        pos_err = ori_err = float("inf")
        for _ in range(MAX_ITERS):
            vel = mink.solve_ik(self.configuration, [self.task], DT, "daqp", limits=self.limits)
            self.configuration.integrate_inplace(vel, DT)
            err = self.task.compute_error(self.configuration)
            pos_err, ori_err = np.linalg.norm(err[:3]), np.linalg.norm(err[3:])
            if pos_err < POS_TOL and ori_err < ORI_TOL:
                return Waypoint(self.configuration.q.copy(), gripper, phase), True, pos_err, ori_err

        self.configuration.q[:] = q_before  # restore so the next attempt isn't corrupted by a failed solve
        self.configuration.update()
        return Waypoint(q_before.copy(), gripper, phase), False, pos_err, ori_err

    def pick_and_place(self, source_xy, target_xy, piece_kind: str):
        grasp_z = PIECE_DIMENSIONS[piece_kind]["grasp_z"]
        sx, sy = source_xy
        tx, ty = target_xy

        sequence = [
            ((sx, sy, BOARD_TOP_Z + LIFT_CLEARANCE), GRIPPER_OPEN, "approach_source"),
            ((sx, sy, BOARD_TOP_Z + grasp_z), GRIPPER_OPEN, "descend_to_grasp"),
            None,  # close gripper -- no motion, handled below
            ((sx, sy, BOARD_TOP_Z + LIFT_CLEARANCE), GRIPPER_CLOSED, "lift"),
            ((tx, ty, BOARD_TOP_Z + LIFT_CLEARANCE), GRIPPER_CLOSED, "transit"),
            ((tx, ty, BOARD_TOP_Z + grasp_z), GRIPPER_CLOSED, "descend_to_place"),
            None,  # open gripper -- no motion, handled below
            ((tx, ty, BOARD_TOP_Z + LIFT_CLEARANCE), GRIPPER_OPEN, "retract"),
        ]

        waypoints, failures = [], []
        for i, step in enumerate(sequence):
            if step is None:
                gripper = GRIPPER_CLOSED if sequence[i + 1][1] == GRIPPER_CLOSED else GRIPPER_OPEN
                phase = "close_gripper" if gripper == GRIPPER_CLOSED else "open_gripper"
                waypoints.append(Waypoint(self.configuration.q.copy(), gripper, phase))
                continue
            pos, gripper, phase = step
            wp, ok, pos_err, ori_err = self.solve_to(pos, gripper, phase)
            waypoints.append(wp)
            if not ok:
                failures.append((phase, pos_err, ori_err))
        return waypoints, failures

    def generate_move(
        self,
        source_square: str,
        target_square: str,
        piece_kind: str,
        is_capture: bool = False,
        captured_kind: str | None = None,
        captured_color: str | None = None,
        graveyard_index: int | None = None,
    ):
        waypoints, failures = [], []

        if is_capture:
            assert captured_kind and captured_color and graveyard_index is not None
            tx_idx, ty_idx = square_to_idx(target_square)
            cap_pos = square_center(tx_idx, ty_idx)[:2]
            grave_pos = graveyard_slot(captured_color, graveyard_index)[:2]
            wps, fails = self.pick_and_place(cap_pos, grave_pos, captured_kind)
            waypoints.extend(wps)
            failures.extend((f"capture_removal:{p}", pe, oe) for p, pe, oe in fails)

        sx_idx, sy_idx = square_to_idx(source_square)
        tx_idx, ty_idx = square_to_idx(target_square)
        src_pos = square_center(sx_idx, sy_idx)[:2]
        tgt_pos = square_center(tx_idx, ty_idx)[:2]
        wps, fails = self.pick_and_place(src_pos, tgt_pos, piece_kind)
        waypoints.extend(wps)
        failures.extend((f"move:{p}", pe, oe) for p, pe, oe in fails)

        return waypoints, failures


def _smoke_test() -> None:
    """Small hand-picked set: near-rank move, far-rank move, and a capture."""
    expert = ScriptedExpert()

    test_moves = [
        dict(source_square="e2", target_square="e4", piece_kind="pawn"),  # near-rank, White's own side
        dict(source_square="d1", target_square="d6", piece_kind="queen"),  # deep into far-rank territory (stress test, not a legal opening move)
        dict(
            source_square="e4",
            target_square="d5",
            piece_kind="pawn",
            is_capture=True,
            captured_kind="pawn",
            captured_color="black",
            graveyard_index=0,
        ),
    ]

    for move in test_moves:
        expert.reset()
        label = f"{move['source_square']}->{move['target_square']} ({move['piece_kind']}{', capture' if move.get('is_capture') else ''})"
        waypoints, failures = expert.generate_move(**move)
        print(f"{label}: {len(waypoints)} waypoints, {len(failures)} failures")
        for phase, pos_err, ori_err in failures:
            print(f"  FAILED at {phase}: pos_err={pos_err * 1000:.2f}mm ori_err={ori_err:.3f}rad")


if __name__ == "__main__":
    _smoke_test()
