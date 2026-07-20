"""Phase 1 deliverable: verify the UR5e+2F85 can reach all 64 chessboard squares.

For each square, solves IK (via mink) for the pinch site at grasp height with a
fixed top-down gripper orientation. Reports any square that fails to converge
within tolerance or that requires a configuration outside joint limits, so the
board placement can be adjusted before Phase 2.

Also reports manipulability (Yoshikawa index, sqrt(det(J J^T)) of the 6x6
arm-joint Jacobian at the pinch site) for every reachable square. A square
being IK-reachable at all is weaker than being reachable at a well-conditioned
pose: low manipulability means the solved configuration sits close to a
kinematic singularity, where small task-space grasp/lift/place motions in
Phase 4's scripted expert demand large or fast joint motions -- fragile,
jerky trajectories even though the IK "succeeded". Re-run whenever board
geometry changes (this is the pre-flight check for Phase 4).

Run: python assets/workspace_reach_check.py
"""

import os
import sys

import mujoco
import mink
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from board_geometry import BOARD_TOP_Z, FILES, RANKS  # noqa: E402
from board_geometry import square_center as _board_square_center  # noqa: E402

SCENE_XML = f"{HERE}/ur5e_chess_scene.xml"

GRASP_Z = BOARD_TOP_Z + 0.03  # mid-height of a pawn, representative grasp point

# gripper pointing straight down: 180deg rotation about world x (w,x,y,z)
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

POS_TOL = 0.005  # 5mm
ORI_TOL = 0.05  # rad
MAX_ITERS = 300
DT = 0.01


def square_center(file_idx: int, rank_idx: int) -> tuple[float, float, float]:
    x, y, _ = _board_square_center(file_idx, rank_idx)
    return x, y, GRASP_Z


def solve_square(configuration, task, model, target_pos):
    target = mink.SE3.from_rotation_and_translation(
        mink.SO3(wxyz=DOWN_QUAT), np.array(target_pos)
    )
    task.set_target(target)

    limits = [mink.ConfigurationLimit(model)]
    q = configuration.q.copy()
    for _ in range(MAX_ITERS):
        vel = mink.solve_ik(configuration, [task], DT, "daqp", limits=limits)
        configuration.integrate_inplace(vel, DT)
        err = task.compute_error(configuration)
        pos_err = np.linalg.norm(err[:3])
        ori_err = np.linalg.norm(err[3:])
        if pos_err < POS_TOL and ori_err < ORI_TOL:
            return True, pos_err, ori_err, configuration.q.copy()

    configuration.q[:] = q  # restore on failure so next attempt starts fresh
    configuration.update()
    return False, pos_err, ori_err, None


ARM_DOF_ADR = list(range(6))  # shoulder_pan/lift, elbow, wrist_1/2/3 -- see joint dofadr in the compiled model


def manipulability(model, data, site_name: str) -> float:
    """Yoshikawa manipulability index sqrt(det(J J^T)) of the 6x6 Jacobian
    (3 position + 3 orientation rows) restricted to the arm's 6 joint columns.
    Near zero means the pose is near a kinematic singularity.
    """
    site_id = model.site(site_name).id
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, ARM_DOF_ADR], jacr[:, ARM_DOF_ADR]])
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def main() -> None:
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    configuration = mink.Configuration(model)
    configuration.update_from_keyframe("home")
    data = mujoco.MjData(model)

    task = mink.FrameTask(
        frame_name="/pinch",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    results = {}
    manip = {}
    warm_start = configuration.q.copy()
    for rank_idx in range(8):
        for file_idx in range(8):
            square = f"{FILES[file_idx]}{RANKS[rank_idx]}"
            target_pos = square_center(file_idx, rank_idx)
            configuration.q[:] = warm_start
            configuration.update()
            ok, pos_err, ori_err, q_sol = solve_square(configuration, task, model, target_pos)
            results[square] = (ok, pos_err, ori_err)
            if ok:
                data.qpos[: model.nq] = configuration.q
                mujoco.mj_forward(model, data)
                manip[square] = manipulability(model, data, "/pinch")
            print(
                f"{square}: {'OK' if ok else 'FAIL'}  pos_err={pos_err*1000:.2f}mm  ori_err={ori_err:.3f}rad"
                + (f"  manip={manip[square]:.4f}" if ok else "")
            )

    n_ok = sum(1 for ok, _, _ in results.values() if ok)
    print(f"\n{n_ok}/64 squares reachable")
    failed = [sq for sq, (ok, _, _) in results.items() if not ok]
    if failed:
        print(f"UNREACHABLE: {failed}")
    else:
        print("All 64 squares reachable within tolerance.")

    worst = sorted(manip.items(), key=lambda kv: kv[1])[:10]
    print("\nworst 10 squares by manipulability (lowest = closest to a singularity):")
    for square, m in worst:
        print(f"  {square}: {m:.4f}")


if __name__ == "__main__":
    main()
