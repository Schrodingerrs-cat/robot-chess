"""Phase 1 deliverable: verify the UR5e+2F85 can reach all 64 chessboard squares.

For each square, solves IK (via mink) for the pinch site at grasp height with a
fixed top-down gripper orientation. Reports any square that fails to converge
within tolerance or that requires a configuration outside joint limits, so the
board placement can be adjusted before Phase 2.

Run: python assets/workspace_reach_check.py
"""

import mujoco
import mink
import numpy as np

HERE = __file__.rsplit("/", 1)[0]
SCENE_XML = f"{HERE}/ur5e_chess_scene.xml"

FILES = "abcdefgh"
RANKS = "12345678"

BOARD_CENTER = (0.5, 0.0, 0.0)
SQUARE = 0.05
BOARD_THICKNESS = 0.02
BOARD_TOP_Z = BOARD_THICKNESS
GRASP_Z = BOARD_TOP_Z + 0.03  # mid-height of a pawn, representative grasp point

# gripper pointing straight down: 180deg rotation about world x (w,x,y,z)
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

POS_TOL = 0.005  # 5mm
ORI_TOL = 0.05  # rad
MAX_ITERS = 300
DT = 0.01


def square_center(file_idx: int, rank_idx: int) -> tuple[float, float, float]:
    origin_x = BOARD_CENTER[0] - 3.5 * SQUARE
    origin_y = BOARD_CENTER[1] - 3.5 * SQUARE
    x = origin_x + file_idx * SQUARE
    y = origin_y + rank_idx * SQUARE
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


def main() -> None:
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    configuration = mink.Configuration(model)
    configuration.update_from_keyframe("home")

    task = mink.FrameTask(
        frame_name="/pinch",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    results = {}
    warm_start = configuration.q.copy()
    for rank_idx in range(8):
        for file_idx in range(8):
            square = f"{FILES[file_idx]}{RANKS[rank_idx]}"
            target_pos = square_center(file_idx, rank_idx)
            configuration.q[:] = warm_start
            configuration.update()
            ok, pos_err, ori_err, q_sol = solve_square(configuration, task, model, target_pos)
            results[square] = (ok, pos_err, ori_err)
            print(f"{square}: {'OK' if ok else 'FAIL'}  pos_err={pos_err*1000:.2f}mm  ori_err={ori_err:.3f}rad")

    n_ok = sum(1 for ok, _, _ in results.values() if ok)
    print(f"\n{n_ok}/64 squares reachable")
    failed = [sq for sq, (ok, _, _) in results.items() if not ok]
    if failed:
        print(f"UNREACHABLE: {failed}")
    else:
        print("All 64 squares reachable within tolerance.")


if __name__ == "__main__":
    main()
