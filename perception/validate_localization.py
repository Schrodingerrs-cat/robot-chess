"""Phase 3 validation: check BoardLocalizer against sim ground truth.

Renders assets/ur5e_chess_scene.xml from its board_cam, runs BoardLocalizer on
that image (using only the detected ArUco markers -- no privileged camera
knowledge), then for every square: projects its true world center to a pixel
using the camera's actual extrinsics/intrinsics (ground truth, independent of
the homography), feeds that pixel through the localizer, and compares the
recovered board-frame (x,y) against the true square center.

Run: python perception/validate_localization.py
"""

import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "assets"))
from board_geometry import FILES, RANKS, square_center  # noqa: E402

from board_localization import BoardLocalizer  # noqa: E402

SCENE_XML = str(Path(__file__).parent.parent / "assets" / "ur5e_chess_scene.xml")
W, H = 1600, 1200


def project_world_to_pixel(campos, forward, up, right, fovy_deg, w, h, point) -> tuple[float, float]:
    """Ground-truth pinhole projection, independent of the ArUco homography."""
    v = np.asarray(point) - campos
    zc, xc, yc = np.dot(v, forward), np.dot(v, right), np.dot(v, up)
    tan_half = np.tan(np.deg2rad(fovy_deg) / 2)
    aspect = w / h
    ndc_x = (xc / zc) / (tan_half * aspect)
    ndc_y = (yc / zc) / tan_half
    return (ndc_x * 0.5 + 0.5) * w, (1 - (ndc_y * 0.5 + 0.5)) * h


def main() -> None:
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam_id = model.camera("board_cam").id
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # data.cam_xpos/cam_xmat, not mjvScene's scene.camera[0] -- the latter is
    # MuJoCo's stereo left-eye camera, offset from true center by half the IPD.
    campos = data.cam_xpos[cam_id].copy()
    cam_xmat = data.cam_xmat[cam_id].reshape(3, 3)
    right, up, forward = cam_xmat[:, 0], cam_xmat[:, 1], -cam_xmat[:, 2]
    fovy_deg = model.cam_fovy[cam_id]

    localizer = BoardLocalizer(bgr)
    print("detected all 4 ArUco corner markers, homography built")

    errors_mm = []
    worst = None
    mismatches = 0
    for file_idx in range(8):
        for rank_idx in range(8):
            true_point = square_center(file_idx, rank_idx)
            true_xy = true_point[:2]
            square = f"{FILES[file_idx]}{RANKS[rank_idx]}"

            true_pixel = project_world_to_pixel(campos, forward, up, right, fovy_deg, W, H, true_point)
            est_xy = localizer.pixel_to_board_xy(true_pixel)
            err_mm = 1000 * np.hypot(est_xy[0] - true_xy[0], est_xy[1] - true_xy[1])
            errors_mm.append(err_mm)
            if worst is None or err_mm > worst[1]:
                worst = (square, err_mm)

            got_square = localizer.pixel_to_square(true_pixel)
            if got_square != square:
                mismatches += 1
                print(f"  square mismatch: expected {square}, got {got_square}")

    errors_mm = np.array(errors_mm)
    print("64/64 squares checked")
    print(f"board-frame recovery error: mean={errors_mm.mean():.2f}mm  max={errors_mm.max():.2f}mm  (worst: {worst[0]})")
    print(f"square-name lookup: {64 - mismatches}/64 correct")


if __name__ == "__main__":
    main()
