"""ArUco marker board localization + homography.

Detects the 4 ArUco corner markers (assets/board_geometry.ARUCO_CORNERS) in a
camera image and uses their known world-frame positions to compute a planar
homography between image pixels and board-frame (world XY, since the board
sits flat and unrotated) coordinates. That homography is what downstream
perception (e.g. a piece detector's pixel bounding boxes) uses to answer
"which square is this piece on" without needing known camera extrinsics.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "assets"))
from board_geometry import ARUCO_CORNERS, FILES, RANKS, SQUARE, square_center  # noqa: E402

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)


def detect_markers(image: np.ndarray) -> dict[int, np.ndarray]:
    """Returns {marker_id: pixel centroid} for each detected corner marker.

    Uses the centroid of each marker's 4 corners rather than individual corner
    correspondences: since each ID is placed at a known, fixed board corner by
    construction (assets/build_scene.py), ID alone identifies which world point
    a detection corresponds to -- no dependence on the marker-corner ordering
    cv2 returns or the marker's in-plane rotation as seen by the camera.
    """
    corners, ids, _ = _DETECTOR.detectMarkers(image)
    if ids is None:
        return {}
    return {int(i): c[0].mean(axis=0) for c, i in zip(corners, ids.flatten())}


class BoardLocalizer:
    """Pixel<->board-plane(world XY) homography from one image's detected markers."""

    def __init__(self, image: np.ndarray):
        detected = detect_markers(image)
        missing = set(ARUCO_CORNERS) - set(detected)
        if missing:
            raise RuntimeError(f"missing ArUco marker ids: {sorted(missing)}")

        ids = sorted(ARUCO_CORNERS)
        pixel_pts = np.array([detected[i] for i in ids], dtype=np.float32)
        world_pts = np.array([ARUCO_CORNERS[i] for i in ids], dtype=np.float32)

        self.pixel_to_world_H, _ = cv2.findHomography(pixel_pts, world_pts)
        self.world_to_pixel_H, _ = cv2.findHomography(world_pts, pixel_pts)

    def pixel_to_board_xy(self, pixel: tuple[float, float]) -> tuple[float, float]:
        out = cv2.perspectiveTransform(np.array([[pixel]], dtype=np.float32), self.pixel_to_world_H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def board_xy_to_pixel(self, xy: tuple[float, float]) -> tuple[float, float]:
        out = cv2.perspectiveTransform(np.array([[xy]], dtype=np.float32), self.world_to_pixel_H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    @staticmethod
    def board_xy_to_square(xy: tuple[float, float]) -> str:
        """Nearest square name to a board-frame (world XY) point."""
        x, y = xy
        file_idx = round((x - square_center(0, 0)[0]) / SQUARE)
        rank_idx = round((y - square_center(0, 0)[1]) / SQUARE)
        file_idx = min(max(file_idx, 0), 7)
        rank_idx = min(max(rank_idx, 0), 7)
        return f"{FILES[file_idx]}{RANKS[rank_idx]}"

    def pixel_to_square(self, pixel: tuple[float, float]) -> str:
        return self.board_xy_to_square(self.pixel_to_board_xy(pixel))

    def square_pixel_centers(self) -> dict[str, tuple[float, float]]:
        """Board-frame square name -> predicted pixel location, via the homography."""
        out = {}
        for file_idx in range(8):
            for rank_idx in range(8):
                x, y, _ = square_center(file_idx, rank_idx)
                out[f"{FILES[file_idx]}{RANKS[rank_idx]}"] = self.board_xy_to_pixel((x, y))
        return out
