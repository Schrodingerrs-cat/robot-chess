"""Single source of truth for chessboard world-frame geometry.

Shared by assets/build_scene.py (placing the board/pieces/markers) and
perception/board_localization.py (interpreting a camera image of them), so the
two can never drift apart.
"""

BOARD_CENTER = (0.5, 0.0, 0.0)
SQUARE = 0.05  # 5cm squares -> 0.4m x 0.4m board
BOARD_THICKNESS = 0.02
BOARD_TOP_Z = BOARD_THICKNESS  # world z of the board's playing surface

FILES = "abcdefgh"
RANKS = "12345678"


def square_center(file_idx: int, rank_idx: int) -> tuple[float, float, float]:
    """file_idx, rank_idx in [0,7]. a1 at (file=0,rank=0), nearest-left corner."""
    origin_x = BOARD_CENTER[0] - 3.5 * SQUARE
    origin_y = BOARD_CENTER[1] - 3.5 * SQUARE
    x = origin_x + file_idx * SQUARE
    y = origin_y + rank_idx * SQUARE
    return x, y, BOARD_TOP_Z


# ArUco corner markers (DICT_4X4_50), on the table well beyond the board's
# border. Even from a nominally top-down camera, a corner placed just past the
# border gets grazed/occluded by the (much taller) board edge from any tiny
# residual camera tilt -- the low-lying marker and the raised board corner sit
# on almost the same sightline. Pushing the markers out removes the ambiguity.
# IDs are fixed by construction, so a detected ID unambiguously identifies
# which board corner it is -- no dependence on marker-corner ordering or
# in-plane rotation.
ARUCO_MARKER_SIZE = 0.03  # 30mm square, plenty of room now it's off-board
ARUCO_CORNERS: dict[int, tuple[float, float]] = {
    0: (BOARD_CENTER[0] - 5.2 * SQUARE, BOARD_CENTER[1] - 5.2 * SQUARE),  # beyond a1
    1: (BOARD_CENTER[0] + 5.2 * SQUARE, BOARD_CENTER[1] - 5.2 * SQUARE),  # beyond h1
    2: (BOARD_CENTER[0] + 5.2 * SQUARE, BOARD_CENTER[1] + 5.2 * SQUARE),  # beyond h8
    3: (BOARD_CENTER[0] - 5.2 * SQUARE, BOARD_CENTER[1] + 5.2 * SQUARE),  # beyond a8
}
