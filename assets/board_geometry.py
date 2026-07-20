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
    """file_idx, rank_idx in [0,7]. a1 at (file=0,rank=0).

    Rank maps to world X (near/far from the arm, which sits at the world
    origin facing +X) and file maps to world Y (left/right), so rank 1
    (White) sits nearest the arm and rank 8 (Black) sits farthest -- the
    robot plays White, seated behind its own back rank like a human player,
    not side-on to the board.
    """
    origin_x = BOARD_CENTER[0] - 3.5 * SQUARE
    origin_y = BOARD_CENTER[1] - 3.5 * SQUARE
    x = origin_x + rank_idx * SQUARE
    y = origin_y + file_idx * SQUARE
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
    1: (BOARD_CENTER[0] + 5.2 * SQUARE, BOARD_CENTER[1] - 5.2 * SQUARE),  # beyond a8
    2: (BOARD_CENTER[0] + 5.2 * SQUARE, BOARD_CENTER[1] + 5.2 * SQUARE),  # beyond h8
    3: (BOARD_CENTER[0] - 5.2 * SQUARE, BOARD_CENTER[1] + 5.2 * SQUARE),  # beyond h1
}


# Off-board holding area for pieces captured by White's moves (Black's own
# captures are procedural/teleported -- see project memory -- so this is only
# ever populated by the scripted expert, never read for Black). Placed beyond
# the h-file edge (y > board's h-file edge at 0.2) rather than beyond the far
# rank: reachability was verified via IK at the tallest piece's grasp height
# (king, grasp_z=0.065 -- see piece_geometry.py) across this x/y range before
# committing to it (all sub-1mm error), whereas extending past the far rank
# risks the arm's full-extension singularity. 2 rows x 8 columns per color
# (16 slots) comfortably covers the max 15 non-king pieces a side can lose.
GRAVEYARD_ROW_Y = {"white": 0.25, "black": 0.35}  # which color's captured pieces
GRAVEYARD_ROW_SPACING = 0.04
GRAVEYARD_COL_X_START = 0.35
GRAVEYARD_COL_SPACING = 0.043
GRAVEYARD_COLS = 8


def graveyard_slot(captured_color: str, index: int) -> tuple[float, float, float]:
    """index-th slot (0-based, up to 15) for a captured piece of captured_color."""
    row, col = divmod(index, GRAVEYARD_COLS)
    x = GRAVEYARD_COL_X_START + col * GRAVEYARD_COL_SPACING
    y = GRAVEYARD_ROW_Y[captured_color] + row * GRAVEYARD_ROW_SPACING
    return x, y, BOARD_TOP_Z
