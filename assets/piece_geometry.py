"""Single source of truth for chess piece body dimensions.

Shared by assets/build_scene.py (building the visual/collision geoms) and
demos/scripted_expert.py (deriving grasp height/aperture per piece type), so
a grasp pose can never silently drift out of sync with the actual rendered
geometry the way board_geometry.py already prevents for square positions.

All z-values are local heights above the piece body's own origin, which sits
at the board's playing surface (BOARD_TOP_Z in board_geometry.py) -- add that
to get a world z. Each piece is built as a base cylinder plus a "head" section
(capsule/sphere/box) that's often wider than the base, so most pieces have a
narrower, purely-cylindrical "neck" zone that's the natural place for a
parallel-jaw gripper to close around without catching on wider geometry above
or below. GRASP_Z is the center of that zone; GRASP_RADIUS is the piece radius
there (informs required gripper aperture -- all are well within the Robotiq
2F-85's ~85mm stroke, so aperture was never the binding constraint, safe
z-height was).
"""

BASE_RADIUS = 0.014  # shared cylindrical base radius for pawn/rook/knight/bishop

# (kind) -> {total_height, grasp_z, grasp_radius, notes}
# derived directly from assets/build_scene.py's add_piece() geom sizes/positions.
PIECE_DIMENSIONS = {
    "pawn": {
        "total_height": 0.050,  # cylinder [0, 0.03] + sphere head [0.026, 0.050]
        "grasp_z": 0.015,  # mid-cylinder, well clear of the sphere head (starts at 0.026)
        "grasp_radius": BASE_RADIUS,
    },
    "rook": {
        "total_height": 0.052,  # cylinder [0, 0.04] + box top [0.04, 0.052]
        "grasp_z": 0.020,  # mid-cylinder, clear of the box top
        "grasp_radius": BASE_RADIUS,
    },
    "knight": {
        "total_height": 0.070,  # cylinder [0, 0.036] + tilted asymmetric head from ~0.034
        "grasp_z": 0.015,
        "grasp_radius": BASE_RADIUS,
        "notes": (
            "FRAGILE: knight's head is a tilted, non-axisymmetric box starting at "
            "z~0.034, only ~2mm above the top of the safe cylindrical base (z=0.036). "
            "grasp_z=0.015 keeps clear margin, but there's very little room to move "
            "grasp_z upward without the gripper catching the asymmetric head -- if the "
            "scripted expert's approach trajectory has any lateral position error, "
            "knight is the piece most likely to get knocked over or grasped off-center. "
            "Every other piece type has 15mm+ of vertical margin in its safe zone; "
            "knight has effectively none."
        ),
    },
    "bishop": {
        "total_height": 0.083,  # cylinder [0,0.04] + capsule neck [0.019,0.081] (overlaps base) + sphere top [0.067,0.083]
        "grasp_z": 0.050,  # capsule-only neck zone [0.04, 0.067], narrower (r=0.011) than the base
        "grasp_radius": 0.011,
    },
    "queen": {
        "total_height": 0.104,  # cylinder [0,0.048] + capsule neck (overlaps base) + sphere top [0.076,0.104]
        "grasp_z": 0.060,  # capsule-only neck zone [0.048, 0.076], r=0.013
        "grasp_radius": 0.013,
    },
    "king": {
        "total_height": 0.114,  # cylinder [0,0.052] + capsule neck (overlaps base) + cross top [0.086,0.114]
        "grasp_z": 0.065,  # capsule-only neck zone [0.052, 0.086], r=0.013
        "grasp_radius": 0.013,
    },
}

# transit/lift clearance: must clear the tallest piece on the board (king) with
# margin, applied uniformly regardless of which piece is being moved, since
# transit has to clear every OTHER piece on the board too, not just its own.
MAX_PIECE_HEIGHT = max(p["total_height"] for p in PIECE_DIMENSIONS.values())
LIFT_CLEARANCE = MAX_PIECE_HEIGHT + 0.04  # +40mm margin above the tallest piece (king)
