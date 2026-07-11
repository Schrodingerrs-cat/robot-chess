"""Compose UR5e + Robotiq 2F85 + chessboard + 32 pieces into one MJCF scene.

Uses mujoco.MjSpec to attach the gripper to the arm's flange (`attachment_site`)
rather than hand-editing XML, so frame alignment is exact. Chessboard is a flat
checker-textured slab; pieces are basic geometric Staunton-style shapes (box/
cylinder/sphere primitives) per project spec -- visual fidelity isn't the goal.

Run: python assets/build_scene.py
Writes: assets/ur5e_chess_scene.xml
"""

import mujoco

HERE = __file__.rsplit("/", 1)[0]
UR5E_XML = f"{HERE}/menagerie/universal_robots_ur5e/ur5e.xml"
GRIPPER_XML = f"{HERE}/menagerie/robotiq_2f85/2f85.xml"
OUT_XML = f"{HERE}/ur5e_chess_scene.xml"

BOARD_CENTER = (0.5, 0.0, 0.0)  # table-frame xy, board sits on table (z=0 slab top)
SQUARE = 0.05  # 5cm squares -> 0.4m x 0.4m board
BOARD_THICKNESS = 0.02
SQUARE_TOP_Z = BOARD_THICKNESS  # top surface of board slab

FILES = "abcdefgh"
RANKS = "12345678"


def square_center(file_idx: int, rank_idx: int) -> tuple[float, float, float]:
    """file_idx, rank_idx in [0,7]. a1 at (file=0,rank=0), nearest-left corner."""
    origin_x = BOARD_CENTER[0] - 3.5 * SQUARE
    origin_y = BOARD_CENTER[1] - 3.5 * SQUARE
    x = origin_x + file_idx * SQUARE
    y = origin_y + rank_idx * SQUARE
    return x, y, SQUARE_TOP_Z


def build_board(spec: mujoco.MjSpec) -> None:
    spec.add_texture(
        name="board_checker",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.85, 0.72, 0.55],
        rgb2=[0.35, 0.22, 0.12],
        width=256,
        height=256,
    )
    spec.add_material(name="board_mat", textures=["", "board_checker", "", "", "", "", "", "", "", ""]).texrepeat = [
        8,
        8,
    ]

    board_body = spec.worldbody.add_body(
        name="chessboard",
        pos=[BOARD_CENTER[0], BOARD_CENTER[1], BOARD_THICKNESS / 2],
    )
    board_body.add_geom(
        name="board_slab",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[4 * SQUARE, 4 * SQUARE, BOARD_THICKNESS / 2],
        material="board_mat",
    )
    # thin border frame (visual only) so the board reads clearly in renders
    border_mat_rgba = [0.25, 0.15, 0.08, 1]
    spec.add_material(name="border_mat", rgba=border_mat_rgba)
    for i, (dx, dy, sx, sy) in enumerate(
        [
            (0, 4.15 * SQUARE, 4.15 * SQUARE, 0.15 * SQUARE),
            (0, -4.15 * SQUARE, 4.15 * SQUARE, 0.15 * SQUARE),
            (4.15 * SQUARE, 0, 0.15 * SQUARE, 4.15 * SQUARE),
            (-4.15 * SQUARE, 0, 0.15 * SQUARE, 4.15 * SQUARE),
        ]
    ):
        board_body.add_geom(
            name=f"border_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[dx, dy, 0],
            size=[sx, sy, BOARD_THICKNESS / 2],
            material="border_mat",
        )


# piece geometry: (kind, height, rgba-agnostic; color set per-piece by side)
def add_piece(spec: mujoco.MjSpec, name: str, kind: str, pos, rgba) -> None:
    body = spec.worldbody.add_body(name=name, pos=[pos[0], pos[1], pos[2]])
    body.add_freejoint(name=f"{name}_free")
    base_r = 0.014

    if kind == "pawn":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[base_r, 0.015, 0], pos=[0, 0, 0.015], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0], pos=[0, 0, 0.038], rgba=rgba)
    elif kind == "rook":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[base_r, 0.02, 0], pos=[0, 0, 0.02], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.014, 0.014, 0.006], pos=[0, 0, 0.046], rgba=rgba)
    elif kind == "knight":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[base_r, 0.018, 0], pos=[0, 0, 0.018], rgba=rgba)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.009, 0.02, 0.016],
            pos=[0.006, 0, 0.05],
            quat=[0.94, 0, 0.34, 0],
            rgba=rgba,
        )
    elif kind == "bishop":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[base_r, 0.02, 0], pos=[0, 0, 0.02], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.011, 0.02, 0], pos=[0, 0, 0.05], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.008, 0, 0], pos=[0, 0, 0.075], rgba=rgba)
    elif kind == "queen":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[base_r + 0.002, 0.024, 0], pos=[0, 0, 0.024], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.013, 0.025, 0], pos=[0, 0, 0.06], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.014, 0, 0], pos=[0, 0, 0.09], rgba=rgba)
    elif kind == "king":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[base_r + 0.002, 0.026, 0], pos=[0, 0, 0.026], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.013, 0.028, 0], pos=[0, 0, 0.065], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.014, 0.004, 0.004], pos=[0, 0, 0.1], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, 0.004, 0.014], pos=[0, 0, 0.1], rgba=rgba)


BACK_RANK = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
WHITE_RGBA = [0.92, 0.92, 0.85, 1]
BLACK_RGBA = [0.12, 0.12, 0.12, 1]


def build_pieces(spec: mujoco.MjSpec) -> None:
    for f in range(8):
        x, y, z = square_center(f, 1)
        add_piece(spec, f"white_pawn_{FILES[f]}2", "pawn", (x, y, z), WHITE_RGBA)
        x, y, z = square_center(f, 6)
        add_piece(spec, f"black_pawn_{FILES[f]}7", "pawn", (x, y, z), BLACK_RGBA)

        x, y, z = square_center(f, 0)
        add_piece(spec, f"white_{BACK_RANK[f]}_{FILES[f]}1", BACK_RANK[f], (x, y, z), WHITE_RGBA)
        x, y, z = square_center(f, 7)
        add_piece(spec, f"black_{BACK_RANK[f]}_{FILES[f]}8", BACK_RANK[f], (x, y, z), BLACK_RGBA)


def main() -> None:
    ur5e = mujoco.MjSpec.from_file(UR5E_XML)
    gripper = mujoco.MjSpec.from_file(GRIPPER_XML)

    flange_site = ur5e.site("attachment_site")
    ur5e.attach(gripper, site=flange_site)

    # attach() keeps the parent's <option>, silently dropping 2f85's elliptic-cone /
    # impratio=10 tuning that its finger-pad contacts need for stable grasps.
    ur5e.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    ur5e.option.impratio = 10

    build_board(ur5e)
    build_pieces(ur5e)

    # ground plane + lighting for rendering/reach checks
    ur5e.worldbody.add_light(name="top", pos=[0, 0, 2], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    ur5e.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.3, 0.3, 0.32, 1],
    )

    # attach()/compile pad the inherited "home" keyframe's qpos with zeros for
    # every joint added after it (gripper + all 32 piece free joints), instead of
    # each joint's actual default -- collapsing every piece to the origin with an
    # invalid zero quaternion. Rebuild the keyframe from qpos0, keeping only the
    # arm's home angles overridden.
    model = ur5e.compile()
    home_arm = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
    qpos = model.qpos0.copy()
    qpos[:6] = home_arm
    key = ur5e.key("home")
    key.qpos = qpos.tolist()

    xml = ur5e.to_xml()
    with open(OUT_XML, "w") as f:
        f.write(xml)
    print(f"wrote {OUT_XML}")

    # sanity compile
    model = mujoco.MjModel.from_xml_path(OUT_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print(f"compiled ok: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms")
    pawn_pos = data.xpos[model.body("white_pawn_a2").id]
    print(f"white_pawn_a2 pos after keyframe reset: {pawn_pos}")


if __name__ == "__main__":
    main()
