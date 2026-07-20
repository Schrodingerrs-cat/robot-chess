"""Weld-based demonstration dataset generator (Phase 4 critical path).

Self-plays a game via Stockfish; for every White move, generates a dense
per-tick kinematic trajectory using demos.scripted_expert's IK solver, with
the moved (and, for captures, the captured) piece kinematically WELDED to
the gripper's pinch site during the carry phases (grasp closed -> lift ->
transit -> descend_to_place), rather than driven through real contact
physics -- see CLAUDE.md's Phase 4 status note for why: physics_executor.py
proved a genuine grasp-stability limit for the four heavier piece types that
this project's novel contribution (Phase 8's perception-noise ->
manipulation-success ablation) doesn't need resolved. Black's moves are
applied procedurally (teleported, no IK/gripper), same as everywhere else in
this repo.

Weld mechanics: the moved piece's offset from the gripper's pinch site is
fixed at the instant the gripper closes, computed from the ACTUALLY solved
pinch-site position (not an assumed nominal one), so small IK residuals get
carried along rather than silently discarded. That offset is held constant
through lift/transit/descend, so the logged piece trajectory reflects a real
(if idealized) carry path, not a teleport. At release, the pre-snap
pinch-tracked position is checked against WELD_SNAP_TOL *before* the piece
is snapped to the exact target square center -- a discrepancy beyond
tolerance fails the episode rather than being silently absorbed by the snap,
so a real IK problem doesn't masquerade as a clean demonstration.

Known gaps, deliberately out of scope: a single pick-and-place is the atomic
action this project's IL policies are trained on. Castling (a coordinated
two-piece move -- a different action-space shape, with king/rook trajectory
collision to reason about) and promotion (needs a piece body the scene
doesn't model -- a promoted queen from a pawn's body) don't fit that
primitive, so a White move of either kind is applied procedurally instead
(like Black's moves), logged as such rather than silently dropped, so the
self-play game stays legal/coherent even though that ply contributes no
training episode. This is consistent with the project being a manipulation-
policy study, not a complete robotic chess-playing system -- extending the
action primitive to cover two-piece moves would be genuine scope creep, the
same call already made on real-contact-physics grasping (see CLAUDE.md's
Phase 4 status note). Self-play length is NOT capped to dodge this: the game
plays on past a castling/promotion ply exactly as it would otherwise, so the
dataset isn't quietly biased toward opening-adjacent positions.

Every saved episode's metadata says demonstration_type="kinematic_weld"
explicitly, and dataset_info.json repeats the caveat at the dataset level:
policies trained on this data never see grasp-failure/recovery behavior,
since a welded piece cannot slip. Converting these raw per-move records into
whatever exact tensor layout Diffusion Policy / ACT / SmolVLA each expect is
Phase 5/6's job, not this module's -- policies/ is still stub-only.

Run: python demos/generate_dataset.py --n-games 1 --plies-per-game 12 --out demos/data/smoke

Game diversity: a single self-play game from Stockfish at a fixed time limit is
close to deterministic run-to-run (search timing jitter can occasionally flip a
move, but that's incidental, not a designed diversity mechanism -- confirmed by
diffing two early smoke runs, which shared their first several plies before
diverging). To get positional diversity across a batch instead of one game's
positions logged N times, `--opening-random-plies` picks a uniformly random
legal move (instead of Stockfish's choice) for that many plies at the start of
EVERY game -- both colors, so White and Black positions both diversify -- before
handing off to Stockfish for the rest of the game. `--seed` makes a batch's
openings reproducible; omitted, a fresh seed is drawn and printed so a specific
run can still be reproduced after the fact.
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field

import chess
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(HERE, "..", "assets")
ENGINE_DIR = os.path.join(HERE, "..", "engine")
sys.path.insert(0, ASSETS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, HERE)
from board_geometry import BOARD_TOP_Z, graveyard_slot, square_center  # noqa: E402
from piece_geometry import LIFT_CLEARANCE, PIECE_DIMENSIONS  # noqa: E402
from scripted_expert import GRIPPER_CLOSED, GRIPPER_OPEN, SCENE_XML, ScriptedExpert, square_to_idx  # noqa: E402
from board_state import BoardState  # noqa: E402
from stockfish_interface import StockfishEngine  # noqa: E402

DEMO_TYPE = "kinematic_weld"
DATASET_NOTE = (
    "Every episode here uses a kinematic weld: the moved piece is rigidly "
    "attached to the gripper's pinch site during the carry phases, not driven "
    "through simulated contact/grasp dynamics. Real contact physics was "
    "attempted (see demos/physics_executor.py) and found a genuine "
    "grasp-stability limit for queen/king/rook/bishop under a flat rigid "
    "parallel-jaw grip on a smooth cylinder -- an open problem in compliant "
    "manipulation research, not a bug in this repo's control code -- and this "
    "project's novel contribution (perception-noise -> manipulation-success "
    "ablation, Phase 8) does not require resolving it. Practically: policies "
    "trained on this data will never see grasp-slip or recovery behavior, "
    "since a welded piece cannot slip. See CLAUDE.md's Phase 4 status note. "
    "Separately: castling and promotion are excluded from this dataset. A "
    "single pick-and-place is the atomic action these IL policies are trained "
    "on; castling needs a coordinated two-piece action and promotion needs a "
    "piece body this scene doesn't model, so both are applied procedurally "
    "(not physically) when they occur in self-play -- the game continues "
    "normally afterward, it is not truncated, so the dataset isn't biased "
    "toward opening-adjacent positions. This is a deliberate scope boundary "
    "consistent with the project being a manipulation-policy study, not a "
    "complete robotic chess-playing system, not an oversight."
)

WELD_SNAP_TOL = 0.008  # 8mm: pre-snap pinch-tracked xy vs target center, beyond this the episode fails rather than being silently corrected

PHASE_SUBSTEPS = {
    "approach_source": 8,
    "descend_to_grasp": 4,
    "lift": 4,
    "transit": 8,
    "descend_to_place": 4,
    "retract": 4,
}
DWELL_TICKS = 3  # close_gripper / open_gripper hold ticks (no motion)
TICK_DT = 0.05  # nominal per-tick timestamp spacing -- logging metadata only, no dynamics is simulated


@dataclass
class Tick:
    t: float
    phase: str
    qpos: list  # arm+gripper joint positions only (not the full model q -- see n_arm_gripper)
    gripper: float
    ee_pos: list
    welded_piece: str | None
    piece_pos: list | None  # world xyz of the welded piece this tick, if any


@dataclass
class MoveEpisode:
    uci: str
    piece_kind: str
    source_square: str
    target_square: str
    is_capture: bool
    captured_kind: str | None
    captured_color: str | None
    success: bool
    failure_reason: str | None
    placement_error: float | None
    ticks: list = field(default_factory=list)


def _captured_square(board_before: chess.Board, move: chess.Move) -> int:
    if board_before.is_en_passant(move):
        return chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
    return move.to_square


class DatasetGenerator:
    def __init__(self, scene_xml: str = SCENE_XML):
        self.expert = ScriptedExpert(scene_xml)
        self.model = self.expert.model
        self.pinch_site_id = self.model.site("/pinch").id
        self.board = BoardState()
        self._t = 0.0

        first_free = next(j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        self.n_arm_gripper = self.model.jnt_qposadr[first_free]

    def reset_arm(self) -> None:
        self.expert.reset()

    def reset_game(self) -> None:
        """Fresh BoardState + every piece body snapped back to its starting
        square -- needed between games in a batch since captures move bodies
        into the graveyard and moves leave others off their starting square."""
        self.board = BoardState()
        for body, square in self.board.body_to_square.items():
            self.set_piece_pose(body, square_center(*square_to_idx(square))[:2])

    def _pinch_pos(self) -> np.ndarray:
        return self.expert.configuration.data.site_xpos[self.pinch_site_id].copy()

    def set_piece_pose(self, body_name: str, xy, z: float = BOARD_TOP_Z) -> None:
        data = self.expert.configuration.data
        body = self.model.body(body_name)
        joint = self.model.joint(body.jntadr[0])
        qadr = joint.qposadr[0]
        data.qpos[qadr : qadr + 3] = [xy[0], xy[1], z]
        data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, data)

    def _emit(self, phase: str, gripper: float, ticks: list, weld) -> None:
        self._t += TICK_DT
        piece_pos = None
        welded_name = None
        if weld is not None:
            welded_name, offset = weld
            piece_pos = (self._pinch_pos() + offset).tolist()
        ticks.append(
            Tick(
                t=self._t,
                phase=phase,
                qpos=self.expert.configuration.q[: self.n_arm_gripper].copy().tolist(),
                gripper=gripper,
                ee_pos=self._pinch_pos().tolist(),
                welded_piece=welded_name,
                piece_pos=piece_pos,
            )
        )

    def _segment(self, start_pos, end_pos, gripper: float, phase: str, n_sub: int, ticks: list, weld=None) -> bool:
        """Re-solves IK at n_sub interpolated Cartesian points rather than
        interpolating joint angles between the two endpoints -- keeps the
        gripper exactly vertical along the way, same reasoning as
        physics_executor.py's _move_ee_to (an off-vertical grasp mid-carry
        would be a bad demonstration to train on regardless of whether
        contact physics is involved)."""
        start_pos = np.array(start_pos, dtype=float)
        end_pos = np.array(end_pos, dtype=float)
        ok_all = True
        for i in range(1, n_sub + 1):
            alpha = i / n_sub
            pos = start_pos + alpha * (end_pos - start_pos)
            _, ok, _, _ = self.expert.solve_to(tuple(pos), gripper, phase)
            ok_all = ok_all and ok
            self._emit(phase, gripper, ticks, weld)
        return ok_all

    def _dwell(self, gripper: float, phase: str, ticks: list, weld=None) -> None:
        for _ in range(DWELL_TICKS):
            self._emit(phase, gripper, ticks, weld)

    def _weld_pick_and_place(self, body_name: str, source_xy, target_xy, piece_kind: str, ticks: list):
        grasp_z = PIECE_DIMENSIONS[piece_kind]["grasp_z"]
        sx, sy = source_xy
        tx, ty = target_xy
        top = BOARD_TOP_Z

        ok = True
        ok &= self._segment(self._pinch_pos(), (sx, sy, top + LIFT_CLEARANCE), GRIPPER_OPEN, "approach_source", PHASE_SUBSTEPS["approach_source"], ticks)
        ok &= self._segment((sx, sy, top + LIFT_CLEARANCE), (sx, sy, top + grasp_z), GRIPPER_OPEN, "descend_to_grasp", PHASE_SUBSTEPS["descend_to_grasp"], ticks)

        # weld offset from the ACTUAL solved pinch position at grasp, not the nominal target -- see module docstring
        grasp_pinch = self._pinch_pos()
        offset = np.array([sx, sy, top]) - grasp_pinch
        weld = (body_name, offset)

        self._dwell(GRIPPER_CLOSED, "close_gripper", ticks, weld=weld)
        ok &= self._segment((sx, sy, top + grasp_z), (sx, sy, top + LIFT_CLEARANCE), GRIPPER_CLOSED, "lift", PHASE_SUBSTEPS["lift"], ticks, weld=weld)
        ok &= self._segment((sx, sy, top + LIFT_CLEARANCE), (tx, ty, top + LIFT_CLEARANCE), GRIPPER_CLOSED, "transit", PHASE_SUBSTEPS["transit"], ticks, weld=weld)
        ok &= self._segment((tx, ty, top + LIFT_CLEARANCE), (tx, ty, top + grasp_z), GRIPPER_CLOSED, "descend_to_place", PHASE_SUBSTEPS["descend_to_place"], ticks, weld=weld)

        pre_snap_pos = self._pinch_pos() + offset
        placement_error = float(np.linalg.norm(pre_snap_pos[:2] - np.array([tx, ty])))

        self._dwell(GRIPPER_OPEN, "open_gripper", ticks, weld=None)
        self.set_piece_pose(body_name, (tx, ty), top)  # snap to exact target center -- see module docstring

        ok &= self._segment((tx, ty, top + grasp_z), (tx, ty, top + LIFT_CLEARANCE), GRIPPER_OPEN, "retract", PHASE_SUBSTEPS["retract"], ticks)

        if placement_error > WELD_SNAP_TOL:
            return False, f"weld_drift={placement_error * 1000:.2f}mm", placement_error
        if not ok:
            return False, "ik_failed", placement_error
        return True, None, placement_error

    def generate_move_episode(self, move: chess.Move, board_before: chess.Board) -> MoveEpisode:
        from_sq, to_sq = chess.square_name(move.from_square), chess.square_name(move.to_square)
        piece_kind = chess.piece_name(board_before.piece_at(move.from_square).piece_type)

        is_capture = board_before.is_capture(move)
        is_en_passant = board_before.is_en_passant(move)
        captured_kind = captured_color = captured_sq_name = captured_body = None
        if is_capture:
            cap_sq = _captured_square(board_before, move)
            captured_sq_name = chess.square_name(cap_sq)
            captured_piece = board_before.piece_at(cap_sq)
            captured_kind = chess.piece_name(captured_piece.piece_type)
            captured_color = "white" if captured_piece.color == chess.WHITE else "black"
            captured_body = self.board.body_at(captured_sq_name)

        ticks: list[Tick] = []
        overall_ok = True
        failure_reason = None
        placement_error = None

        if is_capture:
            grave_idx = self.board.next_graveyard_index(captured_color)
            grave_xy = graveyard_slot(captured_color, grave_idx)[:2]
            cap_xy = square_center(*square_to_idx(captured_sq_name))[:2]
            ok, reason, _ = self._weld_pick_and_place(captured_body, cap_xy, grave_xy, captured_kind, ticks)
            if not ok:
                overall_ok, failure_reason = False, f"capture_removal:{reason}"

        moving_body = self.board.body_at(from_sq)
        src_xy = square_center(*square_to_idx(from_sq))[:2]
        tgt_xy = square_center(*square_to_idx(to_sq))[:2]
        ok, reason, placement_error = self._weld_pick_and_place(moving_body, src_xy, tgt_xy, piece_kind, ticks)
        if not ok:
            overall_ok = False
            failure_reason = failure_reason or reason

        # a failed physical move shouldn't silently update tracked game state --
        # mirrors a real robot, where a botched pick doesn't move the piece
        if overall_ok:
            if is_capture and is_en_passant:
                del self.board.square_to_body[captured_sq_name]
                del self.board.body_to_square[captured_body]
            self.board.apply_move(from_sq, to_sq)

        return MoveEpisode(
            uci=move.uci(),
            piece_kind=piece_kind,
            source_square=from_sq,
            target_square=to_sq,
            is_capture=is_capture,
            captured_kind=captured_kind,
            captured_color=captured_color,
            success=overall_ok,
            failure_reason=failure_reason,
            placement_error=placement_error,
            ticks=ticks,
        )

    def apply_procedural(self, move: chess.Move, board_before: chess.Board) -> None:
        """Teleport-only move: no IK/gripper/weld. Used for Black's moves
        (Black is a procedural stand-in for a human opponent, never
        physically executed -- project decision) and for White castling/
        promotion, which this pass can't express as a single pick-and-place
        (see module docstring)."""
        from_sq, to_sq = chess.square_name(move.from_square), chess.square_name(move.to_square)
        is_capture = board_before.is_capture(move)
        is_en_passant = board_before.is_en_passant(move)
        is_castling = board_before.is_castling(move)

        if is_capture:
            cap_sq = _captured_square(board_before, move)
            captured_sq_name = chess.square_name(cap_sq)
            captured_piece = board_before.piece_at(cap_sq)
            captured_color = "white" if captured_piece.color == chess.WHITE else "black"
            captured_body = self.board.body_at(captured_sq_name)
            grave_idx = self.board.next_graveyard_index(captured_color)
            self.set_piece_pose(captured_body, graveyard_slot(captured_color, grave_idx)[:2])
            if is_en_passant:
                del self.board.square_to_body[captured_sq_name]
                del self.board.body_to_square[captured_body]

        moving_body = self.board.body_at(from_sq)
        self.board.apply_move(from_sq, to_sq)
        self.set_piece_pose(moving_body, square_center(*square_to_idx(to_sq))[:2])

        if is_castling:
            king_rank = chess.square_rank(move.from_square)
            kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
            rook_from = chess.square_name(chess.square(7 if kingside else 0, king_rank))
            rook_to = chess.square_name(chess.square(5 if kingside else 3, king_rank))
            rook_body = self.board.body_at(rook_from)
            self.board.apply_move(rook_from, rook_to)
            self.set_piece_pose(rook_body, square_center(*square_to_idx(rook_to))[:2])


def _save_episode(path: str, episode: MoveEpisode) -> None:
    ticks = episode.ticks
    meta = dict(
        uci=episode.uci,
        piece_kind=episode.piece_kind,
        source_square=episode.source_square,
        target_square=episode.target_square,
        is_capture=episode.is_capture,
        captured_kind=episode.captured_kind or "",
        captured_color=episode.captured_color or "",
        success=episode.success,
        failure_reason=episode.failure_reason or "",
        placement_error=episode.placement_error if episode.placement_error is not None else -1.0,
        demonstration_type=DEMO_TYPE,
    )
    np.savez(
        path,
        t=np.array([tk.t for tk in ticks]),
        phase=np.array([tk.phase for tk in ticks]),
        qpos=np.array([tk.qpos for tk in ticks]),
        gripper=np.array([tk.gripper for tk in ticks]),
        ee_pos=np.array([tk.ee_pos for tk in ticks]),
        welded_piece=np.array([tk.welded_piece or "" for tk in ticks]),
        piece_pos=np.array([tk.piece_pos if tk.piece_pos is not None else [np.nan, np.nan, np.nan] for tk in ticks]),
        meta=json.dumps(meta),
    )


def play_game(
    gen: DatasetGenerator,
    engine: StockfishEngine,
    out_dir: str,
    game_idx: int,
    max_plies: int,
    time_per_move: float,
    rng: random.Random,
    opening_random_plies: int,
) -> tuple[list[dict], dict]:
    """Plays one self-play game and returns (per-White-episode metadata,
    game-level counters). `opening_random_plies` plies at the start (both
    colors) are picked uniformly at random from legal moves instead of from
    Stockfish, so repeated games don't collapse onto the same position
    sequence -- see module docstring."""
    gen.reset_game()
    board = chess.Board()
    episodes_meta = []
    counters = {"white_castling_skips": 0, "white_promotion_skips": 0, "black_moves": 0, "captures": 0}

    ply = 0
    while not board.is_game_over() and ply < max_plies:
        mover = "white" if board.turn == chess.WHITE else "black"
        if ply < opening_random_plies:
            move = rng.choice(list(board.legal_moves))
        else:
            move = engine.best_move(board, time_limit=time_per_move).move
        board_before = board.copy()
        unsupported = board_before.is_castling(move) or move.promotion is not None

        if mover == "white" and not unsupported:
            gen.reset_arm()
            episode = gen.generate_move_episode(move, board_before)
            path = os.path.join(out_dir, f"g{game_idx:03d}_ep_{ply:04d}_{episode.uci}.npz")
            _save_episode(path, episode)
            if episode.is_capture:
                counters["captures"] += 1
            episodes_meta.append(
                {
                    "game": game_idx,
                    "ply": ply,
                    "uci": episode.uci,
                    "piece_kind": episode.piece_kind,
                    "is_capture": episode.is_capture,
                    "success": episode.success,
                    "failure_reason": episode.failure_reason,
                    "placement_error_mm": (episode.placement_error * 1000) if episode.placement_error is not None else None,
                    "n_ticks": len(episode.ticks),
                    "file": os.path.basename(path),
                    "demonstration_type": DEMO_TYPE,
                }
            )
            status = "OK" if episode.success else f"FAILED({episode.failure_reason})"
            print(f"g{game_idx:03d} {ply:4d} WHITE {episode.uci:6s} {episode.piece_kind:6s} {status} ({len(episode.ticks)} ticks)")
        else:
            gen.apply_procedural(move, board_before)
            if mover == "white":
                reason = "castling" if board_before.is_castling(move) else "promotion"
                counters["white_castling_skips" if reason == "castling" else "white_promotion_skips"] += 1
                print(f"g{game_idx:03d} {ply:4d} WHITE {board_before.san(move):8s} [procedural -- {reason} not yet physically supported]")
            else:
                counters["black_moves"] += 1
                print(f"g{game_idx:03d} {ply:4d} BLACK {board_before.san(move):8s} [procedural]")

        board.push(move)
        ply += 1

    counters["plies"] = ply
    counters["finished"] = board.is_game_over()
    counters["final_fen"] = board.fen()
    counters["opening_moves"] = [m.uci() for m in board.move_stack[:opening_random_plies]]
    return episodes_meta, counters


def self_play_batch(
    n_games: int,
    plies_per_game: int,
    out_dir: str,
    time_per_move: float = 0.1,
    seed: int | None = None,
    opening_random_plies: int = 4,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    print(f"seed={seed} n_games={n_games} plies_per_game={plies_per_game} opening_random_plies={opening_random_plies}")
    rng = random.Random(seed)
    gen = DatasetGenerator()
    all_episodes = []
    all_game_counters = []

    with StockfishEngine() as engine:
        for g in range(n_games):
            episodes_meta, counters = play_game(gen, engine, out_dir, g, plies_per_game, time_per_move, rng, opening_random_plies)
            all_episodes.extend(episodes_meta)
            all_game_counters.append(counters)
            n_ok = sum(e["success"] for e in episodes_meta)
            print(f"=== game {g:03d}: opening {counters['opening_moves']}, {len(episodes_meta)} White episodes ({n_ok} ok), "
                  f"{counters['plies']} plies, {'finished' if counters['finished'] else 'ply-capped'} ===")

    unique_openings = len({tuple(c["opening_moves"]) for c in all_game_counters})
    errors_mm = [e["placement_error_mm"] for e in all_episodes if e["placement_error_mm"] is not None]
    summary = {
        "n_games": n_games,
        "unique_openings": unique_openings,
        "seed": seed,
        "opening_random_plies": opening_random_plies,
        "total_plies": sum(c["plies"] for c in all_game_counters),
        "white_episodes": len(all_episodes),
        "white_episodes_success": sum(e["success"] for e in all_episodes),
        "captures": sum(c["captures"] for c in all_game_counters),
        "white_castling_skips": sum(c["white_castling_skips"] for c in all_game_counters),
        "white_promotion_skips": sum(c["white_promotion_skips"] for c in all_game_counters),
        "black_moves": sum(c["black_moves"] for c in all_game_counters),
        "placement_error_mm": {
            "n": len(errors_mm),
            "min": min(errors_mm) if errors_mm else None,
            "mean": sum(errors_mm) / len(errors_mm) if errors_mm else None,
            "max": max(errors_mm) if errors_mm else None,
        },
    }

    with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
        json.dump(
            {
                "demonstration_type": DEMO_TYPE,
                "note": DATASET_NOTE,
                "summary": summary,
                "games": all_game_counters,
                "episodes": all_episodes,
            },
            f,
            indent=2,
        )

    print(f"\n{summary['white_episodes']} White episodes generated ({summary['white_episodes_success']} succeeded), "
          f"{summary['total_plies']} plies total across {n_games} games ({unique_openings} unique openings).")
    print(f"dataset written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=1)
    parser.add_argument("--plies-per-game", type=int, default=12)
    parser.add_argument("--out", type=str, default=os.path.join(HERE, "data", "smoke"))
    parser.add_argument("--time-per-move", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--opening-random-plies", type=int, default=4)
    args = parser.parse_args()
    self_play_batch(args.n_games, args.plies_per_game, args.out, args.time_per_move, args.seed, args.opening_random_plies)


if __name__ == "__main__":
    main()
