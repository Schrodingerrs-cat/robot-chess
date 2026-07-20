# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A UR5e arm + Robotiq 2F-85 gripper, in MuJoCo, plays chess: perceive the board, get a move
from Stockfish, physically execute it (pick/place, including captures). Imitation learning
first (Diffusion Policy and ACT, compared), then optional PPO fine-tuning. Every model here
is either an existing open-source architecture being fine-tuned onto this one task, or a
classical CV method (ArUco+homography) — nothing is trained from scratch, and there is no
VLA/VLM/world-model built from scratch.

Only **White's** moves go through the physical arm/gripper pipeline. Black's moves are
applied procedurally (reposition/remove the piece body directly in the scene — no IK, no
gripper), since the arm is a single-player stand-in for a human opponent, not a two-armed
setup.

## Fixed tech stack — do not substitute without asking

| Component | Choice |
|---|---|
| Simulator | MuJoCo (`mujoco`) |
| Arm | UR5e — `mujoco_menagerie/universal_robots_ur5e` |
| Gripper | Robotiq 2F-85 — `mujoco_menagerie/robotiq_2f85` |
| IK | `kevinzakka/mink` |
| Chess engine | Stockfish (UCI binary) + `python-chess` |
| Perception dataset | ChessReD (fallback: Roboflow Chess Pieces) |
| Detection/seg | GroundingDINO + SAM2, fine-tuned on ChessReD |
| Board localization | ArUco markers + homography (OpenCV) |
| IL policies | Diffusion Policy, ACT (`lerobot`) |
| VLA fine-tune | SmolVLA (`lerobot`) |
| RL fine-tune | PPO (`stable-baselines3`) |

If any of these turn out infeasible (licensing, missing assets, compute), stop and surface
the specific blocker instead of silently swapping in an alternative.

## Phased, checkpointed workflow

Work proceeds in the numbered phases below. **Stop and report status after each phase**
(what worked, what didn't, concrete numbers) rather than running the whole pipeline
unattended — this is how the project has been run so far and the user expects to confirm
before the next phase starts. This applies at the sub-step level too (e.g. within Phase 4:
scripted-expert core, then a small hand-picked test batch, then the full batch) — don't
jump ahead to the next step until the current one's result has been reported.

0. Scaffold (done)
1. Env setup — arm+gripper+board MJCF, workspace reach check (done, 64/64 squares reachable)
2. Chess engine + rules integration (done)
3. Perception — ArUco localization (done, ~3.5mm error), GroundingDINO+SAM2 fine-tune
   (done, 87.9% mean per-piece-type accuracy, worst category 50%)
4. Scripted expert + demo generation (done — see below)
5. IL training — Diffusion Policy vs ACT (stub only)
6. SmolVLA fine-tune, language-conditioned (stub only)
7. PPO RL fine-tune, optional, post-IL (stub only)
8. Full pipeline eval + ablation: learned vs GT-state vs classical-only perception (stub only)

Files under `policies/`, `rl/`, and most of `eval/` are currently one-line docstring stubs
describing what they'll hold — don't assume they contain working code without checking.

**Phase 4 status**: `demos/scripted_expert.py` (kinematic IK waypoint generation) is
working and smoke-tested. `demos/physics_executor.py` (driving those waypoints through
real MuJoCo contact physics) proved a genuine grasp-stability limit — see its module
docstring and the Architecture notes below — and **the project has since decided against
real contact physics for the IL/RL dataset-generation critical path**: the stated novel
contribution (Phase 8's perception-noise → manipulation-success ablation, on top of the
Phase 5/6 policy comparison) does not require resolving flat-rigid-pad-on-smooth-cylinder
grasp stability, which is itself an open problem in compliant manipulation research, not a
bug in this repo's control code. Dataset generation proceeds on `scripted_expert.py`'s
kinematic path (piece kinematically follows the gripper — a weld, not simulated grasp
dynamics). `physics_executor.py`'s gravity-compensation and adaptive-close fixes are kept
in the repo (legitimate control improvements, potentially relevant if Phase 7's RL
fine-tuning ever touches contact dynamics directly) but are off the Phase 4 critical path.
If a piece-geometry fix (e.g. a grasp collar) ever comes back on the table to revisit real
contact-physics grasping, decide the attempt budget up front rather than open-ended
iteration — this is the second time this exact detour has come up.

`demos/generate_dataset.py` is the dataset generator built on that weld path: it self-plays
a game via Stockfish, generates a dense per-tick weld trajectory for every White move, and
applies Black's moves procedurally, same as elsewhere. It's smoke-tested (two runs, 17/17
White episodes succeeded, sub-mm placement error, including two verified captures). A
single pick-and-place is the atomic action these IL policies are trained on — castling
(coordinated two-piece move) and promotion (needs a piece body the scene doesn't model)
don't fit that primitive, so both are **deliberately excluded from the physical dataset**
and applied procedurally instead when they occur in self-play (the game is not truncated or
capped short to dodge this, so the dataset isn't biased toward opening-adjacent positions).
This is the same scope-boundary call as the weld decision above, not an oversight — see the
module docstring and `dataset_info.json`'s generated note for the full framing. The next
step is scaling from the smoke-tested small batch to the full few-hundred-trajectory batch
TODO.md's Phase 4 describes.

## Environment setup

There's no requirements.txt/pyproject.toml yet — packages are installed ad hoc with
`pip install --user`. Python 3.10. Known-needed packages so far: `mujoco`, `mink`,
`python-chess` (imports as `chess`), plus their transitive deps (`qpsolvers`, `daqp`,
`glfw`, `pyopengl`, `etils`, `numpy`). Perception fine-tuning additionally needs `torch`,
`transformers`, `Pillow`.

Two large third-party dependencies are gitignored (not committed) and must be fetched once
per environment — **fresh containers/sessions won't have them**:

```bash
assets/fetch_menagerie.sh    # clones mujoco_menagerie, stages mesh files at assets/assets/
engine/fetch_stockfish.sh    # downloads Stockfish 18 Linux binary to engine/bin/stockfish
```

`assets/fetch_menagerie.sh` accepts no args; `engine/fetch_stockfish.sh` takes an optional
CPU variant arg (default `bmi2`, matches most x86-64 since ~2013).

The piece-detector fine-tune needs ChessReD's **chessred2k subset** (2078 images, not the
full 24.6GB set) staged at `perception/piece_detector/data/` (`annotations.json` +
`images/`) — this is gitignored and not fetched by any script; it has to be placed there
manually before `prepare_dataset.py` will run. `checkpoints/` under the same directory is
also gitignored (fine-tuned weights aren't committed).

## Common commands

```bash
python assets/build_scene.py            # regenerate assets/ur5e_chess_scene.xml from MjSpec
python assets/workspace_reach_check.py  # mink IK reach check + manipulability over all 64 squares
python engine/dry_run_game.py           # engine-vs-engine game, python-chess only, writes PGN

# perception
python perception/validate_localization.py            # ArUco homography vs sim ground truth
python perception/piece_detector/prepare_dataset.py    # ChessReD annotations.json -> train/val/test JSON
python perception/piece_detector/finetune_groundingdino.py [--epochs N] [--balanced-sampling]
python perception/piece_detector/evaluate.py           # per-piece-type accuracy on the test split

# demos (Phase 4)
python demos/scripted_expert.py     # kinematic IK smoke test: near-rank, far-rank, one capture
python demos/physics_executor.py    # same smoke set, executed through real contact physics
python demos/generate_dataset.py --n-games 20 --plies-per-game 25 --out demos/data/batch_v1  # full weld-based dataset batch
python demos/split_dataset.py --data-dir demos/data/batch_v1                                 # writes splits.json (by-game 70/15/15)
```

`assets/_scene_check.py` is a scratch script (renders a debug PNG to `/tmp`) — not part of
the pipeline, just useful for eyeballing the scene after editing `build_scene.py`.

## Architecture notes

**Scene composition (`assets/build_scene.py`)**: the UR5e and 2F-85 are two separate MJCF
files composed programmatically via `mujoco.MjSpec.attach()` at the arm's `attachment_site`,
rather than hand-edited XML — this keeps frame alignment exact when either upstream model
changes. Board + all 32 pieces are added the same way (`spec.worldbody.add_body(...)`),
positioned by a `square_center(file_idx, rank_idx)` helper so square<->world-frame math
stays in one place. `ur5e_chess_scene.xml` is generated output, not hand-maintained. Piece
bodies are named by their **starting** square (e.g. `white_pawn_e2` stays that name even
after e2-e4) — see `demos/board_state.py` below for how current occupancy is tracked
instead. `build_scene.py` also re-tunes physics the attached gripper needs but `attach()`
drops from the parent's `<option>` (elliptic contact cone, `impratio=10`) — re-check this
if you ever change how the gripper is attached.

**Keyframe gotcha**: when `MjSpec.attach()`/compile grows the model (adding the gripper's
joints, then 32 piece free-joints), it pads the *inherited* `home` keyframe's `qpos` with
zeros for every newly-added joint rather than each joint's actual default (`qpos0`). For a
free joint this produces an invalid all-zero quaternion, which silently collapses that body
to the world origin — visually, every chess piece vanishes from the board. `build_scene.py`
works around this by compiling once, reading `model.qpos0`, and rewriting the keyframe as
`qpos0` with only the arm's home joint angles overridden — do this again if you add more
free-jointed bodies to the scene.

**Board/world frame (`assets/board_geometry.py`)**: board center is at world `(0.5, 0, 0)`,
squares are 5cm, so the board spans x:[0.325, 0.675], y:[-0.175, 0.175]. The arm base is at
the world origin, seated behind rank 1 (White's side), not side-on. Rank maps to world X
(near/far from the arm) and file maps to world Y; `a1` is the file=0,rank=0 corner. This
layout is confirmed reachable (`workspace_reach_check.py`: 64/64 squares, worst
manipulability margin on the arm's *own* back rank, not the far rank — reaching in close
folds the elbow near a singularity) — don't relocate the board without re-running that
check. `board_geometry.py` also defines the off-board **graveyard** (`graveyard_slot`):
captured pieces are physically placed in a 2-row×8-col holding area per color, rather than
deleted/hidden, so captures read as a real move in any recorded demo.

**Piece grasp geometry (`assets/piece_geometry.py`)**: single source of truth for
per-piece-type grasp height/aperture (`PIECE_DIMENSIONS`), mirrored from
`build_scene.py`'s `add_piece()` geom sizes — keep both in sync if a piece's shape
changes. Pieces differ enough (pawn 50mm tall vs. king 114mm; knight's head is
asymmetric/tilted with almost no vertical grasp margin, unlike the axisymmetric others)
that a single fixed grasp pose doesn't work across all six types.

**Engine wrapper (`engine/stockfish_interface.py`)**: `StockfishEngine` is a context manager
around `chess.engine.SimpleEngine` (UCI). `legal_moves`/`validate_move`/`apply_move` are
plain `python-chess` rules checks (no engine subprocess needed); `best_move`/`evaluate` talk
to the Stockfish subprocess. Any code that executes a move against tracked game state should
go through `validate_move`/`apply_move` rather than pushing directly, since moves are meant
to eventually come from a physically-executed robot action, not just the rules engine.

**Piece detector (`perception/piece_detector/`)**: GroundingDINO is phrase-grounded — one
text prompt lists all 12 categories period-separated, and a box's label is the index of
which category-phrase it matched, not an arbitrary class id (`dataset.py`). `evaluate.py`
deliberately does **not** use the library's built-in text-label decoder: when the model's
confidence for two adjacent prompt categories both cross threshold, that decoder merges
them into one garbled string, which would silently read as near-zero accuracy for any
category still being learned. It instead reads mean per-category token-span probability
directly and argmaxes. Training history: knight and queen were confused with their
immediate prompt-neighbor (bishop, king) at ~100% initially; `--balanced-sampling`
(oversampling by rarest-category frequency) resolved both over two additional 8-epoch
rounds — final checkpoint clears 87.9% mean / 50% worst-category. If you retrain, don't
assume one flat run will hit these numbers again without the balanced-sampling passes.

**Scripted expert (`demos/scripted_expert.py`)**: pure-kinematic IK waypoint generator.
`ScriptedExpert.pick_and_place()` solves a `mink` IK sequence (approach → descend → close →
lift → transit → descend → open → retract) per move, warm-starting each solve from the
previous one so the path is continuous rather than independent jumps. This module has no
physics — it's the reference trajectory that `physics_executor.py` then tries to execute
for real.

**Physics executor (`demos/physics_executor.py`)**: drives the same waypoints through real
`mj_step` contact physics (originally chosen over a kinematic weld so demonstrations would
reflect actual grasp dynamics — **superseded, see the Phase 4 status note above**: the
project has since reverted to a kinematic weld for the actual IL/RL dataset-generation
critical path, keeping this module only for its two control fixes below). Two real bugs
were found and fixed here and are worth knowing about if you touch robot arm/gripper
control anywhere else in this repo:
  - The UR5e's `<position>` actuators are proportional-only (no integral/feedforward term),
    which left a persistent 10-30mm steady-state tracking error under gravity load — fixed
    with a gravity-compensation feedforward (`qfrc_applied = qfrc_bias`) applied **only** to
    the arm+gripper's own joint indices (never the piece free-joints, or gravity stops
    acting on pieces entirely).
  - The Robotiq gripper's position-servo finger actuator has no "I've got it, stop
    squeezing" logic — commanding max-close forever keeps clamping after first contact,
    which is what caused the still-open issue below. Fixed with an adaptive close that
    stops shortly after first firm contact instead.

  **Known unresolved limitation**: even with both fixes, only pawn and knight achieve a
  grasp that holds through a full lift (at a narrow closing-force margin). Queen, king,
  rook, and bishop — the heavier pieces — pop out of the gripper under lift-induced dynamic
  load at every closing-force margin tried, and raising pad friction made every piece type
  *worse*, not better (ruling out "insufficient friction" as the cause). This looks like a
  normal-direction contact-stability problem inherent to flat rigid parallel-jaw pads on a
  smooth cylindrical cross-section, not a tunable parameter — see the module docstring and
  `assets/build_scene.py`'s reverted-friction-change comment for the full diagnostic trail
  before re-attempting a fix. Don't silently loosen tolerances, gripper force, or board/pad
  geometry to paper over this; it needs a real decision (closed-loop grasp control, or a
  piece-geometry change like a grasp collar) before Phase 4's full batch can rely on it.

**Board-state tracking (`demos/board_state.py`)**: since scene body names are frozen at
their *starting* square, `BoardState` is the mutable square↔body mapping that both the
physics executor (physical White moves) and Black's procedural moves update as a game
progresses — mirrors `build_scene.py`'s `build_pieces()` layout exactly so names match the
compiled scene.

**Dataset generator (`demos/generate_dataset.py`)**: self-plays a game via Stockfish;
generates one episode per White move (dense per-tick trajectory from re-solving IK at
several interpolated points per phase, not the sparse 8 waypoints `scripted_expert.py`
itself produces) and applies Black's moves procedurally. The moved piece (and, for
captures, the captured piece first) is kinematically welded to the gripper's pinch site:
the offset between piece and pinch site is fixed from the *actual* solved pinch position at
the instant the gripper closes, held constant through lift/transit/descend, checked against
`WELD_SNAP_TOL` before the piece is snapped to the exact target square center at release —
a discrepancy beyond tolerance fails the episode rather than being silently absorbed by the
snap. Captures reuse `scripted_expert.py`'s existing capture-then-move waypoint ordering
unchanged (two weld cycles logged back-to-back in one episode: captured piece to graveyard,
then capturing piece to its target) — confirmed working via two recaptures in a smoke run,
not just assumed. Castling and promotion are out of scope for the physical dataset — see
the module docstring and the Phase 4 status note above for why — and are applied
procedurally like Black's moves when they occur in self-play, without truncating the game.
Every episode's `.npz` and the batch's `dataset_info.json` both say
`demonstration_type="kinematic_weld"` explicitly, so it's never ambiguous later why a policy
trained on this data has never seen grasp-slip/recovery behavior.

**Phase 4 is done.** The full batch (`demos/data/batch_v1/`, gitignored like all of
`demos/data/`) is 20 self-played games / 500 plies, 247 White episodes, **247/247 (100%)
succeeded**, placement error 0.11–1.70mm (mean 0.64mm), 44 captures, 13 castling skips, 0
promotion skips — all applied procedurally as designed, game not truncated around them.
`demos/split_dataset.py` produces the train/val/test manifest (`splits.json` alongside
`dataset_info.json`): split **by game**, not by episode, since positions a few plies apart
in the same game are correlated near-duplicates and an episode-level shuffle would leak
train information into the held-out sets — same discipline as the perception dataset's
chessred2k splits. Default 70/15/15 by game count gives batch_v1 a 14/3/3 game split (173/
37/37 episodes); all five non-king piece kinds appear in every split (no king move was
non-castling in this batch, so `king` has zero episodes across the board — worth knowing
before Phase 5 training, not a splitting bug).
