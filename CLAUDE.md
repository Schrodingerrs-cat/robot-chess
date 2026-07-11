# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A UR5e arm + Robotiq 2F-85 gripper, in MuJoCo, plays chess: perceive the board, get a move
from Stockfish, physically execute it (pick/place, including captures). Imitation learning
first (Diffusion Policy and ACT, compared), then optional PPO fine-tuning. Every model here
is either an existing open-source architecture being fine-tuned onto this one task, or a
classical CV method (ArUco+homography) — nothing is trained from scratch, and there is no
VLA/VLM/world-model built from scratch.

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
before the next phase starts.

0. Scaffold (done)
1. Env setup — arm+gripper+board MJCF, workspace reach check (done)
2. Chess engine + rules integration (done)
3. Perception — ArUco localization, GroundingDINO+SAM2 fine-tune (stub only)
4. Scripted expert + demo generation (stub only)
5. IL training — Diffusion Policy vs ACT (stub only)
6. SmolVLA fine-tune, language-conditioned (stub only)
7. PPO RL fine-tune, optional, post-IL (stub only)
8. Full pipeline eval + ablation: learned vs GT-state vs classical-only perception (stub only)

Files under `policies/`, `perception/piece_detector/`, `rl/`, and most of `eval/` are
currently one-line docstring stubs describing what they'll hold — don't assume they contain
working code without checking.

## Environment setup

There's no requirements.txt/pyproject.toml yet — packages are installed ad hoc with
`pip install --user`. Python 3.10. Known-needed packages so far: `mujoco`, `mink`,
`python-chess` (imports as `chess`), plus their transitive deps (`qpsolvers`, `daqp`,
`glfw`, `pyopengl`, `etils`, `numpy`).

Two large third-party dependencies are gitignored (not committed) and must be fetched once
per environment — **fresh containers/sessions won't have them**:

```bash
assets/fetch_menagerie.sh    # clones mujoco_menagerie, stages mesh files at assets/assets/
engine/fetch_stockfish.sh    # downloads Stockfish 18 Linux binary to engine/bin/stockfish
```

`assets/fetch_menagerie.sh` accepts no args; `engine/fetch_stockfish.sh` takes an optional
CPU variant arg (default `bmi2`, matches most x86-64 since ~2013).

## Common commands

```bash
python assets/build_scene.py            # regenerate assets/ur5e_chess_scene.xml from MjSpec
python assets/workspace_reach_check.py   # mink IK reach check over all 64 squares
python engine/dry_run_game.py            # engine-vs-engine game, python-chess only, writes PGN
```

`assets/_scene_check.py` is a scratch script (renders a debug PNG to `/tmp`) — not part of
the pipeline, just useful for eyeballing the scene after editing `build_scene.py`.

## Architecture notes

**Scene composition (`assets/build_scene.py`)**: the UR5e and 2F-85 are two separate MJCF
files composed programmatically via `mujoco.MjSpec.attach()` at the arm's `attachment_site`,
rather than hand-edited XML — this keeps frame alignment exact when either upstream model
changes. Board + all 32 pieces are added the same way (`spec.worldbody.add_body(...)`),
positioned by a `square_center(file_idx, rank_idx)` helper so square<->world-frame math
stays in one place. `ur5e_chess_scene.xml` is generated output, not hand-maintained.

**Keyframe gotcha**: when `MjSpec.attach()`/compile grows the model (adding the gripper's
joints, then 32 piece free-joints), it pads the *inherited* `home` keyframe's `qpos` with
zeros for every newly-added joint rather than each joint's actual default (`qpos0`). For a
free joint this produces an invalid all-zero quaternion, which silently collapses that body
to the world origin — visually, every chess piece vanishes from the board. `build_scene.py`
works around this by compiling once, reading `model.qpos0`, and rewriting the keyframe as
`qpos0` with only the arm's home joint angles overridden — do this again if you add more
free-jointed bodies to the scene.

**Board/world frame**: board center is at world `(0.5, 0, 0)`, squares are 5cm, so the board
spans x:[0.325, 0.675], y:[-0.175, 0.175]. The arm base is at the world origin. `a1` is the
file=0,rank=0 corner. This layout is confirmed reachable (see Phase 1 status) — don't
relocate the board without re-running `workspace_reach_check.py`.

**Engine wrapper (`engine/stockfish_interface.py`)**: `StockfishEngine` is a context manager
around `chess.engine.SimpleEngine` (UCI). `legal_moves`/`validate_move`/`apply_move` are
plain `python-chess` rules checks (no engine subprocess needed); `best_move`/`evaluate` talk
to the Stockfish subprocess. Any code that executes a move against tracked game state should
go through `validate_move`/`apply_move` rather than pushing directly, since moves are meant
to eventually come from a physically-executed robot action, not just the rules engine.
