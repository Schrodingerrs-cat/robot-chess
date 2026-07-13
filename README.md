# robot-chess

Sim manipulator arm plays chess: perceive board, get move from engine, execute pick-place (incl. captures). IL-first (Diffusion Policy / ACT / SmolVLA), optional PPO fine-tune.

Not a from-scratch VLA/VLM/world-model project — grounds existing open-source policy + perception architectures onto one task.

## Stack

| Component | Choice |
|---|---|
| Simulator | MuJoCo (`mujoco`) |
| Arm | UR5e — `mujoco_menagerie/universal_robots_ur5e` |
| Gripper | Robotiq 2F-85 — `mujoco_menagerie/robotiq_2f85` |
| IK | `kevinzakka/mink` |
| Chess engine | Stockfish + `python-chess` |
| Perception dataset | ChessReD (fallback: Roboflow Chess Pieces) |
| Detection/seg | GroundingDINO + SAM2, fine-tuned on ChessReD |
| Board localization | ArUco markers + homography (OpenCV) |
| IL policies | Diffusion Policy, ACT (`lerobot`) |
| VLA fine-tune | SmolVLA (`lerobot`) |
| RL fine-tune | PPO (`stable-baselines3`) |

## Setup

```bash
pip install --user mujoco mink python-chess opencv-python numpy

assets/fetch_menagerie.sh    # clones mujoco_menagerie, stages mesh files at assets/assets/
engine/fetch_stockfish.sh    # downloads the Stockfish 18 Linux binary to engine/bin/stockfish

python assets/build_scene.py # generates assets/ur5e_chess_scene.xml from the two menagerie MJCFs + board/pieces/markers
```

Both fetch scripts write into gitignored directories, so run them once per fresh clone/container before anything else will work.

## Running it

```bash
# interactive 3D viewer -- drag to orbit, space to pause/step, see the arm + board live
python -m mujoco.viewer --mjcf=assets/ur5e_chess_scene.xml

# headless checks
python assets/workspace_reach_check.py    # mink IK reach check over all 64 board squares
python engine/dry_run_game.py             # full engine-vs-engine game, python-chess only, writes a PGN
python perception/validate_localization.py # renders board_cam, checks ArUco homography against ground truth

# piece detector (needs ChessReD's chessred2k subset in perception/piece_detector/data/, not fetched by a script -- see CLAUDE.md)
python perception/piece_detector/prepare_dataset.py       # filter ChessReD annotations to train/val/test JSON
python perception/piece_detector/finetune_groundingdino.py --balanced-sampling  # fine-tune, ~1-2hr/8 epochs on a 6GB GPU
python perception/piece_detector/evaluate.py               # per-piece-type accuracy on the test split
```

## Structure

```
assets/
  build_scene.py            compose ur5e + 2f85 + board + pieces + ArUco markers -> ur5e_chess_scene.xml
  board_geometry.py         shared board/square/marker world-frame coordinates
  workspace_reach_check.py  mink IK reachability check, all 64 squares
  fetch_menagerie.sh        clones mujoco_menagerie (gitignored)
perception/
  board_localization.py     ArUco detection + pixel<->board homography
  validate_localization.py  checks the homography against sim ground truth
  piece_detector/           GroundingDINO fine-tune + SAM2 box-prompted segmentation
    prepare_dataset.py      filters ChessReD annotations to train/val/test JSON
    finetune_groundingdino.py  fine-tunes grounding-dino-tiny on ChessReD piece boxes
    evaluate.py             per-piece-type accuracy on the held-out test split
    segment.py              frozen SAM2, box-prompted from the detector's output
engine/
  stockfish_interface.py    python-chess <-> Stockfish wrapper
  dry_run_game.py           engine-vs-engine dry run, no robot
  fetch_stockfish.sh        downloads the Stockfish binary (gitignored)
demos/
  scripted_expert.py        engine move -> mink IK -> joint trajectory
policies/
  diffusion_policy/
  act/
  smolvla_finetune/
rl/
  ppo_finetune.py
eval/
  success_rate.py
  ablation_perception_noise.py
configs/
```

## Phases

0. Scaffold — done
1. Env setup — arm+gripper+board MJCF, workspace reach check — done, 64/64 squares reachable
2. Chess engine + rules integration — done, dry-run game reached a legal fivefold-repetition draw
3. Perception — done. ArUco localization: 64/64 squares, ~3.5mm error. GroundingDINO fine-tuned on ChessReD: 87.9% mean per-piece-type accuracy (worst category 50%); SAM2 used frozen/box-prompted (no mask ground truth to fine-tune against)
4. Scripted expert + demo generation
5. IL training — Diffusion Policy vs ACT
6. SmolVLA fine-tune (language-conditioned)
7. PPO RL fine-tune (optional, post IL)
8. Full pipeline eval + ablation (learned vs GT-state vs classical-only perception)

Each phase confirmed with user before next starts. No silent stack substitutions. See `CLAUDE.md` for architecture notes and gotchas.
