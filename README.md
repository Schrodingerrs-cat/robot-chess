# robot-chess

Sim manipulator arm plays chess: perceive board, get move from engine, execute pick-place (incl. captures). IL-first (Diffusion Policy / ACT / SmolVLA), optional PPO fine-tune.

Not a from-scratch VLA/VLM/world-model project — grounds existing open-source policy + perception architectures onto one task.

## Stack

| Component | Choice |
|---|---|
| Simulator | MuJoCo (`mujoco`, `dm_control`) |
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

## Structure

```
assets/                 MJCF: ur5e + robotiq_2f85 + chessboard + pieces
perception/
  board_localization.py ArUco + homography
  piece_detector/        GroundingDINO+SAM2 fine-tune
engine/
  stockfish_interface.py python-chess <-> Stockfish
demos/
  scripted_expert.py     engine move -> mink IK -> joint trajectory
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

0. Scaffold (this commit)
1. Env setup — arm+gripper+board MJCF, workspace reach check
2. Chess engine + rules integration
3. Perception — ArUco localization, GroundingDINO+SAM2 fine-tune
4. Scripted expert + demo generation
5. IL training — Diffusion Policy vs ACT
6. SmolVLA fine-tune (language-conditioned)
7. PPO RL fine-tune (optional, post IL)
8. Full pipeline eval + ablation (learned vs GT-state vs classical-only perception)

Each phase confirmed with user before next starts. No silent stack substitutions.
