# Build Prompt: Robot Chess via IL/RL (for Claude Code)

Paste this into Claude Code in a fresh or dedicated repo. It's written to be executed in
phases — Claude Code should stop and show results at each checkpoint before moving on,
not run the whole pipeline unattended.

---

## Project objective

Build an end-to-end pipeline where an **industrial manipulator arm**, in simulation,
plays chess: perceive the board state, receive a move from a chess engine, and physically
execute that move (pick up the piece, move it, place it, including captures). Train the
motor policy via imitation learning (IL) first, then optionally fine-tune with
reinforcement learning (RL). This is a task-specific policy fine-tuning project — we are
**not** building a VLA, VLM, or world model from scratch. We are grounding an existing
open-source policy architecture (and separately, an existing open-source perception model)
onto one well-defined task.

## Fixed tech stack — do not substitute without asking

| Component | Choice | Why |
|---|---|---|
| Simulator | **MuJoCo** (`mujoco` + `mujoco-py`/`dm_control` as needed) | Best contact solver for small, precise contacts (thin chess pieces); fastest iteration for RL |
| Robot arm | **Universal Robots UR5e** — `google-deepmind/mujoco_menagerie/universal_robots_ur5e` | Industry-standard industrial/collaborative arm, widely deployed, well-documented, has a Menagerie MJCF model |
| Gripper | **Robotiq 2F-85** — `google-deepmind/mujoco_menagerie/robotiq_2f85` | Industry-standard parallel-jaw gripper, pairs naturally with the UR5e in Menagerie |
| IK / motion | **`kevinzakka/mink`** | Differentiable IK solver compatible with MuJoCo models, used for joint trajectory generation from Cartesian grasp/place targets |
| Chess engine | **Stockfish**, driven via **`python-chess`** | Free, strong, standard open-source engine; gives both expert demonstrations (best move) and a reward signal (evaluation delta / legality) |
| Perception dataset | **ChessReD (Chess Recognition Dataset)** — `github.com/ThanosM97/end-to-end-chess-recognition` (or the Roboflow "Chess Pieces" dataset as a fallback) | Open-source, real photographed chessboards with full piece/square annotations — use to pretrain/fine-tune the piece detector so it's grounded in real data, not purely synthetic |
| Piece detection/segmentation | **`IDEA-Research/GroundingDINO`** (detection) + **`facebookresearch/sam2`** (segmentation), fine-tuned on ChessReD | Open-vocabulary detector + segmenter, fine-tuned rather than trained from scratch |
| Board localization | Classical CV: 4 **ArUco markers** at board corners + homography (OpenCV) | Deterministic, reliable, plays to controls/CV strength — don't use a learned method where geometry solves it outright |
| IL policy architectures | **Diffusion Policy** (`real-stanford/diffusion_policy`) and **ACT** (via `huggingface/lerobot`) — train both, compare | Two standard, well-supported IL architectures; comparing them is the reproducibility/rigor angle |
| VLA fine-tune | **SmolVLA** (via `huggingface/lerobot`) | Fine-tune (not train from scratch) on language-conditioned move commands, e.g. "move the pawn from e2 to e4" |
| RL fine-tuning | **PPO** via `stable-baselines3` | Standard, well-supported; used only as a fine-tuning pass on top of the IL-pretrained policy, not as the primary training method |
| Chess logic / rules | **`python-chess`** | Move legality, game state, board representation, interfaces directly with Stockfish |

## Repo structure to scaffold (Phase 0)

```
robot-chess/
  assets/                 # MJCF models: ur5e + robotiq_2f85 + chessboard + pieces
  perception/
    board_localization.py # ArUco + homography
    piece_detector/        # GroundingDINO+SAM2 fine-tuning code + configs
  engine/
    stockfish_interface.py # python-chess <-> Stockfish wrapper
  demos/
    scripted_expert.py     # engine move -> mink IK -> joint trajectory demo generator
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
  README.md
```

## Execution phases — confirm with me before moving to the next phase

### Phase 1 — Environment setup
- Pull `universal_robots_ur5e` and `robotiq_2f85` from `mujoco_menagerie`, compose them
  into one MJCF with the gripper attached to the UR5e flange.
- Build a simple chessboard + 32-piece asset set (basic geometric Staunton-style pieces
  are fine — visual fidelity isn't the point).
- Verify the arm can reach every square on the board within its workspace; report any
  squares that are out of reach so we can adjust board placement/size before continuing.

### Phase 2 — Chess engine + rules integration
- Wire up `python-chess` + Stockfish so we can: get legal moves, get Stockfish's chosen
  move from any position, get an evaluation score, and validate a move post-execution.
- Do a dry run: play a full game engine-vs-engine purely in `python-chess` (no robot yet)
  and confirm the move stream/logging works.

### Phase 3 — Perception
- Implement ArUco-marker-based board localization + homography to get board-frame
  coordinates for all 64 squares.
- Fine-tune GroundingDINO + SAM2 on ChessReD (or generate a matched synthetic set from our
  own MuJoCo renders if ChessReD's real-world domain gap is too large — flag this decision
  to me rather than assuming).
- Report detection accuracy per piece type before moving on.

### Phase 4 — Scripted expert + demonstration generation
- For a given engine move (source square, target square, capture flag), use `mink` to
  solve IK for: approach pose → grasp pose → lift → transit → place pose → release, and
  for captures, remove the captured piece first.
- Generate a demonstration dataset (aim for an initial batch of a few hundred move
  trajectories) in a format compatible with both `diffusion_policy` and `lerobot`
  (ACT/SmolVLA) training pipelines.

### Phase 5 — IL training: Diffusion Policy vs. ACT
- Train both architectures on the demonstration dataset.
- Evaluate success rate (correct piece, correct destination, stable placement, no
  collisions) on a held-out set of positions/moves.
- Report a comparison table: success rate, training time, sample efficiency curve.

### Phase 6 — SmolVLA fine-tuning
- Fine-tune SmolVLA on the same demonstrations, but condition on natural-language move
  instructions instead of raw coordinates.
- Evaluate the same way as Phase 5, plus a check on whether it generalizes to
  paraphrased instructions.

### Phase 7 — RL fine-tuning pass (optional, only after Phase 5/6 succeed)
- Take the best-performing IL-pretrained policy and fine-tune with PPO.
- Reward: successful stable placement, no piece knocked over, post-hoc move legality,
  small time/energy penalty. Show me the exact reward function before training.
- Report before/after success rate and sample efficiency.

### Phase 8 — Evaluation & ablation
- Full pipeline eval: engine picks a move → perception reads board state → policy
  executes → verify final board state matches expected.
- Ablation: swap the learned (GroundingDINO+SAM2) piece detector for ground-truth sim
  state, and separately for the classical/ArUco-only pipeline, then measure how policy
  success rate changes. This is the core novel contribution of the project — treat it
  as such in the final report.

## Constraints / ground rules for you (Claude Code)

- Do not silently substitute a different robot, simulator, or dataset — if something in
  this stack turns out infeasible (e.g. licensing, missing assets, compute limits), stop
  and tell me the specific blocker before picking an alternative.
- Do not build a VLA, VLM, or world model from scratch at any point — every model here is
  either an existing pretrained/open-source model being fine-tuned, or a classical CV
  method.
- After each phase, give me a short status report (what worked, what didn't, any numbers)
  before starting the next phase.
- Keep all dataset/model downloads to properly licensed open-source sources; flag
  anything that requires a paid API key or account before assuming I have one.
