"""Train/val/test split for a demos/generate_dataset.py batch.

Splits by GAME, not by episode: every White episode from a given self-play
game stays on the same side of the split. Positions a few plies apart in the
same game are correlated (near-duplicate board states, same opening/piece
layout), so an episode-level shuffle would leak train-time information into
what's supposed to be a held-out test set -- same discipline already applied
to the perception dataset's chessred2k splits (see CLAUDE.md).

Reads an existing `dataset_info.json` (written by generate_dataset.py) and
writes `splits.json` alongside it -- a manifest of which episode files belong
to which split, not a copy of the .npz files themselves.

Run: python demos/split_dataset.py --data-dir demos/data/batch_v1
"""

import argparse
import json
import os
import random


def split_dataset(data_dir: str, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 0) -> dict:
    with open(os.path.join(data_dir, "dataset_info.json")) as f:
        info = json.load(f)
    episodes = info["episodes"]

    game_ids = sorted({e["game"] for e in episodes})
    n = len(game_ids)
    if n < 3:
        raise ValueError(f"need at least 3 games to form train/val/test, got {n}")

    rng = random.Random(seed)
    shuffled = game_ids[:]
    rng.shuffle(shuffled)

    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac))
    n_train = min(n_train, n - 2)  # leave at least 1 game each for val and test
    n_val = max(1, min(n_val, n - n_train - 1))

    split_games = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }

    result = {"seed": seed, "train_frac": train_frac, "val_frac": val_frac, "n_games": n}
    for split, games in split_games.items():
        game_set = set(games)
        split_episodes = [e for e in episodes if e["game"] in game_set]
        piece_counts: dict[str, int] = {}
        for e in split_episodes:
            piece_counts[e["piece_kind"]] = piece_counts.get(e["piece_kind"], 0) + 1
        result[split] = {
            "games": games,
            "n_games": len(games),
            "n_episodes": len(split_episodes),
            "n_captures": sum(e["is_capture"] for e in split_episodes),
            "piece_kind_counts": piece_counts,
            "files": [e["file"] for e in split_episodes],
        }

    with open(os.path.join(data_dir, "splits.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = split_dataset(args.data_dir, args.train_frac, args.val_frac, args.seed)

    for split in ("train", "val", "test"):
        s = result[split]
        print(f"{split:5s}: {s['n_games']:2d} games, {s['n_episodes']:3d} episodes, "
              f"{s['n_captures']:2d} captures, games={s['games']}")
    print(f"splits.json written to {args.data_dir}/splits.json")


if __name__ == "__main__":
    main()
