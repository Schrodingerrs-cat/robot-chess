"""Phase 2 dry run: play a full game engine-vs-engine in python-chess only (no robot).

Confirms the move stream + logging works: for every ply, prints the mover, the
chosen move in SAN, the resulting evaluation, and validates the move was legal
before applying it. Writes a PGN of the finished game.

Run: python engine/dry_run_game.py
"""

import argparse
import sys
import time
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).parent))
from stockfish_interface import StockfishEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-per-move", type=float, default=0.1)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "dry_run_game.pgn")
    args = parser.parse_args()

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "robot-chess Phase 2 dry run"
    game.headers["White"] = "Stockfish"
    game.headers["Black"] = "Stockfish"
    node = game

    t0 = time.time()
    with StockfishEngine() as engine:
        ply = 0
        while not board.is_game_over() and ply < args.max_moves:
            mover = "White" if board.turn == chess.WHITE else "Black"
            engine_move = engine.best_move(board, time_limit=args.time_per_move)

            if not StockfishEngine.validate_move(board, engine_move.move):
                raise RuntimeError(f"engine proposed illegal move {engine_move.move} at ply {ply}")

            san = board.san(engine_move.move)
            board.push(engine_move.move)
            node = node.add_variation(engine_move.move)

            score = engine_move.score.pov(not board.turn) if engine_move.score else None
            print(f"{ply:4d} {mover:5s} {san:8s} eval={score}")
            ply += 1

    elapsed = time.time() - t0
    result = board.result()
    game.headers["Result"] = result
    print(f"\ngame over: {result}  ({board.outcome().termination.name if board.outcome() else 'move limit'})")
    print(f"{ply} plies in {elapsed:.1f}s ({elapsed / max(ply, 1):.3f}s/ply)")

    args.out.write_text(str(game))
    print(f"PGN written to {args.out}")


if __name__ == "__main__":
    main()
