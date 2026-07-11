"""python-chess <-> Stockfish wrapper.

Thin layer over `chess.engine` (UCI) plus `chess.Board` giving the four
operations Phase 2 needs: legal moves, engine's chosen move, position
evaluation, and post-hoc move validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

DEFAULT_STOCKFISH_PATH = Path(__file__).parent / "bin" / "stockfish"


@dataclass
class EngineMove:
    move: chess.Move
    score: chess.engine.PovScore  # from the mover's perspective
    ponder: chess.Move | None = None


class StockfishEngine:
    """Wraps a Stockfish UCI process for one game/session.

    Use as a context manager so the subprocess is always cleaned up:

        with StockfishEngine() as engine:
            move = engine.best_move(board)
    """

    def __init__(self, path: str | Path = DEFAULT_STOCKFISH_PATH, threads: int = 1, hash_mb: int = 128):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Stockfish binary not found at {path}. Run engine/fetch_stockfish.sh first."
            )
        self._engine = chess.engine.SimpleEngine.popen_uci(str(path))
        self._engine.configure({"Threads": threads, "Hash": hash_mb})

    def __enter__(self) -> "StockfishEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._engine.quit()

    @staticmethod
    def legal_moves(board: chess.Board) -> list[chess.Move]:
        """All legal moves in the given position (python-chess rules engine, no Stockfish needed)."""
        return list(board.legal_moves)

    def best_move(self, board: chess.Board, time_limit: float = 0.5, depth: int | None = None) -> EngineMove:
        """Stockfish's chosen move from the given position."""
        limit = chess.engine.Limit(depth=depth) if depth is not None else chess.engine.Limit(time=time_limit)
        result = self._engine.play(board, limit, info=chess.engine.INFO_SCORE)
        assert result.move is not None
        score = result.info.get("score")
        return EngineMove(move=result.move, score=score, ponder=result.ponder)

    def evaluate(self, board: chess.Board, time_limit: float = 0.5, depth: int | None = None) -> chess.engine.PovScore:
        """Static evaluation of the given position, from the side-to-move's perspective."""
        limit = chess.engine.Limit(depth=depth) if depth is not None else chess.engine.Limit(time=time_limit)
        info = self._engine.analyse(board, limit)
        return info["score"]

    @staticmethod
    def validate_move(board: chess.Board, move: chess.Move) -> bool:
        """Post-execution check: was `move` actually legal in `board`'s position?

        Intended to be called against the pre-move board state after the robot has
        physically executed a move, to confirm the commanded move was legal before
        applying it to the tracked game state.
        """
        return move in board.legal_moves

    @staticmethod
    def apply_move(board: chess.Board, move: chess.Move) -> None:
        if not StockfishEngine.validate_move(board, move):
            raise ValueError(f"Illegal move {move} in position {board.fen()}")
        board.push(move)
