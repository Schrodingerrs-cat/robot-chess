"""Tracks which scene body sits on which chess square as a game progresses.

assets/build_scene.py names piece bodies by their STARTING square (e.g.
"white_pawn_e2" stays "white_pawn_e2" even after e2-e4), so a body name alone
can't answer "what's on e4 right now". This module is the mutable square<->body
mapping that both demos/physics_executor.py (physical White moves) and
demos/generate_dataset.py (procedural Black moves) update as a game is played,
mirroring assets/build_scene.py's build_pieces() layout exactly so body names
match the compiled scene.
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "assets"))
from board_geometry import FILES  # noqa: E402

BACK_RANK = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]


class BoardState:
    def __init__(self):
        self.square_to_body: dict[str, str] = {}
        self.body_to_square: dict[str, str] = {}
        self.body_kind: dict[str, str] = {}
        self.body_color: dict[str, str] = {}
        self._graveyard_count = {"white": 0, "black": 0}

        for f in range(8):
            file = FILES[f]
            self._place(f"white_pawn_{file}2", f"{file}2", "pawn", "white")
            self._place(f"black_pawn_{file}7", f"{file}7", "pawn", "black")
            self._place(f"white_{BACK_RANK[f]}_{file}1", f"{file}1", BACK_RANK[f], "white")
            self._place(f"black_{BACK_RANK[f]}_{file}8", f"{file}8", BACK_RANK[f], "black")

    def _place(self, body: str, square: str, kind: str, color: str) -> None:
        self.square_to_body[square] = body
        self.body_to_square[body] = square
        self.body_kind[body] = kind
        self.body_color[body] = color

    def body_at(self, square: str) -> str | None:
        return self.square_to_body.get(square)

    def next_graveyard_index(self, captured_color: str) -> int:
        idx = self._graveyard_count[captured_color]
        self._graveyard_count[captured_color] += 1
        return idx

    def apply_move(self, from_square: str, to_square: str) -> str | None:
        """Updates the mapping for a from->to move. Returns the captured body
        name (if `to_square` was occupied) so the caller can relocate it to the
        graveyard -- this method only updates bookkeeping, it never touches
        scene qpos itself (see physics_executor.py / generate_dataset.py for
        the physical/teleport moves that do).
        """
        captured_body = self.square_to_body.get(to_square)
        if captured_body is not None:
            del self.square_to_body[to_square]
            del self.body_to_square[captured_body]

        moving_body = self.square_to_body.pop(from_square)
        self.square_to_body[to_square] = moving_body
        self.body_to_square[moving_body] = to_square
        return captured_body
