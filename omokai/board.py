from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Action = int


@dataclass(slots=True)
class GameState:
    board_size: int = 15
    exactly_five: bool = False
    board: np.ndarray | None = None
    to_play: int = 1
    move_count: int = 0
    last_action: Action | None = None
    winner: int = 0
    terminal: bool = False

    def __post_init__(self) -> None:
        if self.board is None:
            self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        else:
            self.board = np.asarray(self.board, dtype=np.int8)

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size

    def clone(self) -> "GameState":
        return GameState(
            board_size=self.board_size,
            exactly_five=self.exactly_five,
            board=self.board.copy(),
            to_play=self.to_play,
            move_count=self.move_count,
            last_action=self.last_action,
            winner=self.winner,
            terminal=self.terminal,
        )

    def legal_moves(self) -> np.ndarray:
        if self.terminal:
            return np.zeros(self.action_size, dtype=bool)
        return (self.board.reshape(-1) == 0).astype(bool)

    def apply_action(self, action: Action) -> None:
        if self.terminal:
            raise ValueError("cannot play on a terminal position")
        row, col = divmod(action, self.board_size)
        if self.board[row, col] != 0:
            raise ValueError(f"illegal move at {(row, col)}")
        player = self.to_play
        self.board[row, col] = player
        self.last_action = action
        self.move_count += 1
        self.to_play = -player
        if self._is_winning_move(row, col, player):
            self.winner = player
            self.terminal = True
        elif self.move_count == self.action_size:
            self.winner = 0
            self.terminal = True

    def outcome_for_player(self, player: int) -> float:
        if not self.terminal:
            raise ValueError("game is not terminal")
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def feature_planes(self) -> np.ndarray:
        own = (self.board == self.to_play).astype(np.float32)
        opp = (self.board == -self.to_play).astype(np.float32)
        last = np.zeros_like(own, dtype=np.float32)
        if self.last_action is not None:
            row, col = divmod(self.last_action, self.board_size)
            last[row, col] = 1.0
        color = np.full_like(own, 1.0 if self.to_play == 1 else 0.0, dtype=np.float32)
        return np.stack([own, opp, last, color], axis=0)

    def _is_winning_move(self, row: int, col: int, player: int) -> bool:
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1 + self._count_dir(row, col, dr, dc, player) + self._count_dir(row, col, -dr, -dc, player)
            if self.exactly_five:
                if count == 5:
                    return True
            else:
                if count >= 5:
                    return True
        return False

    def _count_dir(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        total = 0
        r = row + dr
        c = col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
            total += 1
            r += dr
            c += dc
        return total
