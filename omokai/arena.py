from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .board import GameState
from .mcts import MCTS
from .network import PolicyValueNet
from .openings import build_opening_sequences


@dataclass(slots=True)
class ArenaResult:
    games: int
    candidate_wins: int
    best_wins: int
    draws: int
    candidate_black_wins: int
    candidate_white_wins: int

    @property
    def candidate_win_rate(self) -> float:
        return 0.0 if self.games == 0 else self.candidate_wins / self.games

class Arena:
    def __init__(
        self,
        candidate_model: PolicyValueNet,
        best_model: PolicyValueNet,
        device: torch.device,
        board_size: int,
        exactly_five: bool,
        simulations: int,
        c_puct: float,
        use_amp: bool,
    ) -> None:
        self.board_size = board_size
        self.exactly_five = exactly_five
        self.simulations = simulations
        self.candidate_search = MCTS(
            model=candidate_model,
            device=device,
            c_puct=c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            use_amp=use_amp,
        )
        self.best_search = MCTS(
            model=best_model,
            device=device,
            c_puct=c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            use_amp=use_amp,
        )
        candidate_model.eval()
        best_model.eval()

    def evaluate(self, games: int) -> ArenaResult:
        target_games = max(2, games)
        pair_count = max(1, target_games // 2)
        openings = build_opening_sequences(self.board_size, pair_count)
        candidate_wins = 0
        best_wins = 0
        draws = 0
        candidate_black_wins = 0
        candidate_white_wins = 0

        for opening in openings:
            for candidate_color in (1, -1):
                winner = self._play_game(opening, candidate_color)
                if winner == 0:
                    draws += 1
                    continue
                if winner == candidate_color:
                    candidate_wins += 1
                    if candidate_color == 1:
                        candidate_black_wins += 1
                    else:
                        candidate_white_wins += 1
                else:
                    best_wins += 1

        return ArenaResult(
            games=pair_count * 2,
            candidate_wins=candidate_wins,
            best_wins=best_wins,
            draws=draws,
            candidate_black_wins=candidate_black_wins,
            candidate_white_wins=candidate_white_wins,
        )

    def _play_game(self, opening: Sequence[int], candidate_color: int) -> int:
        state = GameState(board_size=self.board_size, exactly_five=self.exactly_five)
        for action in opening:
            state.apply_action(action)

        while not state.terminal:
            search = self.candidate_search if state.to_play == candidate_color else self.best_search
            result = search.search_batch([state], self.simulations, [0.0], add_noise=False)[0]
            state.apply_action(result.action)
        return state.winner
