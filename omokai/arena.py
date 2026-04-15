from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import torch

from .board import GameState
from .evaluator import BatchedEvaluator, ModelEvaluator
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
        search_threads: int = 1,
        inference_batch_size: int = 256,
        inference_wait_ms: float = 1.0,
    ) -> None:
        self.board_size = board_size
        self.exactly_five = exactly_five
        self.simulations = simulations
        self.c_puct = c_puct
        self.search_threads = max(1, search_threads)
        self.candidate_evaluator = self._build_evaluator(
            model=candidate_model,
            device=device,
            use_amp=use_amp,
            inference_batch_size=inference_batch_size,
            inference_wait_ms=inference_wait_ms,
        )
        self.best_evaluator = self._build_evaluator(
            model=best_model,
            device=device,
            use_amp=use_amp,
            inference_batch_size=inference_batch_size,
            inference_wait_ms=inference_wait_ms,
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

        try:
            if self.search_threads <= 1 or len(openings) <= 1:
                pair_results = [self._play_opening_pair(opening) for opening in openings]
            else:
                with ThreadPoolExecutor(max_workers=min(self.search_threads, len(openings))) as executor:
                    pair_results = list(executor.map(self._play_opening_pair, openings))
        finally:
            self.candidate_evaluator.close()
            self.best_evaluator.close()

        for wins, losses, pair_draws, black_wins, white_wins in pair_results:
            candidate_wins += wins
            best_wins += losses
            draws += pair_draws
            candidate_black_wins += black_wins
            candidate_white_wins += white_wins

        return ArenaResult(
            games=pair_count * 2,
            candidate_wins=candidate_wins,
            best_wins=best_wins,
            draws=draws,
            candidate_black_wins=candidate_black_wins,
            candidate_white_wins=candidate_white_wins,
        )

    def _build_evaluator(
        self,
        model: PolicyValueNet,
        device: torch.device,
        use_amp: bool,
        inference_batch_size: int,
        inference_wait_ms: float,
    ) -> ModelEvaluator | BatchedEvaluator:
        if self.search_threads <= 1:
            return ModelEvaluator(model=model, device=device, use_amp=use_amp)
        return BatchedEvaluator(
            model=model,
            device=device,
            use_amp=use_amp,
            max_batch_size=inference_batch_size,
            wait_ms=inference_wait_ms,
        )

    def _play_opening_pair(self, opening: Sequence[int]) -> tuple[int, int, int, int, int]:
        candidate_wins = 0
        best_wins = 0
        draws = 0
        candidate_black_wins = 0
        candidate_white_wins = 0
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
        return candidate_wins, best_wins, draws, candidate_black_wins, candidate_white_wins

    def _play_game(self, opening: Sequence[int], candidate_color: int) -> int:
        candidate_search = MCTS(
            c_puct=self.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            evaluator=self.candidate_evaluator,
        )
        best_search = MCTS(
            c_puct=self.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            evaluator=self.best_evaluator,
        )
        state = GameState(board_size=self.board_size, exactly_five=self.exactly_five)
        for action in opening:
            state.apply_action(action)

        candidate_root = None
        best_root = None
        while not state.terminal:
            if state.to_play == candidate_color:
                result = candidate_search.search_batch([state], self.simulations, [0.0], add_noise=False, roots=[candidate_root])[0]
                candidate_root = result.next_root
                if best_root is not None:
                    best_root = best_root.children.get(result.action)
            else:
                result = best_search.search_batch([state], self.simulations, [0.0], add_noise=False, roots=[best_root])[0]
                best_root = result.next_root
                if candidate_root is not None:
                    candidate_root = candidate_root.children.get(result.action)
            state.apply_action(result.action)
        return state.winner
