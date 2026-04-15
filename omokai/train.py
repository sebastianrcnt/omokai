from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
import random
import signal
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .arena import Arena
from .board import GameState
from .checkpoint import load_checkpoint, load_training_state, save_checkpoint, save_training_state
from .config import RunConfig, load_config
from .device import amp_context, is_rocm_build, resolve_device
from .evaluator import BatchedEvaluator, Evaluator, ModelEvaluator
from .logging_utils import configure_debug_logging, get_debug_logger, log_event
from .mcts import MCTS
from .network import PolicyValueNet, clone_model
from .openings import sample_balanced_openings
from .replay import PendingSample, ReplayBuffer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(slots=True)
class SelfPlayChunkResult:
    source: str
    completed_games: list[tuple[list[PendingSample], int]]
    black_wins: int
    white_wins: int
    draws: int
    total_moves: int
    white_to_move_games: int


def split_evenly(items: Sequence[list[int]], parts: int) -> list[list[list[int]]]:
    if not items:
        return []
    parts = max(1, min(parts, len(items)))
    base, extra = divmod(len(items), parts)
    chunks: list[list[list[int]]] = []
    offset = 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        if size <= 0:
            continue
        chunks.append(list(items[offset : offset + size]))
        offset += size
    return chunks


class Trainer:
    def __init__(self, config: RunConfig, resume_path: str | None = None) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.checkpoint_dir / "metrics.jsonl"
        self.progress_file = self.checkpoint_dir / "runtime_progress.json"
        self.debug_log_file = self.checkpoint_dir / "training.debug.log"
        self.debug_jsonl_file = self.checkpoint_dir / "training.debug.jsonl"
        self._configure_debug_logging()
        self.replay = ReplayBuffer(config.optimization.replay_capacity)
        self.model = PolicyValueNet(config.rules.board_size, config.network).to(self.device)
        self.best_model = clone_model(self.model).to(self.device)
        self.best_model.eval()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.use_amp and self.device.type == "cuda")
        self.selfplay_rng = random.Random(config.seed + 1_048_583)
        self.iteration = 0
        self.total_updates = 0
        self.best_iteration = 0
        self.best_arena_win_rate = 0.0
        self.best_checkpoint_metadata: dict[str, object] = {
            "iteration": 0,
            "elapsed_hours": 0.0,
            "best_iteration": 0,
            "best_arena_win_rate": 0.0,
            "total_updates": 0,
            "status": "initial_best",
        }
        self.elapsed_seconds_offset = 0.0
        self.start_time = time.monotonic()
        self.last_progress_save_time = self.start_time
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._handle_stop_signal)
        signal.signal(signal.SIGTERM, self._handle_stop_signal)
        self.debug_logger.debug(
            "trainer initialized experiment=%s device=%s checkpoint_dir=%s resume_path=%s",
            self.config.experiment_name,
            self.device,
            self.checkpoint_dir,
            resume_path,
        )
        if resume_path:
            self._restore_from_checkpoint(resume_path)

    def _configure_debug_logging(self) -> None:
        configure_debug_logging(self.debug_log_file, jsonl_path=self.debug_jsonl_file, level=logging.DEBUG)
        self.debug_logger = get_debug_logger("trainer")
        log_event(
            self.debug_logger,
            logging.DEBUG,
            "debug_logging_configured",
            "debug logging configured path=%s jsonl_path=%s",
            self.debug_log_file,
            self.debug_jsonl_file,
            log_path=self.debug_log_file,
            jsonl_path=self.debug_jsonl_file,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimization.learning_rate,
            weight_decay=self.config.optimization.weight_decay,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.config.optimization.min_learning_rate >= self.config.optimization.learning_rate:
            return None
        schedule_updates = max(1, self.config.optimization.updates_per_iteration * 16)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=schedule_updates,
            eta_min=self.config.optimization.min_learning_rate,
        )

    def run(self) -> None:
        torch.set_float32_matmul_precision("high")
        startup_metadata = {
            "iteration": self.iteration,
            "status": "startup",
            "best_iteration": self.best_iteration,
            "total_updates": self.total_updates,
        }
        log_event(
            self.debug_logger,
            logging.INFO,
            "run_started",
            "run starting iteration=%d best_iteration=%d total_updates=%d device=%s",
            self.iteration,
            self.best_iteration,
            self.total_updates,
            self.device,
            iteration=self.iteration,
            best_iteration=self.best_iteration,
            total_updates=self.total_updates,
            device=str(self.device),
        )
        self.save_model_checkpoints(startup_metadata)
        self.save_runtime_state(startup_metadata)

        while not self._should_stop():
            self.iteration += 1
            simulations = self.current_simulations()
            selfplay_plan = self.current_selfplay_plan()
            iteration_started_at = time.monotonic()
            log_event(
                self.debug_logger,
                logging.INFO,
                "iteration_started",
                "iteration %d started simulations=%d selfplay_plan=%s",
                self.iteration,
                simulations,
                selfplay_plan,
                iteration=self.iteration,
                simulations=simulations,
                selfplay_plan=selfplay_plan,
                best_iteration=self.best_iteration,
                replay_games=self.replay.games_seen,
                replay_samples=len(self.replay),
            )
            selfplay_stats = self.generate_selfplay(simulations, selfplay_plan)

            if self._should_stop():
                log_event(
                    self.debug_logger,
                    logging.WARNING,
                    "stop_requested",
                    "stop requested during selfplay at iteration=%d",
                    self.iteration,
                    iteration=self.iteration,
                    stage="selfplay",
                )
                self.save_runtime_state(
                    {
                        "iteration": self.iteration,
                        "simulations": simulations,
                        "stopped_during": "selfplay",
                        "best_iteration": self.best_iteration,
                        "total_updates": self.total_updates,
                    }
                )
                break

            if self.replay.games_seen < self.config.optimization.warmup_games:
                log_event(
                    self.debug_logger,
                    logging.INFO,
                    "warmup_iteration",
                    "iteration %d warmup replay_games=%d warmup_games=%d",
                    self.iteration,
                    self.replay.games_seen,
                    self.config.optimization.warmup_games,
                    iteration=self.iteration,
                    replay_games=self.replay.games_seen,
                    warmup_games=self.config.optimization.warmup_games,
                    duration_seconds=time.monotonic() - iteration_started_at,
                )
                warmup_metadata = {
                    "iteration": self.iteration,
                    "simulations": simulations,
                    "warmup": True,
                    "best_iteration": self.best_iteration,
                    "total_updates": self.total_updates,
                }
                self._log(
                    {
                        "iteration": self.iteration,
                        "elapsed_hours": round(self.elapsed_hours, 4),
                        "simulations": simulations,
                        "status": "warmup",
                        "selfplay_source": selfplay_plan["label"],
                        "best_iteration": self.best_iteration,
                        "total_updates": self.total_updates,
                        **selfplay_stats,
                    }
                )
                self.save_model_checkpoints(warmup_metadata)
                self.save_runtime_state(warmup_metadata)
                continue

            training_stats = self.train_model(self.model)
            log_event(
                self.debug_logger,
                logging.INFO,
                "training_iteration_finished",
                "iteration %d training finished stats=%s",
                self.iteration,
                training_stats,
                iteration=self.iteration,
                training=training_stats,
            )

            if self._should_stop():
                log_event(
                    self.debug_logger,
                    logging.WARNING,
                    "stop_requested",
                    "stop requested during training at iteration=%d",
                    self.iteration,
                    iteration=self.iteration,
                    stage="training",
                )
                self.save_runtime_state(
                    {
                        "iteration": self.iteration,
                        "simulations": simulations,
                        "stopped_during": "training",
                        "training": training_stats,
                        "best_iteration": self.best_iteration,
                        "total_updates": self.total_updates,
                    }
                )
                break

            arena_stats = self.evaluate_candidate(simulations)
            log_event(
                self.debug_logger,
                logging.INFO,
                "arena_iteration_finished",
                "iteration %d arena finished stats=%s",
                self.iteration,
                arena_stats,
                iteration=self.iteration,
                arena=arena_stats,
            )
            if arena_stats["accepted"]:
                self.best_model.load_state_dict(self.model.state_dict())
                self.best_model.eval()
                self.best_iteration = self.iteration
                self.best_arena_win_rate = float(arena_stats["arena_win_rate"])
                log_event(
                    self.debug_logger,
                    logging.INFO,
                    "best_model_promoted",
                    "new best accepted iteration=%d arena_win_rate=%.4f",
                    self.best_iteration,
                    self.best_arena_win_rate,
                    iteration=self.iteration,
                    best_iteration=self.best_iteration,
                    best_arena_win_rate=self.best_arena_win_rate,
                )
                self.best_checkpoint_metadata = {
                    "iteration": self.iteration,
                    "elapsed_hours": self.elapsed_hours,
                    "simulations": simulations,
                    "selfplay_source": selfplay_plan["label"],
                    "accepted": arena_stats["accepted"],
                    "best_iteration": self.best_iteration,
                    "best_arena_win_rate": self.best_arena_win_rate,
                    "total_updates": self.total_updates,
                    "training": training_stats,
                    "arena": arena_stats,
                }

            metadata = {
                "iteration": self.iteration,
                "elapsed_hours": self.elapsed_hours,
                "simulations": simulations,
                "selfplay_source": selfplay_plan["label"],
                "accepted": arena_stats["accepted"],
                "best_iteration": self.best_iteration,
                "best_arena_win_rate": self.best_arena_win_rate,
                "total_updates": self.total_updates,
                "training": training_stats,
                "arena": arena_stats,
            }
            self.save_model_checkpoints(metadata)
            self.save_runtime_state(metadata)
            self._log(
                {
                    "iteration": self.iteration,
                    "elapsed_hours": round(self.elapsed_hours, 4),
                    "simulations": simulations,
                    "selfplay_source": selfplay_plan["label"],
                    "accepted": arena_stats["accepted"],
                    "best_iteration": self.best_iteration,
                    "best_arena_win_rate": round(self.best_arena_win_rate, 4),
                    "total_updates": self.total_updates,
                    **selfplay_stats,
                    **training_stats,
                    **arena_stats,
                }
            )
            log_event(
                self.debug_logger,
                logging.INFO,
                "iteration_completed",
                "iteration %d completed accepted=%s best_iteration=%d replay_games=%d replay_samples=%d",
                self.iteration,
                arena_stats["accepted"],
                self.best_iteration,
                self.replay.games_seen,
                len(self.replay),
                iteration=self.iteration,
                accepted=arena_stats["accepted"],
                best_iteration=self.best_iteration,
                best_arena_win_rate=self.best_arena_win_rate,
                replay_games=self.replay.games_seen,
                replay_samples=len(self.replay),
                duration_seconds=time.monotonic() - iteration_started_at,
                simulations=simulations,
                selfplay_source=selfplay_plan["label"],
            )

    @property
    def elapsed_hours(self) -> float:
        return (self.elapsed_seconds_offset + time.monotonic() - self.start_time) / 3600.0

    def _time_exceeded(self) -> bool:
        return self.config.max_hours is not None and self.elapsed_hours >= self.config.max_hours

    def _should_stop(self) -> bool:
        return self.stop_requested or self._time_exceeded()

    def current_simulations(self) -> int:
        if not self.config.selfplay.simulation_schedule:
            raise ValueError("simulation_schedule must not be empty")

        if self.config.max_hours is not None:
            fraction = min(1.0, self.elapsed_hours / max(self.config.max_hours, 1.0e-6))
        elif self.config.selfplay.simulation_ramp_iterations is not None:
            ramp = max(1, self.config.selfplay.simulation_ramp_iterations)
            fraction = min(1.0, max(0, self.iteration - 1) / ramp)
        else:
            return int(self.config.selfplay.simulation_schedule[-1]["simulations"])

        simulations = int(self.config.selfplay.simulation_schedule[0]["simulations"])
        for point in self.config.selfplay.simulation_schedule:
            if fraction >= float(point["fraction"]):
                simulations = int(point["simulations"])
        return simulations

    def current_selfplay_plan(self) -> dict[str, float | int | str]:
        if self.best_iteration == 0:
            return {"label": "candidate", "candidate_games": self.config.selfplay.games_per_iteration}

        candidate_mix_fraction = max(0.0, min(1.0, self.config.selfplay.candidate_mix_fraction))
        mix_iterations = max(0, self.config.selfplay.mixed_iterations_after_promotion)
        if mix_iterations > 0 and self.iteration <= self.best_iteration + mix_iterations and candidate_mix_fraction > 0.0:
            candidate_games = int(round(self.config.selfplay.games_per_iteration * candidate_mix_fraction))
            candidate_games = min(self.config.selfplay.games_per_iteration, max(0, candidate_games))
            if candidate_games >= self.config.selfplay.games_per_iteration:
                return {"label": "candidate", "candidate_games": candidate_games}
            return {"label": "mixed", "candidate_games": candidate_games}
        return {"label": "best", "candidate_games": 0}

    def _build_evaluator(self, model: PolicyValueNet) -> Evaluator:
        context = "selfplay"
        if self.config.selfplay.search_threads <= 1:
            return ModelEvaluator(
                model=model,
                device=self.device,
                use_amp=self.config.use_amp,
                logger=self.debug_logger.getChild("evaluator"),
                iteration=self.iteration,
                context=context,
            )
        return BatchedEvaluator(
            model=model,
            device=self.device,
            use_amp=self.config.use_amp,
            max_batch_size=self.config.selfplay.inference_batch_size,
            wait_ms=self.config.selfplay.inference_wait_ms,
            logger=self.debug_logger.getChild("evaluator"),
            iteration=self.iteration,
            context=context,
        )

    def _parallel_source_workers(self, candidate_games: int, best_games: int) -> dict[str, int]:
        total_workers = max(1, self.config.selfplay.search_threads)
        if total_workers <= 1:
            return {"candidate": int(candidate_games > 0), "best": int(best_games > 0 and candidate_games == 0)}
        if candidate_games <= 0:
            return {"candidate": 0, "best": min(total_workers, best_games)}
        if best_games <= 0:
            return {"candidate": min(total_workers, candidate_games), "best": 0}

        total_games = candidate_games + best_games
        candidate_workers = max(1, min(candidate_games, round(total_workers * candidate_games / total_games)))
        best_workers = max(1, min(best_games, total_workers - candidate_workers))
        while candidate_workers + best_workers < total_workers:
            if candidate_games - candidate_workers >= best_games - best_workers and candidate_workers < candidate_games:
                candidate_workers += 1
            elif best_workers < best_games:
                best_workers += 1
            else:
                break
        workers = {"candidate": candidate_workers, "best": best_workers}
        log_event(
            self.debug_logger,
            logging.DEBUG,
            "selfplay_worker_split",
            "selfplay worker split candidate_games=%d best_games=%d workers=%s",
            candidate_games,
            best_games,
            workers,
            iteration=self.iteration,
            candidate_games=candidate_games,
            best_games=best_games,
            workers=workers,
        )
        return workers

    def _selfplay_temperature(self, move_count: int) -> float:
        cfg = self.config.selfplay
        end = float(cfg.temperature_end)
        if cfg.temperature_moves <= 0:
            return end
        if move_count >= cfg.temperature_moves:
            return end
        frac = move_count / float(cfg.temperature_moves)
        return 1.0 + (end - 1.0) * frac

    def _consume_selfplay_chunk(
        self,
        result: SelfPlayChunkResult,
        counters: dict[str, int | float],
    ) -> None:
        discount = float(self.config.optimization.value_discount)
        for history, winner in result.completed_games:
            self.replay.add_game(history, winner, value_discount=discount)
        counters["black_wins"] += result.black_wins
        counters["white_wins"] += result.white_wins
        counters["draws"] += result.draws
        counters["total_moves"] += result.total_moves
        counters["completed_games"] += len(result.completed_games)
        counters["white_to_move_games"] += result.white_to_move_games
        if result.source == "candidate":
            counters["candidate_games"] += len(result.completed_games)
        else:
            counters["best_games"] += len(result.completed_games)

    def _run_selfplay_chunk(
        self,
        source: str,
        openings: list[list[int]],
        simulations: int,
        worker_batch_size: int,
        evaluator: Evaluator,
    ) -> SelfPlayChunkResult:
        log_event(
            self.debug_logger,
            logging.DEBUG,
            "selfplay_chunk_started",
            "selfplay chunk started source=%s openings=%d worker_batch_size=%d simulations=%d",
            source,
            len(openings),
            worker_batch_size,
            simulations,
            iteration=self.iteration,
            source=source,
            openings=len(openings),
            worker_batch_size=worker_batch_size,
            simulations=simulations,
        )
        search = MCTS(
            c_puct=self.config.selfplay.c_puct,
            dirichlet_alpha=self.config.selfplay.dirichlet_alpha,
            dirichlet_epsilon=self.config.selfplay.dirichlet_epsilon,
            evaluator=evaluator,
        )
        completed_games: list[tuple[list[PendingSample], int]] = []
        black_wins = 0
        white_wins = 0
        draws = 0
        total_moves = 0
        white_to_move_games = 0
        pending_openings = list(openings)

        while pending_openings and not self._should_stop():
            batch_openings = pending_openings[:worker_batch_size]
            del pending_openings[:worker_batch_size]
            states: list[GameState] = []
            histories: list[list[PendingSample]] = []
            roots = []
            for opening in batch_openings:
                state = GameState(self.config.rules.board_size, self.config.rules.exactly_five)
                for action in opening:
                    state.apply_action(action)
                if state.to_play == -1:
                    white_to_move_games += 1
                states.append(state)
                histories.append([])
                roots.append(None)

            while states and not self._should_stop():
                temperatures = [self._selfplay_temperature(state.move_count) for state in states]
                results = search.search_batch(
                    states,
                    simulations,
                    temperatures,
                    add_noise=True,
                    roots=roots,
                    leaves_per_batch=self.config.selfplay.leaves_per_batch,
                    virtual_loss=self.config.selfplay.virtual_loss,
                )

                next_states: list[GameState] = []
                next_histories: list[list[PendingSample]] = []
                next_roots = []
                for state, history, result in zip(states, histories, results, strict=True):
                    history.append(
                        PendingSample(
                            board=state.board.copy(),
                            to_play=state.to_play,
                            last_action=state.last_action,
                            policy=result.visit_policy.copy(),
                        )
                    )
                    state.apply_action(result.action)
                    if state.terminal:
                        completed_games.append((history, state.winner))
                        total_moves += len(history)
                        if state.winner == 1:
                            black_wins += 1
                        elif state.winner == -1:
                            white_wins += 1
                        else:
                            draws += 1
                    else:
                        next_states.append(state)
                        next_histories.append(history)
                        next_roots.append(result.next_root)
                states = next_states
                histories = next_histories
                roots = next_roots

        result = SelfPlayChunkResult(
            source=source,
            completed_games=completed_games,
            black_wins=black_wins,
            white_wins=white_wins,
            draws=draws,
            total_moves=total_moves,
            white_to_move_games=white_to_move_games,
        )
        log_event(
            self.debug_logger,
            logging.DEBUG,
            "selfplay_chunk_finished",
            "selfplay chunk finished source=%s games=%d black_wins=%d white_wins=%d draws=%d avg_moves=%.2f",
            source,
            len(result.completed_games),
            result.black_wins,
            result.white_wins,
            result.draws,
            0.0 if not result.completed_games else result.total_moves / max(1, len(result.completed_games)),
            iteration=self.iteration,
            source=source,
            games=len(result.completed_games),
            black_wins=result.black_wins,
            white_wins=result.white_wins,
            draws=result.draws,
            avg_moves=0.0 if not result.completed_games else result.total_moves / max(1, len(result.completed_games)),
            white_to_move_games=result.white_to_move_games,
        )
        return result

    def _maybe_save_progress(self, games_completed: int, last_saved_games: int, metadata: dict[str, object]) -> int:
        interval_batches = max(0, self.config.checkpoint.progress_interval_batches)
        interval_games = interval_batches * max(1, self.config.selfplay.batch_size)
        elapsed = time.monotonic() - self.last_progress_save_time
        should_save = False
        if interval_games > 0 and games_completed - last_saved_games >= interval_games:
            should_save = True
        if self.config.checkpoint.progress_interval_seconds > 0 and elapsed >= self.config.checkpoint.progress_interval_seconds:
            should_save = True
        if not should_save:
            return last_saved_games
        self.save_progress_state(metadata)
        self.last_progress_save_time = time.monotonic()
        log_event(
            self.debug_logger,
            logging.DEBUG,
            "selfplay_progress_saved",
            "progress state saved iteration=%d games_completed=%d",
            int(metadata.get("iteration", self.iteration)),
            games_completed,
            iteration=int(metadata.get("iteration", self.iteration)),
            games_completed=games_completed,
            progress_interval_batches=self.config.checkpoint.progress_interval_batches,
        )
        return games_completed

    def generate_selfplay(self, simulations: int, selfplay_plan: dict[str, float | int | str]) -> dict[str, float | int]:
        total_games = self.config.selfplay.games_per_iteration
        candidate_games = int(selfplay_plan["candidate_games"])
        best_games = total_games - candidate_games
        openings = sample_balanced_openings(self.config.rules.board_size, total_games, self.selfplay_rng)
        source_openings = {
            "candidate": openings[:candidate_games],
            "best": openings[candidate_games:],
        }
        counters: dict[str, int | float] = {
            "black_wins": 0,
            "white_wins": 0,
            "draws": 0,
            "total_moves": 0,
            "completed_games": 0,
            "candidate_games": 0,
            "best_games": 0,
            "white_to_move_games": 0,
        }
        last_saved_games = 0
        source_models = {"candidate": self.model, "best": self.best_model}
        evaluators: dict[str, Evaluator] = {}
        started_at = time.monotonic()
        log_event(
            self.debug_logger,
            logging.INFO,
            "selfplay_started",
            "selfplay started iteration=%d simulations=%d total_games=%d candidate_games=%d best_games=%d search_threads=%d batch_size=%d leaves_per_batch=%d virtual_loss=%.3f",
            self.iteration,
            simulations,
            total_games,
            candidate_games,
            best_games,
            self.config.selfplay.search_threads,
            self.config.selfplay.batch_size,
            self.config.selfplay.leaves_per_batch,
            self.config.selfplay.virtual_loss,
            iteration=self.iteration,
            simulations=simulations,
            total_games=total_games,
            candidate_games=candidate_games,
            best_games=best_games,
            search_threads=self.config.selfplay.search_threads,
            batch_size=self.config.selfplay.batch_size,
            leaves_per_batch=self.config.selfplay.leaves_per_batch,
            virtual_loss=self.config.selfplay.virtual_loss,
        )

        if self.config.selfplay.search_threads <= 1:
            worker_batch_size = max(1, self.config.selfplay.batch_size)
            try:
                for source in ("candidate", "best"):
                    if not source_openings[source]:
                        continue
                    evaluators[source] = self._build_evaluator(source_models[source])
                    result = self._run_selfplay_chunk(source, source_openings[source], simulations, worker_batch_size, evaluators[source])
                    self._consume_selfplay_chunk(result, counters)
                    last_saved_games = self._maybe_save_progress(
                        int(counters["completed_games"]),
                        last_saved_games,
                        {
                            "iteration": self.iteration,
                            "status": "selfplay_progress",
                            "games_completed_in_iteration": int(counters["completed_games"]),
                            "best_iteration": self.best_iteration,
                            "total_updates": self.total_updates,
                        },
                    )
            finally:
                for evaluator in evaluators.values():
                    evaluator.close()
        else:
            worker_counts = self._parallel_source_workers(candidate_games, best_games)
            total_workers = max(1, worker_counts["candidate"] + worker_counts["best"])
            worker_batch_size = max(1, (self.config.selfplay.batch_size + total_workers - 1) // total_workers)
            try:
                for source in ("candidate", "best"):
                    if source_openings[source]:
                        evaluators[source] = self._build_evaluator(source_models[source])
                with ThreadPoolExecutor(max_workers=total_workers) as executor:
                    futures = []
                    for source in ("candidate", "best"):
                        chunks = split_evenly(source_openings[source], worker_counts[source])
                        for chunk in chunks:
                            futures.append(
                                executor.submit(
                                    self._run_selfplay_chunk,
                                    source,
                                    chunk,
                                    simulations,
                                    worker_batch_size,
                                    evaluators[source],
                                )
                            )
                    for future in as_completed(futures):
                        result = future.result()
                        self._consume_selfplay_chunk(result, counters)
                        last_saved_games = self._maybe_save_progress(
                            int(counters["completed_games"]),
                            last_saved_games,
                            {
                                "iteration": self.iteration,
                                "status": "selfplay_progress",
                                "games_completed_in_iteration": int(counters["completed_games"]),
                                "best_iteration": self.best_iteration,
                                "total_updates": self.total_updates,
                            },
                        )
            finally:
                for evaluator in evaluators.values():
                    evaluator.close()

        total_played = int(counters["black_wins"] + counters["white_wins"] + counters["draws"])
        avg_moves = 0.0 if total_played == 0 else float(counters["total_moves"]) / total_played
        duration_seconds = time.monotonic() - started_at
        log_event(
            self.debug_logger,
            logging.INFO,
            "selfplay_finished",
            "selfplay finished iteration=%d games=%d candidate_games=%d best_games=%d black_wins=%d white_wins=%d draws=%d avg_moves=%.2f replay_games=%d replay_samples=%d",
            self.iteration,
            total_played,
            int(counters["candidate_games"]),
            int(counters["best_games"]),
            int(counters["black_wins"]),
            int(counters["white_wins"]),
            int(counters["draws"]),
            avg_moves,
            self.replay.games_seen,
            len(self.replay),
            iteration=self.iteration,
            games=total_played,
            candidate_games=int(counters["candidate_games"]),
            best_games=int(counters["best_games"]),
            black_wins=int(counters["black_wins"]),
            white_wins=int(counters["white_wins"]),
            draws=int(counters["draws"]),
            avg_moves=avg_moves,
            white_to_move_games=int(counters["white_to_move_games"]),
            replay_games=self.replay.games_seen,
            replay_samples=len(self.replay),
            duration_seconds=duration_seconds,
            games_per_second=0.0 if duration_seconds <= 0 else total_played / duration_seconds,
            positions_per_second=0.0 if duration_seconds <= 0 else float(counters["total_moves"]) / duration_seconds,
        )
        return {
            "selfplay_games": total_played,
            "selfplay_black_wins": int(counters["black_wins"]),
            "selfplay_white_wins": int(counters["white_wins"]),
            "selfplay_draws": int(counters["draws"]),
            "selfplay_avg_moves": round(avg_moves, 2),
            "selfplay_candidate_games": int(counters["candidate_games"]),
            "selfplay_best_games": int(counters["best_games"]),
            "selfplay_white_to_move_games": int(counters["white_to_move_games"]),
            "replay_samples": len(self.replay),
            "replay_games": self.replay.games_seen,
        }

    def train_model(self, model: PolicyValueNet) -> dict[str, float]:
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        updates_done = 0

        policy_weight = float(self.config.optimization.policy_loss_weight)
        value_weight = float(self.config.optimization.value_loss_weight)
        recency_temperature = float(self.config.optimization.recency_temperature)
        started_at = time.monotonic()
        log_event(
            self.debug_logger,
            logging.INFO,
            "training_started",
            "training started iteration=%d updates_per_iteration=%d batch_size=%d policy_weight=%.3f value_weight=%.3f recency_temperature=%.3f",
            self.iteration,
            self.config.optimization.updates_per_iteration,
            self.config.optimization.batch_size,
            policy_weight,
            value_weight,
            recency_temperature,
            iteration=self.iteration,
            updates_per_iteration=self.config.optimization.updates_per_iteration,
            batch_size=self.config.optimization.batch_size,
            policy_weight=policy_weight,
            value_weight=value_weight,
            recency_temperature=recency_temperature,
        )
        for _ in range(self.config.optimization.updates_per_iteration):
            if self._should_stop():
                break
            states, target_policy, target_value = self.replay.sample_batch(
                self.config.optimization.batch_size,
                self.device,
                recency_temperature=recency_temperature,
            )
            self.optimizer.zero_grad(set_to_none=True)
            with amp_context(self.device, self.config.use_amp):
                logits, value = model(states)
                policy_loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(value, target_value)
                loss = policy_weight * policy_loss + value_weight * value_loss
            self.scaler.scale(loss).backward()
            if self.config.optimization.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.config.optimization.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += float(loss.item())
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            updates_done += 1
            if updates_done <= 3 or updates_done % 16 == 0:
                log_event(
                    self.debug_logger,
                    logging.DEBUG,
                    "training_progress",
                    "training progress iteration=%d update=%d/%d loss=%.6f policy_loss=%.6f value_loss=%.6f lr=%.8f",
                    self.iteration,
                    updates_done,
                    self.config.optimization.updates_per_iteration,
                    float(loss.item()),
                    float(policy_loss.item()),
                    float(value_loss.item()),
                    float(self.optimizer.param_groups[0]["lr"]),
                    iteration=self.iteration,
                    update=updates_done,
                    total_updates_in_iteration=self.config.optimization.updates_per_iteration,
                    loss=float(loss.item()),
                    policy_loss=float(policy_loss.item()),
                    value_loss=float(value_loss.item()),
                    learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                )

        self.total_updates += updates_done
        updates = float(max(1, updates_done))
        duration_seconds = time.monotonic() - started_at
        stats = {
            "train_loss": round(total_loss / updates, 6),
            "policy_loss": round(total_policy_loss / updates, 6),
            "value_loss": round(total_value_loss / updates, 6),
            "learning_rate": round(float(self.optimizer.param_groups[0]["lr"]), 8),
            "updates_done": updates_done,
        }
        log_event(
            self.debug_logger,
            logging.INFO,
            "training_finished",
            "training finished iteration=%d updates_done=%d train_loss=%.6f policy_loss=%.6f value_loss=%.6f",
            self.iteration,
            updates_done,
            stats["train_loss"],
            stats["policy_loss"],
            stats["value_loss"],
            iteration=self.iteration,
            duration_seconds=duration_seconds,
            updates_done=updates_done,
            updates_per_second=0.0 if duration_seconds <= 0 else updates_done / duration_seconds,
            train_loss=stats["train_loss"],
            policy_loss=stats["policy_loss"],
            value_loss=stats["value_loss"],
            learning_rate=stats["learning_rate"],
        )
        return stats

    def evaluate_candidate(self, selfplay_simulations: int) -> dict[str, float | int | bool | str]:
        if self.config.arena.games <= 0 or self.config.arena.simulations <= 0:
            return {
                "accepted": True,
                "arena_phase": "disabled",
                "arena_games": 0,
                "arena_candidate_black_win_rate": 0.0,
                "arena_candidate_white_win_rate": 0.0,
                "arena_white_win_rate_threshold": 0.0,
                "arena_passes_white_gate": True,
                "arena_candidate_wins": 0,
                "arena_best_wins": 0,
                "arena_draws": 0,
                "arena_candidate_black_wins": 0,
                "arena_candidate_white_wins": 0,
                "arena_win_rate": 1.0,
            }

        arena_phase = "strict"
        arena_games = self.config.arena.games
        arena_simulations = self.config.arena.simulations
        arena_accept_win_rate = self.config.arena.accept_win_rate
        arena_min_white_win_rate = self.config.arena.min_white_win_rate
        if self.best_iteration == 0:
            arena_phase = "bootstrap"
            arena_games = self.config.arena.bootstrap_games
            arena_simulations = self.config.arena.bootstrap_simulations
            arena_accept_win_rate = self.config.arena.bootstrap_accept_win_rate
            arena_min_white_win_rate = self.config.arena.bootstrap_min_white_win_rate
            arena_simulations = max(arena_simulations, selfplay_simulations)
        started_at = time.monotonic()
        log_event(
            self.debug_logger,
            logging.INFO,
            "arena_started",
            "arena started iteration=%d phase=%s games=%d simulations=%d accept_win_rate=%.4f min_white_win_rate=%.4f",
            self.iteration,
            arena_phase,
            arena_games,
            arena_simulations,
            arena_accept_win_rate,
            arena_min_white_win_rate,
            iteration=self.iteration,
            phase=arena_phase,
            games=arena_games,
            simulations=arena_simulations,
            accept_win_rate=arena_accept_win_rate,
            min_white_win_rate=arena_min_white_win_rate,
        )

        arena = Arena(
            candidate_model=self.model,
            best_model=self.best_model,
            device=self.device,
            board_size=self.config.rules.board_size,
            exactly_five=self.config.rules.exactly_five,
            simulations=arena_simulations,
            c_puct=self.config.selfplay.c_puct,
            use_amp=self.config.use_amp,
            search_threads=self.config.selfplay.search_threads,
            inference_batch_size=self.config.selfplay.inference_batch_size,
            inference_wait_ms=self.config.selfplay.inference_wait_ms,
            leaves_per_batch=self.config.selfplay.leaves_per_batch,
            virtual_loss=self.config.selfplay.virtual_loss,
            logger=self.debug_logger.getChild("arena"),
            iteration=self.iteration,
            phase=arena_phase,
        )
        result = arena.evaluate(arena_games)
        side_games = max(1, result.games // 2)
        candidate_black_win_rate = result.candidate_black_wins / side_games
        candidate_white_win_rate = result.candidate_white_wins / side_games
        passes_white_gate = candidate_white_win_rate >= arena_min_white_win_rate
        accepted = result.candidate_win_rate >= arena_accept_win_rate and passes_white_gate
        stats = {
            "accepted": accepted,
            "arena_phase": arena_phase,
            "arena_games": result.games,
            "arena_simulations": arena_simulations,
            "arena_accept_win_rate": round(arena_accept_win_rate, 4),
            "arena_white_win_rate_threshold": round(arena_min_white_win_rate, 4),
            "arena_passes_white_gate": passes_white_gate,
            "arena_candidate_wins": result.candidate_wins,
            "arena_best_wins": result.best_wins,
            "arena_draws": result.draws,
            "arena_candidate_black_wins": result.candidate_black_wins,
            "arena_candidate_white_wins": result.candidate_white_wins,
            "arena_candidate_black_win_rate": round(candidate_black_win_rate, 4),
            "arena_candidate_white_win_rate": round(candidate_white_win_rate, 4),
            "arena_win_rate": round(result.candidate_win_rate, 4),
        }
        log_event(
            self.debug_logger,
            logging.INFO,
            "arena_finished",
            "arena finished iteration=%d phase=%s accepted=%s arena_win_rate=%.4f white_win_rate=%.4f",
            self.iteration,
            arena_phase,
            accepted,
            stats["arena_win_rate"],
            stats["arena_candidate_white_win_rate"],
            iteration=self.iteration,
            duration_seconds=time.monotonic() - started_at,
            **stats,
        )
        return stats

    def save_model_checkpoints(self, metadata: dict[str, object]) -> None:
        latest_path = self.checkpoint_dir / "latest.pt"
        best_path = self.checkpoint_dir / "best.pt"
        latest_metadata = {**metadata, "checkpoint_role": "candidate"}
        best_metadata = {
            **self.best_checkpoint_metadata,
            "checkpoint_role": "best",
            "best_iteration": self.best_iteration,
            "best_arena_win_rate": self.best_arena_win_rate,
        }
        save_checkpoint(latest_path, self.model, self.config, latest_metadata)
        save_checkpoint(best_path, self.best_model, self.config, best_metadata)
        log_event(
            self.debug_logger,
            logging.DEBUG,
            "model_checkpoints_saved",
            "saved model checkpoints latest=%s best=%s iteration=%d",
            latest_path,
            best_path,
            self.iteration,
            iteration=self.iteration,
            latest_path=latest_path,
            best_path=best_path,
            best_iteration=self.best_iteration,
        )
        save_interval = 1 if self.config.checkpoint.save_every_iteration else max(0, self.config.checkpoint.save_iteration_interval)
        if save_interval > 0 and self.iteration % save_interval == 0:
            save_checkpoint(self.checkpoint_dir / f"iter_{self.iteration:04d}.pt", self.model, self.config, latest_metadata)
            log_event(
                self.debug_logger,
                logging.DEBUG,
                "iteration_checkpoint_saved",
                "saved iteration checkpoint iteration=%d interval=%d",
                self.iteration,
                save_interval,
                iteration=self.iteration,
                interval=save_interval,
            )

    def save_progress_state(self, metadata: dict[str, object]) -> None:
        progress_metadata = {
            **metadata,
            "iteration": int(metadata.get("iteration", self.iteration)),
            "best_iteration": int(metadata.get("best_iteration", self.best_iteration)),
            "best_arena_win_rate": float(metadata.get("best_arena_win_rate", self.best_arena_win_rate)),
            "total_updates": int(metadata.get("total_updates", self.total_updates)),
            "elapsed_seconds": self.elapsed_seconds_offset + time.monotonic() - self.start_time,
            "replay_games": self.replay.games_seen,
            "replay_samples": len(self.replay),
        }
        self.progress_file.write_text(json.dumps(progress_metadata, ensure_ascii=False) + "\n", encoding="utf-8")
        log_event(
            self.debug_logger,
            logging.DEBUG,
            "progress_file_saved",
            "wrote progress file path=%s metadata=%s",
            self.progress_file,
            progress_metadata,
            iteration=progress_metadata["iteration"],
            progress_path=self.progress_file,
            metadata=progress_metadata,
        )

    def save_runtime_state(self, metadata: dict[str, object]) -> None:
        runtime_metadata = {
            **metadata,
            "iteration": int(metadata.get("iteration", self.iteration)),
            "best_iteration": int(metadata.get("best_iteration", self.best_iteration)),
            "best_arena_win_rate": float(metadata.get("best_arena_win_rate", self.best_arena_win_rate)),
            "total_updates": int(metadata.get("total_updates", self.total_updates)),
            "elapsed_seconds": self.elapsed_seconds_offset + time.monotonic() - self.start_time,
        }
        save_training_state(
            self.checkpoint_dir / "trainer_state.pt",
            self.model,
            self.best_model,
            self.config,
            replay_state=self.replay.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=None if self.scheduler is None else self.scheduler.state_dict(),
            scaler_state=self.scaler.state_dict(),
            metadata=runtime_metadata,
        )
        log_event(
            self.debug_logger,
            logging.DEBUG,
            "runtime_state_saved",
            "saved runtime state path=%s iteration=%d replay_games=%d replay_samples=%d",
            self.checkpoint_dir / "trainer_state.pt",
            runtime_metadata["iteration"],
            self.replay.games_seen,
            len(self.replay),
            iteration=runtime_metadata["iteration"],
            trainer_state_path=self.checkpoint_dir / "trainer_state.pt",
            replay_games=self.replay.games_seen,
            replay_samples=len(self.replay),
        )
        self.save_progress_state(runtime_metadata)
        self.last_progress_save_time = time.monotonic()

    def _restore_from_checkpoint(self, resume_path: str) -> None:
        path = Path(resume_path)
        requested_config = self.config
        replay_state = None
        trainer_extras: dict[str, object] = {}
        if path.name == "trainer_state.pt":
            model, loaded_config, metadata, replay_state, trainer_extras = load_training_state(path, device="cpu")
        else:
            model, loaded_config, metadata = load_checkpoint(path, device="cpu")

        self._validate_resume_config(requested_config, loaded_config)
        self.config = requested_config
        self.device = resolve_device(self.config.device)
        self.checkpoint_dir = self.config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.checkpoint_dir / "metrics.jsonl"
        self.progress_file = self.checkpoint_dir / "runtime_progress.json"
        self.debug_log_file = self.checkpoint_dir / "training.debug.log"
        self.debug_jsonl_file = self.checkpoint_dir / "training.debug.jsonl"
        self._configure_debug_logging()
        self.replay = ReplayBuffer(self.config.optimization.replay_capacity)
        if replay_state:
            self.replay.load_state_dict(replay_state)
        self.model = model.to(self.device)
        self.best_model = clone_model(self.model).to(self.device)
        self.best_model.eval()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.use_amp and self.device.type == "cuda")
        self.iteration = int(metadata.get("iteration", 0))
        self.total_updates = int(metadata.get("total_updates", 0))
        self.best_iteration = int(metadata.get("best_iteration", 0))
        self.best_arena_win_rate = float(metadata.get("best_arena_win_rate", 0.0))
        self.best_checkpoint_metadata = {
            "iteration": self.best_iteration,
            "elapsed_hours": float(metadata.get("elapsed_hours", 0.0)),
            "best_iteration": self.best_iteration,
            "best_arena_win_rate": self.best_arena_win_rate,
            "total_updates": self.total_updates,
            "status": "restored_best",
        }
        self.elapsed_seconds_offset = float(metadata.get("elapsed_seconds", 0.0))
        self.start_time = time.monotonic()
        self.last_progress_save_time = self.start_time

        best_model_state = trainer_extras.get("best_model_state")
        if isinstance(best_model_state, dict):
            self.best_model.load_state_dict(best_model_state)
        else:
            best_path = self.checkpoint_dir / "best.pt"
            if best_path.exists():
                best_model, _, best_metadata = load_checkpoint(best_path, device="cpu")
                self.best_model = best_model.to(self.device)
                self.best_iteration = int(best_metadata.get("best_iteration", self.best_iteration))
                self.best_arena_win_rate = float(best_metadata.get("best_arena_win_rate", self.best_arena_win_rate))
                self.best_checkpoint_metadata = dict(best_metadata)
        self.best_model.eval()

        optimizer_state = trainer_extras.get("optimizer_state")
        if isinstance(optimizer_state, dict):
            self.optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_state_to_device()

        scheduler_state = trainer_extras.get("scheduler_state")
        if self.scheduler is not None and isinstance(scheduler_state, dict):
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = trainer_extras.get("scaler_state")
        if isinstance(scaler_state, dict):
            self.scaler.load_state_dict(scaler_state)

        self._log(
            {
                "event": "resume",
                "resume_path": str(path),
                "iteration": self.iteration,
                "best_iteration": self.best_iteration,
                "replay_games": self.replay.games_seen,
                "replay_samples": len(self.replay),
                "total_updates": self.total_updates,
                "elapsed_hours": round(self.elapsed_hours, 4),
            }
        )
        log_event(
            self.debug_logger,
            logging.INFO,
            "resume_completed",
            "resume complete path=%s iteration=%d best_iteration=%d replay_games=%d replay_samples=%d total_updates=%d",
            path,
            self.iteration,
            self.best_iteration,
            self.replay.games_seen,
            len(self.replay),
            self.total_updates,
            resume_path=path,
            iteration=self.iteration,
            best_iteration=self.best_iteration,
            replay_games=self.replay.games_seen,
            replay_samples=len(self.replay),
            total_updates=self.total_updates,
        )

    def _validate_resume_config(self, requested: RunConfig, loaded: RunConfig) -> None:
        if requested.rules != loaded.rules:
            raise ValueError("resume config rules do not match checkpoint rules")
        if requested.network != loaded.network:
            raise ValueError("resume config network does not match checkpoint network")
        log_event(
            self.debug_logger,
            logging.DEBUG,
            "resume_config_validated",
            "resume config validated against checkpoint config",
            rules=str(requested.rules),
            network=str(requested.network),
        )

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _handle_stop_signal(self, signum: int, _frame: object) -> None:
        self.stop_requested = True
        log_event(
            self.debug_logger,
            logging.WARNING,
            "stop_signal_received",
            "received stop signal signum=%d",
            signum,
            signal=signum,
            iteration=self.iteration,
        )
        print(json.dumps({"event": "signal", "signal": signum, "message": "stop requested"}, ensure_ascii=False), flush=True)

    def _log(self, payload: dict[str, object]) -> None:
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train OmokAI with self-play reinforcement learning.")
    parser.add_argument("--config", type=str, default="configs/rocm_24h.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    if args.max_hours is not None:
        config.max_hours = args.max_hours
    if args.device is not None:
        config.device = args.device
    set_seed(config.seed)

    trainer = Trainer(config=config, resume_path=args.resume)
    print(
        json.dumps(
            {
                "event": "startup",
                "device": str(trainer.device),
                "cuda_available": torch.cuda.is_available(),
                "rocm_build": is_rocm_build(),
                "torch_version": torch.__version__,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    log_event(
        trainer.debug_logger,
        logging.INFO,
        "startup_emitted",
        "startup event emitted device=%s cuda_available=%s rocm_build=%s torch_version=%s",
        trainer.device,
        torch.cuda.is_available(),
        is_rocm_build(),
        torch.__version__,
        device=str(trainer.device),
        cuda_available=torch.cuda.is_available(),
        rocm_build=is_rocm_build(),
        torch_version=torch.__version__,
    )
    try:
        trainer.run()
    except BaseException:
        log_event(
            trainer.debug_logger,
            logging.ERROR,
            "training_exception",
            "training process terminated with an exception",
            iteration=trainer.iteration,
            best_iteration=trainer.best_iteration,
        )
        trainer.debug_logger.exception("training process terminated with an exception")
        raise


if __name__ == "__main__":
    main()
