from __future__ import annotations

import argparse
import json
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .arena import Arena
from .board import GameState
from .checkpoint import load_checkpoint, load_training_state, save_checkpoint, save_training_state
from .config import RunConfig, load_config
from .device import amp_context, is_rocm_build, resolve_device
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


class Trainer:
    def __init__(self, config: RunConfig, resume_path: str | None = None) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.checkpoint_dir / "metrics.jsonl"
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
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._handle_stop_signal)
        signal.signal(signal.SIGTERM, self._handle_stop_signal)
        if resume_path:
            self._restore_from_checkpoint(resume_path)

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
        self.save_model_checkpoints(startup_metadata)
        self.save_runtime_state(startup_metadata)

        while not self._should_stop():
            self.iteration += 1
            simulations = self.current_simulations()
            selfplay_plan = self.current_selfplay_plan()
            selfplay_stats = self.generate_selfplay(simulations, selfplay_plan)

            if self._should_stop():
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

            if self._should_stop():
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
            if arena_stats["accepted"]:
                self.best_model.load_state_dict(self.model.state_dict())
                self.best_model.eval()
                self.best_iteration = self.iteration
                self.best_arena_win_rate = float(arena_stats["arena_win_rate"])
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

    def generate_selfplay(self, simulations: int, selfplay_plan: dict[str, float | int | str]) -> dict[str, float | int]:
        searches: dict[str, MCTS] = {}

        def resolve_search(source: str) -> MCTS:
            if source not in searches:
                model = self.model if source == "candidate" else self.best_model
                searches[source] = MCTS(
                    model=model,
                    device=self.device,
                    c_puct=self.config.selfplay.c_puct,
                    dirichlet_alpha=self.config.selfplay.dirichlet_alpha,
                    dirichlet_epsilon=self.config.selfplay.dirichlet_epsilon,
                    use_amp=self.config.use_amp,
                )
            return searches[source]

        games_left = self.config.selfplay.games_per_iteration
        candidate_games_left = int(selfplay_plan["candidate_games"])
        best_games_left = games_left - candidate_games_left
        openings = sample_balanced_openings(self.config.rules.board_size, games_left, self.selfplay_rng)
        black_wins = 0
        white_wins = 0
        draws = 0
        total_moves = 0
        selfplay_candidate_games = 0
        selfplay_best_games = 0
        selfplay_white_to_move_games = 0

        while games_left > 0 and not self._should_stop():
            if candidate_games_left > 0 and best_games_left > 0:
                candidate_ratio = candidate_games_left / max(1, candidate_games_left + best_games_left)
                batch_source = "candidate" if self.selfplay_rng.random() < candidate_ratio else "best"
            elif candidate_games_left > 0:
                batch_source = "candidate"
            else:
                batch_source = "best"

            batch_games_left = candidate_games_left if batch_source == "candidate" else best_games_left
            batch_size = min(self.config.selfplay.batch_size, games_left, batch_games_left)
            states: list[GameState] = []
            histories: list[list[PendingSample]] = []
            for _ in range(batch_size):
                opening = openings.pop(0)
                state = GameState(self.config.rules.board_size, self.config.rules.exactly_five)
                for action in opening:
                    state.apply_action(action)
                if state.to_play == -1:
                    selfplay_white_to_move_games += 1
                states.append(state)
                histories.append([])

            while states and not self._should_stop():
                temperatures = [1.0 if state.move_count < self.config.selfplay.temperature_moves else 0.0 for state in states]
                results = resolve_search(batch_source).search_batch(states, simulations, temperatures, add_noise=True)

                next_states: list[GameState] = []
                next_histories: list[list[PendingSample]] = []
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
                        self.replay.add_game(history, state.winner)
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
                states = next_states
                histories = next_histories
            games_left -= batch_size
            if batch_source == "candidate":
                candidate_games_left -= batch_size
                selfplay_candidate_games += batch_size
            else:
                best_games_left -= batch_size
                selfplay_best_games += batch_size
            self.save_runtime_state(
                {
                    "iteration": self.iteration,
                    "status": "selfplay_progress",
                    "games_completed_in_iteration": self.config.selfplay.games_per_iteration - games_left,
                    "best_iteration": self.best_iteration,
                    "total_updates": self.total_updates,
                }
            )

        total_games = black_wins + white_wins + draws
        avg_moves = 0.0 if total_games == 0 else total_moves / total_games
        return {
            "selfplay_games": total_games,
            "selfplay_black_wins": black_wins,
            "selfplay_white_wins": white_wins,
            "selfplay_draws": draws,
            "selfplay_avg_moves": round(avg_moves, 2),
            "selfplay_candidate_games": selfplay_candidate_games,
            "selfplay_best_games": selfplay_best_games,
            "selfplay_white_to_move_games": selfplay_white_to_move_games,
            "replay_samples": len(self.replay),
            "replay_games": self.replay.games_seen,
        }

    def train_model(self, model: PolicyValueNet) -> dict[str, float]:
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        updates_done = 0

        for _ in range(self.config.optimization.updates_per_iteration):
            if self._should_stop():
                break
            states, target_policy, target_value = self.replay.sample_batch(self.config.optimization.batch_size, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with amp_context(self.device, self.config.use_amp):
                logits, value = model(states)
                policy_loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(value, target_value)
                loss = policy_loss + value_loss
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

        self.total_updates += updates_done
        updates = float(max(1, updates_done))
        return {
            "train_loss": round(total_loss / updates, 6),
            "policy_loss": round(total_policy_loss / updates, 6),
            "value_loss": round(total_value_loss / updates, 6),
            "learning_rate": round(float(self.optimizer.param_groups[0]["lr"]), 8),
            "updates_done": updates_done,
        }

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

        arena = Arena(
            candidate_model=self.model,
            best_model=self.best_model,
            device=self.device,
            board_size=self.config.rules.board_size,
            exactly_five=self.config.rules.exactly_five,
            simulations=arena_simulations,
            c_puct=self.config.selfplay.c_puct,
            use_amp=self.config.use_amp,
        )
        result = arena.evaluate(arena_games)
        side_games = max(1, result.games // 2)
        candidate_black_win_rate = result.candidate_black_wins / side_games
        candidate_white_win_rate = result.candidate_white_wins / side_games
        passes_white_gate = candidate_white_win_rate >= arena_min_white_win_rate
        accepted = result.candidate_win_rate >= arena_accept_win_rate and passes_white_gate
        return {
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
        if self.config.checkpoint.save_every_iteration:
            save_checkpoint(self.checkpoint_dir / f"iter_{self.iteration:04d}.pt", self.model, self.config, latest_metadata)

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

    def _restore_from_checkpoint(self, resume_path: str) -> None:
        path = Path(resume_path)
        replay_state = None
        trainer_extras: dict[str, object] = {}
        if path.name == "trainer_state.pt":
            model, loaded_config, metadata, replay_state, trainer_extras = load_training_state(path, device="cpu")
        else:
            model, loaded_config, metadata = load_checkpoint(path, device="cpu")

        self.config = loaded_config
        self.device = resolve_device(self.config.device)
        self.checkpoint_dir = self.config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.checkpoint_dir / "metrics.jsonl"
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

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _handle_stop_signal(self, signum: int, _frame: object) -> None:
        self.stop_requested = True
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
    trainer.run()


if __name__ == "__main__":
    main()
