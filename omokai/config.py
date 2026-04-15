from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RulesConfig:
    board_size: int = 15
    exactly_five: bool = False


@dataclass(slots=True)
class NetworkConfig:
    input_planes: int = 4
    channels: int = 128
    blocks: int = 8
    value_hidden: int = 192
    se_reduction: int = 8


@dataclass(slots=True)
class SelfPlayConfig:
    games_per_iteration: int = 48
    batch_size: int = 12
    temperature_moves: int = 12
    dirichlet_alpha: float = 0.20
    dirichlet_epsilon: float = 0.25
    c_puct: float = 1.6
    simulation_ramp_iterations: int | None = None
    candidate_mix_fraction: float = 0.0
    mixed_iterations_after_promotion: int = 0
    simulation_schedule: list[dict[str, float | int]] = field(
        default_factory=lambda: [
            {"fraction": 0.0, "simulations": 48},
            {"fraction": 0.30, "simulations": 96},
            {"fraction": 0.70, "simulations": 160},
        ]
    )


@dataclass(slots=True)
class OptimizationConfig:
    batch_size: int = 256
    updates_per_iteration: int = 320
    learning_rate: float = 1.0e-3
    min_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    replay_capacity: int = 120_000
    warmup_games: int = 48


@dataclass(slots=True)
class ArenaConfig:
    games: int = 20
    simulations: int = 96
    accept_win_rate: float = 0.55
    min_white_win_rate: float = 0.0
    bootstrap_games: int = 12
    bootstrap_simulations: int = 96
    bootstrap_accept_win_rate: float = 0.50
    bootstrap_min_white_win_rate: float = 0.0


@dataclass(slots=True)
class CheckpointConfig:
    directory: str = "checkpoints/default"
    save_every_iteration: bool = True


@dataclass(slots=True)
class RunConfig:
    experiment_name: str = "omokai"
    seed: int = 17
    max_hours: float | None = 24.0
    device: str = "auto"
    use_amp: bool = True
    rules: RulesConfig = field(default_factory=RulesConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.checkpoint.directory)


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _to_dataclass(cfg: dict[str, Any]) -> RunConfig:
    raw_max_hours = cfg.get("max_hours", 24.0)
    return RunConfig(
        experiment_name=cfg.get("experiment_name", "omokai"),
        seed=int(cfg.get("seed", 17)),
        max_hours=None if raw_max_hours is None else float(raw_max_hours),
        device=str(cfg.get("device", "auto")),
        use_amp=bool(cfg.get("use_amp", True)),
        rules=RulesConfig(**cfg.get("rules", {})),
        network=NetworkConfig(**cfg.get("network", {})),
        selfplay=SelfPlayConfig(**cfg.get("selfplay", {})),
        optimization=OptimizationConfig(**cfg.get("optimization", {})),
        arena=ArenaConfig(**cfg.get("arena", {})),
        checkpoint=CheckpointConfig(**cfg.get("checkpoint", {})),
    )


def load_config(path: str | Path | None) -> RunConfig:
    default = RunConfig()
    default_dict = asdict(default)
    if path is None:
        return default
    raw = yaml.safe_load(Path(path).read_text()) or {}
    merged = _merge_dict(default_dict, raw)
    return _to_dataclass(merged)
