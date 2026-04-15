from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import RunConfig, _to_dataclass
from .network import PolicyValueNet


def save_checkpoint(
    path: str | Path,
    model: PolicyValueNet,
    config: RunConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "metadata": metadata or {},
    }
    torch.save(payload, target)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[PolicyValueNet, RunConfig, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = _to_dataclass(payload["config"])
    model = PolicyValueNet(config.rules.board_size, config.network)
    model.load_state_dict(payload["model_state"])
    return model, config, payload.get("metadata", {})


def save_training_state(
    path: str | Path,
    model: PolicyValueNet,
    best_model: PolicyValueNet,
    config: RunConfig,
    replay_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any] | None,
    scaler_state: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "trainer_state",
        "model_state": model.state_dict(),
        "best_model_state": best_model.state_dict(),
        "config": asdict(config),
        "metadata": metadata or {},
        "replay_state": replay_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
    }
    torch.save(payload, target)


def load_training_state(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[PolicyValueNet, RunConfig, dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = _to_dataclass(payload["config"])
    model = PolicyValueNet(config.rules.board_size, config.network)
    model.load_state_dict(payload["model_state"])
    extras = {
        "best_model_state": payload.get("best_model_state"),
        "optimizer_state": payload.get("optimizer_state"),
        "scheduler_state": payload.get("scheduler_state"),
        "scaler_state": payload.get("scaler_state"),
    }
    return model, config, payload.get("metadata", {}), payload.get("replay_state"), extras


def list_checkpoints(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*.pt") if path.name != "trainer_state.pt")
