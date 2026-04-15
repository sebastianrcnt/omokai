from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .features import apply_symmetry_batch, encode_feature_planes_batch


@dataclass(slots=True)
class PendingSample:
    board: np.ndarray
    to_play: int
    last_action: int | None
    policy: np.ndarray


@dataclass(slots=True)
class ReplaySample:
    board: np.ndarray
    to_play: int
    last_action: int | None
    policy: np.ndarray
    value: float


def encode_sample(board: np.ndarray, to_play: int, last_action: int | None) -> np.ndarray:
    last_action_value = -1 if last_action is None else int(last_action)
    planes = encode_feature_planes_batch(
        boards=np.asarray(board, dtype=np.int8)[None, ...],
        to_play=np.asarray([to_play], dtype=np.int8),
        last_actions=np.asarray([last_action_value], dtype=np.int32),
        board_size=board.shape[0],
    )
    return planes[0]


def apply_symmetry(planes: np.ndarray, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    board_size = planes.shape[-1]
    policy_grid = policy.reshape(board_size, board_size)
    transform = np.random.randint(8)
    if transform >= 4:
        planes = planes[:, :, ::-1]
        policy_grid = policy_grid[:, ::-1]
        transform -= 4
    if transform:
        planes = np.rot90(planes, transform, axes=(-2, -1))
        policy_grid = np.rot90(policy_grid, transform)
    return np.ascontiguousarray(planes), np.ascontiguousarray(policy_grid.reshape(-1))


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.samples: deque[ReplaySample] = deque(maxlen=capacity)
        self.games_seen = 0

    def __len__(self) -> int:
        return len(self.samples)

    def add_game(
        self,
        history: Iterable[PendingSample],
        winner: int,
        value_discount: float = 1.0,
    ) -> None:
        history_list = list(history)
        total = len(history_list)
        if total == 0:
            return
        discount = float(np.clip(value_discount, 0.0, 1.0))
        for idx, item in enumerate(history_list):
            if winner == 0:
                value = 0.0
            else:
                outcome = 1.0 if winner == item.to_play else -1.0
                remaining = total - 1 - idx
                value = outcome * (discount ** remaining)
            self.samples.append(
                ReplaySample(
                    board=np.asarray(item.board, dtype=np.int8),
                    to_play=int(item.to_play),
                    last_action=item.last_action,
                    policy=np.asarray(item.policy, dtype=np.float32),
                    value=float(value),
                )
            )
        self.games_seen += 1

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
        recency_temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.samples) < batch_size:
            raise ValueError("not enough replay data to sample a batch")
        total = len(self.samples)
        if recency_temperature > 0.0 and total > 1:
            positions = np.arange(total, dtype=np.float64)
            log_weights = recency_temperature * (positions - (total - 1))
            log_weights -= log_weights.max()
            weights = np.exp(log_weights)
            weights /= weights.sum()
            indices = np.random.choice(total, size=batch_size, replace=False, p=weights)
        else:
            indices = np.random.choice(total, size=batch_size, replace=False)
        batch = [self.samples[index] for index in indices]
        boards = np.stack([sample.board for sample in batch], axis=0).astype(np.int8, copy=False)
        to_play = np.fromiter((sample.to_play for sample in batch), dtype=np.int8, count=batch_size)
        last_actions = np.fromiter(
            ((-1 if sample.last_action is None else sample.last_action) for sample in batch),
            dtype=np.int32,
            count=batch_size,
        )
        policy_batch = np.stack([sample.policy for sample in batch], axis=0).astype(np.float32, copy=False)
        value_batch = np.fromiter((sample.value for sample in batch), dtype=np.float32, count=batch_size)
        planes_batch = encode_feature_planes_batch(boards, to_play, last_actions, boards.shape[-1])
        planes_batch, policy_batch = apply_symmetry_batch(planes_batch, policy_batch)
        states = torch.from_numpy(np.ascontiguousarray(planes_batch)).to(device, non_blocking=True)
        policy = torch.from_numpy(np.ascontiguousarray(policy_batch)).to(device, non_blocking=True)
        value = torch.from_numpy(value_batch).to(device, non_blocking=True)
        return states, policy, value

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "games_seen": self.games_seen,
            "samples": [
                {
                    "board": sample.board,
                    "to_play": sample.to_play,
                    "last_action": sample.last_action,
                    "policy": sample.policy,
                    "value": sample.value,
                }
                for sample in self.samples
            ],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.samples = deque(maxlen=self.capacity)
        for item in state.get("samples", []):
            payload = dict(item)
            self.samples.append(
                ReplaySample(
                    board=np.asarray(payload["board"], dtype=np.int8),
                    to_play=int(payload["to_play"]),
                    last_action=payload["last_action"],
                    policy=np.asarray(payload["policy"], dtype=np.float32),
                    value=float(payload["value"]),
                )
            )
        self.games_seen = int(state.get("games_seen", 0))
