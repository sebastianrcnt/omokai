from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


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
    own = (board == to_play).astype(np.float32)
    opp = (board == -to_play).astype(np.float32)
    last = np.zeros_like(own, dtype=np.float32)
    if last_action is not None:
        row, col = divmod(last_action, board.shape[0])
        last[row, col] = 1.0
    color = np.full_like(own, 1.0 if to_play == 1 else 0.0, dtype=np.float32)
    return np.stack([own, opp, last, color], axis=0)


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

    def add_game(self, history: Iterable[PendingSample], winner: int) -> None:
        count = 0
        for item in history:
            value = 0.0 if winner == 0 else (1.0 if winner == item.to_play else -1.0)
            self.samples.append(
                ReplaySample(
                    board=np.asarray(item.board, dtype=np.int8),
                    to_play=int(item.to_play),
                    last_action=item.last_action,
                    policy=np.asarray(item.policy, dtype=np.float32),
                    value=value,
                )
            )
            count += 1
        if count:
            self.games_seen += 1

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.samples) < batch_size:
            raise ValueError("not enough replay data to sample a batch")
        indices = np.random.choice(len(self.samples), size=batch_size, replace=False)
        planes_batch = []
        policy_batch = []
        value_batch = []
        for index in indices:
            sample = self.samples[index]
            planes = encode_sample(sample.board, sample.to_play, sample.last_action)
            planes, policy = apply_symmetry(planes, sample.policy)
            planes_batch.append(planes)
            policy_batch.append(policy)
            value_batch.append(sample.value)
        states = torch.from_numpy(np.stack(planes_batch)).to(device, non_blocking=True)
        policy = torch.from_numpy(np.stack(policy_batch)).to(device, non_blocking=True)
        value = torch.tensor(value_batch, dtype=torch.float32, device=device)
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
