from __future__ import annotations

import numpy as np

from .board import GameState


def states_to_feature_planes(states: list[GameState]) -> np.ndarray:
    if not states:
        raise ValueError("states must not be empty")
    board_size = states[0].board_size
    boards = np.stack([state.board for state in states], axis=0).astype(np.int8, copy=False)
    to_play = np.fromiter((state.to_play for state in states), dtype=np.int8, count=len(states))
    last_actions = np.fromiter(
        ((-1 if state.last_action is None else state.last_action) for state in states),
        dtype=np.int32,
        count=len(states),
    )
    return encode_feature_planes_batch(boards, to_play, last_actions, board_size)


def encode_feature_planes_batch(
    boards: np.ndarray,
    to_play: np.ndarray,
    last_actions: np.ndarray,
    board_size: int | None = None,
) -> np.ndarray:
    if boards.ndim != 3:
        raise ValueError("boards must have shape [batch, board_size, board_size]")
    if board_size is None:
        board_size = int(boards.shape[-1])

    own = (boards == to_play[:, None, None]).astype(np.float32, copy=False)
    opp = (boards == -to_play[:, None, None]).astype(np.float32, copy=False)
    last = np.zeros_like(own, dtype=np.float32)
    valid_last = last_actions >= 0
    if np.any(valid_last):
        rows = last_actions[valid_last] // board_size
        cols = last_actions[valid_last] % board_size
        last[np.flatnonzero(valid_last), rows, cols] = 1.0
    color_value = (to_play == 1).astype(np.float32)[:, None, None]
    color = np.broadcast_to(color_value, own.shape)

    planes = np.empty((boards.shape[0], 4, board_size, board_size), dtype=np.float32)
    planes[:, 0] = own
    planes[:, 1] = opp
    planes[:, 2] = last
    planes[:, 3] = color
    return planes


def apply_symmetry_batch(planes: np.ndarray, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if planes.ndim != 4:
        raise ValueError("planes must have shape [batch, channels, board_size, board_size]")
    if policy.ndim != 2:
        raise ValueError("policy must have shape [batch, action_size]")

    batch_size = planes.shape[0]
    board_size = planes.shape[-1]
    policy_grid = policy.reshape(batch_size, board_size, board_size)
    transforms = np.random.randint(8, size=batch_size)
    transformed_planes = np.empty_like(planes)
    transformed_policy = np.empty_like(policy_grid)

    for transform in range(8):
        mask = transforms == transform
        if not np.any(mask):
            continue
        planes_slice = planes[mask]
        policy_slice = policy_grid[mask]
        mirror = transform >= 4
        rotate = transform - 4 if mirror else transform
        if mirror:
            planes_slice = planes_slice[:, :, :, ::-1]
            policy_slice = policy_slice[:, :, ::-1]
        if rotate:
            planes_slice = np.rot90(planes_slice, rotate, axes=(-2, -1))
            policy_slice = np.rot90(policy_slice, rotate, axes=(-2, -1))
        transformed_planes[mask] = np.ascontiguousarray(planes_slice)
        transformed_policy[mask] = np.ascontiguousarray(policy_slice)

    return transformed_planes, transformed_policy.reshape(batch_size, -1)
