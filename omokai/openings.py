from __future__ import annotations

import random
from typing import Iterable


def build_opening_sequences(board_size: int, count: int) -> list[list[int]]:
    center = board_size // 2
    templates = [
        (),
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
        ((0, 0), (0, -1)),
        ((0, 0), (-1, 0)),
        ((0, 0), (1, -1)),
        ((0, 0), (-1, 1)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (0, 1)),
        ((0, 0), (1, 1), (0, 1)),
        ((0, 0), (0, 1), (0, -1)),
        ((0, 0), (1, 0), (-1, 0)),
        ((0, 0), (1, 1), (-1, -1)),
        ((0, 0), (1, -1), (-1, 1)),
    ]
    openings: list[list[int]] = []
    for template in templates:
        opening: list[int] = []
        used_actions: set[int] = set()
        valid = True
        for dr, dc in template:
            row = center + dr
            col = center + dc
            if not (0 <= row < board_size and 0 <= col < board_size):
                valid = False
                break
            action = row * board_size + col
            if action in used_actions:
                valid = False
                break
            used_actions.add(action)
            opening.append(action)
        if valid:
            openings.append(opening)
    if not openings:
        return [[] for _ in range(count)]
    return [list(openings[index % len(openings)]) for index in range(count)]


def apply_symmetry_to_action(action: int, board_size: int, transform: int) -> int:
    row, col = divmod(action, board_size)
    if transform >= 4:
        col = board_size - 1 - col
        transform -= 4
    for _ in range(transform):
        row, col = col, board_size - 1 - row
    return row * board_size + col


def transform_opening(opening: Iterable[int], board_size: int, transform: int) -> list[int]:
    transformed = [apply_symmetry_to_action(action, board_size, transform) for action in opening]
    # Template design should already avoid collisions, but keep this guard to avoid invalid starts.
    if len(set(transformed)) != len(transformed):
        return list(opening)
    return transformed


def sample_balanced_openings(board_size: int, count: int, rng: random.Random) -> list[list[int]]:
    openings = build_opening_sequences(board_size, count)
    transformed = [transform_opening(opening, board_size, rng.randrange(8)) for opening in openings]
    rng.shuffle(transformed)
    return transformed
