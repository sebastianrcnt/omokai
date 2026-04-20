"""Dump expected features for a few hand-built positions (Python ground truth)."""
import json
from pathlib import Path

import numpy as np

from omokai.board import GameState
from omokai.features import states_to_feature_planes


def encode(state: GameState) -> dict:
    planes = states_to_feature_planes([state])[0]  # [4, N, N]
    return {
        "board_size": state.board_size,
        "to_play": int(state.to_play),
        "last_action": -1 if state.last_action is None else int(state.last_action),
        "board": state.board.flatten().astype(int).tolist(),
        "planes": planes.flatten().astype(float).tolist(),
    }


def main() -> None:
    cases = []

    # Empty 9x9 board
    s = GameState(board_size=9)
    cases.append({"name": "empty_9", **encode(s)})

    # A few moves
    s = GameState(board_size=9)
    for a in [40, 30, 31, 50, 22]:
        s.apply_action(a)
    cases.append({"name": "after_5_moves", **encode(s)})

    # Same-color view: black to play
    s = GameState(board_size=9)
    for a in [40, 30, 41, 31]:
        s.apply_action(a)
    cases.append({"name": "black_to_play", **encode(s)})

    # White to play with last move
    s = GameState(board_size=9)
    for a in [40, 30, 41]:
        s.apply_action(a)
    cases.append({"name": "white_to_play", **encode(s)})

    out = Path(__file__).resolve().parent / "expected.json"
    out.write_text(json.dumps(cases, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
