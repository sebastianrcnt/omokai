#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SVG_WIDTH = 1280
SVG_HEIGHT = 1180
MARGIN_LEFT = 88
MARGIN_RIGHT = 28
MARGIN_TOP = 112
MARGIN_BOTTOM = 52
PANEL_GAP = 28
PANEL_HEIGHT = 210
PLOT_WIDTH = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT


def read_metrics(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "iteration" not in payload:
            continue
        rows.append(payload)
    rows.sort(key=lambda item: int(item["iteration"]))
    return rows


def svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def fmt_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def panel_top(index: int) -> float:
    return MARGIN_TOP + index * (PANEL_HEIGHT + PANEL_GAP)


def x_scale(iteration: int, min_iter: int, max_iter: int) -> float:
    if max_iter == min_iter:
        return MARGIN_LEFT + PLOT_WIDTH / 2
    fraction = (iteration - min_iter) / (max_iter - min_iter)
    return MARGIN_LEFT + fraction * PLOT_WIDTH


def y_bounds(series: list[float], floor_zero: bool = False) -> tuple[float, float]:
    if not series:
        return (0.0, 1.0)
    low = min(series)
    high = max(series)
    if floor_zero:
        low = min(0.0, low)
    if math.isclose(low, high):
        delta = 1.0 if math.isclose(low, 0.0) else abs(low) * 0.2
        lower = low - delta
        if floor_zero:
            lower = max(0.0, lower)
        return (lower, high + delta)
    pad = (high - low) * 0.08
    lower = low - pad
    if floor_zero:
        lower = max(0.0, lower)
    return (lower, high + pad)


def y_scale(value: float, y0: float, y1: float, panel_y: float) -> float:
    if math.isclose(y0, y1):
        return panel_y + PANEL_HEIGHT / 2
    fraction = (value - y0) / (y1 - y0)
    return panel_y + PANEL_HEIGHT - fraction * PANEL_HEIGHT


def build_path(
    rows: list[dict[str, object]],
    key: str,
    min_iter: int,
    max_iter: int,
    y0: float,
    y1: float,
    panel_y: float,
) -> str:
    points: list[str] = []
    for row in rows:
        value = to_float(row.get(key))
        if value is None:
            continue
        x = x_scale(int(row["iteration"]), min_iter, max_iter)
        y = y_scale(value, y0, y1, panel_y)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def draw_panel_background(parts: list[str], title: str, subtitle: str, panel_y: float) -> None:
    parts.append(
        f'<rect x="{MARGIN_LEFT}" y="{panel_y:.1f}" width="{PLOT_WIDTH}" height="{PANEL_HEIGHT}" '
        'fill="#ffffff" stroke="#d7dde7" stroke-width="1.2" rx="10" />'
    )
    parts.append(
        f'<text x="{MARGIN_LEFT}" y="{panel_y - 14:.1f}" fill="#192231" font-size="20" font-weight="700">{svg_escape(title)}</text>'
    )
    parts.append(
        f'<text x="{MARGIN_LEFT + 170}" y="{panel_y - 14:.1f}" fill="#637085" font-size="13">{svg_escape(subtitle)}</text>'
    )


def draw_grid(parts: list[str], min_iter: int, max_iter: int, y0: float, y1: float, panel_y: float) -> None:
    for tick in range(5):
        fraction = tick / 4
        y = panel_y + PANEL_HEIGHT - fraction * PANEL_HEIGHT
        value = y0 + fraction * (y1 - y0)
        parts.append(
            f'<line x1="{MARGIN_LEFT}" y1="{y:.2f}" x2="{MARGIN_LEFT + PLOT_WIDTH}" y2="{y:.2f}" '
            'stroke="#edf1f6" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{MARGIN_LEFT - 12}" y="{y + 4:.2f}" fill="#6f7d90" font-size="12" text-anchor="end">{svg_escape(fmt_number(value))}</text>'
        )

    for tick in range(6):
        fraction = tick / 5
        x = MARGIN_LEFT + fraction * PLOT_WIDTH
        iteration = round(min_iter + fraction * (max_iter - min_iter))
        parts.append(
            f'<line x1="{x:.2f}" y1="{panel_y}" x2="{x:.2f}" y2="{panel_y + PANEL_HEIGHT}" '
            'stroke="#f4f7fb" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{panel_y + PANEL_HEIGHT + 18:.2f}" fill="#6f7d90" font-size="12" text-anchor="middle">{iteration}</text>'
        )


def draw_series(
    parts: list[str],
    rows: list[dict[str, object]],
    key: str,
    label: str,
    color: str,
    min_iter: int,
    max_iter: int,
    y0: float,
    y1: float,
    panel_y: float,
    legend_x: float,
    legend_y: float,
) -> None:
    path = build_path(rows, key, min_iter, max_iter, y0, y1, panel_y)
    if path:
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" points="{path}" />'
        )
    parts.append(
        f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 18}" y2="{legend_y}" stroke="{color}" stroke-width="3" stroke-linecap="round" />'
    )
    parts.append(
        f'<text x="{legend_x + 24}" y="{legend_y + 4}" fill="#304055" font-size="13">{svg_escape(label)}</text>'
    )


def draw_accept_markers(
    parts: list[str],
    rows: list[dict[str, object]],
    min_iter: int,
    max_iter: int,
    y0: float,
    y1: float,
    panel_y: float,
) -> None:
    for row in rows:
        if not bool(row.get("accepted")):
            continue
        arena_value = to_float(row.get("arena_win_rate"))
        if arena_value is None:
            continue
        x = x_scale(int(row["iteration"]), min_iter, max_iter)
        y = y_scale(arena_value, y0, y1, panel_y)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="#12b886" stroke="#ffffff" stroke-width="1.4" />')


def render_svg(rows: list[dict[str, object]], output_path: Path, input_path: Path) -> None:
    if not rows:
        raise ValueError("No metrics rows found.")

    min_iter = int(rows[0]["iteration"])
    max_iter = int(rows[-1]["iteration"])
    latest = rows[-1]

    loss_values = [
        value
        for row in rows
        for key in ("train_loss", "policy_loss", "value_loss")
        if (value := to_float(row.get(key))) is not None
    ]
    loss_y0, loss_y1 = y_bounds(loss_values, floor_zero=True)

    arena_values = [value for row in rows if (value := to_float(row.get("arena_win_rate"))) is not None]
    threshold_values = [value for row in rows if (value := to_float(row.get("arena_accept_win_rate"))) is not None]
    arena_y0, arena_y1 = y_bounds(arena_values + threshold_values + [0.0, 1.0], floor_zero=True)
    arena_y0 = min(0.0, arena_y0)
    arena_y1 = max(1.0, arena_y1)

    move_values = [value for row in rows if (value := to_float(row.get("selfplay_avg_moves"))) is not None]
    move_y0, move_y1 = y_bounds(move_values, floor_zero=True)

    replay_values = [value for row in rows if (value := to_float(row.get("replay_samples"))) is not None]
    replay_y0, replay_y1 = y_bounds(replay_values, floor_zero=True)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f6f8fb" />',
        '<text x="36" y="48" fill="#162033" font-size="28" font-weight="700">OmokAI Training Metrics</text>',
        f'<text x="36" y="76" fill="#607086" font-size="15">{svg_escape(input_path)}</text>',
        (
            f'<text x="36" y="98" fill="#425168" font-size="14">'
            f'latest iteration {int(latest["iteration"])}'
            f' | selfplay_source {svg_escape(latest.get("selfplay_source", "-"))}'
            f' | accepted {svg_escape(latest.get("accepted", "-"))}'
            f' | best_iteration {svg_escape(latest.get("best_iteration", "-"))}'
            f' | total_updates {svg_escape(latest.get("total_updates", "-"))}'
            '</text>'
        ),
    ]

    loss_panel_y = panel_top(0)
    draw_panel_background(parts, "Loss", "train_loss / policy_loss / value_loss", loss_panel_y)
    draw_grid(parts, min_iter, max_iter, loss_y0, loss_y1, loss_panel_y)
    draw_series(parts, rows, "train_loss", "train_loss", "#2b8a3e", min_iter, max_iter, loss_y0, loss_y1, loss_panel_y, MARGIN_LEFT + 20, loss_panel_y + 22)
    draw_series(parts, rows, "policy_loss", "policy_loss", "#1c7ed6", min_iter, max_iter, loss_y0, loss_y1, loss_panel_y, MARGIN_LEFT + 170, loss_panel_y + 22)
    draw_series(parts, rows, "value_loss", "value_loss", "#e8590c", min_iter, max_iter, loss_y0, loss_y1, loss_panel_y, MARGIN_LEFT + 330, loss_panel_y + 22)

    arena_panel_y = panel_top(1)
    draw_panel_background(parts, "Arena", "arena_win_rate, acceptance threshold, accepted markers", arena_panel_y)
    draw_grid(parts, min_iter, max_iter, arena_y0, arena_y1, arena_panel_y)
    draw_series(parts, rows, "arena_win_rate", "arena_win_rate", "#7c3aed", min_iter, max_iter, arena_y0, arena_y1, arena_panel_y, MARGIN_LEFT + 20, arena_panel_y + 22)
    draw_series(parts, rows, "arena_accept_win_rate", "accept threshold", "#d6336c", min_iter, max_iter, arena_y0, arena_y1, arena_panel_y, MARGIN_LEFT + 210, arena_panel_y + 22)
    draw_accept_markers(parts, rows, min_iter, max_iter, arena_y0, arena_y1, arena_panel_y)

    move_panel_y = panel_top(2)
    draw_panel_background(parts, "Self-Play Length", "average moves per self-play game", move_panel_y)
    draw_grid(parts, min_iter, max_iter, move_y0, move_y1, move_panel_y)
    draw_series(parts, rows, "selfplay_avg_moves", "selfplay_avg_moves", "#0ca678", min_iter, max_iter, move_y0, move_y1, move_panel_y, MARGIN_LEFT + 20, move_panel_y + 22)

    replay_panel_y = panel_top(3)
    draw_panel_background(parts, "Replay Growth", "replay_samples", replay_panel_y)
    draw_grid(parts, min_iter, max_iter, replay_y0, replay_y1, replay_panel_y)
    draw_series(parts, rows, "replay_samples", "replay_samples", "#f08c00", min_iter, max_iter, replay_y0, replay_y1, replay_panel_y, MARGIN_LEFT + 20, replay_panel_y + 22)

    parts.append(
        f'<text x="{SVG_WIDTH - 36}" y="{SVG_HEIGHT - 18}" fill="#718096" font-size="12" text-anchor="end">generated by scripts/plot_metrics.py</text>'
    )
    parts.append("</svg>")

    output_path.write_text("\n".join(parts), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render OmokAI metrics.jsonl into an SVG dashboard.")
    parser.add_argument("input", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output SVG path")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_path = args.input
    output_path = args.output or input_path.with_suffix(".svg")
    rows = read_metrics(input_path)
    render_svg(rows, output_path, input_path)
    print(output_path)


if __name__ == "__main__":
    main()
