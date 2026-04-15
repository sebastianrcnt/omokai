#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def read_metrics(path: Path) -> list[dict[str, object]]:
    rows = [row for row in read_jsonl(path) if "iteration" in row]
    rows.sort(key=lambda item: int(item["iteration"]))
    return rows


def read_debug_events(path: Path) -> list[dict[str, object]]:
    rows = [row for row in read_jsonl(path) if row.get("event")]
    rows.sort(key=lambda item: (int(item.get("iteration", -1)), str(item.get("timestamp", ""))))
    return rows


def to_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def to_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def metric_series(rows: list[dict[str, object]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        iteration = to_int(row.get("iteration"))
        value = to_float(row.get(key))
        if iteration is None or value is None:
            continue
        xs.append(iteration)
        ys.append(value)
    return xs, ys


def aggregate_debug_series(
    rows: list[dict[str, object]],
    event_name: str,
    value_key: str,
    reducer: str = "last",
) -> tuple[list[int], list[float]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row.get("event") != event_name:
            continue
        iteration = to_int(row.get("iteration"))
        value = to_float(row.get(value_key))
        if iteration is None or value is None:
            continue
        grouped.setdefault(iteration, []).append(value)
    xs = sorted(grouped)
    ys: list[float] = []
    for iteration in xs:
        values = grouped[iteration]
        if reducer == "mean":
            ys.append(float(mean(values)))
        elif reducer == "max":
            ys.append(float(max(values)))
        else:
            ys.append(float(values[-1]))
    return xs, ys


def aggregate_debug_p90(
    rows: list[dict[str, object]],
    event_name: str,
    value_key: str,
) -> tuple[list[int], list[float]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row.get("event") != event_name:
            continue
        iteration = to_int(row.get("iteration"))
        value = to_float(row.get(value_key))
        if iteration is None or value is None:
            continue
        grouped.setdefault(iteration, []).append(value)
    xs = sorted(grouped)
    ys = [float(np.percentile(grouped[iteration], 90)) for iteration in xs]
    return xs, ys


def latest_metric(rows: list[dict[str, object]], key: str) -> float | None:
    for row in reversed(rows):
        value = to_float(row.get(key))
        if value is not None:
            return value
    return None


def latest_event_value(rows: list[dict[str, object]], event_name: str, key: str) -> float | None:
    for row in reversed(rows):
        if row.get("event") != event_name:
            continue
        value = to_float(row.get(key))
        if value is not None:
            return value
    return None


def accepted_iterations(rows: list[dict[str, object]]) -> list[int]:
    values: list[int] = []
    for row in rows:
        iteration = to_int(row.get("iteration"))
        if iteration is None:
            continue
        if bool(row.get("accepted")):
            values.append(iteration)
    return values


def metric_row_map(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {
        int(row["iteration"]): row
        for row in rows
        if to_int(row.get("iteration")) is not None
    }


def format_number(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def trend_direction(values: list[float]) -> str:
    if len(values) < 4:
        return "표본 부족"
    half = max(1, len(values) // 2)
    early = mean(values[:half])
    late = mean(values[-half:])
    delta = late - early
    threshold = max(0.01, abs(early) * 0.02)
    if delta <= -threshold:
        return "개선"
    if delta >= threshold:
        return "악화"
    return "정체"


def build_summary(metrics_rows: list[dict[str, object]], debug_rows: list[dict[str, object]]) -> list[str]:
    latest = metrics_rows[-1]
    latest_iteration = int(latest["iteration"])
    best_iteration = to_int(latest.get("best_iteration"))

    _, policy_loss = metric_series(metrics_rows, "policy_loss")
    _, arena_win_rate = metric_series(metrics_rows, "arena_win_rate")
    _, white_win_rate = metric_series(metrics_rows, "arena_candidate_white_win_rate")
    latest_arena = latest_metric(metrics_rows, "arena_win_rate")
    latest_accept = latest_metric(metrics_rows, "arena_accept_win_rate")
    latest_white = latest_metric(metrics_rows, "arena_candidate_white_win_rate")
    latest_white_gate = latest_metric(metrics_rows, "arena_white_win_rate_threshold")
    latest_iter_duration = latest_event_value(debug_rows, "iteration_completed", "duration_seconds")
    latest_selfplay_gps = latest_event_value(debug_rows, "selfplay_finished", "games_per_second")
    latest_updates_per_sec = latest_event_value(debug_rows, "training_finished", "updates_per_second")
    latest_batch_mean = latest_event_value(debug_rows, "batched_inference", "positions")

    lines = [
        f"Latest iter {latest_iteration} | best {best_iteration if best_iteration is not None else '-'} | accepted {latest.get('accepted', '-')}",
        f"Loss trend: policy {trend_direction(policy_loss)} | latest {format_number(latest_metric(metrics_rows, 'policy_loss'), 4)}",
        f"Arena: {format_number(latest_arena, 4)} vs gate {format_number(latest_accept, 4)} | white {format_number(latest_white, 4)} vs gate {format_number(latest_white_gate, 4)}",
    ]
    if latest_iter_duration is not None or latest_selfplay_gps is not None or latest_updates_per_sec is not None:
        lines.append(
            "Speed: "
            f"iter {format_number(latest_iter_duration, 1)}s | "
            f"self-play {format_number(latest_selfplay_gps, 2)} games/s | "
            f"train {format_number(latest_updates_per_sec, 2)} updates/s | "
            f"batch {format_number(latest_batch_mean, 0)} pos"
        )
    else:
        lines.append("Speed: structured debug JSONL data not available yet for this run")
    if arena_win_rate and white_win_rate:
        lines.append(
            f"Interpretation: loss는 내려갈수록 좋고, arena/white 선은 gate 위에 오래 머물수록 좋습니다. "
            f"iteration time은 낮을수록, inference batch는 높을수록 GPU 활용에 유리합니다."
        )
    return lines


def style_axis(axis: plt.Axes, title: str, subtitle: str) -> None:
    axis.set_title(f"{title}\n{subtitle}", loc="left", fontsize=12, fontweight="bold")
    axis.grid(True, alpha=0.25, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def draw_plot(
    metrics_rows: list[dict[str, object]],
    debug_rows: list[dict[str, object]],
    output_path: Path,
    metrics_path: Path,
    debug_path: Path,
) -> list[str]:
    if not metrics_rows:
        raise ValueError("No metrics rows found.")

    accepted = accepted_iterations(metrics_rows)
    latest = metrics_rows[-1]
    row_by_iteration = metric_row_map(metrics_rows)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), dpi=150)
    fig.patch.set_facecolor("#f7f8fb")
    for axis in axes.flat:
        axis.set_facecolor("white")

    summary_lines = build_summary(metrics_rows, debug_rows)
    fig.suptitle("OmokAI Training Overview", fontsize=18, fontweight="bold", x=0.06, ha="left", y=0.985)
    fig.text(
        0.06,
        0.945,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox={"facecolor": "#eef2f8", "edgecolor": "#d7deea", "boxstyle": "round,pad=0.5"},
    )

    loss_ax = axes[0, 0]
    style_axis(loss_ax, "Loss", "내려갈수록 학습 목표와 더 잘 맞고 있다는 뜻입니다.")
    for key, label, color in (
        ("train_loss", "train", "#2b6cb0"),
        ("policy_loss", "policy", "#c05621"),
        ("value_loss", "value", "#2f855a"),
    ):
        xs, ys = metric_series(metrics_rows, key)
        if xs:
            loss_ax.plot(xs, ys, label=label, color=color, linewidth=2.2)
    if accepted:
        accepted_points = [
            (iteration, to_float(row_by_iteration[iteration].get("train_loss")))
            for iteration in accepted
            if iteration in row_by_iteration
        ]
        accepted_points = [(iteration, value) for iteration, value in accepted_points if value is not None]
        if accepted_points:
            loss_ax.scatter(
                [iteration for iteration, _ in accepted_points],
                [value for _, value in accepted_points],
                color="#12b886",
                s=26,
                label="accepted",
                zorder=3,
            )
    loss_ax.set_xlabel("iteration")
    loss_ax.legend(frameon=False, ncol=4, fontsize=9)

    arena_ax = axes[0, 1]
    style_axis(arena_ax, "Arena", "파란선과 초록선이 각 gate보다 위에 있으면 승급 조건에 가까운 상태입니다.")
    for key, label, color, linestyle in (
        ("arena_win_rate", "arena", "#1a365d", "-"),
        ("arena_accept_win_rate", "arena gate", "#1a365d", "--"),
        ("arena_candidate_white_win_rate", "white", "#975a16", "-"),
        ("arena_white_win_rate_threshold", "white gate", "#975a16", "--"),
    ):
        xs, ys = metric_series(metrics_rows, key)
        if xs:
            arena_ax.plot(xs, ys, label=label, color=color, linestyle=linestyle, linewidth=2.0)
    arena_ax.set_ylim(0.0, 1.0)
    arena_ax.set_xlabel("iteration")
    arena_ax.legend(frameon=False, ncol=2, fontsize=9)

    duration_ax = axes[1, 0]
    style_axis(duration_ax, "Stage Duration", "낮을수록 빠릅니다. total이 유지되면서 self-play와 training이 내려가면 구조 개선이 먹힌 겁니다.")
    for event_name, key, label, color in (
        ("selfplay_finished", "duration_seconds", "self-play", "#3182ce"),
        ("training_finished", "duration_seconds", "training", "#dd6b20"),
        ("arena_finished", "duration_seconds", "arena", "#38a169"),
        ("iteration_completed", "duration_seconds", "total", "#1a202c"),
    ):
        xs, ys = aggregate_debug_series(debug_rows, event_name, key)
        if xs:
            duration_ax.plot(xs, ys, label=label, color=color, linewidth=2.0)
    duration_ax.set_xlabel("iteration")
    duration_ax.set_ylabel("seconds")
    duration_ax.legend(frameon=False, ncol=4, fontsize=9)

    throughput_ax = axes[1, 1]
    style_axis(throughput_ax, "Throughput", "높을수록 좋습니다. self-play/training이 같이 오르면 GPU와 CPU가 덜 놀고 있다는 뜻입니다.")
    for event_name, key, label, color in (
        ("selfplay_finished", "games_per_second", "games/s", "#2b6cb0"),
        ("selfplay_finished", "positions_per_second", "positions/s", "#2f855a"),
        ("training_finished", "updates_per_second", "updates/s", "#c05621"),
    ):
        xs, ys = aggregate_debug_series(debug_rows, event_name, key)
        if xs:
            throughput_ax.plot(xs, ys, label=label, color=color, linewidth=2.0)
    throughput_ax.set_xlabel("iteration")
    throughput_ax.legend(frameon=False, ncol=3, fontsize=9)

    batch_ax = axes[2, 0]
    style_axis(batch_ax, "Inference Batch", "평균 batch가 커질수록 GPU가 더 크게 한 번에 일합니다.")
    avg_x, avg_y = aggregate_debug_series(debug_rows, "batched_inference", "positions", reducer="mean")
    p90_x, p90_y = aggregate_debug_p90(debug_rows, "batched_inference", "positions")
    max_x, max_y = aggregate_debug_series(debug_rows, "batched_inference", "max_batch_size", reducer="last")
    if avg_x:
        batch_ax.plot(avg_x, avg_y, label="avg positions", color="#805ad5", linewidth=2.0)
    if p90_x:
        batch_ax.plot(p90_x, p90_y, label="p90 positions", color="#553c9a", linewidth=2.0)
    if max_x:
        batch_ax.plot(max_x, max_y, label="max batch", color="#9f7aea", linestyle="--", linewidth=1.8)
    batch_ax.set_xlabel("iteration")
    batch_ax.legend(frameon=False, ncol=3, fontsize=9)

    replay_ax = axes[2, 1]
    style_axis(replay_ax, "Replay And Game Length", "replay는 올라갈수록 데이터가 쌓이는 것이고, avg moves는 전술/규칙 편향 변화를 같이 봐야 합니다.")
    xs, replay = metric_series(metrics_rows, "replay_samples")
    if xs:
        replay_ax.plot(xs, replay, label="replay samples", color="#1a365d", linewidth=2.0)
    replay_ax.set_xlabel("iteration")
    replay_ax.set_ylabel("samples")
    moves_ax = replay_ax.twinx()
    moves_x, moves_y = metric_series(metrics_rows, "selfplay_avg_moves")
    if moves_x:
        moves_ax.plot(moves_x, moves_y, label="avg moves", color="#d69e2e", linewidth=2.0)
    white_x, white_y = metric_series(metrics_rows, "selfplay_white_to_move_games")
    if white_x:
        moves_ax.plot(white_x, white_y, label="white-to-move games", color="#c05621", linewidth=1.7, linestyle="--")
    moves_ax.set_ylabel("moves / games")
    replay_handles, replay_labels = replay_ax.get_legend_handles_labels()
    moves_handles, moves_labels = moves_ax.get_legend_handles_labels()
    replay_ax.legend(replay_handles + moves_handles, replay_labels + moves_labels, frameon=False, ncol=3, fontsize=9, loc="upper left")

    for axis in axes.flat:
        axis.tick_params(labelsize=9)
    moves_ax.tick_params(labelsize=9)

    fig.text(
        0.06,
        0.02,
        f"metrics: {metrics_path} | debug: {debug_path} | latest iteration: {latest['iteration']}",
        fontsize=9,
        color="#4a5568",
    )
    plt.subplots_adjust(top=0.84, hspace=0.34, wspace=0.2, bottom=0.06, left=0.07, right=0.96)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return summary_lines


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot OmokAI metrics with matplotlib.")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--debug-log", type=Path, default=None, help="Path to structured training.debug.jsonl")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    metrics_path = args.metrics
    debug_path = args.debug_log or metrics_path.parent / "training.debug.jsonl"
    output_path = args.output or metrics_path.with_suffix(".png")

    metrics_rows = read_metrics(metrics_path)
    debug_rows = read_debug_events(debug_path)
    summary_lines = draw_plot(metrics_rows, debug_rows, output_path, metrics_path, debug_path)
    print(f"saved plot: {output_path}")
    print(f"metrics input: {metrics_path}")
    print(f"debug input: {debug_path}")
    for line in summary_lines:
        print(f"- {line}")


if __name__ == "__main__":
    main()
