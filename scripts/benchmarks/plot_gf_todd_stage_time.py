#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


ALGORITHMS = ("vartodd_fasttodd_mimic", "fasttodd")
COLORS = {
    "vartodd_fasttodd_mimic": "#2f7bbd",
    "fasttodd": "#d46a3a",
}
LABELS = {
    "vartodd_fasttodd_mimic": "VarTODD",
    "fasttodd": "FastTODD",
}

CHART_TITLE = "TODD stage and TOHPE action time by GF circuit"
CHART_SUBTITLE = "TODD uses todd_seconds / todd_stages; TOHPE uses tohpe_seconds / tohpe_actions"
X_AXIS_LABEL = "GF circuit"
TIME_Y_AXIS_LABEL = "seconds per TODD stage"
TOHPE_TIME_Y_AXIS_LABEL = "seconds per TOHPE action"
TIME_PANEL_TITLE = "Time per TODD stage"
TOHPE_PANEL_TITLE = "Time per TOHPE action"

SUMMARY_CHART_TITLE = "GF total runtime and final T-count reduction"
SUMMARY_CHART_SUBTITLE = "Total time uses wall_seconds; time-bar labels show rounded wall time"
SUMMARY_X_AXIS_LABEL = "GF circuit"
TOTAL_TIME_Y_AXIS_LABEL = "wall seconds"
REDUCTION_Y_AXIS_LABEL = "T-count reduction"
TOTAL_TIME_PANEL_TITLE = "Total runtime"
REDUCTION_PANEL_TITLE = "Final T-count reduction"

TIME_PER_STAGE_KEY = "time_per_stage"
TOHPE_PER_ACTION_KEY = "time_per_tohpe_action"
ITERATIONS_KEY = "stages"
TODD_STAGES_KEY = "todd_stages"
TOHPE_STAGES_KEY = "tohpe_stages"
TOHPE_ACTIONS_KEY = "tohpe_actions"
WALL_SECONDS_KEY = "wall_seconds"
INITIAL_T_COUNT_KEY = "initial_t_count"
FINAL_T_COUNT_KEY = "final_t_count"
REDUCTION_KEY = "reduction"


def circuit_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"gf2\^(\d+)", name)
    return (int(match.group(1)) if match else 10**9, name)


def nice_log_ticks(min_value: float, max_value: float, max_ticks: int = 12) -> list[float]:
    low = math.floor(math.log10(min_value))
    high = math.ceil(math.log10(max_value))
    ticks = []
    for power in range(low, high + 1):
        for multiplier in (1, 2, 5):
            value = multiplier * (10**power)
            if min_value <= value <= max_value:
                ticks.append(value)
    if len(ticks) > max_ticks:
        ticks = []
        for power in range(low, high + 1):
            value = 10**power
            if min_value <= value <= max_value:
                ticks.append(value)
    return ticks


def nice_linear_ticks(max_value: float, steps: int = 5) -> list[float]:
    if max_value <= 0:
        return [0.0]

    raw_step = max(1.0, max_value / steps)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = 10 * magnitude
    for multiplier in (1, 2, 5, 10):
        candidate = multiplier * magnitude
        if candidate >= raw_step:
            step = candidate
            break

    top = math.ceil(max_value / step) * step
    tick_count = int(round(top / step))
    return [i * step for i in range(tick_count + 1)]


def fmt_seconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1_000_000:.0f} us"
    if value < 1:
        return f"{value * 1000:.0f} ms"
    return f"{value:.0f} s"


def fmt_bar_seconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1_000_000:.0f}us"
    if value < 1:
        return f"{value * 1000:.1f}ms"
    if value < 10:
        return f"{value:.1f}s"
    if value < 60:
        return f"{value:.0f}s"
    if value < 3600:
        return f"{value / 60:.1f}m"
    return f"{value / 3600:.1f}h"


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_count(value: float) -> str:
    if value == round(value):
        return str(int(round(value)))
    return f"{value:g}"


def load_rows(csv_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            algorithm = row["algorithm"]
            if algorithm not in ALGORITHMS:
                continue
            todd_stages = int(row["todd_stages"])
            tohpe_stages = int(row["tohpe_stages"])
            tohpe_actions = int(row["tohpe_actions"])
            if todd_stages <= 0 or tohpe_actions <= 0:
                continue
            circuit = row["circuit"]
            initial_t_count = int(row["initial_t_count"])
            final_t_count = int(row["final_t_count"])
            grouped.setdefault(circuit, {})[algorithm] = {
                TIME_PER_STAGE_KEY: float(row["todd_seconds"]) / todd_stages,
                TOHPE_PER_ACTION_KEY: float(row["tohpe_seconds"]) / tohpe_actions,
                ITERATIONS_KEY: float(row["stages"]),
                TODD_STAGES_KEY: float(todd_stages),
                TOHPE_STAGES_KEY: float(tohpe_stages),
                TOHPE_ACTIONS_KEY: float(tohpe_actions),
                WALL_SECONDS_KEY: float(row["wall_seconds"]),
                INITIAL_T_COUNT_KEY: float(initial_t_count),
                FINAL_T_COUNT_KEY: float(final_t_count),
                REDUCTION_KEY: float(initial_t_count - final_t_count),
            }
    return grouped


def complete_circuits(grouped: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    circuits = sorted(
        [c for c, values in grouped.items() if all(a in values for a in ALGORITHMS)],
        key=circuit_sort_key,
    )
    if not circuits:
        raise ValueError("no circuits with both VarTODD and FastTODD rows")
    return circuits


def render_stage_svg(grouped: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    circuits = complete_circuits(grouped)
    todd_values = [grouped[c][a][TIME_PER_STAGE_KEY] for c in circuits for a in ALGORITHMS]
    tohpe_values = [grouped[c][a][TOHPE_PER_ACTION_KEY] for c in circuits for a in ALGORITHMS]
    todd_min = min(todd_values) * 0.75
    todd_max = max(todd_values) * 1.25
    tohpe_min = min(tohpe_values) * 0.75
    tohpe_max = max(tohpe_values) * 1.25
    todd_ticks = nice_log_ticks(todd_min, todd_max)
    tohpe_ticks = nice_log_ticks(tohpe_min, tohpe_max)

    width = max(1160, 74 * len(circuits) + 190)
    height = 900
    left = 96
    right = 34
    top = 100
    panel_gap = 84
    panel_height = 285
    plot_width = width - left - right
    todd_top = top
    tohpe_top = todd_top + panel_height + panel_gap
    todd_log_min = math.log10(todd_min)
    todd_log_max = math.log10(todd_max)
    tohpe_log_min = math.log10(tohpe_min)
    tohpe_log_max = math.log10(tohpe_max)

    def todd_y_for(value: float) -> float:
        t = (math.log10(value) - todd_log_min) / (todd_log_max - todd_log_min)
        return todd_top + panel_height * (1 - t)

    def tohpe_y_for(value: float) -> float:
        t = (math.log10(value) - tohpe_log_min) / (tohpe_log_max - tohpe_log_min)
        return tohpe_top + panel_height * (1 - t)

    group_width = plot_width / len(circuits)
    bar_width = min(24, group_width * 0.28)
    bar_gap = 6
    todd_baseline = todd_y_for(todd_min)
    tohpe_baseline = tohpe_y_for(tohpe_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933} .tick{fill:#52616b;font-size:12px} .label{font-size:13px} .bar-note{font-size:10px;fill:#334e68;font-weight:700} .panel-title{font-size:16px;font-weight:700} .title{font-size:24px;font-weight:700} .subtitle{font-size:14px;fill:#52616b}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" class="title">{escape(CHART_TITLE)}</text>',
        f'<text x="{left}" y="58" class="subtitle">{escape(CHART_SUBTITLE)}</text>',
        f'<text x="{left}" y="{todd_top - 18}" class="panel-title">{escape(TIME_PANEL_TITLE)}</text>',
        f'<text x="{left}" y="{tohpe_top - 18}" class="panel-title">{escape(TOHPE_PANEL_TITLE)}</text>',
    ]

    for tick in todd_ticks:
        y = todd_y_for(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e4e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">{fmt_seconds(tick)}</text>')

    for tick in tohpe_ticks:
        y = tohpe_y_for(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e4e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">{fmt_seconds(tick)}</text>')

    parts.append(f'<line x1="{left}" y1="{todd_top}" x2="{left}" y2="{todd_baseline:.2f}" stroke="#9aa5b1" stroke-width="1"/>')
    parts.append(f'<line x1="{left}" y1="{todd_baseline:.2f}" x2="{width - right}" y2="{todd_baseline:.2f}" stroke="#9aa5b1" stroke-width="1"/>')
    parts.append(f'<line x1="{left}" y1="{tohpe_top}" x2="{left}" y2="{tohpe_baseline:.2f}" stroke="#9aa5b1" stroke-width="1"/>')
    parts.append(f'<line x1="{left}" y1="{tohpe_baseline:.2f}" x2="{width - right}" y2="{tohpe_baseline:.2f}" stroke="#9aa5b1" stroke-width="1"/>')

    for i, circuit in enumerate(circuits):
        center = left + group_width * (i + 0.5)
        for j, algorithm in enumerate(ALGORITHMS):
            time_value = grouped[circuit][algorithm][TIME_PER_STAGE_KEY]
            tohpe_value = grouped[circuit][algorithm][TOHPE_PER_ACTION_KEY]
            iterations = grouped[circuit][algorithm][ITERATIONS_KEY]
            todd_stages = grouped[circuit][algorithm][TODD_STAGES_KEY]
            tohpe_stages = grouped[circuit][algorithm][TOHPE_STAGES_KEY]
            tohpe_actions = grouped[circuit][algorithm][TOHPE_ACTIONS_KEY]
            x = center + (j - 0.5) * (bar_width + bar_gap)
            todd_y = todd_y_for(time_value)
            tohpe_y = tohpe_y_for(tohpe_value)
            parts.append(
                f'<rect x="{x:.2f}" y="{todd_y:.2f}" width="{bar_width:.2f}" height="{todd_baseline - todd_y:.2f}" '
                f'fill="{COLORS[algorithm]}" rx="2"><title>{escape(LABELS[algorithm])} {escape(circuit)}: {time_value:.6g} s/TODD stage; iterations {fmt_count(iterations)}; TODD stages {fmt_count(todd_stages)}</title></rect>'
            )
            parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{max(todd_top + 13, todd_y - 5):.2f}" text-anchor="middle" class="bar-note">{fmt_bar_seconds(time_value)}</text>')
            parts.append(
                f'<rect x="{x:.2f}" y="{tohpe_y:.2f}" width="{bar_width:.2f}" height="{tohpe_baseline - tohpe_y:.2f}" '
                f'fill="{COLORS[algorithm]}" rx="2"><title>{escape(LABELS[algorithm])} {escape(circuit)}: {tohpe_value:.6g} s/TOHPE action; iterations {fmt_count(iterations)}; TOHPE stages {fmt_count(tohpe_stages)}; TOHPE actions {fmt_count(tohpe_actions)}</title></rect>'
            )
            parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{max(tohpe_top + 13, tohpe_y - 5):.2f}" text-anchor="middle" class="bar-note">{fmt_bar_seconds(tohpe_value)}</text>')
        parts.append(
            f'<text x="{center:.2f}" y="{tohpe_baseline + 24:.2f}" text-anchor="end" transform="rotate(-45 {center:.2f} {tohpe_baseline + 24:.2f})" class="tick">{escape(circuit)}</text>'
        )

    legend_x = width - right - 250
    for i, algorithm in enumerate(ALGORITHMS):
        y = 30 + i * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="16" height="16" fill="{COLORS[algorithm]}" rx="2"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{y + 1}" class="label">{LABELS[algorithm]}</text>')

    time_label_y = todd_top + panel_height / 2
    tohpe_label_y = tohpe_top + panel_height / 2
    parts.append(f'<text x="22" y="{time_label_y:.2f}" transform="rotate(-90 22 {time_label_y:.2f})" text-anchor="middle" class="label">{escape(TIME_Y_AXIS_LABEL)}</text>')
    parts.append(f'<text x="22" y="{tohpe_label_y:.2f}" transform="rotate(-90 22 {tohpe_label_y:.2f})" text-anchor="middle" class="label">{escape(TOHPE_TIME_Y_AXIS_LABEL)}</text>')
    parts.append(f'<text x="{left + plot_width / 2:.2f}" y="{height - 20}" text-anchor="middle" class="label">{escape(X_AXIS_LABEL)}</text>')
    parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def render_summary_svg(grouped: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    circuits = complete_circuits(grouped)
    wall_values = [grouped[c][a][WALL_SECONDS_KEY] for c in circuits for a in ALGORITHMS]
    reduction_values = [grouped[c][a][REDUCTION_KEY] for c in circuits for a in ALGORITHMS]
    wall_min = min(wall_values) * 0.75
    wall_max = max(wall_values) * 1.25
    wall_ticks = nice_log_ticks(wall_min, wall_max)
    reduction_ticks = nice_linear_ticks(max(reduction_values))
    reduction_max = max(reduction_ticks)

    width = max(1160, 74 * len(circuits) + 190)
    height = 900
    left = 96
    right = 34
    top = 100
    panel_gap = 84
    panel_height = 285
    plot_width = width - left - right
    wall_top = top
    reduction_top = wall_top + panel_height + panel_gap
    wall_log_min = math.log10(wall_min)
    wall_log_max = math.log10(wall_max)

    def wall_y_for(value: float) -> float:
        t = (math.log10(value) - wall_log_min) / (wall_log_max - wall_log_min)
        return wall_top + panel_height * (1 - t)

    def reduction_y_for(value: float) -> float:
        if reduction_max == 0:
            return reduction_top + panel_height
        return reduction_top + panel_height * (1 - value / reduction_max)

    group_width = plot_width / len(circuits)
    bar_width = min(24, group_width * 0.28)
    bar_gap = 6
    wall_baseline = wall_y_for(wall_min)
    reduction_baseline = reduction_y_for(0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933} .tick{fill:#52616b;font-size:12px} .label{font-size:13px} .bar-note{font-size:10px;fill:#334e68;font-weight:700} .panel-title{font-size:16px;font-weight:700} .title{font-size:24px;font-weight:700} .subtitle{font-size:14px;fill:#52616b}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" class="title">{escape(SUMMARY_CHART_TITLE)}</text>',
        f'<text x="{left}" y="58" class="subtitle">{escape(SUMMARY_CHART_SUBTITLE)}</text>',
        f'<text x="{left}" y="{wall_top - 18}" class="panel-title">{escape(TOTAL_TIME_PANEL_TITLE)}</text>',
        f'<text x="{left}" y="{reduction_top - 18}" class="panel-title">{escape(REDUCTION_PANEL_TITLE)}</text>',
    ]

    for tick in wall_ticks:
        y = wall_y_for(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e4e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">{fmt_seconds(tick)}</text>')

    for tick in reduction_ticks:
        y = reduction_y_for(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e4e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">{fmt_count(tick)}</text>')

    parts.append(f'<line x1="{left}" y1="{wall_top}" x2="{left}" y2="{wall_baseline:.2f}" stroke="#9aa5b1" stroke-width="1"/>')
    parts.append(f'<line x1="{left}" y1="{wall_baseline:.2f}" x2="{width - right}" y2="{wall_baseline:.2f}" stroke="#9aa5b1" stroke-width="1"/>')
    parts.append(f'<line x1="{left}" y1="{reduction_top}" x2="{left}" y2="{reduction_baseline:.2f}" stroke="#9aa5b1" stroke-width="1"/>')
    parts.append(f'<line x1="{left}" y1="{reduction_baseline:.2f}" x2="{width - right}" y2="{reduction_baseline:.2f}" stroke="#9aa5b1" stroke-width="1"/>')

    for i, circuit in enumerate(circuits):
        center = left + group_width * (i + 0.5)
        for j, algorithm in enumerate(ALGORITHMS):
            wall_value = grouped[circuit][algorithm][WALL_SECONDS_KEY]
            reduction_value = grouped[circuit][algorithm][REDUCTION_KEY]
            initial_t_count = grouped[circuit][algorithm][INITIAL_T_COUNT_KEY]
            final_t_count = grouped[circuit][algorithm][FINAL_T_COUNT_KEY]
            x = center + (j - 0.5) * (bar_width + bar_gap)
            wall_y = wall_y_for(wall_value)
            reduction_y = reduction_y_for(reduction_value)
            parts.append(
                f'<rect x="{x:.2f}" y="{wall_y:.2f}" width="{bar_width:.2f}" height="{wall_baseline - wall_y:.2f}" '
                f'fill="{COLORS[algorithm]}" rx="2"><title>{escape(LABELS[algorithm])} {escape(circuit)}: {wall_value:.6g} wall seconds</title></rect>'
            )
            parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{max(wall_top + 13, wall_y - 5):.2f}" text-anchor="middle" class="bar-note">{fmt_bar_seconds(wall_value)}</text>')
            parts.append(
                f'<rect x="{x:.2f}" y="{reduction_y:.2f}" width="{bar_width:.2f}" height="{reduction_baseline - reduction_y:.2f}" '
                f'fill="{COLORS[algorithm]}" rx="2"><title>{escape(LABELS[algorithm])} {escape(circuit)}: reduction {fmt_count(reduction_value)}; T-count {fmt_count(initial_t_count)} -> {fmt_count(final_t_count)}</title></rect>'
            )
            parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{max(reduction_top + 13, reduction_y - 5):.2f}" text-anchor="middle" class="bar-note">{fmt_count(reduction_value)}</text>')
        parts.append(
            f'<text x="{center:.2f}" y="{reduction_baseline + 24:.2f}" text-anchor="end" transform="rotate(-45 {center:.2f} {reduction_baseline + 24:.2f})" class="tick">{escape(circuit)}</text>'
        )

    legend_x = width - right - 250
    for i, algorithm in enumerate(ALGORITHMS):
        y = 30 + i * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="16" height="16" fill="{COLORS[algorithm]}" rx="2"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{y + 1}" class="label">{LABELS[algorithm]}</text>')

    wall_label_y = wall_top + panel_height / 2
    reduction_label_y = reduction_top + panel_height / 2
    parts.append(f'<text x="22" y="{wall_label_y:.2f}" transform="rotate(-90 22 {wall_label_y:.2f})" text-anchor="middle" class="label">{escape(TOTAL_TIME_Y_AXIS_LABEL)}</text>')
    parts.append(f'<text x="22" y="{reduction_label_y:.2f}" transform="rotate(-90 22 {reduction_label_y:.2f})" text-anchor="middle" class="label">{escape(REDUCTION_Y_AXIS_LABEL)}</text>')
    parts.append(f'<text x="{left + plot_width / 2:.2f}" y="{height - 20}" text-anchor="middle" class="label">{escape(SUMMARY_X_AXIS_LABEL)}</text>')
    parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GF benchmark comparisons for VarTODD and FastTODD.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmark_results/gf_comparison.csv"),
        help="input comparison CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/gf_todd_stage_time.svg"),
        help="output SVG path for the TODD/TOHPE stage-time plot",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("benchmark_results/gf_total_time_reduction.svg"),
        help="output SVG path for the total-time/reduction plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = load_rows(args.csv)
    render_stage_svg(grouped, args.out)
    print(f"wrote {args.out}")
    render_summary_svg(grouped, args.summary_out)
    print(f"wrote {args.summary_out}")


if __name__ == "__main__":
    main()
