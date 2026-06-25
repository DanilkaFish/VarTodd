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


def circuit_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"gf2\^(\d+)", name)
    return (int(match.group(1)) if match else 10**9, name)


def nice_log_ticks(min_value: float, max_value: float) -> list[float]:
    low = math.floor(math.log10(min_value))
    high = math.ceil(math.log10(max_value))
    ticks = []
    for power in range(low, high + 1):
        for multiplier in (1, 2, 5):
            value = multiplier * (10**power)
            if min_value <= value <= max_value:
                ticks.append(value)
    return ticks


def fmt_seconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1_000_000:.0f} us"
    if value < 1:
        return f"{value * 1000:.0f} ms"
    return f"{value:.0f} s"


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def load_rows(csv_path: Path) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            algorithm = row["algorithm"]
            if algorithm not in ALGORITHMS:
                continue
            todd_actions = int(row["todd_actions"])
            if todd_actions <= 0:
                continue
            circuit = row["circuit"]
            grouped.setdefault(circuit, {})[algorithm] = float(row["todd_seconds"]) / todd_actions
    return grouped


def render_svg(grouped: dict[str, dict[str, float]], out_path: Path) -> None:
    circuits = sorted(
        [c for c, values in grouped.items() if all(a in values for a in ALGORITHMS)],
        key=circuit_sort_key,
    )
    if not circuits:
        raise ValueError("no circuits with both VarTODD and FastTODD rows")

    values = [grouped[c][a] for c in circuits for a in ALGORITHMS]
    min_value = min(values) * 0.75
    max_value = max(values) * 1.25
    ticks = nice_log_ticks(min_value, max_value)

    width = max(1100, 72 * len(circuits) + 180)
    height = 680
    left = 92
    right = 34
    top = 78
    bottom = 170
    plot_width = width - left - right
    plot_height = height - top - bottom
    log_min = math.log10(min_value)
    log_max = math.log10(max_value)

    def y_for(value: float) -> float:
        t = (math.log10(value) - log_min) / (log_max - log_min)
        return top + plot_height * (1 - t)

    group_width = plot_width / len(circuits)
    bar_width = min(24, group_width * 0.28)
    bar_gap = 6
    baseline = y_for(min_value)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933} .tick{fill:#52616b;font-size:12px} .label{font-size:13px} .title{font-size:24px;font-weight:700} .subtitle{font-size:14px;fill:#52616b}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" class="title">Average TODD stage time by GF circuit</text>',
        f'<text x="{left}" y="58" class="subtitle">Computed as todd_seconds / todd_actions from {escape(str(out_path.parent / "gf_comparison.csv"))}; log-scale seconds</text>',
    ]

    for tick in ticks:
        y = y_for(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e4e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">{fmt_seconds(tick)}</text>')

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline:.2f}" stroke="#9aa5b1" stroke-width="1"/>')
    parts.append(f'<line x1="{left}" y1="{baseline:.2f}" x2="{width - right}" y2="{baseline:.2f}" stroke="#9aa5b1" stroke-width="1"/>')

    for i, circuit in enumerate(circuits):
        center = left + group_width * (i + 0.5)
        for j, algorithm in enumerate(ALGORITHMS):
            value = grouped[circuit][algorithm]
            x = center + (j - 0.5) * (bar_width + bar_gap)
            y = y_for(value)
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{baseline - y:.2f}" '
                f'fill="{COLORS[algorithm]}" rx="2"><title>{escape(LABELS[algorithm])} {escape(circuit)}: {value:.6g} s/stage</title></rect>'
            )
        parts.append(
            f'<text x="{center:.2f}" y="{baseline + 24:.2f}" text-anchor="end" transform="rotate(-45 {center:.2f} {baseline + 24:.2f})" class="tick">{escape(circuit)}</text>'
        )

    legend_x = width - right - 250
    for i, algorithm in enumerate(ALGORITHMS):
        y = 30 + i * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="16" height="16" fill="{COLORS[algorithm]}" rx="2"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{y + 1}" class="label">{LABELS[algorithm]}</text>')

    parts.append(f'<text x="22" y="{top + plot_height / 2:.2f}" transform="rotate(-90 22 {top + plot_height / 2:.2f})" text-anchor="middle" class="label">seconds per TODD stage</text>')
    parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot average TODD-stage time for VarTODD and FastTODD.")
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
        help="output SVG path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_svg(load_rows(args.csv), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
