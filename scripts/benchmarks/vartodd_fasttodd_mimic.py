#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repo_root()))

try:
    from pyvartodd.Release.pyvartodd import (  # type: ignore
        ActionPool,
        ActionSelection,
        ExplorationScore,
        FinalizationScore,
        Matrix,
        PolicyConfig,
        PolicyScores,
        SamplingBudget,
        SourcePool,
        ToddSearch,
        TohpePrefixSearch,
        TohpeSearch,
        ZBucketSearch,
        policy_iteration,
    )
except ModuleNotFoundError:
    from pyvartodd.pyvartodd import (  # type: ignore
        ActionPool,
        ActionSelection,
        ExplorationScore,
        FinalizationScore,
        Matrix,
        PolicyConfig,
        PolicyScores,
        SamplingBudget,
        SourcePool,
        ToddSearch,
        TohpePrefixSearch,
        TohpeSearch,
        ZBucketSearch,
        policy_iteration,
    )


CSV_FIELDS = [
    "algorithm",
    "circuit",
    "initial_t_count",
    "final_t_count",
    "wall_seconds",
    "tohpe_seconds",
    "todd_seconds",
    "stages",
    "tohpe_stages",
    "todd_stages",
    "tohpe_actions",
    "todd_actions",
    "phase_polynomials",
]

VANDAELE_TOHPE_SAMPLING = ("1", "0", "0", "2")
VANDAELE_FASTTODD_SAMPLING = ("all", "0", "0", "2")


class _TCountReporter:
    def __init__(self) -> None:
        self._last: int | None = None

    def emit(self, t_count: int, label: str) -> None:
        if self._last == t_count:
            return
        self._last = t_count
        print(f"T-count {t_count}: {label}", file=sys.stderr, flush=True)


def _sampling_budget(values: Iterable[str]) -> list[object]:
    raw = list(values)
    if len(raw) != 4:
        raise ValueError("sampling budget must be exactly four values: one_hot sparse dense sparse_max_weight")
    one_hot: object
    if str(raw[0]).lower() == "all":
        one_hot = "all"
    else:
        one_hot = max(0, int(raw[0]))
    out: list[object] = [one_hot]
    out.extend(max(0, int(v)) for v in raw[1:])
    return out


def _load_matrix(path: Path) -> Matrix:
    arr = np.asarray(np.load(path) != 0, dtype=np.bool_)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D npy matrix, got shape={arr.shape}")
    return Matrix.from_numpy(np.ascontiguousarray(arr))


def _circuit_name(path: Path) -> str:
    name = path.name
    for suffix in (".matrix.npy", ".npy"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _duration_seconds(value: str | None) -> float | None:
    if value is None or value == "" or value == "0":
        return None
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*([smhd]?)\s*", value)
    if match is None:
        raise ValueError(f"invalid duration: {value!r}; use seconds or suffix s/m/h/d")
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    multiplier = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[unit]
    return amount * multiplier


def _policy_config(args: argparse.Namespace, *, stage: str) -> PolicyConfig:
    if stage not in {"tohpe", "todd"}:
        raise ValueError(f"unknown stage: {stage}")

    pool_size = 1 if stage == "tohpe" else 0
    todd_size = 1 if stage == "todd" else 0
    tohpe_sampling = SamplingBudget(
        one_hot=args.tohpe_sampling[0],
        sparse=args.tohpe_sampling[1],
        dense=args.tohpe_sampling[2],
        sparse_max_weight=args.tohpe_sampling[3],
    )
    todd_sampling = SamplingBudget(
        one_hot=args.todd_sampling[0],
        sparse=args.todd_sampling[1],
        dense=args.todd_sampling[2],
        sparse_max_weight=args.todd_sampling[3],
    )
    return PolicyConfig(
        scores=PolicyScores(
            exploration=ExplorationScore(weights=[1.0, 0.0, 0.0, 0.0, 0.0], pow=1.0),
            final=FinalizationScore(weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], pow=1.0),
        ),
        selection=ActionSelection(count=1, mode="best", temperature=0.0),
        pool=ActionPool(final_size=1),
        tohpe=TohpeSearch(pool=SourcePool(keep=1, reserve=0)),
        tohpeprefix=TohpePrefixSearch(
            sampling=tohpe_sampling,
            pool=SourcePool(keep=pool_size, reserve=0),
            actions_per_bucket=args.tohpe_z_choices,
            buckets=ZBucketSearch(min_buckets=args.z_min_buckets, max_buckets=args.z_max_buckets),
        ),
        todd=ToddSearch(
            sampling=todd_sampling,
            pool=SourcePool(keep=todd_size, reserve=0),
            actions_per_bucket=args.todd_actions_per_bucket,
            buckets=ZBucketSearch(min_buckets=args.z_min_buckets, max_buckets=args.z_max_buckets),
        ),
    )


def _apply_one(state: Matrix, pcfg: PolicyConfig, *, seed: int, add_seed: int) -> tuple[Matrix | None, object | None]:
    result = policy_iteration(cur_mat=state, policy_cfg=pcfg, seed=seed, add_seed=add_seed)
    if not result.chosen or not result.states:
        return None, None
    next_state = result.states[-1]
    if next_state.rows >= state.rows:
        return None, None
    return next_state, result.chosen[-1]


def run_mimic(matrix_path: Path, args: argparse.Namespace) -> dict[str, object]:
    start = time.perf_counter()
    limit_seconds = _duration_seconds(args.time_limit)
    deadline = start + limit_seconds if limit_seconds is not None else None
    state = _load_matrix(matrix_path)
    initial_rows = int(state.rows)
    progress = _TCountReporter()
    progress.emit(initial_rows, "initial")
    tohpe_cfg = _policy_config(args, stage="tohpe")
    todd_cfg = _policy_config(args, stage="todd")

    tohpe_seconds = 0.0
    todd_seconds = 0.0
    stages = 0
    tohpe_stages = 0
    todd_stages = 0
    tohpe_actions = 0
    todd_actions = 0

    for outer in range(args.max_stages):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        stages += 1

        tohpe_stages += 1
        action_in_stage = 0
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            t0 = time.perf_counter()
            next_state, _candidate = _apply_one(
                state,
                tohpe_cfg,
                seed=args.seed,
                add_seed=outer * 1_000_000 + action_in_stage,
            )
            tohpe_seconds += time.perf_counter() - t0
            if next_state is None:
                break
            state = next_state
            tohpe_actions += 1
            action_in_stage += 1
            progress.emit(int(state.rows), f"TOHPE stage {stages}, action {action_in_stage}")
        if deadline is not None and time.perf_counter() >= deadline:
            break

        t0 = time.perf_counter()
        next_state, _candidate = _apply_one(state, todd_cfg, seed=args.seed, add_seed=outer)
        todd_seconds += time.perf_counter() - t0
        todd_stages += 1
        if next_state is None:
            break
        state = next_state
        todd_actions += 1
        progress.emit(int(state.rows), f"FastTODD stage {stages}, action {todd_actions}")
    else:
        raise RuntimeError(f"max stages exceeded: {args.max_stages}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.output_dir / matrix_path.name, state.to_numpy())

    return {
        "algorithm": "vartodd_fasttodd_mimic",
        "circuit": _circuit_name(matrix_path),
        "initial_t_count": initial_rows,
        "final_t_count": int(state.rows),
        "wall_seconds": f"{time.perf_counter() - start:.9f}",
        "tohpe_seconds": f"{tohpe_seconds:.9f}",
        "todd_seconds": f"{todd_seconds:.9f}",
        "stages": stages,
        "tohpe_stages": tohpe_stages,
        "todd_stages": todd_stages,
        "tohpe_actions": tohpe_actions,
        "todd_actions": todd_actions,
        "phase_polynomials": 1,
    }


def _write_row(row: dict[str, object], csv_path: Path | None, *, header: bool) -> None:
    if csv_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)
        if header:
            writer.writeheader()
        writer.writerow(row)
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VarTODD's C++ policy_iteration in a FastTODD-shaped loop: "
            "TOHPE-only to local convergence, then one full-TODD action, repeated."
        )
    )
    parser.add_argument("matrix", type=Path, help="input parity matrix .npy")
    parser.add_argument("--csv", type=Path, default=None, help="append one aggregate CSV row")
    parser.add_argument("--header", action="store_true", help="write CSV header before the row")
    parser.add_argument("--output-dir", type=Path, default=None, help="optional final matrix output directory")
    parser.add_argument("--time-limit", default=None, help="optional cooperative time limit, e.g. 4h, 30m, 60s")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-stages", type=int, default=10000)
    parser.add_argument("--z-min-buckets", type=int, default=0)
    parser.add_argument("--z-max-buckets", type=int, default=1_000_000_000)
    parser.add_argument("--tohpe-z-choices", type=int, default=1)
    parser.add_argument("--todd-actions-per-bucket", type=int, default=1)
    parser.add_argument(
        "--tohpe-sampling",
        nargs=4,
        default=list(VANDAELE_TOHPE_SAMPLING),
        metavar=("ONE_HOT", "SPARSE", "DENSE", "SPARSE_MAX_WEIGHT"),
    )
    parser.add_argument(
        "--todd-sampling",
        nargs=4,
        default=list(VANDAELE_FASTTODD_SAMPLING),
        metavar=("ONE_HOT", "SPARSE", "DENSE", "SPARSE_MAX_WEIGHT"),
    )
    args = parser.parse_args()
    args.tohpe_sampling = _sampling_budget(args.tohpe_sampling)
    args.todd_sampling = _sampling_budget(args.todd_sampling)
    return args


def main() -> None:
    args = parse_args()
    row = run_mimic(args.matrix, args)
    _write_row(row, args.csv, header=args.header)


if __name__ == "__main__":
    main()
