#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPTIMIZER = ROOT / "scripts/base_search/1239.py"
DEFAULT_MATRIX = "gf2^32_3228310.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 1239 optimizer multiple times in parallel and print elapsed time per run.")
    parser.add_argument("-k", "--runs", type=int, default=3, help="Number of independent runs to execute.")
    parser.add_argument("--matrix", default=DEFAULT_MATRIX, help="Matrix filename or path passed to the optimizer.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use for the child process.")
    return parser.parse_args()


def run_single_instance(idx: int, total_runs: int, args: argparse.Namespace) -> tuple[int, float, int]:
    print(f"\n=== Launching run {idx}/{total_runs} ===")
    start = time.perf_counter()
    completed = subprocess.run(
        [args.python, str(OPTIMIZER), "--matrix", args.matrix],
        cwd=ROOT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    return idx, elapsed, completed.returncode


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    print(f"Running {args.runs} instances of the optimizer in parallel with matrix {args.matrix}")
    total_time = 0.0
    failed = False
    results: list[tuple[int, float, int]] = []

    with ThreadPoolExecutor(max_workers=args.runs) as executor:
        futures = {
            executor.submit(run_single_instance, idx, args.runs, args): idx
            for idx in range(1, args.runs + 1)
        }
        for future in as_completed(futures):
            idx, elapsed, returncode = future.result()
            results.append((idx, elapsed, returncode))

    results.sort(key=lambda item: item[0])
    for idx, elapsed, returncode in results:
        print(f"Elapsed time for run {idx}: {elapsed:.2f}s")
        if returncode != 0:
            failed = True
            print(f"Run {idx} failed with exit code {returncode}")
        total_time += elapsed

    if not failed:
        avg = total_time / args.runs
        print(f"\nAverage time per run: {avg:.2f}s")


if __name__ == "__main__":
    main()
