"""Run a chosen scripts/base_search/*.py strategy against a matrix and final rank.

Usage (from anywhere):
    python3 scripts/base_search/run_search.py full_pso \\
        --matrix data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^16_1612310.npy \\
        --final-rank 380
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

STRATEGIES = [
    "anti_greedy_strategy",
    "wide_z_two_stage_descent",
    "full_cmaes",
    "full_de",
    "full_pso",
    "greedy_strategy",
    "large_instance_pso",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("script", choices=STRATEGIES, help="Strategy module under scripts/base_search")
    parser.add_argument("--matrix", required=True, help="Path to a .npy matrix (absolute or relative to repo root)")
    parser.add_argument("--final-rank", type=int, required=True, help="Target final rank for this run")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # TARGET_FINAL_RANK is read from this env var once, at import time, by
    # scripts.optimization_core.helper -- must be set before that module (or
    # anything importing it) is loaded.
    os.environ["VARTODD_TARGET_FINAL_RANK"] = str(args.final_rank)

    from scripts.optimization_core.helper import Matrix, load_matrix_array

    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = ROOT_DIR / matrix_path
    mat = Matrix.from_numpy(load_matrix_array(matrix_path))

    module = importlib.import_module(f"scripts.base_search.{args.script}")
    result = module.entrypoint(mat)
    print(result)


if __name__ == "__main__":
    main()
