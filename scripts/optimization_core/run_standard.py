import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path
from typing import Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.optimization_core.helper import Matrix, Tensor3D, load_matrix_array

DATA_ROOT = ROOT_DIR / "data/init_npy"
DEFAULT_INIT_CIRCUIT = "gf_mult_Vandaele_wo_ancilla"
DEFAULT_MODULE_PATH = "scripts/base_search/full_pso.py"
DEFAULT_STOP_BEFORE_GF_DEGREE = 32
GF_DEGREE_RE = re.compile(r"^gf2\^(\d+)_")
TIME_TO_FINAL_RANK_RE = re.compile(
    r"time_to_final_rank_seconds:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

def get_matrix(name: str) -> Matrix:
    return Matrix.from_numpy(load_matrix_array(DATA_ROOT / f"{name}.npy"))


def discover_names(init_circuit: str, stop_before_gf_degree: int) -> list[str]:
    root = DATA_ROOT / init_circuit
    if not root.is_dir():
        raise FileNotFoundError(f"init circuit directory does not exist: {root}")

    records = []
    for path in root.rglob("*.npy"):
        name = path.relative_to(root).with_suffix("").as_posix()
        match = GF_DEGREE_RE.match(Path(name).name)
        if match is None:
            raise ValueError(f"cannot determine GF degree from problem name: {name}")
        degree = int(match.group(1))
        if degree < stop_before_gf_degree:
            records.append((degree, name))
    return [name for _, name in sorted(records)]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a baseline optimizer over init matrices and save validated outputs."
    )
    parser.add_argument(
        "module_path",
        nargs="?",
        default=DEFAULT_MODULE_PATH,
        help="Optimizer module path, e.g. scripts/base_search/full_pso.py",
    )
    parser.add_argument(
        "workers",
        nargs="?",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of matrices to process in parallel.",
    )
    parser.add_argument(
        "--init-circuit",
        default=DEFAULT_INIT_CIRCUIT,
        help=f"Subdirectory under {DATA_ROOT} to scan.",
    )
    parser.add_argument(
        "--stop-before-gf-degree",
        type=int,
        default=DEFAULT_STOP_BEFORE_GF_DEGREE,
        help="Keep matrices whose GF degree is strictly less than this value.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Explicit matrix names relative to --init-circuit, without .npy.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print selected matrix names and exit without running the optimizer.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSON report path; validated baseline matrices are saved beside it.",
    )
    return parser.parse_args(argv)


def _module_name(module_path: str) -> str:
    path = Path(module_path)
    if path.suffix == ".py":
        try:
            path = path.resolve().relative_to(ROOT_DIR)
        except ValueError:
            path = path.with_suffix("")
        return ".".join(path.with_suffix("").parts)
    return module_path.replace("/", ".").removesuffix(".py")


def _module_stem(module_path: str) -> str:
    return Path(module_path).name.removesuffix(".py")


def _resolve_project_path(path: str) -> Path:
    out = Path(path)
    return out if out.is_absolute() else ROOT_DIR / out


def validate(
    result: tuple[np.ndarray, str, str],
    name: Optional[str] = None,
) -> dict[str, object]:
    if name is None:
        raise ValueError("validate requires a matrix name")

    context = get_matrix(name)
    result, report, paths = result
    res = Matrix.from_numpy(result)
    if Tensor3D(context) != Tensor3D(res):
        raise RuntimeError(
            f"tensor mismatch for {name}: "
            f"expected {context.rows}x{context.cols}, got {res.rows}x{res.cols}"
        )
    print(report + paths)
    return {
        "result": result,
        "mcts info": report + paths,
        "paths": paths,
    }


def _extract_time_to_final_rank(paths: str) -> float | None:
    match = TIME_TO_FINAL_RANK_RE.search(paths)
    return float(match.group(1)) if match else None


def _write_json_report(output_path: Path, records: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)


def _run_one(
    name: str,
    *,
    module_path: str,
    last_name: str,
    init_circuit: str,
    output_path: Path,
    initial_rank: int,
) -> tuple[dict[str, object], list]:
    entrypoint = import_module(module_path).entrypoint
    start_time = time.perf_counter()
    en = entrypoint(get_matrix(init_circuit + "/" + name))
    execution_seconds = time.perf_counter() - start_time
    if isinstance(en, tuple) and len(en) == 2:
        en, tcount = en
    else:
        tcount = []
    res = validate(en, init_circuit + "/" + name)
    result_rank = int(res["result"].shape[0])
    safe_name = name.replace("/", "__")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_filename = output_path.parent / f"{last_name}-{safe_name}-{result_rank}.npy"
    np.save(output_filename, res["result"])
    print(f"Results for {name} saved to {output_filename}:\n\tFinal rank = {result_rank}")
    return {
        "problem_name": init_circuit + "/" + name,
        "initial_rank": initial_rank,
        "final_rank": result_rank,
        "execution_seconds": execution_seconds,
        "time_to_final_rank_seconds": _extract_time_to_final_rank(str(res["paths"])),
        "paths": res["paths"],
        "result_path": str(output_filename.resolve()),
    }, tcount


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    module_path = _module_name(args.module_path)
    last_name = _module_stem(args.module_path)
    workers = max(1, int(args.workers))
    init_circuit = args.init_circuit
    output_path = _resolve_project_path(args.output)
    names = args.names or discover_names(init_circuit, args.stop_before_gf_degree)

    print(
        f"Selected {len(names)} matrices from {DATA_ROOT / init_circuit} "
        f"with GF degree < {args.stop_before_gf_degree}:"
    )
    for name in names:
        print(f"  {name}")
    if args.list_only:
        return
    if not names:
        raise SystemExit("no matrices selected")

    initial_ranks = [get_matrix(init_circuit + "/" + name).rows for name in names]
    records: list[Optional[dict[str, object]]] = [None] * len(names)
    tcounts: list[list] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _run_one,
                name,
                module_path=module_path,
                last_name=last_name,
                init_circuit=init_circuit,
                output_path=output_path,
                initial_rank=initial_ranks[index],
            ): index
            for index, name in enumerate(names)
        }
        for future in as_completed(futures):
            index = futures[future]
            record, tcount = future.result()
            records[index] = record
            tcounts.append(tcount)
    _write_json_report(output_path, [record for record in records if record is not None])
    print([tcount for tcount in tcounts])


if __name__ == "__main__":
    main()
