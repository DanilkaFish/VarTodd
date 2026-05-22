import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path
from typing import Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from helper import Matrix, Tensor3D, load_matrix_array, normalize_matrix_array

DATA_ROOT = ROOT_DIR / "data/init_npy"
DEFAULT_INIT_CIRCUIT = "gf_mult_Vandaele_wo_ancilla"
DEFAULT_M_INIT = 600
DEFAULT_N_INIT = 200

def get_matrix(name: str) -> Matrix:
    return Matrix.from_numpy(load_matrix_array(DATA_ROOT / f"{name}.npy"))


def _matrix_shape(path: Path) -> tuple[int, int]:
    arr = normalize_matrix_array(np.load(path, mmap_mode="r"))
    if arr.ndim != 2:
        raise ValueError(f"{path} is not a 2D matrix, shape={arr.shape}")
    return int(arr.shape[0]), int(arr.shape[1])


def discover_names(init_circuit: str, m_init: int, n_init: int) -> list[str]:
    root = DATA_ROOT / init_circuit
    if not root.is_dir():
        raise FileNotFoundError(f"init circuit directory does not exist: {root}")

    records = []
    for path in sorted(root.rglob("*.npy")):
        rows, cols = _matrix_shape(path)
        if rows < m_init and cols < n_init:
            name = path.relative_to(root).with_suffix("").as_posix()
            records.append((rows, cols, name))
    records.sort()
    return [name for _, _, name in records]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a baseline optimizer over init matrices and save validated outputs."
    )
    parser.add_argument("module_path", help="Optimizer module path, e.g. scripts/base_search/full_pso.py")
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
        "--m-init",
        type=int,
        default=DEFAULT_M_INIT,
        help="Keep matrices with rows strictly less than this value.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=DEFAULT_N_INIT,
        help="Keep matrices with columns strictly less than this value.",
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
        "--output-root",
        default=str(ROOT_DIR / "data/baseline_npy"),
        help="Directory where validated baseline matrices are saved.",
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
    context = get_matrix(name)
    result, report, best_path = result
    res = Matrix.from_numpy(result)
    if Tensor3D(context) != Tensor3D(res):
        print(f"{context.rows=} {context.cols=}")
        print(f"{res.rows=} {res.cols=}")
        raise RuntimeError("AHTUNG")
    print(report + best_path)
    return {"result": result,
            "mcts info": report + best_path,
            }

def _run_one(
    name: str,
    *,
    module_path: str,
    last_name: str,
    init_circuit: str,
    output_root: Path,
) -> tuple[str, list]:
    entrypoint = import_module(module_path).entrypoint
    en = entrypoint(get_matrix(init_circuit + "/" + name))
    if isinstance(en, tuple) and len(en) == 2:
        en, tcount = en
    else:
        tcount = []
    res = validate(en, init_circuit + "/" + name)
    result_rank = res["result"].shape[0]
    safe_name = name.replace("/", "__")
    output_dir = output_root / init_circuit
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{last_name}-{safe_name}-{result_rank}.npy"
    np.save(output_filename, res["result"])
    print(f"Results for {name} saved to {output_filename}:\n\tFinal rank = {result_rank}")
    return name, tcount

if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    module_path = _module_name(args.module_path)
    last_name = _module_stem(args.module_path)
    workers = max(1, int(args.workers))
    init_circuit = args.init_circuit
    output_root = _resolve_project_path(args.output_root)
    names = args.names or discover_names(init_circuit, args.m_init, args.n_init)

    print(
        f"Selected {len(names)} matrices from {DATA_ROOT / init_circuit} "
        f"with rows < {args.m_init} and cols < {args.n_init}:"
    )
    for name in names:
        rows, cols = _matrix_shape(DATA_ROOT / init_circuit / f"{name}.npy")
        print(f"  {name} ({rows}x{cols})")
    if args.list_only:
        raise SystemExit(0)
    if not names:
        raise SystemExit("no matrices selected")

    tcounts = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _run_one,
                name,
                module_path=module_path,
                last_name=last_name,
                init_circuit=init_circuit,
                output_root=output_root,
            ): name
            for name in names
        }
        for future in as_completed(futures):
            name, tcount = future.result()
            tcounts.append(tcount)
    print([tcount for tcount in tcounts])
