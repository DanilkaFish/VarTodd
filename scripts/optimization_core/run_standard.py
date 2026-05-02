import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
from typing import Tuple

import numpy as np

from helper import Matrix, Tensor3D

DATA_ROOT = "data/init_npy"

def get_matrix(name: str) -> Matrix:
    return Matrix.from_numpy(np.load(f"{DATA_ROOT}/{name}.npy"))

def validate(
    result: Tuple[np.ndarray, str],
    name: str= None
) -> dict[str, float]:
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
) -> Tuple[str, list]:
    entrypoint = import_module(module_path).entrypoint
    en = entrypoint(get_matrix(init_circuit + "/" + name))
    if isinstance(en, Tuple) and len(en) == 2:
        en, tcount = en
    else:
        tcount = []
    res = validate(en, init_circuit + "/" + name)
    result_rank = res["result"].shape[0]
    output_filename = f"data/baseline_npy/{init_circuit}/{last_name}-{name}-{result_rank}"
    np.save(output_filename, res["result"])
    print(f"Results for {name} saved to {output_filename}:\n\tFinal rank = {result_rank}")
    return name, tcount

if __name__ == "__main__":
    print("hello")
    if len(sys.argv) < 2:
        raise SystemExit("usage: python run_standard.py <module_path> [workers]")
    module_path = sys.argv[1].replace('/', '.').replace('.py', '')
    last_name = sys.argv[1].split('/')[-1].replace('.py', '')
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, min(8, os.cpu_count() or 1))
    init_circuit = "other"
    # init_circuit = "gf_mult_Vandaele_wo_ancilla"
    # init_circuit = "gf_mult_Khoruzhii_best"
    names = [
        # "gf2^3",
        # "gf2^4",
        # "gf2^5",
        # "gf2^6",
        # "gf2^7",
        # "gf2^8",
        # "gf2^9",
        # "gf2^10",
        # "gf2^3_310",
        # "gf2^4_410",
        # "gf2^5_520",
        # "gf2^6_610",
        # "gf2^7_710",
        # "gf2^8_84320",
        # "gf2^9_940",
        # "gf2^10_1030",
        # "gf2^11_1120",
        # "gf2^12_126410",
        # "gf2^13_134310",
        # "gf2^14_148610",
        # "gf2^15_1510",
        # "gf2^16_1612310"
        # "gf2^32_3228320"
        # "mod_adder_1024"
        "ham15_high"
    ]
    aux_info = {}
    tcounts = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _run_one,
                name,
                module_path=module_path,
                last_name=last_name,
                init_circuit=init_circuit,
            ): name
            for name in names
        }
        for future in as_completed(futures):
            name, tcount = future.result()
            tcounts.append(tcount)
    print([tcount for tcount in tcounts])
