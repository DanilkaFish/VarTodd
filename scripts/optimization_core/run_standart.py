import sys
from typing import List, Tuple
import numpy as np
from helper import Matrix, Tensor3D, get_matrix
from importlib import import_module

def get_matrix(name:str=None) -> Matrix:
    return Matrix.from_numpy(np.load(f"data/init_npy/{name}.matrix.npy") )

def validate(
    result: Tuple[np.ndarray, str],
    name: str= None
) -> dict[str, float]:
    context = get_matrix(name)
    result, report, best_path = result
    if np.any(Tensor3D(context) != Tensor3D(Matrix.from_numpy(result))):
        raise RuntimeError("AHTUNG")
    
    return {"result": result,
            "mcts info": report + best_path,
            }

if __name__ == "__main__":
    module_path = sys.argv[1].replace('/', '.').replace('.py', '')
    entrypoint = import_module(module_path).entrypoint
    names = [
        # "gf2^3_mult_fr_310",
        # "gf2^4_mult_fr_410",
        # "gf2^5_mult_fr_54320",
        "gf2^6_mult_fr_610",
        # "gf2^7_mult_fr_730",
        # "gf2^8_mult_fr_84310",
        # "gf2^9_mult_fr_940",
        # "gf2^10_mult_fr_1030",
        # "gf2^11_mult_fr_1120"
        # "1733init173-00231"
        # # "1731init129-00168"

    ]
        # "gf2^10_mult_1030.qc.matrix.npy").T )
    results = []
    aux_info = {}
    tcounts = []
    for name in names:
        en = entrypoint(get_matrix(name))
        res = validate(en, name)
        # tcounts.append(tcount)
        results.append(res["result"])
        output_filename = f"data/baseline_npy/{name}-{res["result"].shape[0]}"
        np.save(output_filename, res["result"])
        print(f"Results for {name} saved to {output_filename}: Final rank = {res["result"].shape[0]}")
        # print(f"Results for {name} saved to {output_filename}")

    # if r is None:
    #     output_filename = f"{module_path.replace(".", "/")}.txt"
    # else:
    #     output_filename = f"{module_path.replace(".", "/")}{r=}.txt"
    # with open(output_filename, 'w') as f:
    #     f.write(f"{results}\n")

    #     for name, fitness, tcount in zip(names, results, tcounts):
    #         f.write(f"{name}: {fitness} {tcount}\n")
    #     f.write(f"{aux_info}\n")
    
    # print(f"Results saved to {output_filename}")
    # res = validate(entrypoint())
    # print(res["mcts info"])