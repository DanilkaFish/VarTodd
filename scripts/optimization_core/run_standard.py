import sys
from typing import List, Tuple
import numpy as np
from helper import Matrix, Tensor3D, get_matrix
from importlib import import_module

def get_matrix(name:str=None) -> Matrix:
    return Matrix.from_numpy(np.load(f"data/init_npy/{name}.npy") )

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

if __name__ == "__main__":
    module_path = sys.argv[1].replace('/', '.').replace('.py', '')
    last_name = sys.argv[1].split('/')[-1].replace('.py', '')
    entrypoint = import_module(module_path).entrypoint
    init_circtit = "gf_mult_Vandaele_wo_ancilla"
    names = [
        # "gf_mult_Khoruzhii_best/gf2^3",
        # "gf_mult_Khoruzhii_best/gf2^4",
        # "gf_mult_Khoruzhii_best/gf2^5",
        # "gf_mult_Khoruzhii_best/gf2^6",
        # "gf_mult_Khoruzhii_best/gf2^7",
        # "gf_mult_Khoruzhii_best/gf2^8",
        # "gf_mult_Khoruzhii_best/gf2^9",
        # "gf_mult_Khoruzhii_best/gf2^10",
        "gf2^3_310",
        "gf2^4_410",
        "gf2^5_520",
        "gf2^6_610",
        "gf2^7_710",
        "gf2^8_84320",
        "gf2^9_940",
        "gf2^10_1030",
        "gf2^11_1120",
        "gf2^12_126410",
        "gf2^13_134310",
        "gf2^14_148610",
        "gf2^15_1510",
        "gf2^16_1612310",

        # "gf2^3_mult_fr_310",
        # "gf2^3_mult_fr_310",

        # "gf2^3_mult_fr_310",
        # "gf2^4_mult_fr_410",
        # "gf2^5_mult_fr_54320",
        # "gf2^6_mult_fr_610",
        # "gf_mult_Vandaele_wo_ancilla/gf2^10_1030",
        # "gf2^8_mult_fr_84310",
        # "gf2^9_mult_fr_940",
        # "gf2^10_mult_fr_1030",
        # "gf2^11_mult_fr_1120"
        # "1733init173-00231"
        # # "1731init129-00168"
    ]
    results = []
    aux_info = {}
    tcounts = []
    for name in names:
        en = entrypoint(get_matrix(init_circtit + "/" + name))
        if isinstance(en, Tuple) and len(en) == 2:
            en, tcount = en
        else:
            tcount = []
        res = validate(en, init_circtit + "/" + name)
        tcounts.append(tcount)
        results.append(res["result"])
        output_filename = f"data/baseline_npy/{init_circtit}/{last_name}-{name}-{res["result"].shape[0]}"

        np.save(output_filename, res["result"])
        print(f"Results for {name} saved to {output_filename}:\n\tFinal rank = {res["result"].shape[0]}")
    print([tcount for tcount in tcounts])