import sys
from typing import List, Tuple
import numpy as np
from helper import Matrix, Tensor3D, get_matrix
from importlib import import_module


def validate(
    result: Tuple[np.ndarray, str]
) -> dict[str, float]:
    context = get_matrix()
    result, report, best_path = result
    if np.any(Tensor3D(context) != Tensor3D(Matrix.from_numpy(result))):
        return {"fitness": float('inf'),
                "is_valid": 0.0,
                "mcts info": report + best_path,
                }
    
    return {"fitness":  result.shape[0], 
            "is_valid": 1.0,
            "mcts info": report + best_path,
            }

# names = {"gf10": }
if __name__ == "__main__":
    module_path = sys.argv[1].replace('/', '.').replace('.py', '')
    entrypoint = import_module(module_path).entrypoint

    res = validate(entrypoint())
    print(res["mcts info"])