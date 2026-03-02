from typing import Iterable
import cma
import nlopt
from pyswarm import pso
import numpy as np
from scripts.optimization_core.helper import get_matrix, BaseEvaluator, Matrix, find_rank, ExplorationScore, FinalizationScore
import random
from itertools import product
np.random.seed(42)
random.seed(40)

def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))


class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(4)]

    def policy_mapping(self):
        lr = 0
        ranks = [lr]
        w_pars = [[self.map_par(_w_tanh, r) for _ in range(5)] for r in  ranks]
        w_pool = [ExplorationScore(*w_par) for w_par in  w_pars]
        w_final = [FinalizationScore(*w_par, 1) for r, w_par in zip(ranks, w_pars)]

        self.set_pool_weights(ranks, w_pool)
        self.set_final_weights(ranks, w_final)
        self.set_min_pool_size(4)
        
        #order correspond to grid dict
        self.set_try_only_tohpe(1)
        self.set_max_tohpe(5)
        self.set_max_pool_size(200)

        self.set_num_samples(60)
        self.set_min_z_to_research(1000)
        self.set_max_reduction(20)
        self.set_temperature(0.4)
        self.set_gen_part(1.0)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(20)
        self.set_tohpe_num_best(1)
        self.set_todd_width(5)
        self.set_beamsearch_width(10)

    def __call__(self, params: Iterable):
        tcounts = self.run(params, self.seeds)
        best = float(np.min(tcounts))
        std = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return best + 0.02 * std

def entrypoint(mat: Matrix):
    fun = Evaluator(mat=mat,  max_depth=200)
    pools = [np.random.normal(0, 2.0, 5).tolist() for _ in range(4)]
    fun(pools[0])
    print(fun.get_best()[1])
    x0 = fun.set_up_new_init(0, 500)

    if x0 is None:
        return fun.get_best()
    fun(pools[1])
    
    x0 = fun.set_up_new_init(0, 460)
    if x0 is None:
        return fun.get_best()
    for pool in pools[2:]:
        fun(pool)

    return fun.get_best()