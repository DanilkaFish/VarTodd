from typing import Iterable
import numpy as np
from scripts.optimization_core.helper import get_matrix, BaseEvaluator, Matrix, find_rank, ExplorationScore, FinalizationScore
import random

np.random.seed(42)
random.seed(40)

def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0/beta) * np.log(np.exp(-beta*(xs - m)).sum()))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(64)]

    def policy_mapping(self):
        lr = 0
        ranks = [lr]
        w_pool = [ExplorationScore([-1,0,0,0,0],
                                  pow=1) for r in  ranks]
        w_final = [FinalizationScore([-1,0,0,0,0,0],
                                  pow=1
                                  ) for r in ranks]
        self.set_pool_scores(ranks, w_pool)
        self.set_final_scores(ranks, w_final)
        self.set_min_pool_size(1)
        self.set_min_z_to_research(100000)
        self.set_temperature(0.0)
        self.set_num_samples(30)
        self.set_max_pool_size(1)
        self.set_max_tohpe(1)
        self.set_try_only_tohpe(1)
        self.set_max_reduction(100)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(1)
        self.set_tohpe_num_best(1)
        self.set_gen_part(1.0)
        self.set_todd_width(1)
        self.set_beamsearch_width(1)

    def __call__(self, params: Iterable):
        tcounts = self.run(params, self.seeds, max_workers=4)
        return min(tcounts) 

def entrypoint(mat: Matrix):
    fun = Evaluator(mat=mat, fin_rank=170, max_depth=200)
    x = fun.extract_active()
    fun(x)
    return fun.get_best(), fun.tcount


# [27, 43, 67, 91, 117, 137, 193, 219, 257, 281, 333, 359, 397, 415]