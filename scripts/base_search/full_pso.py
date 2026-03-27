from typing import Iterable

import numpy as np
import pyswarms as ps
import random

from scripts.optimization_core.helper import (
    BaseEvaluator,
    ExplorationScore,
    FinalizationScore,
    Matrix,
)

np.random.seed(42)
random.seed(40)

def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))

def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0/beta) * np.log(np.exp(-beta*(xs - m)).sum()))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(2)]

    def policy_mapping(self):
        lr = 0
        ranks = [lr]
        w_pool = [ExplorationScore([self.map_par(_w_tanh,0), 
                                self.map_par(_w_tanh, 0), 
                                self.map_par(_w_tanh, 0), 
                                self.map_par(_w_tanh, 0),
                                self.map_par(_w_tanh, 0)],
                                  pow=1) for r in  ranks]
        w_final = [FinalizationScore([self.map_par(_w_tanh,0), 
                                    self.map_par(_w_tanh, 0), 
                                    self.map_par(_w_tanh, 0), 
                                    self.map_par(_w_tanh, 0),
                                    self.map_par(_w_tanh, 0), 
                                    self.map_par(_w_tanh, 0)],
                                  pow=1
                                  ) for r in ranks]
        self.set_pool_scores(ranks, w_pool)
        self.set_final_scores(ranks, w_final)
        self.set_min_pool_size(1)
        self.set_min_z_to_research(10 + 250 * self.map_par(sigmoid, 0))
        self.set_temperature(0.2)
        self.set_num_samples(8 + 48 * self.map_par(sigmoid, 0))
        self.set_max_pool_size(12)
        self.set_max_tohpe(2)
        self.set_try_only_tohpe(0)
        self.set_max_reduction(12)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(1)
        self.set_tohpe_num_best(2)
        self.set_gen_part(0.1 + 0.5 * self.map_par(sigmoid, 0))
        self.set_todd_width(2)
        self.set_beamsearch_width(2)

    def __call__(self, params: Iterable):
        tcounts = self.run(params, self.seeds, max_workers=1)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.02 * spread
    

def run_opt(fun: Evaluator, num_eval: int=10) -> Evaluator:
    x = fun.extract_active()
    def objective_function(positions):
        """Alternative wrapper if fun takes parameters directly"""
        return np.array([fun(p) for p in positions])
    
    n_params = len(x)
    lb = [-1.0] * n_params
    ub = [ 1.0] * n_params

    # Create bounds
    bounds = (np.array(lb), np.array(ub))
    options = {
        'c1': 0.4,  # cognitive parameter
        'c2': 0.4,  # social parameter
        'w': 0.7    # inertia weight
    }

    # Create optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=16,
        dimensions=n_params, 
        options=options,
        bounds=bounds
    )

    # Run optimization
    best_cost, best_position = optimizer.optimize(
        objective_function, 
        iters=num_eval,
        verbose=True
    )
    return best_position
        
def entrypoint(mat: Matrix):
    fun = Evaluator(mat=mat, fin_rank=170, max_depth=100)
    num_eval = 50
    x0 = run_opt(fun, num_eval)
    fun(x0)
    print(f"{fun.best_pathes[0].final_node.state.rows=}")

    best_rank = fun.best_pathes[0].final_node.state.rows
    mid_rank_thr = max(best_rank + 1, (fun.init_rank + best_rank) // 2)
    late_rank_thr = max(best_rank + 1, best_rank + (mid_rank_thr - best_rank) // 2)

    for rank_thr in (mid_rank_thr, late_rank_thr):
        x_active = fun.set_up_new_init(0, rank_thr=rank_thr, xopt=None)
        if x_active is None:
            break
        x0 = run_opt(fun, 30)
        fun(x0)
    return fun.get_best()