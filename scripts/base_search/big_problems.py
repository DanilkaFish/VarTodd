from typing import Iterable
import cma
import nlopt
from pyswarm import pso
import numpy as np
from scripts.optimization_core.helper import get_matrix, BaseEvaluator, Matrix, find_rank, ExplorationScore, FinalizationScore
import random

np.random.seed(42)
random.seed(42)

def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.0) -> float:
    return float(scale * np.tanh(z / sharp))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(4)]

    def policy_mapping(self):
        ranks = [440, 0]
        
        w_pool = [ExplorationScore([self.map_par(_w_tanh, r), 
                                 self.map_par(_w_tanh, r),
                                 self.map_par(_w_tanh, r),
                                 self.map_par(_w_tanh, r),
                                 self.map_par(_w_tanh, r)],
                                 pow=1
                                 ) for r in ranks]
        
        w_final = [FinalizationScore([self.map_par(_w_tanh, r), 
                                    self.map_par(_w_tanh, r),
                                    self.map_par(_w_tanh, r),
                                    self.map_par(_w_tanh, r),
                                    self.map_par(_w_tanh, r),
                                    self.map_par(_w_tanh, r)],
                                    [self.map_par(_w_tanh, r), 
                                    self.map_par(_w_tanh, r),
                                    self.map_par(_w_tanh, r),
                                    self.map_par(_w_tanh, r),
                                    self.map_par(_w_tanh, r),
                                    self.map_par(_w_tanh, r)],
                                    pow=1
                                   ) for r in ranks]
        
        self.set_pool_scores(ranks, w_pool)
        self.set_final_scores(ranks, w_final)
        
        self.set_min_pool_size(2)
        self.set_max_pool_size(ranks, [32,  64,])
        self.set_min_z_to_research(ranks, [300,  120])
        self.set_temperature(0.12)
        
        self.set_num_samples(16)
        
        self.set_max_tohpe(60)
        self.set_try_only_tohpe(1)
        
        self.set_max_reduction(20)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(8)
        self.set_tohpe_num_best(4)
        self.set_gen_part(0.7)
        self.set_todd_width(ranks, [2, 6])  
        self.set_beamsearch_width(ranks, [2, 12])

    def __call__(self, params: Iterable):
        tcounts = self.run(params, self.seeds, max_workers=4)
        if not tcounts: return 1000.0
        return float(np.min(tcounts)) + 0.01 * float(np.std(tcounts))

def entrypoint(mat):
    fun = Evaluator(mat=mat, max_depth=250)
    x_active = fun.extract_active()
    if not x_active: return fun.get_best()
        
    lb = [-2.0] * len(x_active)
    ub = [2.0] * len(x_active)
    xopt, fopt = pso(fun, lb, ub, swarmsize=6, maxiter=6, debug=True)
    
    x_active = fun.set_up_new_init(0, rank_thr=440, xopt=xopt)
    if x_active is not None:
        lb2 = [-1.5] * len(x_active)
        ub2 = [1.5] * len(x_active)
        xopt, fopt = pso(fun, lb2, ub2, swarmsize=8, maxiter=6, debug=True)
    
    x_active = fun.set_up_new_init(0, rank_thr=412, xopt=xopt)
    if x_active is not None:
        options = {"seed": 42, "maxiter": 15, "popsize": 10, "verbose": -9, "maxfevals": 80}
        xopt, _ = cma.fmin2(fun, x_active, 0.2, options=options)

    return fun.get_best()