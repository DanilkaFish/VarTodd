from typing import Iterable
import cma
import nlopt
from pyswarm import pso
import numpy as np
from scripts.optimization_core.helper import get_matrix, BaseEvaluator, Matrix, find_rank, ExplorationScore, FinalizationScore
import random

np.random.seed(42)
random.seed(40)

def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))

def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0/beta) * np.log(np.exp(-beta*(xs - m)).sum()))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(6)]

    def policy_mapping(self):
        ranks = [0, 195, 205]
        
        w_pool = [ExplorationScore(wred=self.map_par(_w_tanh, r), 
                                 wdim=self.map_par(_w_tanh, r), 
                                 wbucket=self.map_par(_w_tanh, r), 
                                 wvw=self.map_par(_w_tanh, r),
                                 wz=self.map_par(_w_tanh, r)) for r in ranks]
        
        w_final = [FinalizationScore(wred=self.map_par(_w_tanh, r), 
                                   wdim=self.map_par(_w_tanh, r), 
                                   wbucket=self.map_par(_w_tanh, r), 
                                   wvw=self.map_par(_w_tanh, r), 
                                   wz=self.map_par(_w_tanh, r),
                                   wtohpe=self.map_par(_w_tanh, r)) for r in ranks]
        
        self.set_pool_weights(ranks, w_pool)
        self.set_final_weights(ranks, w_final)
        self.set_min_pool_size(2)
        self.set_min_z_to_research(150) 
        self.set_temperature(0.5)
        
        num_samples_vals = [8, 6, 4] 
        self.set_num_samples(ranks, num_samples_vals)
        self.set_max_pool_size(6)
        self.set_max_tohpe(6)
        
        try_only_tohpe_vals = [1, 1, 0]  
        self.set_try_only_tohpe(ranks, try_only_tohpe_vals)
        
        self.set_max_reduction(10)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(2)
        self.set_tohpe_num_best(8)
        self.set_gen_part(1.0)
        self.set_todd_width(2)  
        self.set_beamsearch_width(3)  

    def __call__(self, params: Iterable): 
        tcounts = self.run(params, self.seeds)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.01 * spread
    

def entrypoint(mat):
    fun = Evaluator(mat=mat, max_depth=100)
    
    x_active = fun.extract_active()
    n_params = len(x_active)
    if n_params == 0:
        return fun.get_best()
        
    lb = [-1.2] * n_params
    ub = [1.2] * n_params
    
    xopt, fopt = pso(fun, lb, ub, swarmsize=8, maxiter=8, debug=True)
    x_active2 = fun.set_up_new_init(0, rank_thr=195, xopt=xopt)
    if x_active is not None and len(x_active2) > 0:
        lb2 = [-1.2] * len(x_active2)
        ub2 = [1.2] * len(x_active2)
        xopt2, fopt2 = pso(fun, lb2, ub2, swarmsize=6, maxiter=5, debug=True)
    
    return fun.get_best()