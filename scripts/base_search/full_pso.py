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
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(4)]

    def policy_mapping(self):
        lr = 0
        ranks = [lr]
        w_pool = [ExplorationScore(wred=self.map_par(_w_tanh, r), 
                                  wdim=self.map_par(_w_tanh, r), 
                                  wbucket=self.map_par(_w_tanh, r), 
                                  wvw=self.map_par(_w_tanh, r),
                                  wz=self.map_par(_w_tanh, r)) for r in  ranks]
        w_final = [FinalizationScore(wred=self.map_par(_w_tanh, r), 
                                  wdim=self.map_par(_w_tanh, r), 
                                  wbucket=self.map_par(_w_tanh, r), 
                                  wvw=self.map_par(_w_tanh, r), 
                                  wz=self.map_par(_w_tanh, r),
                                  wtohpe=self.map_par(_w_tanh, r)
                                  ) for r in ranks]
        self.set_pool_weights(ranks, w_pool)
        self.set_final_weights(ranks, w_final)
        self.set_min_pool_size(3)
        self.set_min_z_to_research(self.map_par(lambda x: 100 + 1500*sigmoid(x), 0))
        self.set_temperature(0.5)
        self.set_num_samples(self.map_par(lambda x: 1 + 100*sigmoid(x), 0))
        self.set_top_pool(20)
        self.set_max_tohpe(4)
        self.set_try_only_tohpe(1)
        self.set_max_reduction(30)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(4)
        self.set_tohpe_num_best(7)
        self.set_gen_part(self.map_par(lambda x: sigmoid(x), 0))
        self.set_todd_width(3)
        self.set_beamsearch_width(4)

    def __call__(self, params: Iterable):
        tcounts = self.run(params, self.seeds)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.02 * spread
    

def run_opt(fun: Evaluator, num_eval: int=10) -> Evaluator:
    x = fun.extract_active()
    n_params = len(x)
    lb = [-2.2] * n_params
    ub = [ 2.2] * n_params
    xopt, fopt = pso(fun, lb, ub, swarmsize=20, maxiter=num_eval, debug=True)
    return xopt

def run_cma(fun: Evaluator) -> Evaluator:
    x0 = fun.extract_active()
    sigma0 = 0.1
    cma_opts = {
        "seed": 42,
        "verbose": 1,
        "maxfevals": 220,
        "popsize": 8,
        "CMA_active": True,
    }
    x_cma, f_cma = cma.fmin2(fun, x0, sigma0, options=cma_opts)
    return x_cma

        
def entrypoint(mat: Matrix):
    fun = Evaluator(mat=mat, fin_rank=170, max_depth=100)
    num_eval = 20
    x0 = run_opt(fun, num_eval)
    fun.insert(x0)
    x0 = run_cma(fun)
    ranks = [mat.rows - 10*i for i in range(100)]
    for r in ranks:
        new_mat = find_rank(fun.best_pathes[0], rank=r)
        if new_mat is None:
            break
        fun.set_up_new_init(new_mat, xopt=x0)
        x0 = run_opt(fun, num_eval)
        fun.insert(x0)
        x0 = run_cma(fun)
    return fun.get_best()