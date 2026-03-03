from typing import Iterable
import cma
import nlopt
from pyswarm import pso
import numpy as np
from scripts.optimization_core.helper import get_matrix, BaseEvaluator, Matrix, find_rank, ExplorationScore, FinalizationScore, ScoringConfig, ScoringFunction
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
    return 2.0 / (1.0 + np.exp(-x))
def exp(x):
    return np.exp(x)
class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(4)]

    def policy_mapping(self):
        lr = 0
        ranks = [lr]
        w_pool = [ExplorationScore(wred=self.map_par(sigmoid, r), 
                                  wdim=self.map_par(sigmoid, r), 
                                  wbucket=self.map_par(sigmoid, r), 
                                  wvw=self.map_par(sigmoid, r),
                                  wz=self.map_par(sigmoid, r), 
                                  sc=ScoringConfig(ScoringFunction.DISTANCE, 2)) for r in  ranks]
        w_final = [FinalizationScore(wred=self.map_par(sigmoid, r), 
                                  wdim=self.map_par(sigmoid, r), 
                                  wbucket=self.map_par(sigmoid, r), 
                                  wvw=self.map_par(sigmoid, r), 
                                  wz=self.map_par(sigmoid, r),
                                  wtohpe=self.map_par(sigmoid, r),
                                  sc=ScoringConfig(ScoringFunction.DISTANCE, 2, 0)
                                  ) for r in ranks]
        self.set_pool_weights(ranks, w_pool)
        self.set_final_weights(ranks, w_final)
        self.set_min_pool_size(4)
        self.set_min_z_to_research(1500)
        self.set_temperature(0.5)
        self.set_num_samples(20)
        self.set_max_pool_size(20)
        self.set_max_tohpe(4)
        self.set_try_only_tohpe(1)
        self.set_max_reduction(15)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(4)
        self.set_tohpe_num_best(20)
        self.set_gen_part(1.0)
        self.set_todd_width(3)
        self.set_beamsearch_width(4)

    def __call__(self, params: Iterable):
        tcounts = self.run(params, self.seeds, max_workers=4)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.02 * spread
    

def run_opt(fun: Evaluator, num_eval: int=10) -> Evaluator:
    x = fun.extract_active()
    n_params = len(x)
    lb = [-3.0] * n_params
    ub = [ 3.0] * n_params
    xopt, fopt = pso(fun, lb, ub, swarmsize=20, maxiter=num_eval, debug=True)
    return xopt

def run_cma(fun: Evaluator) -> Evaluator:
    x0 = fun.extract_active()
    sigma0 = 0.3
    cma_opts = {
        "seed": 42,
        "verbose": 1,
        "maxfevals": 120,
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
    ranks = [mat.rows - 30*i for i in range(100)]
    for r in ranks:
        x_active = fun.set_up_new_init(0, rank_thr=r, xopt=None)
        x0 = run_cma(fun)
        if x_active is None:
            break
        x0 = run_opt(fun, num_eval)
    return fun.get_best()

# {'num_samples': [(334, 314, 215, 0), (20, 20, 20, 20)], 'top_pool': [(334, 314, 215, 0), (20, 20, 20, 20)], 'try_only_tohpe': [(334, 314, 215, 0), (1, 1, 1, 1)], 'max_tohpe': [(334, 314, 215, 0), (4, 4, 4, 4)], 'num_tohpe_sample': [(334, 314, 215, 0), (7, 7, 7, 7)], 'max_from_single_ns': [(334, 314, 215, 0), (4, 4, 4, 4)], 'min_reduction': [(334, 314, 215, 0), (1, 1, 1, 1)], 'max_reduction': [(334, 314, 215, 0), (30, 30, 30, 30)], 'pool_weights': [(334, 314, 215, 0), ([3.05, 1.258, -2.704, -2.543, 0.314], [1.226, -1.352, -3.112, -1.762, 2.803], [1.812, 0.338, 2.59, -2.568, 2.969], [-0.881, -1.023, 1.808, 0.775, -1.525])], 'final_weights': [(334, 314, 215, 0), ([3.078, -3.078, -3.375, -3.294, -3.423, 3.562], [2.785, -1.552, -1.13, 1.885, -0.178, 2.001], [-0.007, 0.871, -0.217, 2.643, 2.897, 0.343], [2.131, -1.467, -0.018, 2.591, -1.914, -1.241])], 'max_z_to_research': [(334, 314, 215, 0), (300.0, 300.0, 300.0, 300.0)], 'gen_part': [(334, 314, 215, 0), (1.0, 1.0, 1.0, 1.0)], 'temperature': [(334, 314, 215, 0), (0.5, 0.5, 0.5, 0.5)]}