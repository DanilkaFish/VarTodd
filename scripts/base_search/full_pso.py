from typing import Iterable
import cma
import nlopt
from concurrent.futures import ProcessPoolExecutor
import pyswarms as ps
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
    
def exp(x):
    return np.exp(x)

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(4)]

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
        self.set_min_pool_size(2)
        self.set_min_z_to_research(10 + 1000*self.map_par(sigmoid, 0))
        self.set_temperature(0.5)
        self.set_num_samples(0 + 63*self.map_par(sigmoid, 0))
        self.set_max_pool_size(30)
        self.set_max_tohpe(4)
        self.set_try_only_tohpe(0)
        self.set_max_reduction(20)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(2)
        self.set_tohpe_num_best(5)
        self.set_gen_part(0.1 + 0.9*self.map_par(sigmoid, 0))
        self.set_todd_width(3)
        self.set_beamsearch_width(5)

    def __call__(self, params: Iterable):
        tcounts = self.run(params, self.seeds, max_workers=1)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.02 * spread
    

def run_opt(fun: Evaluator, num_eval: int=10) -> Evaluator:
    x = fun.extract_active()
    def objective_function(positions):
        """Alternative wrapper if fun takes parameters directly"""
        
        with ProcessPoolExecutor(max_workers=16) as ex:
            futures = [
                ex.submit(fun, position)
                for position in positions
            ]
            costs = [f.result() for f in futures]

        # for i in range(n_particles):
        #     costs[i] = fun(positions[i])
        
        return np.array(costs)
    
    n_params = len(x)
    lb = [-2.0] * n_params
    ub = [2.0] * n_params

    # Create bounds
    bounds = (np.array(lb), np.array(ub))
    options = {
        'c1': 0.5,  # cognitive parameter
        'c2': 0.3,  # social parameter
        'w': 0.9    # inertia weight
    }

    # Create optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=32, 
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
    num_eval = 40
    x0 = run_opt(fun, num_eval)
    fun(x0)
    print(f"{fun.best_pathes[0].final_node.state.rows=}")
    # x0 = run_cma(fun)
    ranks = [fun.best_pathes[0].final_node.state.rows + 60 - 20*i for i in range(20)]
    for r in ranks:
        x_active = fun.set_up_new_init(0, rank_thr=r, xopt=None)
        if x_active is None:
            break
        x0 = run_opt(fun, 20)
        fun(x0)
    return fun.get_best()

# {'num_samples': [(334, 314, 215, 0), (20, 20, 20, 20)], 'top_pool': [(334, 314, 215, 0), (20, 20, 20, 20)], 'try_only_tohpe': [(334, 314, 215, 0), (1, 1, 1, 1)], 'max_tohpe': [(334, 314, 215, 0), (4, 4, 4, 4)], 'num_tohpe_sample': [(334, 314, 215, 0), (7, 7, 7, 7)], 'max_from_single_ns': [(334, 314, 215, 0), (4, 4, 4, 4)], 'min_reduction': [(334, 314, 215, 0), (1, 1, 1, 1)], 'max_reduction': [(334, 314, 215, 0), (30, 30, 30, 30)], 'pool_scores': [(334, 314, 215, 0), ([3.05, 1.258, -2.704, -2.543, 0.314], [1.226, -1.352, -3.112, -1.762, 2.803], [1.812, 0.338, 2.59, -2.568, 2.969], [-0.881, -1.023, 1.808, 0.775, -1.525])], 'final_scores': [(334, 314, 215, 0), ([3.078, -3.078, -3.375, -3.294, -3.423, 3.562], [2.785, -1.552, -1.13, 1.885, -0.178, 2.001], [-0.007, 0.871, -0.217, 2.643, 2.897, 0.343], [2.131, -1.467, -0.018, 2.591, -1.914, -1.241])], 'max_z_to_research': [(334, 314, 215, 0), (300.0, 300.0, 300.0, 300.0)], 'gen_part': [(334, 314, 215, 0), (1.0, 1.0, 1.0, 1.0)], 'temperature': [(334, 314, 215, 0), (0.5, 0.5, 0.5, 0.5)]}