from typing import Iterable
import numpy as np
import random
import cma
from helper import BaseEvaluator, Matrix, ExplorationScore, FinalizationScore

# Fixed RNGs for reproducible sampling behavior
np.random.seed(42)
random.seed(40)

def _w_tanh(z: float, scale: float = 3.0, sharp: float = 1.2) -> float:
    return float(scale * np.tanh(z / sharp))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

class Evaluator(BaseEvaluator):
    # Keep seeds fixed but slightly reduced to 3 to balance variance and runtime cost
    seeds = [42, 1337, 9001]

    def policy_mapping(self):
        # Lowered thresholds to match current best performance (around 405)
        ranks = [415, 395, 0]

        # Map core weights including previously ignored 'vw' and 'z' features
        wred = self.map_par(_w_tanh, 0)
        wdim = self.map_par(_w_tanh, 0)
        wvw = self.map_par(_w_tanh, 415)  # Activate later to reduce dimensionality early
        wz = self.map_par(_w_tanh, 415)
        
        fwred = self.map_par(_w_tanh, 0)
        fwvw = self.map_par(_w_tanh, 395)

        budget = self.map_par(sigmoid, 0)

        # Enable vw and z weights in scores to improve action discrimination
        w_pool = [ExplorationScore(weights=[wred, wdim, 0.0, wvw, wz], pow=1) for _ in ranks]
        w_final = [FinalizationScore(
            weights=[fwred, 0.0, 0.0, fwvw, 0.0, 0.0],
            centers=[1, 0, 0, 0, 0, 0],
            pow=2
        ) for _ in ranks]

        self.set_pool_scores(ranks, w_pool)
        self.set_final_scores(ranks, w_final)

        # Adaptive search widths based on budget
        beam = int(1 + 2 * budget)
        todd_w = int(1 + 2 * budget)
        max_pool = int(10 + 20 * budget)
        num_samples = int(4 + 8 * (1.0 - budget))
        
        # Stricter min_z cap to manage runtime while allowing exploration
        min_z = int(min(8 + 80 * budget, 45))
        
        self.set_min_pool_size(2)
        self.set_min_z_to_research(min_z)
        self.set_temperature(0.4)
        self.set_num_samples(ranks, [num_samples, num_samples, max(4, num_samples - 2)])
        self.set_max_pool_size(max_pool)
        
        # Force TODD earlier than Parent 2 to ensure deep exploration
        self.set_try_only_tohpe(ranks, [1, 0, 0])
        
        self.set_max_reduction(12)  # Relaxed clamp to allow big jumps
        self.set_min_reduction(1)
        self.set_max_from_single_ns(2)
        self.set_tohpe_num_best(6)
        self.set_gen_part(0.3 + 0.5 * budget)
        self.set_todd_width(todd_w)
        self.set_beamsearch_width(beam)

    def __call__(self, params: Iterable):
        # Tightened early rejection threshold to +12 to save runtime
        early = self.run(params, self.seeds[:1])
        if self.best_rank < 1e6 and early[0] > self.best_rank + 12:
            return float(early[0] + 10.0)

        rest = self.run(params, self.seeds[1:])
        tcounts = early + rest
        # Use 0.6 quantile for a slightly more optimistic but still robust signal
        q = float(np.quantile(tcounts, 0.6))
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return q + 0.05 * spread

def run_cma(fun: Evaluator, maxfevals: int = 50, sigma: float = 0.35) -> np.ndarray:
    x0 = fun.extract_active()
    if x0 is None or len(x0) == 0:
        x0 = np.zeros(1)

    options = {
        "maxfevals": maxfevals,
        "popsize": 8,
        "bounds": [-2.0, 2.0],
        "verbose": 0,
        "seed": 42,
    }

    xopt, es = cma.fmin2(
        lambda x: float(fun(x)),
        x0,
        sigma,
        options=options,
    )
    return np.array(xopt)

def entrypoint():
    fun = Evaluator(path_name="init", max_depth=200)

    # Phase 1: Broad search
    xopt = run_cma(fun, maxfevals=70, sigma=0.4)
    fun(xopt)

    # Phase 2: Refined exploitation of the tail
    best_rank = fun.best_rank
    x_active = fun.set_up_new_init(0, rank_thr=int(best_rank + 10), xopt=xopt)
    if x_active is not None and len(x_active) > 0:
        xopt = run_cma(fun, maxfevals=50, sigma=0.2)
        fun(xopt)

    return fun.get_best()