from typing import Iterable
import random

import numpy as np

try:
    import pyswarms as ps
except Exception:  # pragma: no cover - evaluation env normally provides it
    ps = None

from helper import BaseEvaluator, ExplorationScore, FinalizationScore

np.random.seed(42)
random.seed(40)


def _w_tanh(z: float, scale: float = 2.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(float(z) / sharp))


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-x)))


class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(2)]
    deep_search = False
    deep_max_z = 1200000

    def policy_mapping(self):
        ranks = [max(1, int(self.init_rank * 0.82)), 0]

        wred = self.map_par(_w_tanh, 0)
        wdim = self.map_par(_w_tanh, 0)
        wbucket = self.map_par(_w_tanh, 0)
        fwred = self.map_par(_w_tanh, 0)
        fwdim = self.map_par(_w_tanh, 0)
        budget = self.map_par(sigmoid, 0)

        self.set_pool_scores(
            ranks,
            [
                ExplorationScore(weights=[wred, wdim, wbucket, 0.0, 0.0], pow=1),
                ExplorationScore(weights=[wred, 0.5 * wdim, 0.5 * wbucket, 0.0, 0.0], pow=1),
            ],
        )
        self.set_final_scores(
            ranks,
            [
                FinalizationScore(weights=[fwred, fwdim, wbucket, 0.0, 0.0, 0.0], centers=[1, 0, 0, 0, 0, 0], pow=2),
                FinalizationScore(weights=[fwred, 0.5 * fwdim, 0.4 * wbucket, 0.0, 0.0, 0.0], centers=[1, 0, 0, 0, 0, 0], pow=2),
            ],
        )

        beam = 1
        self.set_min_pool_size(1)
        self.set_min_z_to_research(ranks, [22 + 48 * budget, 14 + 30 * budget])
        self.set_max_z_to_research(ranks, [10000, self.deep_max_z if self.deep_search else 10000])
        self.set_temperature(0.24 + 0.16 * budget)
        self.set_num_samples(ranks, [20, 10])
        self.set_max_pool_size(ranks, [3 + int(2 * budget), 2 + int(2 * budget)])
        self.set_max_tohpe(ranks, [2, 1])
        self.set_try_only_tohpe(ranks, [1, 1])
        self.set_max_reduction(6)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(2)
        self.set_tohpe_num_best(ranks, [3, 2])
        self.set_gen_part(ranks, [0.08 + 0.10 * budget, 0.05 + 0.08 * budget])
        self.set_todd_width(ranks, [beam, 1])
        self.set_beamsearch_width(ranks, [beam, 1])

    def __call__(self, params: Iterable):
        ranks = self.run(params, self.seeds)
        return float(np.quantile(ranks, 0.65) + 0.01 * np.std(ranks))


def _fallback_pso(fun: Evaluator, evals: int, low: float, high: float) -> np.ndarray:
    n_params = len(fun.extract_active())
    best = np.zeros(n_params, dtype=float)
    best_cost = float(fun(best))
    for _ in range(max(0, evals - 1)):
        cand = np.random.uniform(low, high, n_params)
        cost = float(fun(cand))
        if cost < best_cost:
            best_cost = cost
            best = cand
    return best


def run_pso(fun: Evaluator, iters: int = 2, particles: int = 3) -> np.ndarray:
    n_params = len(fun.extract_active())
    low, high = -1.0, 1.0
    if n_params == 0:
        return np.asarray([], dtype=float)
    if ps is None:
        return _fallback_pso(fun, iters * particles, low, high)

    def objective(positions):
        return np.asarray([fun(pos) for pos in positions], dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.6, "c2": 0.5, "w": 0.75},
        bounds=(np.full(n_params, low), np.full(n_params, high)),
    )
    _, best_position = optimizer.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best_position, dtype=float)


def entrypoint(mat):
    fun = Evaluator(path_name="init", max_depth=235)
    xopt = run_pso(fun, iters=2, particles=3)
    fun(xopt)
    if fun.best_paths:
        fun.deep_search = True
        x_active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 25), xopt=xopt)
        if x_active is not None and len(x_active) > 0:
            fun(np.asarray(x_active, dtype=float))
    return fun.get_best()
