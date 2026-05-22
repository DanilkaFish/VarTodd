from typing import Iterable
import random

import cma
import numpy as np

from helper import BaseEvaluator, ExplorationScore, FinalizationScore

np.random.seed(43)
random.seed(41)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20, 20)))))


class Evaluator(BaseEvaluator):
    """Beam-search dedicated program: spend compute on width before deep TODD."""

    seeds = [random.randint(1, 10000) for _ in range(3)]
    phase = "beam"

    def _target_rank(self) -> int:
        return int(round(self.init_rank * 0.68))

    def _ranks(self):
        target = self._target_rank()
        gap = self.init_rank - target
        return [int(round(target + 0.50 * gap)), int(round(target + 0.18 * gap)), 0]

    def _margin(self, fraction: float) -> int:
        return max(12, int(round((self.init_rank - self._target_rank()) * fraction)))

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float, sparse_fraction: float = 0.08):
        one_hot = max(0, int(round(sample_count * one_hot_fraction)))
        sparse = max(0, int(round(sample_count * sparse_fraction)))
        dense = max(0, int(sample_count) - one_hot - sparse)
        return [one_hot, sparse, dense]

    def policy_mapping(self):
        ranks = self._ranks()
        bucket_bias = self.map_par(sigmoid, 0)
        z_budget = self.map_par(sigmoid, 0)
        width_bias = self.map_par(sigmoid, 0)

        pool = [0.35, -0.25, 1.0 + 0.8 * bucket_bias, 1 - 2*self.map_par(sigmoid, 0),  1 - 2*self.map_par(sigmoid, 0)]
        final = [-0.45, -0.15, 1.1 + 0.6 * bucket_bias, 1 - 2*self.map_par(sigmoid, 0), 1 - 2*self.map_par(sigmoid, 0), 0.0]
        self.set_pool_scores(ExplorationScore(weights=pool, pow=1))
        self.set_final_scores(
            FinalizationScore(weights=final, centers=[0, 0, 0, 0, 0, 0], pow=2)
        )

        deep = self.phase == "deep"
        beam = 3 + int(width_bias > 0.55)
        self.set_beamsearch_width(ranks, [beam, 3, 2])
        self.set_todd_width(ranks, [1, 1, 1])
        self.set_min_pool_size(ranks, [2, 1, 1])
        sample_caps = [
            self._sample_caps(10, 0.34 + 0.22 * bucket_bias),
            self._sample_caps(8, 0.52 + 0.22 * bucket_bias),
            self._sample_caps(8, 0.64 + 0.16 * bucket_bias),
        ]
        self.set_tohpe_vector_samples(ranks, sample_caps)
        self.set_todd_vector_samples(ranks, sample_caps)
        self.set_sparse_max_weight(4)
        self.set_max_pool_size(ranks, [10, 8, 6])
        self.set_tohpe_pool_size(ranks, [5, 4, 3])
        self.set_todd_pool_size(ranks, [3 if deep else 0, 3 if deep else 0, 2 if deep else 0])
        self.set_min_tohpe_actions(ranks, [2, 1, 1])
        self.set_min_todd_actions(ranks, [1 if deep else 0, 1 if deep else 0, 1 if deep else 0])
        self.set_tohpe_sample(ranks, [5, 4, 3])
        self.set_temperature(ranks, [0.22, 0.14, 0.08])
        self.set_min_z_to_research(ranks, [900 + 2600 * z_budget, 650 + 1800 * z_budget, 450 + 1200 * z_budget])
        self.set_max_z_to_research(ranks, [240_000, 440_000 if deep else 240_000, 1_000_000 if deep else 300_000])
        self.set_min_reduction(1)
        self.set_max_reduction(10)
        self.set_max_from_single_ns(2)

    def __call__(self, params: Iterable):
        ranks = self.run(params, self.seeds)
        return float(np.quantile(ranks, 0.72) + 0.010 * np.std(ranks))


def run_cma(fun: Evaluator, x0: np.ndarray | None = None, maxfevals: int = 30, sigma: float = 0.35) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)
    if x0 is None:
        x0 = np.zeros(n_params, dtype=float)

    def objective(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(fun(x))
        return np.asarray([fun(row) for row in x], dtype=float)

    xopt, _ = cma.fmin2(
        objective,
        np.asarray(x0, dtype=float),
        sigma,
        options={
            "maxfevals": maxfevals,
            "popsize": 4,
            "bounds": [-1.5, 1.5],
            "verbose": -9,
            "seed": 43,
        },
        restarts=0,
        bipop=False,
    )
    return np.asarray(xopt, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=390)
    xopt = run_cma(fun, maxfevals=88, sigma=0.40)

    if fun.best_paths:
        fun.phase = "deep"
        active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + fun._margin(0.22)), xopt=xopt)
        if active is not None and len(active):
            run_cma(fun, np.asarray(active, dtype=float), maxfevals=20, sigma=0.18)

    return fun.get_best()
