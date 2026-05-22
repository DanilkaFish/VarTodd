from typing import Iterable
import random

import numpy as np
import pyswarms as ps

from helper import BaseEvaluator, ExplorationScore, FinalizationScore

np.random.seed(42)
random.seed(40)


def squash(x: float, scale: float = 2.0) -> float:
    return float(scale * np.tanh(float(x)))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20, 20)))))


class Evaluator(BaseEvaluator):
    """Standard parameter optimization with identical exploration/final scores."""

    seeds = [random.randint(1, 10000) for _ in range(4)]
    phase = "scout"

    def _target_rank(self) -> int:
        return int(round(self.init_rank * 0.68))

    def _ranks(self):
        target = self._target_rank()
        return [
            int(round(target + 0.55 * (self.init_rank - target))),
            int(round(target + 0.25 * (self.init_rank - target))),
            0,
        ]

    def _margin(self, fraction: float) -> int:
        return max(12, int(round((self.init_rank - self._target_rank()) * fraction)))

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float, sparse_fraction: float = 0.15):
        one_hot = max(0, int(round(sample_count * one_hot_fraction)))
        sparse = max(0, int(round(sample_count * sparse_fraction)))
        dense = max(0, int(sample_count) - one_hot - sparse)
        return [one_hot, sparse, dense]

    def policy_mapping(self):
        ranks = self._ranks()
        score = [self.map_par(squash, 0) for _ in range(5)]
        budget = self.map_par(sigmoid, 0)
        sample_bias = self.map_par(sigmoid, 0)

        self.set_pool_scores(ExplorationScore(weights=score, pow=1))
        self.set_final_scores(
            FinalizationScore(weights=score + [0.0], centers=[0, 0, 0, 0, 0, 0], pow=1)
        )

        deep = self.phase == "deep"
        sample_count = 14 + int(42 * sample_bias)
        caps = [
            self._sample_caps(sample_count + 8, 0.38 + 0.20 * budget),
            self._sample_caps(sample_count, 0.50 + 0.20 * budget),
            self._sample_caps(max(8, sample_count - 6), 0.62 + 0.18 * budget),
        ]
        top_pool = [14, 12, 10]
        tohpe_pool = [5, 4, 3]
        todd_pool = [5 if deep else 0, 5 if deep else 0, 4 if deep else 0]
        self.set_min_pool_size(ranks, [2, 1, 1])
        self.set_tohpe_vector_samples(ranks, caps)
        self.set_todd_vector_samples(ranks, caps)
        self.set_sparse_max_weight(4 + int(6 * sample_bias))
        self.set_max_pool_size(ranks, top_pool)
        self.set_tohpe_pool_size(ranks, tohpe_pool)
        self.set_todd_pool_size(ranks, todd_pool)
        self.set_min_tohpe_actions(ranks, [1, 1, 1])
        self.set_min_todd_actions(ranks, [1 if deep else 0, 1 if deep else 0, 1 if deep else 0])
        self.set_tohpe_sample(ranks, [4, 3, 2])
        self.set_temperature(ranks, [0.20, 0.14, 0.08])
        self.set_min_z_to_research(ranks, [900 + 2500 * budget, 700 + 1800 * budget, 450 + 1200 * budget])
        self.set_max_z_to_research(ranks, [280_000, 500_000 if deep else 240_000, 1_200_000 if deep else 300_000])
        self.set_min_reduction(1)
        self.set_max_reduction(12)
        self.set_max_from_single_ns(2)
        self.set_beamsearch_width(1)
        self.set_todd_width(1)

    def __call__(self, params: Iterable):
        ranks = self.run(params, self.seeds)
        return float(np.quantile(ranks, 0.70) + 0.012 * np.std(ranks))


def run_pso(fun: Evaluator, iters: int, particles: int, low: float = -1.2, high: float = 1.2) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    def objective(positions):
        return np.asarray([fun(pos) for pos in positions], dtype=float)

    opt = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.65, "c2": 0.55, "w": 0.76},
        bounds=(np.full(n_params, low), np.full(n_params, high)),
    )
    _, best = opt.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=380)
    xopt = run_pso(fun, iters=14, particles=8)

    if fun.best_paths:
        fun.phase = "deep"
        active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + fun._margin(0.18)), xopt=xopt)
        if active is not None and len(active):
            run_pso(fun, iters=5, particles=4, low=-0.6, high=0.6)

    return fun.get_best()
