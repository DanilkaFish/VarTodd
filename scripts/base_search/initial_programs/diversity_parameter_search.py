from typing import Iterable
import random

import cma
import numpy as np
import pyswarms as ps

from helper import BaseEvaluator, ExplorationScore, FinalizationScore

np.random.seed(45)
random.seed(43)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20, 20)))))


def centered(x: float) -> float:
    return float(2.0 * sigmoid(x) - 1.0)


class Evaluator(BaseEvaluator):
    """Diversity-parameter search with sparse scores and many action candidates."""

    seeds = [random.randint(1, 10000) for _ in range(3)]
    phase = "diverse"

    def _target_rank(self) -> int:
        return int(round(self.init_rank * 0.68))

    def _ranks(self, split_bias: float):
        target = self._target_rank()
        gap = self.init_rank - target
        upper = int(round(target + (0.58 - 0.16 * split_bias) * gap))
        lower = int(round(target + (0.24 - 0.08 * split_bias) * gap))
        return [upper, lower, 0]

    def _margin(self, fraction: float) -> int:
        return max(12, int(round((self.init_rank - self._target_rank()) * fraction)))

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float, sparse_fraction: float):
        one_hot = max(0, int(round(sample_count * one_hot_fraction)))
        sparse = max(0, int(round(sample_count * sparse_fraction)))
        dense = max(0, int(sample_count) - one_hot - sparse)
        return [one_hot, sparse, dense]

    def policy_mapping(self):
        red_bias = self.map_par(centered, 0)
        z_ham_bias = self.map_par(centered, 0)
        bucket_bias = self.map_par(centered, 0)
        tohpe_dim_bias = self.map_par(centered, 0)
        sample_bias = self.map_par(sigmoid, 0)
        pool_bias = self.map_par(sigmoid, 0)
        min_z_bias = self.map_par(sigmoid, 0)
        gen_bias = self.map_par(sigmoid, 0)
        tohpe_bias = self.map_par(sigmoid, 0)
        split_bias = self.map_par(sigmoid, 0)
        ranks = self._ranks(split_bias)

        pool = [0.55 + 0.45 * red_bias, 0.0, 0.75 + 0.75 * bucket_bias, z_ham_bias, 0.0]
        final = [0.60 + 0.40 * red_bias, 0.0, 0.85 + 0.55 * bucket_bias, z_ham_bias, 0.0, tohpe_dim_bias]
        self.set_pool_scores(ExplorationScore(weights=pool, pow=1))
        self.set_final_scores(
            FinalizationScore(weights=final, pow=2)
        )

        deep = self.phase == "deep"
        sample_counts = [20 + int(36 * sample_bias), 16 + int(24 * sample_bias), 10 + int(16 * sample_bias)]
        one_hot_fractions = [0.28 + 0.34 * gen_bias, 0.42 + 0.30 * gen_bias, 0.58 + 0.22 * gen_bias]
        sparse_fraction = 0.18 + 0.18 * sample_bias
        sample_caps = [
            self._sample_caps(sample_counts[0], one_hot_fractions[0], sparse_fraction),
            self._sample_caps(sample_counts[1], one_hot_fractions[1], sparse_fraction),
            self._sample_caps(sample_counts[2], one_hot_fractions[2], sparse_fraction * 0.7),
        ]
        top_pool = [14 + int(18 * pool_bias), 11 + int(14 * pool_bias), 8 + int(8 * pool_bias)]
        tohpe_pool = [5 + int(5 * tohpe_bias), 4 + int(4 * tohpe_bias), 3 + int(3 * tohpe_bias)]
        todd_pool = [max(1, top_pool[i] - tohpe_pool[i]) for i in range(3)]
        self.set_min_pool_size(ranks, [3 + int(6 * pool_bias), 2 + int(4 * pool_bias), 1 + int(2 * pool_bias)])
        self.set_tohpe_vector_samples(ranks, sample_caps)
        self.set_todd_vector_samples(ranks, sample_caps)
        self.set_sparse_max_weight(5 + int(9 * sample_bias))
        self.set_max_pool_size(ranks, top_pool)
        self.set_tohpe_pool_size(ranks, tohpe_pool)
        self.set_todd_pool_size(ranks, todd_pool)
        self.set_min_tohpe_actions(ranks, [2, 1, 1])
        self.set_min_todd_actions(ranks, [1, 1, 1])
        self.set_tohpe_sample(ranks, [5 + int(6 * tohpe_bias), 4 + int(5 * tohpe_bias), 3 + int(3 * tohpe_bias)])
        self.set_temperature(ranks, [0.22 + 0.10 * gen_bias, 0.14 + 0.07 * gen_bias, 0.08 + 0.05 * gen_bias])
        self.set_bucket_temperature(0.08 + 0.22 * gen_bias)
        self.set_bucket_random_fraction(0.01 + 0.04 * gen_bias)
        self.set_max_per_signature(2)
        self.set_min_z_to_research(ranks, [800 + 3600 * min_z_bias, 650 + 2500 * min_z_bias, 450 + 1600 * min_z_bias])
        self.set_max_z_to_research(ranks, [240_000, 620_000, 1_200_000 if deep else 340_000])
        self.set_min_reduction(1)
        self.set_max_reduction(12)
        self.set_max_from_single_ns(2)
        self.set_beamsearch_width(1)
        self.set_todd_width(1)

    def __call__(self, params: Iterable):
        ranks = self.run(params, self.seeds)
        return float(np.quantile(ranks, 0.68) + 0.014 * np.std(ranks))


def run_pso(fun: Evaluator, iters: int, particles: int) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    def objective(positions):
        return np.asarray([fun(pos) for pos in positions], dtype=float)

    opt = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.75, "c2": 0.55, "w": 0.80},
        bounds=(np.full(n_params, -1.5), np.full(n_params, 1.5)),
    )
    _, best = opt.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best, dtype=float)


def run_cma(fun: Evaluator, x0: np.ndarray, maxfevals: int, sigma: float) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float)
    if x0.size == 0:
        return x0

    def objective(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(fun(x))
        return np.asarray([fun(row) for row in x], dtype=float)

    xopt, _ = cma.fmin2(
        objective,
        x0,
        sigma,
        options={
            "maxfevals": maxfevals,
            "popsize": 4,
            "bounds": [-1.8, 1.8],
            "verbose": -9,
            "seed": 45,
        },
        restarts=0,
        bipop=False,
    )
    return np.asarray(xopt, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=400)
    xopt = run_pso(fun, iters=10, particles=5)
    xopt = run_cma(fun, xopt, maxfevals=16, sigma=0.25)

    if fun.best_paths:
        fun.phase = "deep"
        active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + fun._margin(0.18)), xopt=xopt)
        if active is not None and len(active):
            run_cma(fun, np.asarray(active, dtype=float), maxfevals=14, sigma=0.15)

    return fun.get_best()
