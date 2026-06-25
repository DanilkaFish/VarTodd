from typing import Iterable
import random

import cma
import numpy as np
import pyswarms as ps

from helper import (
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    ExplorationScore,
    FinalizationScore,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
)

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
        self.set_scores(
            PolicyScores(
                exploration=ExplorationScore(weights=pool, pow=1),
                final=FinalizationScore(weights=final, pow=2),
            )
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
        final_pool_sizes = [14 + int(18 * pool_bias), 11 + int(14 * pool_bias), 8 + int(8 * pool_bias)]
        tohpe_pool = [5 + int(5 * tohpe_bias), 4 + int(4 * tohpe_bias), 3 + int(3 * tohpe_bias)]
        todd_pool = [max(1, final_pool_sizes[i] - tohpe_pool[i]) for i in range(3)]
        sparse_weight = 5 + int(9 * sample_bias)
        samplings = [SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=sparse_weight) for c in sample_caps]
        temps = [0.22 + 0.10 * gen_bias, 0.14 + 0.07 * gen_bias, 0.08 + 0.05 * gen_bias]
        min_z = [800 + 3600 * min_z_bias, 650 + 2500 * min_z_bias, 450 + 1600 * min_z_bias]
        max_z = [240_000, 620_000, 1_200_000 if deep else 340_000]
        bucket_temp = 0.08 + 0.22 * gen_bias
        bucket_rand = 0.01 + 0.04 * gen_bias
        self.set_action_selection(ranks, [ActionSelection(count=1, mode="softmax", temperature=t) for t in temps])
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in final_pool_sizes])
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(sampling=samplings[i], pool=SourcePool(keep=tohpe_pool[i], reserve=reserve), z_choices=z)
                for i, (reserve, z) in enumerate(zip([2, 1, 1], [5 + int(6 * tohpe_bias), 4 + int(5 * tohpe_bias), 3 + int(3 * tohpe_bias)]))
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=samplings[i],
                    pool=SourcePool(keep=todd_pool[i], reserve=1),
                    actions_per_bucket=2,
                    buckets=ZBucketSearch(
                        min_buckets=min_z[i],
                        max_buckets=max_z[i],
                        temperature=bucket_temp,
                        random_fraction=bucket_rand,
                    ),
                )
                for i in range(3)
            ],
        )
        self.set_widths(actions=1, beam=1)

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
