from typing import Iterable
import random

import numpy as np
import pyswarms as ps

from helper import BaseEvaluator, ExplorationScore, FinalizationScore

np.random.seed(46)
random.seed(44)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20, 20)))))


class Evaluator(BaseEvaluator):
    """Three-stage scout/mid/refine budget schedule with explicit phase changes."""

    seeds = [random.randint(1, 10000) for _ in range(3)]
    phase = "scout"

    def _target_rank(self) -> int:
        return int(round(self.init_rank * 0.68))

    def _ranks(self, split_bias: float):
        target = self._target_rank()
        gap = self.init_rank - target
        upper = int(round(target + (0.62 - 0.12 * split_bias) * gap))
        lower = int(round(target + (0.28 - 0.08 * split_bias) * gap))
        return [upper, lower, 0]

    def _margin(self, fraction: float) -> int:
        return max(12, int(round((self.init_rank - self._target_rank()) * fraction)))

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float, sparse_fraction: float = 0.12):
        one_hot = max(0, int(round(sample_count * one_hot_fraction)))
        sparse = max(0, int(round(sample_count * sparse_fraction)))
        dense = max(0, int(sample_count) - one_hot - sparse)
        return [one_hot, sparse, dense]

    def policy_mapping(self):
        red = self.map_par(sigmoid, 0)
        bucket = self.map_par(sigmoid, 0)
        scout_width = self.map_par(sigmoid, 0)
        refine_z = self.map_par(sigmoid, 0)
        gen = self.map_par(sigmoid, 0)
        sample_bias = self.map_par(sigmoid, 0)
        split_bias = self.map_par(sigmoid, 0)
        ranks = self._ranks(split_bias)

        pool = [0.25 + red, -0.15, 0.30 + 1.2 * bucket, 0.15, 0.0]
        final = [0.25 + red, -0.10, 0.45 + 1.0 * bucket, 0.10, 0.0, 0.0]
        self.set_pool_scores(ExplorationScore(weights=pool, pow=1))
        self.set_final_scores(
            FinalizationScore(weights=final, centers=[0.2, 0, 0, 0, 0, 0], pow=2)
        )

        deep = self.phase == "refine"
        mid = self.phase == "mid"
        beam = 1 + int(scout_width > 0.65)
        sample_counts = [18 + int(28 * sample_bias), 14 + int(20 * sample_bias), 10 + int(14 * sample_bias)]
        one_hot_fractions = [0.32 + 0.42 * gen, 0.46 + 0.32 * gen, 0.62 + 0.22 * gen]
        sample_caps = [
            self._sample_caps(sample_counts[0], one_hot_fractions[0]),
            self._sample_caps(sample_counts[1], one_hot_fractions[1]),
            self._sample_caps(sample_counts[2], one_hot_fractions[2]),
        ]
        top_pool = [12 + int(14 * scout_width), 10 + int(8 * scout_width), 8]
        tohpe_pool = [5 + int(4 * scout_width), 4, 3]
        todd_pool = [
            max(1, top_pool[0] - tohpe_pool[0]) if deep else 0,
            max(1, top_pool[1] - tohpe_pool[1]) if (deep or mid) else 0,
            max(1, top_pool[2] - tohpe_pool[2]) if deep else 0,
        ]
        self.set_beamsearch_width(ranks, [beam, 1, 1])
        self.set_todd_width(1)
        self.set_min_pool_size(ranks, [2, 1, 1])
        self.set_tohpe_vector_samples(ranks, sample_caps)
        self.set_todd_vector_samples(ranks, sample_caps)
        self.set_sparse_max_weight(4 + int(5 * sample_bias))
        self.set_max_pool_size(ranks, top_pool)
        self.set_tohpe_pool_size(ranks, tohpe_pool)
        self.set_todd_pool_size(ranks, todd_pool)
        self.set_min_tohpe_actions(ranks, [2, 1, 1])
        self.set_min_todd_actions(ranks, [1 if deep else 0, 1 if (deep or mid) else 0, 1 if deep else 0])
        self.set_tohpe_sample(ranks, [5 + int(6 * scout_width), 4, 3])
        self.set_temperature(ranks, [0.22, 0.14, 0.08])
        self.set_min_z_to_research(ranks, [850 + 3400 * refine_z, 650 + 2400 * refine_z, 450 + 1600 * refine_z])
        self.set_max_z_to_research(
            ranks,
            [240_000 if not mid else 460_000, 600_000 if (mid or deep) else 300_000, 1_250_000 if deep else 340_000],
        )
        self.set_min_reduction(1)
        self.set_max_reduction(12)
        self.set_max_from_single_ns(2)

    def __call__(self, params: Iterable):
        ranks = self.run(params, self.seeds)
        return float(np.quantile(ranks, 0.70) + 0.012 * np.std(ranks))


def run_pso(fun: Evaluator, iters: int, particles: int, radius: float = 1.3) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    def objective(positions):
        return np.asarray([fun(pos) for pos in positions], dtype=float)

    opt = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.70, "c2": 0.55, "w": 0.78},
        bounds=(np.full(n_params, -radius), np.full(n_params, radius)),
    )
    _, best = opt.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=400)
    xopt = run_pso(fun, iters=16, particles=8)

    if fun.best_paths:
        fun.phase = "mid"
        active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + fun._margin(0.40)), xopt=xopt)
        if active is not None and len(active):
            xopt = run_pso(fun, iters=8, particles=4, radius=0.7)

    if fun.best_paths:
        fun.phase = "refine"
        active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + fun._margin(0.16)), xopt=xopt)
        if active is not None and len(active):
            fun(np.asarray(active, dtype=float))

    return fun.get_best()
