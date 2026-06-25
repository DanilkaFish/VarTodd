from typing import Iterable
import random

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
        self.set_scores(
            PolicyScores(
                exploration=ExplorationScore(weights=pool, pow=1),
                final=FinalizationScore(weights=final, centers=[0.2, 0, 0, 0, 0, 0], pow=2),
            )
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
        final_pool_sizes = [12 + int(14 * scout_width), 10 + int(8 * scout_width), 8]
        tohpe_pool = [5 + int(4 * scout_width), 4, 3]
        todd_pool = [
            max(1, final_pool_sizes[0] - tohpe_pool[0]) if deep else 0,
            max(1, final_pool_sizes[1] - tohpe_pool[1]) if (deep or mid) else 0,
            max(1, final_pool_sizes[2] - tohpe_pool[2]) if deep else 0,
        ]
        sparse_weight = 4 + int(5 * sample_bias)
        samplings = [SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=sparse_weight) for c in sample_caps]
        temps = [0.22, 0.14, 0.08]
        min_z = [850 + 3400 * refine_z, 650 + 2400 * refine_z, 450 + 1600 * refine_z]
        max_z = [240_000 if not mid else 460_000, 600_000 if (mid or deep) else 300_000, 1_250_000 if deep else 340_000]
        self.set_widths(beam=list(zip(ranks, [beam, 1, 1])), actions=1)
        self.set_action_selection(ranks, [ActionSelection(count=1, mode="softmax", temperature=t) for t in temps])
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in final_pool_sizes])
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(sampling=samplings[i], pool=SourcePool(keep=tohpe_pool[i], reserve=reserve), z_choices=z)
                for i, (reserve, z) in enumerate(zip([2, 1, 1], [5 + int(6 * scout_width), 4, 3]))
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=samplings[i],
                    pool=SourcePool(keep=todd_pool[i], reserve=reserve),
                    actions_per_bucket=2,
                    buckets=ZBucketSearch(min_buckets=min_z[i], max_buckets=max_z[i]),
                )
                for i, reserve in enumerate([1 if deep else 0, 1 if (deep or mid) else 0, 1 if deep else 0])
            ],
        )
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
