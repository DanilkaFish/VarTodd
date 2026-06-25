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

        self.set_scores(
            PolicyScores(
                exploration=ExplorationScore(weights=score, pow=1),
                final=FinalizationScore(weights=score + [0.0], centers=[0, 0, 0, 0, 0, 0], pow=1),
            )
        )

        deep = self.phase == "deep"
        sample_count = 14 + int(42 * sample_bias)
        caps = [
            self._sample_caps(sample_count + 8, 0.38 + 0.20 * budget),
            self._sample_caps(sample_count, 0.50 + 0.20 * budget),
            self._sample_caps(max(8, sample_count - 6), 0.62 + 0.18 * budget),
        ]
        final_pool_sizes = [14, 12, 10]
        tohpe_pool = [5, 4, 3]
        todd_pool = [5 if deep else 0, 5 if deep else 0, 4 if deep else 0]
        sparse_weight = 4 + int(6 * sample_bias)
        samplings = [SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=sparse_weight) for c in caps]
        temps = [0.20, 0.14, 0.08]
        min_z = [900 + 2500 * budget, 700 + 1800 * budget, 450 + 1200 * budget]
        max_z = [280_000, 500_000 if deep else 240_000, 1_200_000 if deep else 300_000]
        self.set_action_selection(ranks, [ActionSelection(count=1, mode="softmax", temperature=t) for t in temps])
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in final_pool_sizes])
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(sampling=samplings[i], pool=SourcePool(keep=tohpe_pool[i], reserve=1), z_choices=z)
                for i, z in enumerate([4, 3, 2])
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=samplings[i],
                    pool=SourcePool(keep=todd_pool[i], reserve=1 if deep else 0),
                    actions_per_bucket=2,
                    buckets=ZBucketSearch(min_buckets=min_z[i], max_buckets=max_z[i]),
                )
                for i in range(3)
            ],
        )
        self.set_widths(actions=1, beam=1)

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
