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

np.random.seed(44)
random.seed(42)


def squash(x: float) -> float:
    return float(np.tanh(float(x)))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20, 20)))))


class Evaluator(BaseEvaluator):
    """Many-restart program: cheap diverse restarts across thresholds and score bias."""

    seeds = [random.randint(1, 10000) for _ in range(2)]
    phase = "restart"

    def _target_rank(self) -> int:
        return int(round(self.init_rank * 0.68))

    def _margin(self, fraction: float) -> int:
        return max(12, int(round((self.init_rank - self._target_rank()) * fraction)))

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float, sparse_fraction: float = 0.08):
        one_hot = max(0, int(round(sample_count * one_hot_fraction)))
        sparse = max(0, int(round(sample_count * sparse_fraction)))
        dense = max(0, int(sample_count) - one_hot - sparse)
        return [one_hot, sparse, dense]

    def policy_mapping(self):
        red = self.map_par(squash, 0)
        bucket = self.map_par(squash, 0)
        diversity = self.map_par(sigmoid, 0)
        z_budget = self.map_par(sigmoid, 0)

        pool = [0.55 + red, -0.30, 0.80 + bucket, 0.10, 0.0]
        final = [0.55 + red, -0.20, 0.90 + bucket, 0.05, 0.0, 0.0]
        self.set_scores(
            PolicyScores(
                exploration=ExplorationScore(weights=pool, pow=1),
                final=FinalizationScore(weights=final, centers=[0, 0, 0, 0, 0, 0], pow=2),
            )
        )

        deep = self.phase == "deep"
        final_pool_size = 8 + int(10 * diversity)
        tohpe_pool = 4
        sample_caps = self._sample_caps(10 + int(10 * diversity), 0.42 + 0.35 * diversity)
        sampling = SamplingBudget(one_hot=sample_caps[0], sparse=sample_caps[1], dense=sample_caps[2], sparse_max_weight=4)
        self.set_action_selection(ActionSelection(count=1, mode="softmax", temperature=0.16 + 0.10 * diversity))
        self.set_action_pool(ActionPool(final_size=final_pool_size))
        self.set_tohpe_search(TohpeSearch(sampling=sampling, pool=SourcePool(keep=tohpe_pool, reserve=1), z_choices=5))
        self.set_todd_search(
            ToddSearch(
                sampling=sampling,
                pool=SourcePool(keep=max(1, final_pool_size - tohpe_pool) if deep else 0, reserve=1 if deep else 0),
                actions_per_bucket=2,
                buckets=ZBucketSearch(min_buckets=500 + 2200 * z_budget, max_buckets=900_000 if deep else 180_000),
            )
        )
        self.set_widths(actions=1, beam=1)

    def __call__(self, params: Iterable):
        ranks = self.run(params, self.seeds)
        return float(min(ranks) + 0.02 * np.std(ranks))


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=380)
    xopt = run_pso(fun, iters=14, particles=7)

    if fun.best_paths:
        for margin in (fun._margin(0.46), fun._margin(0.26), fun._margin(0.14)):
            active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + margin), xopt=xopt)
            if active is None or not len(active):
                continue
            xopt = run_pso(fun, iters=8, particles=6)

        fun.phase = "deep"
        active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + fun._margin(0.10)), xopt=xopt)
        if active is not None and len(active):
            fun(np.asarray(active, dtype=float))

    return fun.get_best()


def run_pso(fun: Evaluator, iters: int, particles: int) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    def objective(positions):
        return np.asarray([fun(pos) for pos in positions], dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.72, "c2": 0.55, "w": 0.78},
        bounds=(np.full(n_params, -1.3), np.full(n_params, 1.3)),
    )
    _, best_position = optimizer.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best_position, dtype=float)
