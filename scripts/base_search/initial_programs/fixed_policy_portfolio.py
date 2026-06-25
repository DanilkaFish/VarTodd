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

np.random.seed(48)
random.seed(46)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20, 20)))))


def unit(x: float) -> float:
    return float(np.clip(x, 0.0, 0.999))


class Evaluator(BaseEvaluator):
    """Fixed policy families with PSO-optimized portfolio coordinates."""

    seeds = [random.randint(1, 10000) for _ in range(3)]
    phase = "portfolio"

    def _target_rank(self) -> int:
        return int(round(self.init_rank * 0.68))

    def _margin(self, fraction: float) -> int:
        return max(12, int(round((self.init_rank - self._target_rank()) * fraction)))

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float, sparse_fraction: float = 0.12):
        one_hot = max(0, int(round(sample_count * one_hot_fraction)))
        sparse = max(0, int(round(sample_count * sparse_fraction)))
        dense = max(0, int(sample_count) - one_hot - sparse)
        return [one_hot, sparse, dense]

    def policy_mapping(self):
        policy_id = self.map_par(unit, 0)
        diversity = self.map_par(unit, 0)
        z_budget = self.map_par(unit, 0)
        sample_bias = self.map_par(unit, 0)

        policies = [
            ([1.10, -0.25, 0.30, -0.10, 0.00], [1.05, -0.20, 0.35, -0.10, 0.00, 0.0]),
            ([0.30, -0.10, 1.35, 0.00, 0.10], [0.35, -0.10, 1.25, 0.00, 0.10, 0.0]),
            ([0.50, 0.45, 0.25, -0.30, 0.00], [0.50, 0.35, 0.25, -0.25, 0.00, 0.0]),
            ([0.25, -0.20, 0.65, 0.45, 0.25], [0.30, -0.10, 0.70, 0.35, 0.20, 0.0]),
        ]
        idx = min(len(policies) - 1, int(policy_id * len(policies)))
        pool, final = policies[idx]
        self.set_scores(
            PolicyScores(
                exploration=ExplorationScore(weights=pool, pow=1),
                final=FinalizationScore(weights=final, centers=[0, 0, 0, 0, 0, 0], pow=2),
            )
        )

        deep = self.phase == "deep"
        min_pool = 2 + int(5 * diversity)
        sample_count = 10 + int(24 * sample_bias)
        final_pool_size = 10 + int(16 * diversity)
        tohpe_pool = 5 + int(5 * diversity)
        todd_pool = max(1, final_pool_size - tohpe_pool)
        sample_caps = self._sample_caps(sample_count, 0.40 + 0.45 * diversity)
        sampling = SamplingBudget(
            one_hot=sample_caps[0],
            sparse=sample_caps[1],
            dense=sample_caps[2],
            sparse_max_weight=4 + int(5 * sample_bias),
        )
        self.set_action_selection(ActionSelection(count=1, mode="softmax", temperature=0.16))
        self.set_action_pool(ActionPool(final_size=final_pool_size))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=sampling,
                pool=SourcePool(keep=tohpe_pool, reserve=min(2, tohpe_pool)),
                z_choices=5 + int(8 * diversity),
            )
        )
        self.set_todd_search(
            ToddSearch(
                sampling=sampling,
                pool=SourcePool(keep=todd_pool, reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(min_buckets=700 + 3600 * z_budget, max_buckets=1_100_000 if deep else 240_000),
            )
        )
        self.set_widths(actions=1, beam=1)

    def __call__(self, params: Iterable):
        ranks = self.run(params, self.seeds)
        return float(np.quantile(ranks, 0.70) + 0.010 * np.std(ranks))


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=390)
    xopt = run_pso(fun, iters=8, particles=6)

    if fun.best_paths:
        fun.phase = "deep"
        for margin in (fun._margin(0.26), fun._margin(0.14)):
            active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + margin), xopt=xopt)
            if active is not None and len(active):
                xopt = run_pso(fun, iters=4, particles=3)

    return fun.get_best()


def run_pso(fun: Evaluator, iters: int, particles: int) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    def objective(positions):
        clipped = np.clip(np.asarray(positions, dtype=float), 0.0, 0.999)
        return np.asarray([fun(pos) for pos in clipped], dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.70, "c2": 0.55, "w": 0.76},
        bounds=(np.zeros(n_params), np.full(n_params, 0.999)),
    )
    _, best_position = optimizer.optimize(objective, iters=iters, verbose=False)
    return np.asarray(np.clip(best_position, 0.0, 0.999), dtype=float)
