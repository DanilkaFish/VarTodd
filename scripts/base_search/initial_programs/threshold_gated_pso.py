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

np.random.seed(127)
random.seed(127)


def squash(x: float, scale: float = 3.5, sharp: float = 1.4) -> float:
    return float(scale * np.tanh(float(x) / sharp))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def softmin(xs: Iterable[float], beta: float = 5.0) -> float:
    values = np.asarray(list(xs), dtype=float)
    m = values.min()
    return float(m - (1.0 / beta) * np.log(np.exp(-beta * (values - m)).sum()))


class Evaluator(BaseEvaluator):
    """PSO seed that uses map_par thresholds to shrink policy search after restarts."""

    seeds = [random.randint(1, 10000) for _ in range(2)]

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float, sparse_fraction: float):
        total = max(1, int(round(sample_count)))
        one_hot = max(0, int(round(total * one_hot_fraction)))
        sparse = max(0, int(round(total * sparse_fraction)))
        dense = max(0, total - one_hot - sparse)
        return [one_hot, sparse, dense]

    def policy_mapping(self):
        ranks = [520, 465, 420, 390]

        early_z = self.map_par(sigmoid, 520)
        mid_z = self.map_par(sigmoid, 465)
        late_z = self.map_par(sigmoid, 390)
        sample_bias = self.map_par(sigmoid, 465)
        late_sample_bias = self.map_par(sigmoid, 420)
        hot_bias = self.map_par(sigmoid, 390)

        pool = [
            0.8 + self.map_par(squash, 0),
            self.map_par(squash, 520),
            0.5 + self.map_par(squash, 465),
            self.map_par(squash, 420),
            self.map_par(squash, 420),
        ]
        final = [
            1.2 + self.map_par(squash, 0),
            self.map_par(squash, 520),
            0.8 + self.map_par(squash, 465),
            self.map_par(squash, 420),
            self.map_par(squash, 390),
            0.35 * self.map_par(squash, 465),
        ]
        self.set_scores(
            PolicyScores(
                exploration=ExplorationScore(weights=pool, pow=1),
                final=FinalizationScore(weights=final, pow=1),
            )
        )

        early_caps = self._sample_caps(16 + int(40 * sample_bias), 0.18 + 0.34 * hot_bias, 0.05)
        mid_caps = self._sample_caps(12 + int(34 * sample_bias), 0.28 + 0.38 * hot_bias, 0.03)
        late_caps = self._sample_caps(10 + int(36 * late_sample_bias), 0.36 + 0.42 * hot_bias, 0.00)
        final_caps = self._sample_caps(8 + int(26 * late_sample_bias), 0.50 + 0.28 * hot_bias, 0.00)

        caps = [early_caps, mid_caps, late_caps, final_caps]
        samplings = [SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=2) for c in caps]
        min_z = [
            40 + int(260 * early_z),
            120 + int(900 * mid_z),
            220 + int(1800 * late_z),
            320 + int(3200 * late_z),
        ]
        max_z = [260_000, 520_000, 950_000, 1_500_000]
        temps = [0.20, 0.15, 0.10, 0.07]
        self.set_widths(beam=list(zip(ranks, [2, 1, 1, 1])), actions=list(zip(ranks, [1, 1, 1, 1])))
        self.set_action_selection(ranks, [ActionSelection(count=1, mode="softmax", temperature=t) for t in temps])
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in [12, 12, 10, 8]])
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(sampling=samplings[i], pool=SourcePool(keep=keep, reserve=1), z_choices=z)
                for i, (keep, z) in enumerate(zip([3, 3, 2, 2], [3, 3, 2, 2]))
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=samplings[i],
                    pool=SourcePool(keep=keep, reserve=1),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=min_z[i], max_buckets=max_z[i]),
                )
                for i, keep in enumerate([8, 10, 12, 12])
            ],
        )
    def __call__(self, params: Iterable):
        ranks = self.run(params, self.seeds)
        spread = float(np.std(ranks)) if len(ranks) > 1 else 0.0
        return softmin(ranks, beta=5.0) + 0.015 * spread


def run_pso(
    fun: Evaluator,
    iters: int,
    particles: int,
    low: float = -1.0,
    high: float = 1.0,
) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    def objective(positions):
        return np.asarray([fun(position) for position in positions], dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.45, "c2": 0.45, "w": 0.72},
        bounds=(np.full(n_params, low), np.full(n_params, high)),
    )
    _, best_position = optimizer.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best_position, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=390)
    xopt = run_pso(fun, iters=28, particles=7)

    if fun.best_paths:
        active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 35), xopt=xopt)
        if active is not None and len(active):
            run_pso(fun, iters=14, particles=5, low=-0.55, high=0.55)

    return fun.get_best()
