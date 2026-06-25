from typing import Iterable

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


def squash(x: float, scale: float = 3.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(float(x) / sharp))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


class Evaluator(BaseEvaluator):
    """PSO ab initio restarts with partial score schedules."""

    seeds = [314, 2718, 1618]

    def policy_mapping(self):
        ranks = [500, 430, 0]

        w_red = self.map_par(squash, 430)
        w_dim = self.map_par(squash, 430)
        w_bucket = self.map_par(squash, 430)
        w_vw = self.map_par(squash, 430)
        w_z = self.map_par(squash, 430)
        f_red = self.map_par(squash, 430)
        f_dim = self.map_par(squash, 430)
        f_bucket = self.map_par(squash, 430)
        f_vw = self.map_par(squash, 430)
        f_z = self.map_par(squash, 430)
        f_tohpe = self.map_par(squash, 430)
        budget = self.map_par(sigmoid, 430)

        pool_high = ExplorationScore(weights=[w_red, w_dim, w_bucket, w_vw, w_z], pow=1.0)
        pool_mid = ExplorationScore(
            weights=[0.85 * w_red, 0.9 * w_dim, 1.05 * w_bucket, 0.7 * w_vw, 0.7 * w_z],
            pow=1.0,
        )
        pool_low = ExplorationScore(
            weights=[0.7 * w_red, 0.8 * w_dim, 1.1 * w_bucket, 0.8 * w_vw, 0.9 * w_z],
            pow=2.0,
        )
        final_high = FinalizationScore(
            weights=[f_red, f_dim, f_bucket, f_vw, f_z, f_tohpe],
            centers=[1, 0, 0, 0, 0, 0],
            pow=2.0,
        )
        final_mid = FinalizationScore(
            weights=[0.9 * f_red, f_dim, 0.95 * f_bucket, 0.75 * f_vw, 0.8 * f_z, 0.8 * f_tohpe],
            centers=[1, 0, 0, 0, 0, 0],
            pow=2.0,
        )
        final_low = FinalizationScore(
            weights=[0.8 * f_red, 0.9 * f_dim, 0.9 * f_bucket, 0.85 * f_vw, f_z, f_tohpe],
            centers=[1, 0, 0, 0, 0, 0],
            pow=2.0,
        )

        beam = int(3 + 3 * budget)
        todd_width = int(2 + 4 * budget)
        final_pool_size = int(10 + 10 * budget)
        min_z = int(15 + 45 * budget)
        tohpe_pool = int(2 + budget)
        one_hot_hi = int(8 + 4 * (1.0 - budget))
        one_hot_lo = int(6 + 3 * (1.0 - budget))

        caps = [[one_hot_hi, 1, 8], [one_hot_lo, 1, 6], [6, 1, 5]]
        samplings = [
            SamplingBudget(one_hot=cap[0], sparse=cap[1], dense=cap[2], sparse_max_weight=2)
            for cap in caps
        ]
        temps = [0.9, 0.55, 0.35]
        max_z = [600_000, 500_000, 350_000]
        self.set_scores(
            ranks,
            [
                PolicyScores(exploration=pool_high, final=final_high),
                PolicyScores(exploration=pool_mid, final=final_mid),
                PolicyScores(exploration=pool_low, final=final_low),
            ],
        )
        self.set_action_selection(
            ranks,
            [ActionSelection(count=todd_width, mode="softmax", temperature=temp) for temp in temps],
        )
        self.set_action_pool(ActionPool(final_size=final_pool_size))
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(sampling=samplings[i], pool=SourcePool(keep=tohpe_pool, reserve=reserve), z_choices=z)
                for i, (reserve, z) in enumerate(zip([1, 0, 0], [4, 3, 2]))
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=samplings[i],
                    pool=SourcePool(keep=keep, reserve=0),
                    actions_per_bucket=3,
                    buckets=ZBucketSearch(min_buckets=min_z, max_buckets=max_z[i]),
                )
                for i, keep in enumerate([8, 10, 8])
            ],
        )
        self.set_widths(beam=beam, actions=todd_width)
        self.set_rollout_depth([430, 0], [0, 1])
        self.set_rollout_topk([430, 0], [0, 4])

    def __call__(self, params: Iterable):
        early = self.run(params, self.seeds[:1])
        if self.best_rank < 100000 and early[0] > self.best_rank + 20:
            return float(early[0] + 6.0)
        ranks = early + self.run(params, self.seeds[1:])
        return float(np.quantile(ranks, 0.60) + 0.015 * np.std(ranks))


def run_pso(fun: Evaluator, iters: int, particles: int):
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    def objective(positions):
        return np.asarray([fun(pos) for pos in positions], dtype=float)

    opt = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.7, "c2": 0.65, "w": 0.75},
        bounds=(np.full(n_params, -2.0), np.full(n_params, 2.0)),
    )
    _, best = opt.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=390)
    xopt = run_pso(fun, iters=16, particles=8)
    best_rank = fun.best_rank
    for gap, iters, particles in [(15, 12, 7), (30, 10, 6), (45, 8, 5)]:
        xopt = fun.set_up_new_init(0, rank_thr=int(best_rank + gap), xopt=xopt)
        if xopt is not None and len(xopt):
            xopt = run_pso(fun, iters=iters, particles=particles)
    return fun.get_best()
