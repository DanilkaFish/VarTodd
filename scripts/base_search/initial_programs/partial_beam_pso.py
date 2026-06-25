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

np.random.seed(53)


def squash(x: float, scale: float = 3.5, sharp: float = 1.35) -> float:
    return float(scale * np.tanh(float(x) / sharp))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


class Evaluator(BaseEvaluator):
    """Wide beam/TODD PSO with few score knobs and ab initio path building."""

    seeds = [911, 1729, 2718, 3141]

    def policy_mapping(self):
        ranks = [500, 450, 410, 0]

        w_red = self.map_par(squash, 0)
        w_bucket = self.map_par(squash, 450)
        w_vw = self.map_par(squash, 410)
        w_z = self.map_par(squash, 410)
        f_red = self.map_par(lambda x: squash(x, scale=4.2, sharp=1.2), 0)
        f_vw = self.map_par(squash, 410)
        f_z = self.map_par(squash, 410)
        f_tohpe = self.map_par(squash, 410)
        sample_bias = self.map_par(sigmoid, 450)
        z_bias = self.map_par(sigmoid, 410)

        pool_simple = ExplorationScore(weights=[w_red, 0.0, w_bucket, 0.0, 0.0], pow=1.0)
        pool_mixed = ExplorationScore(weights=[w_red, 0.0, w_bucket, w_vw, w_z], pow=1.0)
        final_simple = FinalizationScore(
            weights=[f_red, 0.0, 0.0, 0.0, f_z, 0.0],
            centers=[1, 0, 0, 0, 0, 0],
            pow=2.0,
        )
        final_mixed = FinalizationScore(
            weights=[f_red, 0.0, 0.0, f_vw, f_z, f_tohpe],
            centers=[1, 0, 0, 0, 0, 0],
            pow=2.0,
        )

        one_hot = 8 + int(10 * (1.0 - sample_bias))
        dense = 8 + int(16 * sample_bias)
        caps = [one_hot, 0, dense]
        deep_caps = [one_hot + 4, 0, dense + 6]
        min_z_mid = 20 + int(80 * z_bias)
        min_z_deep = 35 + int(160 * z_bias)

        caps_schedule = [caps, caps, deep_caps, deep_caps]
        samplings = [SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=2) for c in caps_schedule]
        action_counts = [4, 5, 7, 7]
        temps = [0.70, 0.50, 0.35, 0.25]
        min_z = [15, min_z_mid, min_z_deep, min_z_deep]
        max_z = [400_000, 700_000, 1_000_000, 900_000]
        self.set_scores(
            ranks,
            [
                PolicyScores(exploration=pool_simple, final=final_simple),
                PolicyScores(exploration=pool_mixed, final=final_simple),
                PolicyScores(exploration=pool_mixed, final=final_mixed),
                PolicyScores(exploration=pool_mixed, final=final_mixed),
            ],
        )
        self.set_action_selection(
            ranks,
            [ActionSelection(count=count, mode="softmax", temperature=temp) for count, temp in zip(action_counts, temps)],
        )
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in [12, 12, 10, 8]])
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(sampling=samplings[i], pool=SourcePool(keep=keep, reserve=0), z_choices=z)
                for i, (keep, z) in enumerate(zip([3, 3, 2, 1], [4, 3, 2, 2]))
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=samplings[i],
                    pool=SourcePool(keep=keep, reserve=0),
                    actions_per_bucket=3,
                    buckets=ZBucketSearch(min_buckets=min_z[i], max_buckets=max_z[i]),
                )
                for i, keep in enumerate([8, 10, 12, 12])
            ],
        )
        self.set_widths(beam=list(zip(ranks, action_counts)), actions=list(zip(ranks, action_counts)))
        self.set_rollout_depth([410, 0], [0, 1])
        self.set_rollout_topk([410, 0], [0, 6])
    def __call__(self, params: Iterable):
        first = self.run(params, self.seeds[:2])
        med = float(np.median(first))
        if self.best_rank < 100000 and med > self.best_rank + 16:
            return med + 6.0
        ranks = first + self.run(params, self.seeds[2:])
        return float(np.quantile(ranks, 0.50) + 0.02 * np.std(ranks))


def run_pso(fun: Evaluator, iters: int, particles: int, low: float = -1.7, high: float = 1.7):
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    def objective(positions):
        return np.asarray([fun(position) for position in positions], dtype=float)

    opt = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.55, "c2": 0.55, "w": 0.78},
        bounds=(np.full(n_params, low), np.full(n_params, high)),
    )
    _, best = opt.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=390)
    xopt = run_pso(fun, iters=18, particles=8)
    xopt = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 40), xopt=xopt)
    if xopt is not None and len(xopt):
        xopt = run_pso(fun, iters=10, particles=6, low=-0.9, high=0.9)
    xopt = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 16), xopt=xopt)
    if xopt is not None and len(xopt):
        run_pso(fun, iters=7, particles=5, low=-0.45, high=0.45)
    return fun.get_best()
