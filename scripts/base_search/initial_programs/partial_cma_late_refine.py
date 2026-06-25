from typing import Iterable

import cma
import numpy as np

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

np.random.seed(52)


def squash(x: float, scale: float = 3.8, sharp: float = 1.25) -> float:
    return float(scale * np.tanh(float(x) / sharp))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


class Evaluator(BaseEvaluator):
    """CMA frontier builder using the old successful partial-score shape."""

    seeds = [42, 1337, 9001, 7, 123, 456]

    def policy_mapping(self):
        ranks = [470, 420, 395, 0]

        w_red = self.map_par(squash, 0)
        w_bucket = self.map_par(squash, 0)
        f_red = self.map_par(lambda x: squash(x, scale=4.8, sharp=1.15), 0)
        f_bucket = self.map_par(squash, 420)
        f_vw = self.map_par(squash, 420)
        f_z = self.map_par(squash, 395)
        f_tohpe = self.map_par(squash, 395)
        sample_bias = self.map_par(sigmoid, 395)
        width_bias = self.map_par(sigmoid, 420)

        pool_early = ExplorationScore(weights=[w_red, 0.0, w_bucket, 0.0, 0.0], pow=1.0)
        pool_late = ExplorationScore(weights=[w_red, 0.0, w_bucket, f_vw, f_z], pow=1.0)
        final_early = FinalizationScore(
            weights=[f_red, 0.0, 0.0, 0.0, f_z, 0.0],
            centers=[1, 0, 0, 0, 0, 0],
            pow=2.0,
        )
        final_mid = FinalizationScore(
            weights=[f_red, 0.0, f_bucket, f_vw, f_z, 0.5 * f_tohpe],
            centers=[1, 0, 0, 0, 0, 0],
            pow=2.0,
        )
        final_late = FinalizationScore(
            weights=[f_red, 0.0, f_bucket, f_vw, f_z, f_tohpe],
            centers=[1, 0, 0, 0, 0, 0],
            pow=2.0,
        )

        one_hot = 8 + int(10 * (1.0 - sample_bias))
        dense_mid = 8 + int(12 * sample_bias)
        dense_late = 12 + int(16 * sample_bias)
        beam = 4 + int(4 * width_bias)
        todd_width = 3 + int(4 * width_bias)

        caps_schedule = [[one_hot, 0, dense_mid], [one_hot, 0, dense_mid], [one_hot, 0, dense_late], [10, 0, dense_late]]
        samplings = [SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=2) for c in caps_schedule]
        action_counts = [todd_width, todd_width, todd_width + 2, todd_width + 2]
        beam_counts = [beam, beam, beam + 2, beam + 2]
        temps = [0.65, 0.45, 0.30, 0.22]
        min_z = [15, 20, 35, 45]
        max_z = [450_000, 700_000, 1_100_000, 1_300_000]
        self.set_scores(
            ranks,
            [
                PolicyScores(exploration=pool_early, final=final_early),
                PolicyScores(exploration=pool_late, final=final_mid),
                PolicyScores(exploration=pool_late, final=final_late),
                PolicyScores(exploration=pool_late, final=final_late),
            ],
        )
        self.set_action_selection(
            ranks,
            [ActionSelection(count=count, mode="softmax", temperature=temp) for count, temp in zip(action_counts, temps)],
        )
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in [8, 10, 12, 10]])
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(sampling=samplings[i], pool=SourcePool(keep=keep, reserve=0), z_choices=z)
                for i, (keep, z) in enumerate(zip([2, 2, 1, 1], [3, 3, 2, 2]))
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
        self.set_widths(beam=list(zip(ranks, beam_counts)), actions=list(zip(ranks, action_counts)))
        self.set_rollout_depth([395, 0], [0, 1])
        self.set_rollout_topk([395, 0], [0, 6])
    def __call__(self, params: Iterable):
        early = self.run(params, self.seeds[:3])
        med = float(np.median(early))
        if self.best_rank < 100000 and med > self.best_rank + 10:
            return med + 5.0
        ranks = early + self.run(params, self.seeds[3:])
        return float(np.quantile(ranks, 0.45) + 0.025 * np.std(ranks))


def run_cma(fun: Evaluator, x0=None, maxfevals: int = 48, sigma: float = 0.28, popsize: int = 9, seed: int = 52):
    low, high = -2.0, 2.0
    x0 = fun.extract_active() if x0 is None else np.asarray(x0, dtype=float)
    if x0.size == 0:
        return x0
    x0 = np.clip(x0, low + 1e-9, high - 1e-9)
    xopt, _ = cma.fmin2(
        lambda x: float(fun(np.asarray(x, dtype=float))),
        x0,
        sigma,
        options={
            "maxfevals": maxfevals,
            "popsize": popsize,
            "bounds": [low, high],
            "verbose": -9,
            "seed": seed,
        },
        restarts=0,
        bipop=False,
    )
    return np.asarray(xopt, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=390)
    xopt = run_cma(fun, maxfevals=60, sigma=0.32, popsize=9, seed=52)
    xopt = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 45), xopt=xopt)
    if xopt is not None and len(xopt):
        xopt = run_cma(fun, xopt, maxfevals=36, sigma=0.18, popsize=8, seed=420)
    xopt = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 18), xopt=xopt)
    if xopt is not None and len(xopt):
        run_cma(fun, xopt, maxfevals=24, sigma=0.10, popsize=6, seed=395)
    return fun.get_best()
