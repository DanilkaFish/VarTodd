from typing import Iterable

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

np.random.seed(43)


def squash(x: float, scale: float = 3.5, sharp: float = 1.4) -> float:
    return float(scale * np.tanh(float(x) / sharp))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


class Evaluator(BaseEvaluator):
    """PSO scout plus CMA refine, using partial y/z-centered score controls."""

    seeds = [101, 202, 303]

    def policy_mapping(self):
        ranks = [500, 430, 0]

        w_red = self.map_par(squash, 500)
        w_dim = self.map_par(squash, 500)
        w_bucket = self.map_par(squash, 500)
        w_vw = self.map_par(squash, 430)
        w_z = self.map_par(squash, 430)
        w_tohpe = self.map_par(squash, 430)
        center_vw = self.map_par(sigmoid, 430)
        center_z = self.map_par(sigmoid, 430)
        budget = self.map_par(sigmoid, 0)

        pool = ExplorationScore(
            weights=[w_red, w_dim, w_bucket, w_vw, w_z],
            centers=[0, 0, 0, center_vw, center_z],
            pow=1.0,
        )
        final = FinalizationScore(
            weights=[w_red, w_dim, w_bucket, w_vw, w_z, w_tohpe],
            centers=[0, 0, 0, center_vw, center_z, 0],
            pow=1.0,
        )

        beam = int(2 + 4 * budget)
        todd_width = int(1 + 4 * budget)
        final_pool_size = int(10 + 12 * budget)
        mid_z = int(40 + 80 * budget)
        high_z = int(70 + 90 * budget)
        one_hot = int(8 + 6 * (1.0 - budget))
        dense = int(5 + 5 * (1.0 - budget))

        caps = [[one_hot, 1, dense], [one_hot, 1, dense], [8, 1, dense]]
        samplings = [SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=2) for c in caps]
        temps = [0.75, 0.55, 0.35]
        min_z = [high_z, mid_z, mid_z]
        max_z = [700_000, 500_000, 400_000]
        self.set_scores(ranks, [PolicyScores(exploration=pool, final=final) for _ in ranks])
        self.set_action_selection(ranks, [ActionSelection(count=todd_width, mode="softmax", temperature=t) for t in temps])
        self.set_action_pool(ActionPool(final_size=final_pool_size))
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(sampling=samplings[i], pool=SourcePool(keep=keep, reserve=0), z_choices=2)
                for i, keep in enumerate([3, 2, 1])
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=samplings[i],
                    pool=SourcePool(keep=keep, reserve=0),
                    actions_per_bucket=2,
                    buckets=ZBucketSearch(min_buckets=min_z[i], max_buckets=max_z[i]),
                )
                for i, keep in enumerate([10, 12, 10])
            ],
        )
        self.set_widths(beam=beam, actions=todd_width)
        self.set_rollout_depth([430, 0], [0, 1])
        self.set_rollout_topk([430, 0], [0, 4])

    def __call__(self, params: Iterable):
        early = self.run(params, self.seeds[:1])
        if self.best_rank < 100000 and early[0] > self.best_rank + 18:
            return float(early[0] + 5.0)
        ranks = early + self.run(params, self.seeds[1:])
        return float(np.quantile(ranks, 0.60) + 0.015 * np.std(ranks))


def run_pso(fun: Evaluator, iters: int = 10, particles: int = 8):
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    def objective(positions):
        return np.asarray([fun(pos) for pos in positions], dtype=float)

    opt = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.8, "c2": 0.8, "w": 0.75},
        bounds=(np.full(n_params, -2.0), np.full(n_params, 2.0)),
    )
    _, best = opt.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best, dtype=float)


def local_refine(fun: Evaluator, x0: np.ndarray, steps: int = 8, sigma: float = 0.18):
    best = np.asarray(x0, dtype=float)
    best_score = float(fun(best))
    for _ in range(steps):
        candidate = best + np.random.normal(0.0, sigma, size=best.shape)
        score = float(fun(candidate))
        if score < best_score:
            best, best_score = candidate, score
    return best


def run_cma(fun: Evaluator, x0=None, maxfevals: int = 30, sigma: float = 0.28):
    low, high = -1.5, 1.5
    if x0 is None:
        x0 = np.zeros_like(fun.extract_active())
    x0 = np.asarray(x0, dtype=float)
    if x0.size == 0:
        return x0
    x0 = np.clip(x0, low + 1e-9, high - 1e-9)
    xopt, _ = cma.fmin2(
        lambda x: float(fun(np.asarray(x, dtype=float))),
        x0,
        sigma,
        options={
            "maxfevals": maxfevals,
            "popsize": 6,
            "bounds": [low, high],
            "verbose": -9,
            "seed": 43,
        },
        restarts=0,
        bipop=False,
    )
    return np.asarray(xopt, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=390)
    xopt = run_pso(fun, iters=10, particles=8)
    xopt = local_refine(fun, xopt, steps=8, sigma=0.18)
    xopt = fun.set_up_new_init(0, rank_thr=455, xopt=xopt)
    if xopt is not None and len(xopt):
        xopt = run_cma(fun, xopt, maxfevals=30, sigma=0.28)
        local_refine(fun, xopt, steps=6, sigma=0.12)
    return fun.get_best()
