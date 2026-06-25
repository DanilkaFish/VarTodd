from typing import Iterable
import random

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

np.random.seed(52)
random.seed(52)


def weight(x: float) -> float:
    return float(4.0 * np.tanh(float(x) / 1.5))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def softmin(xs: Iterable[float], beta: float = 6.0) -> float:
    values = np.asarray(list(xs), dtype=float)
    m = values.min()
    return float(m - (1.0 / beta) * np.log(np.exp(-beta * (values - m)).sum()))


class Evaluator(BaseEvaluator):
    """Full-score policy optimized by PSO scout and CMA local refinement."""

    seeds = [random.randint(1, 10000) for _ in range(2)]

    @staticmethod
    def sample_caps(sample_count: int, one_hot_fraction: float):
        one_hot = max(0, int(round(sample_count * one_hot_fraction)))
        dense = max(0, int(sample_count) - one_hot)
        return [one_hot, 1, dense]

    def policy_mapping(self):
        ranks = [0]
        pool_score = ExplorationScore([self.map_par(weight) for _ in range(5)], pow=1)
        final_score = FinalizationScore([self.map_par(weight) for _ in range(6)], pow=1)
        self.set_scores(ranks, [PolicyScores(exploration=pool_score, final=final_score)])

        z_budget = self.map_par(sigmoid)
        sample_budget = self.map_par(sigmoid)
        one_hot_fraction = 0.1 + 0.5 * self.map_par(sigmoid)
        sample_caps = self.sample_caps(8 + int(48 * sample_budget), one_hot_fraction)

        sampling = SamplingBudget(
            one_hot=sample_caps[0],
            sparse=sample_caps[1],
            dense=sample_caps[2],
            sparse_max_weight=2,
        )
        self.set_action_selection(ActionSelection(count=1, mode="softmax", temperature=0.2))
        self.set_action_pool(ActionPool(final_size=12))
        self.set_tohpe_search(TohpeSearch(sampling=sampling, pool=SourcePool(keep=2, reserve=0), z_choices=2))
        self.set_todd_search(
            ToddSearch(
                sampling=sampling,
                pool=SourcePool(keep=12, reserve=0),
                actions_per_bucket=1,
                buckets=ZBucketSearch(min_buckets=10 + int(250 * z_budget), max_buckets=100000),
            )
        )
        self.set_widths(actions=1, beam=1)

    def __call__(self, params: Iterable):
        ranks = self.run(params, self.seeds)
        spread = float(np.std(ranks)) if len(ranks) > 1 else 0.0
        return softmin(ranks, beta=6.0) + 0.02 * spread


def run_pso(fun: Evaluator, iters: int) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    def objective(positions):
        return np.asarray([fun(position) for position in positions], dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=8,
        dimensions=n_params,
        options={"c1": 0.9, "c2": 0.5, "w": 0.7},
        bounds=(np.full(n_params, -1.0), np.full(n_params, 1.0)),
    )
    _, best_position = optimizer.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best_position, dtype=float)


def run_cma(fun: Evaluator, x0: np.ndarray, maxfevals: int) -> np.ndarray:
    low, high = -1.0, 1.0
    def objective(position):
        return float(fun(np.asarray(position, dtype=float)))

    x0 = np.asarray(x0, dtype=float)
    if x0.size == 0:
        return x0
    x0 = np.clip(x0, low + 1e-9, high - 1e-9)
    xopt, _ = cma.fmin2(
        objective,
        x0,
        0.25,
        options={
            "bounds": [low, high],
            "maxfevals": maxfevals,
            "popsize": 4,
            "seed": 52,
            "verbose": -9,
        },
        restarts=0,
        bipop=False,
    )
    return np.asarray(xopt, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", fin_rank=170, max_depth=100)
    x0 = run_pso(fun, iters=28)
    if fun.best_paths:
        xopt = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 55), xopt=x0)
        if xopt is not None and len(xopt):
            run_cma(fun, xopt, 50)
        xopt = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 10), xopt=xopt)
        if xopt is not None and len(xopt):
            run_cma(fun, xopt, 50)
    return fun.get_best()
