from typing import Iterable

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from helper import (
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    PARAM_ALWAYS_ACTIVE,
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


def _w_tanh(z: float, scale: float = 5.5, sharp: float = 1.2) -> float:
    return float(scale * np.tanh(float(z) / sharp))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


class PymooProblem(ElementwiseProblem):
    def __init__(self, fun: "Evaluator", low: float, high: float):
        self.fun = fun
        n_params = len(fun.extract_active())
        super().__init__(
            n_var=n_params,
            n_obj=1,
            xl=np.full(n_params, low),
            xu=np.full(n_params, high),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = float(self.fun(np.asarray(x, dtype=float)))


class Evaluator(BaseEvaluator):
    """9e0 multi_pso_413: multi-seed PSO with per-rank score profiles."""

    seeds = [1825, 410, 4507, 4013]

    def policy_mapping(self):
        ranks = [10_000, 440, 421, 405]
        map_ranks = [PARAM_ALWAYS_ACTIVE, 440, 421, 405]

        pool_scores = []
        for rank_gate in map_ranks:
            weights = [self.map_par(_w_tanh, rank_gate) for _ in range(5)]
            if rank_gate <= 415:
                weights[0] = -abs(weights[0]) - 0.5
            pool_scores.append(ExplorationScore(weights=weights, pow=1))

        final_scores = []
        for rank_gate in map_ranks:
            weights = [self.map_par(_w_tanh, rank_gate) for _ in range(6)]
            centers = [self.map_par(sigmoid, rank_gate) for _ in range(6)]
            final_scores.append(FinalizationScore(weights=weights, centers=centers, pow=2))

        tohpe_caps = [[16, 0, 0], [12, 2, 2], [8, 2, 2], [6, 1, 1]]
        todd_caps = [[0, 0, 0], [10, 1, 1], [8, 2, 2], [6, 1, 1]]
        sparse_weights = [2, 2, 3, 4]
        action_counts = [2, 2, 4, 4]
        min_z = [300, 300, 200, 100]
        max_z = [180_000, 360_000, 700_000, 1_400_000]
        self.set_scores(
            ranks,
            [PolicyScores(exploration=pool_scores[i], final=final_scores[i]) for i in range(4)],
        )
        self.set_action_selection(
            ranks,
            [ActionSelection(count=count, mode="softmax", temperature=0.15) for count in action_counts],
        )
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in [32, 32, 48, 64]])
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(
                    sampling=SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=sparse_weights[i]),
                    pool=SourcePool(keep=keep, reserve=reserve),
                    z_choices=z,
                )
                for i, (c, keep, reserve, z) in enumerate(zip(tohpe_caps, [60, 60, 40, 20], [1, 1, 0, 0], [8, 8, 4, 2]))
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=sparse_weights[i]),
                    pool=SourcePool(keep=keep, reserve=reserve),
                    actions_per_bucket=4,
                    buckets=ZBucketSearch(min_buckets=min_z[i], max_buckets=max_z[i]),
                )
                for i, (c, keep, reserve) in enumerate(zip(todd_caps, [0, 12, 18, 20], [0, 0, 1, 1]))
            ],
        )
        self.set_widths(beam=list(zip(ranks, [2, 2, 6, 4])), actions=list(zip(ranks, action_counts)))

    def __call__(self, params: Iterable[float]) -> float:
        ranks = self.run(params, self.seeds)
        if not ranks:
            return 1000.0
        return float(np.min(ranks) + 0.015 * np.std(ranks))


def run_pso(
    fun: Evaluator,
    n_eval: int = 24,
    pop_size: int = 4,
    low: float = -2.5,
    high: float = 2.5,
) -> np.ndarray:
    if len(fun.extract_active()) == 0:
        return np.asarray([], dtype=float)
    algorithm = PSO(pop_size=pop_size, w=0.7, c1=0.6, c2=0.6, adaptive=False)
    result = minimize(
        PymooProblem(fun, low, high),
        algorithm,
        termination=("n_eval", n_eval),
        seed=44,
        verbose=False,
    )
    return np.asarray(result.X if result.X is not None else fun.extract_active(), dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=250)
    xopt = run_pso(fun, n_eval=20, pop_size=4)
    x_active = fun.set_up_new_init(0, rank_thr=430, xopt=xopt)
    if x_active is not None and len(x_active) > 0:
        run_pso(fun, n_eval=36, pop_size=6, low=-1.5, high=1.5)
    return fun.get_best()
