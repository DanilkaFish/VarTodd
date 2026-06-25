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


def _w_tanh(z: float, scale: float = 5.5, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(float(z) / sharp))


def softmin(xs: Iterable[float], beta: float = 6.0) -> float:
    values = np.asarray(list(xs), dtype=float)
    if values.size == 0:
        return 10_000.0
    m = float(values.min())
    return float(m - np.log(np.exp(-beta * (values - m)).sum()) / beta)


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
    """9e0 bs_413: narrow TOHPE scout with a late beam-width tail."""

    seeds = [6312]

    def policy_mapping(self):
        ranks = [10_000, 450, 421]
        pool_thresholds = [PARAM_ALWAYS_ACTIVE, PARAM_ALWAYS_ACTIVE, 468, 421, 421]
        final_thresholds = [PARAM_ALWAYS_ACTIVE, PARAM_ALWAYS_ACTIVE, 468, 421, 421, 421]

        pool_weights = [self.map_par(_w_tanh, thr) for thr in pool_thresholds]
        final_weights = [self.map_par(_w_tanh, thr) for thr in final_thresholds]

        tohpe_caps = [[10, 0, 0], [8, 1, 1], [6, 2, 2]]
        todd_caps = [[0, 0, 0], [2, 0, 0], [4, 1, 1]]
        sparse_weights = [2, 2, 3]
        action_counts = [1, 4, 18]
        max_buckets = [220_000, 420_000, 700_000]
        self.set_scores(
            PolicyScores(
                exploration=ExplorationScore(weights=pool_weights, pow=1),
                final=FinalizationScore(weights=final_weights, pow=2),
            )
        )
        self.set_action_selection(
            ranks,
            [ActionSelection(count=count, mode="softmax", temperature=0.2) for count in action_counts],
        )
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in [1, 12, 25]])
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(
                    sampling=SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=sparse_weights[i]),
                    pool=SourcePool(keep=6, reserve=1),
                    z_choices=10,
                )
                for i, c in enumerate(tohpe_caps)
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=sparse_weights[i]),
                    pool=SourcePool(keep=keep, reserve=0),
                    actions_per_bucket=5,
                    buckets=ZBucketSearch(min_buckets=350, max_buckets=max_buckets[i]),
                )
                for i, (c, keep) in enumerate(zip(todd_caps, [0, 2, 5]))
            ],
        )
        self.set_widths(beam=list(zip(ranks, [1, 4, 24])), actions=list(zip(ranks, action_counts)))

    def __call__(self, params: Iterable[float]) -> float:
        ranks = self.run(params, self.seeds)
        return float(softmin(ranks, beta=6.0) + 0.01 * np.std(ranks))


def run_pso(
    fun: Evaluator,
    n_eval: int = 40,
    low: float = -2.0,
    high: float = 2.0,
) -> np.ndarray:
    if len(fun.extract_active()) == 0:
        return np.asarray([], dtype=float)
    algorithm = PSO(pop_size=4, w=0.9, c1=1.0, c2=1.0, adaptive=False)
    result = minimize(
        PymooProblem(fun, low, high),
        algorithm,
        termination=("n_eval", n_eval),
        seed=42,
        verbose=False,
    )
    return np.asarray(result.X if result.X is not None else fun.extract_active(), dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=200)
    run_pso(fun)
    return fun.get_best()
