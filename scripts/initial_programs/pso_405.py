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


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


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
    """9e0 pso_405: PSO version of the compact TOHPE-first policy."""

    seeds = [6312, 6891]

    def policy_mapping(self):
        ranks = [10_000, 460, 421]
        pool_weights = [
            self.map_par(_w_tanh, thr)
            for thr in [PARAM_ALWAYS_ACTIVE, PARAM_ALWAYS_ACTIVE, 468, 421, 421]
        ]
        final_weights = [
            self.map_par(_w_tanh, thr)
            for thr in [PARAM_ALWAYS_ACTIVE, PARAM_ALWAYS_ACTIVE, 468, 421, 421, 421]
        ]
        z_scale = 10.0 + 500.0 * self.map_par(sigmoid, 421)

        todd_caps = [[0, 0, 0], [2, 0, 0], [4, 1, 1]]
        sparse_weights = [2, 2, 3]
        action_counts = [2, 2, 4]
        max_buckets = [260_000, 540_000, 980_000]
        self.set_scores(
            PolicyScores(
                exploration=ExplorationScore(weights=pool_weights, pow=1),
                final=FinalizationScore(weights=final_weights, pow=2),
            )
        )
        self.set_action_selection(
            ranks,
            [ActionSelection(count=count, mode="softmax", temperature=0.7) for count in action_counts],
        )
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in [20, 48, 100]])
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(
                    sampling=SamplingBudget(one_hot=30, sparse=0, dense=0, sparse_max_weight=sparse_weights[i]),
                    pool=SourcePool(keep=6, reserve=1),
                    z_choices=1,
                )
                for i in range(3)
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=sparse_weights[i]),
                    pool=SourcePool(keep=keep, reserve=0),
                    actions_per_bucket=2,
                    buckets=ZBucketSearch(min_buckets=int(z_scale), max_buckets=max_buckets[i]),
                )
                for i, (c, keep) in enumerate(zip(todd_caps, [0, 2, 6]))
            ],
        )
        self.set_widths(beam=list(zip(ranks, [3, 3, 6])), actions=list(zip(ranks, action_counts)))

    def __call__(self, params: Iterable[float]) -> float:
        ranks = self.run(params, self.seeds)
        return float(softmin(ranks, beta=6.0) + 0.01 * np.std(ranks))


def run_pso(
    fun: Evaluator,
    n_eval: int = 24,
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
        seed=45,
        verbose=False,
    )
    return np.asarray(result.X if result.X is not None else fun.extract_active(), dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=200)
    xopt = run_pso(fun, n_eval=24)
    x_active = fun.set_up_new_init(0, rank_thr=470, xopt=xopt)
    if x_active is not None and len(x_active) > 0:
        run_pso(fun, n_eval=24, low=-1.2, high=1.2)
    return fun.get_best()
