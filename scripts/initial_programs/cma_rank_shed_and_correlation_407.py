from typing import Iterable

import numpy as np
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
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
    """9e0 cma_rank_shed: rank-scheduled correlated score offsets."""

    seeds = [6312, 6891]

    def policy_mapping(self):
        ranks = [10_000, 460, 421]
        pool_base = [
            self.map_par(_w_tanh, thr)
            for thr in [PARAM_ALWAYS_ACTIVE, PARAM_ALWAYS_ACTIVE, 468, 421, 421]
        ]
        final_base = [
            self.map_par(_w_tanh, thr)
            for thr in [PARAM_ALWAYS_ACTIVE, PARAM_ALWAYS_ACTIVE, 468, 421, 421, 421]
        ]
        red_diffs = [0.0, self.map_par(_w_tanh, 460), self.map_par(_w_tanh, 421)]
        weight_diffs = [0.0, self.map_par(_w_tanh, 460), self.map_par(_w_tanh, 421)]
        tohpe_mult = [1.0, 0.25 + self.map_par(sigmoid, 460), 0.0]
        z_scale = 10.0 + 200.0 * self.map_par(sigmoid, 421)

        pool_scores = [
            ExplorationScore(
                weights=[
                    pool_base[0] + rdiff,
                    pool_base[1],
                    pool_base[2] + rdiff,
                    pool_base[3] + wdiff,
                    pool_base[4] + wdiff,
                ],
                pow=1,
            )
            for rdiff, wdiff in zip(red_diffs, weight_diffs)
        ]
        final_scores = [
            FinalizationScore(
                weights=[
                    final_base[0] + rdiff,
                    final_base[1],
                    final_base[2] + rdiff,
                    final_base[3] + wdiff,
                    final_base[4] + wdiff,
                    mult * final_base[5],
                ],
                centers=[0.0, 0.1, 0.5, 0.5, 0.5, 0.0],
                pow=2,
            )
            for rdiff, wdiff, mult in zip(red_diffs, weight_diffs, tohpe_mult)
        ]

        tohpe_caps = [[8, 0, 0], [5, 1, 0], [3, 1, 0]]
        todd_caps = [[8, 0, 0], [4, 1, 1], [0, 0, 0]]
        sparse_weights = [2, 3, 4]
        max_buckets = [260_000, 540_000, 980_000]
        self.set_scores(
            ranks,
            [PolicyScores(exploration=pool_scores[i], final=final_scores[i]) for i in range(3)],
        )
        self.set_action_selection(
            ranks,
            [ActionSelection(count=2, mode="softmax", temperature=0.2) for _ in ranks],
        )
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in [20, 48, 100]])
        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(
                    sampling=SamplingBudget(one_hot=c[0], sparse=c[1], dense=c[2], sparse_max_weight=sparse_weights[i]),
                    pool=SourcePool(keep=6, reserve=reserve),
                    z_choices=10,
                )
                for i, (c, reserve) in enumerate(zip(tohpe_caps, [0, 1, 1]))
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
                for i, (c, keep) in enumerate(zip(todd_caps, [6, 0, 0]))
            ],
        )
        self.set_widths(beam=list(zip(ranks, [2, 2, 6])), actions=2)

    def __call__(self, params: Iterable[float]) -> float:
        ranks = self.run(params, self.seeds)
        return float(softmin(ranks, beta=6.0) + 0.01 * np.std(ranks))


def run_cma(
    fun: Evaluator,
    x0: np.ndarray | None = None,
    n_eval: int = 30,
    sigma: float = 0.5,
    low: float = -2.0,
    high: float = 2.0,
) -> np.ndarray:
    current = np.asarray(fun.extract_active() if x0 is None else x0, dtype=float)
    if current.size == 0:
        return current
    if current.size != len(fun.extract_active()):
        current = np.asarray(fun.extract_active(), dtype=float)
    current = np.clip(current, low + 1e-6, high - 1e-6)
    algorithm = CMAES(x0=current, sigma=sigma, pop_size=9)
    result = minimize(
        PymooProblem(fun, low, high),
        algorithm,
        termination=("n_eval", n_eval),
        seed=43,
        verbose=False,
    )
    return np.asarray(result.X if result.X is not None else current, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=200)
    xopt = run_cma(fun, n_eval=30)
    x_active = fun.set_up_new_init(0, rank_thr=460, xopt=xopt)
    if x_active is not None and len(x_active) > 0:
        run_cma(fun, np.asarray(x_active, dtype=float), n_eval=50, sigma=0.35)
    return fun.get_best()
