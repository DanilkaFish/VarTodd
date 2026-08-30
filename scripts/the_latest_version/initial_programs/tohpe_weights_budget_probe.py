"""Light TODD seed: search the joint policy, then refine score weights."""

from collections.abc import Iterable
from helper import (
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
    policy,
)
import numpy as np
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_pso_then_cma_es"
SEEDS = [21, 22]
JOINT_TRIALS = 192
SCORE_EVALS = 96
TODD_ROLE = "light"
LOWER_BOUND = -2.4
UPPER_BOUND = 2.4


def objective(ranks: list[int]) -> float:
    values = np.asarray(ranks, dtype=float)
    return float(values.min() + 0.015 * values.std())


@policy.exploration
def explore_score(k, p, fn):
    """Linear in the five exploration knobs."""
    return (k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2)
            + k.nyw * p.w(3) + k.nzw * p.w(4))


@policy.final
def final_score(k, p, fn):
    """Linear, but z density is scored by distance from a tuned center."""
    return (k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2)
            + k.nyw * p.w(3) + fn.abs(k.nzw - p.w(6)) * p.w(4)
            + k.ntohpe * p.w(5))


class Evaluator(BaseEvaluator):
    """TOHPE supplies most candidates; light TODD is an escape path."""

    def float_range(self, low: float, high: float, *, group: str) -> float:
        return self.map_par(lambda x: low + (high - low) * (0.5 + 0.5 * np.tanh(float(x) / 1.6)), group=group)

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(
            lambda x: min(high, low + int((high - low + 1) * (0.5 + 0.5 * np.tanh(float(x) / 1.6)))), group=group
        )

    def policy_mapping(self):
        # map_par order is unchanged: five exploration weights, six final
        # weights, then the z-density center.
        explore_w = [self.float_range(-4.0, 4.0, group="scores") for _ in range(5)]
        final_w = [self.float_range(-4.0, 4.0, group="scores") for _ in range(6)]
        final_w.append(self.float_range(0.0, 1.0, group="scores"))
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind(explore_w),
                final=final_score.bind(final_w),
            )
        )
        tohpe_sampling = SamplingBudget(
            one_hot="all",
            sparse=self.int_range(16, 96, group="search"),
            dense=self.int_range(8, 64, group="search"),
            sparse_max_weight=self.int_range(2, 5, group="search"),
        )
        self.set_action_selection(ActionSelection(beamwidth=2, mode="best"))
        self.set_action_pool(ActionPool(final_size=self.int_range(32, 72, group="search")))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=tohpe_sampling,
                pool=SourcePool(keep=self.int_range(20, 64, group="search"), reserve=2),
                z_choices=self.int_range(3, 14, group="search"),
            )
        )
        self.set_todd_search(
            ToddSearch(
                sampling=SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
                pool=SourcePool(keep=2, reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(min_buckets=8, max_buckets=32, limit_bucket=256),
            )
        )

    def __call__(self, params: Iterable[float]) -> float:
        return objective(self.run(params, SEEDS))


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        dimensions = len(evaluator.extract_active())
        super().__init__(
            n_var=dimensions,
            n_obj=1,
            xl=np.full(dimensions, LOWER_BOUND),
            xu=np.full(dimensions, UPPER_BOUND),
        )
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def optimize_pso(evaluator: Evaluator, evaluations: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    if len(current) == 0:
        evaluator([])
        return current
    result = minimize(
        Problem(evaluator),
        PSO(pop_size=16, w=0.7, c1=1.4, c2=1.4, adaptive=True),
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    best = np.asarray(result.X if result.X is not None else current, dtype=float)
    evaluator.insert(best)
    evaluator.reinit()
    return best


def optimize_cma(evaluator: Evaluator, evaluations: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    if len(current) == 0:
        evaluator([])
        return current
    result = minimize(
        Problem(evaluator),
        CMAES(x0=np.clip(current, LOWER_BOUND, UPPER_BOUND), sigma=0.65, pop_size=8, restarts=0),
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    best = np.asarray(result.X if result.X is not None else current, dtype=float)
    evaluator.insert(best)
    evaluator.reinit()
    return best


def entrypoint():
    evaluator = Evaluator(path_name="init", max_depth=500)
    evaluator.select_all_parameter_groups()
    optimize_pso(evaluator, JOINT_TRIALS, seed=21)
    evaluator.select_parameter_groups("scores")
    optimize_cma(evaluator, SCORE_EVALS, seed=24)
    return evaluator.get_best()
