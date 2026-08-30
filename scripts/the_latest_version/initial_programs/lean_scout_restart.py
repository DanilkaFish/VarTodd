"""Light TODD seed: CMA-ES search with two policy restarts."""

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
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_cma_es_with_restarts"
SEEDS = [29, 30]
SCOUT_POINTS = 160
REFINE_EVALS = 80
REOPEN_MARGINS = [80, 36]
LOWER_BOUND = -1.0
UPPER_BOUND = 1.0


def objective(ranks: list[int]) -> float:
    values = np.asarray(ranks, dtype=float)
    return float(values.min() + 0.02 * values.std())


@policy.exploration
def explore_score(k, p, fn):
    """Linear in the five exploration knobs."""
    return (k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2)
            + k.nyw * p.w(3) + k.nzw * p.w(4))


@policy.final
def final_score(k, p, fn):
    """Linear, but y weight is scored by distance from a tuned center."""
    return (k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2)
            + fn.abs(k.nyw - p.w(6)) * p.w(3) + k.nzw * p.w(4)
            + k.ntohpe * p.w(5))


class Evaluator(BaseEvaluator):
    def float_range(self, low: float, high: float) -> float:
        return self.map_par(lambda x: low + (high - low) * np.clip((float(x) + 1.0) / 2.0, 0.0, 1.0))

    def int_range(self, low: int, high: int) -> int:
        return self.map_par(lambda x: int(round(low + (high - low) * np.clip((float(x) + 1.0) / 2.0, 0.0, 1.0))))

    def policy_mapping(self):
        # map_par order is unchanged: five exploration weights, six final
        # weights, then the y-weight center.
        explore_w = [self.float_range(-4, 4) for _ in range(5)]
        final_w = [self.float_range(-4, 4) for _ in range(6)]
        final_w.append(self.float_range(0.0, 1.0))
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind(explore_w),
                final=final_score.bind(final_w),
            )
        )
        samples = SamplingBudget(
            one_hot=20, sparse=self.int_range(4, 20), dense=self.int_range(4, 20), sparse_max_weight=3
        )
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.2))
        self.set_action_pool(ActionPool(final_size=self.int_range(16, 36)))
        self.set_tohpe_search(TohpeSearch(samples, SourcePool(keep=self.int_range(5, 14), reserve=1), z_choices=3))
        self.set_todd_search(
            ToddSearch(
                SamplingBudget(one_hot=12, sparse=4, dense=4, sparse_max_weight=2),
                SourcePool(keep=3, reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(min_buckets=64, max_buckets=4096, limit_bucket=4096),
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


def optimize_cma(evaluator: Evaluator, evaluations: int, seed: int, sigma: float) -> np.ndarray:
    current = evaluator.extract_active()
    if len(current) == 0:
        evaluator([])
        return current
    result = minimize(
        Problem(evaluator),
        CMAES(x0=np.clip(current, LOWER_BOUND, UPPER_BOUND), sigma=sigma, pop_size=8, restarts=0),
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
    params = optimize_cma(evaluator, SCOUT_POINTS, seed=29, sigma=0.8)
    for index, margin in enumerate(REOPEN_MARGINS):
        restarted = evaluator.set_up_new_init(0, rank_thr=evaluator.best_rank + margin, xopt=params)
        if restarted is None:
            break
        params = optimize_cma(evaluator, REFINE_EVALS, seed=30 + index, sigma=0.35)
    return evaluator.get_best()
