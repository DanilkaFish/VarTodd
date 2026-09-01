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


# TOHPE reduction target band.
#
# For each sampled y, the cheap TOHPE z search ranks its z candidates by
#     max(TARGET_MIN_RED - red, red - TARGET_MAX_RED, 0)
# so candidates whose reduction lands inside [TARGET_MIN_RED, TARGET_MAX_RED]
# come first, and outside the band the ones nearest to it follow. The chosen z
# then enter the action pool on the exploration score as usual -- the band only
# decides which z get that far, it does not replace the scoring.
#
# Choosing a *size* of reduction rather than always the largest is the point:
#   [10, 10] aims at the greatest reductions -- the classic greedy behaviour,
#            productive in the early bands where large reductions are plentiful;
#   [3, 4]   aims at medium reductions, which keeps more of the structure intact
#            for later steps instead of spending it in one move;
#   [2, 3]   aims at small reductions, matching the terminal band where large
#            ones have stopped appearing and the search lives on 1-2 per step.
# TARGET_MIN_RED must be > 0: a zero reduction is not a usable action.
TARGET_MIN_RED = 2
TARGET_MAX_RED = 3


@policy.exploration
def explore_score(k, p, fn):
    """Reduction on a compressed scale, with a tuned y-weight band.

    sqrt flattens the reduction scale so a rank-9 action does not swamp three
    rank-3 ones while the descent is still cheap; the restarts then re-tune
    around whatever band the scout found. The y-weight term is a soft band
    rather than a monotone preference: exp of a negative square peaks at the
    center and falls off either side, so the policy can prefer a *particular*
    y weight instead of only "more" or "less".
    """
    compressed = fn.sqrt(fn.max(k.nred, 0.0)) * p.w(0)
    offset = k.nyw - p.w(2)
    band = fn.exp(-(offset * offset) * fn.abs(p.w(3)))
    return compressed + band * p.w(1) + k.ndim * p.w(4)


@policy.final
def final_score(k, p, fn):
    """Reduction, discounted by how much of the pool already beats it.

    (1 - nrank_red) is 1 for the population's best candidate and falls toward 0
    for the worst, so multiplying by it makes a mediocre reduction worth
    strictly less than the same reduction in a weak pool. That is a product of
    two knobs; an additive rank term would let a large reduction buy its way
    back regardless of the company it is in.
    """
    standing = 1.0 - k.nrank_red
    return (k.nred * p.w(0) * fn.max(standing, 0.05)
            + k.ndim * p.w(1)
            + k.ntohpe * p.w(2)
            + fn.abs(k.nyw - p.w(4)) * p.w(3))


class Evaluator(BaseEvaluator):
    def float_range(self, low: float, high: float) -> float:
        return self.map_par(lambda x: low + (high - low) * np.clip((float(x) + 1.0) / 2.0, 0.0, 1.0))

    def int_range(self, low: int, high: int) -> int:
        return self.map_par(lambda x: int(round(low + (high - low) * np.clip((float(x) + 1.0) / 2.0, 0.0, 1.0))))

    def policy_mapping(self):
        # The last final parameter is a y-weight center, so it is mapped to
        # [0, 1] rather than the symmetric coefficient range.
        explore_w = [self.float_range(-4, 4) for _ in range(explore_score.n_params)]
        final_w = [self.float_range(-4, 4) for _ in range(final_score.n_params - 1)]
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
        self.set_tohpe_search(TohpeSearch(samples, SourcePool(keep=self.int_range(5, 14), reserve=1), z_choices=3, target_min_red=TARGET_MIN_RED, target_max_red=TARGET_MAX_RED))
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
