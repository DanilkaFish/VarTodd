"""Score reduction per unit of bucket work, a ratio the weight vector could not express.

The classic score is a weighted sum, so it can reward large reductions and
penalize large buckets, but it cannot ask for reduction *relative to* the bucket
it came from. In the terminal band, where researched z counts grow huge and
acceptance collapses, "which candidate pays for its search cost" is a different
question from "which candidate reduces most".

The ratio lives in the finalization score, which runs once per merged
candidate; exploration stays a weighted sum so the pool fills at the usual
rate.

The inner z search never consults the exploration score -- it ranks z by
distance to the TOHPE reduction band below -- so reading k.nbucket here costs
nothing beyond the arithmetic. The score decides what enters the pool.
"""

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
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_de"
SEEDS = [31, 32]
N_EVAL = 240
TODD_ROLE = "ratio_efficiency"
LOWER_BOUND = -2.0
UPPER_BOUND = 2.0

# Guards the ratio denominator. bucket is a raw count, so a floor of 1 is the
# smallest bucket that can exist and the guard never distorts a real value.
MIN_BUCKET = 1.0


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
TARGET_MIN_RED = 3
TARGET_MAX_RED = 4


@policy.exploration
def explore_score(k, p, fn):
    """Plain linear, kept cheap so the pool fills at the usual rate."""
    return (k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2)
            + k.nyw * p.w(3) + k.nzw * p.w(4))


@policy.final
def final_score(k, p, fn):
    """Reduction, plus reduction per unit of bucket work.

    p.w(1) trades the absolute reduction term against the efficiency term, so
    the optimizer can settle anywhere between pure greedy and pure ratio rather
    than being restricted to one of them.
    """
    # Raw over raw: nred and nbucket both carry a 1/bn factor that would
    # cancel, so a ratio of the normalized knobs is the same quantity with a
    # redundant division. red/bucket is what is actually wanted.
    efficiency = k.red / fn.max(k.bucket, MIN_BUCKET)
    return (k.nred * p.w(0)
            + efficiency * p.w(1)
            + k.ndim * p.w(2)
            + k.ntohpe * p.w(3))


class Evaluator(BaseEvaluator):
    """Linear exploration, ratio-aware finalization."""

    def float_range(self, low: float, high: float, *, group: str = "scores") -> float:
        return self.map_par(
            lambda x: low + (high - low) * (0.5 + 0.5 * np.tanh(float(x) / 2.0)), group=group
        )

    def int_range(self, low: int, high: int, *, group: str = "search") -> int:
        return self.map_par(
            lambda x: min(high, low + int((high - low + 1) * (0.5 + 0.5 * np.tanh(float(x) / 2.0)))),
            group=group,
        )

    def policy_mapping(self):
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind([self.float_range(-4, 4) for _ in range(5)]),
                final=final_score.bind([self.float_range(-4, 4) for _ in range(4)]),
            )
        )
        samples = SamplingBudget(
            one_hot="all",
            sparse=self.int_range(8, 48),
            dense=self.int_range(8, 48),
            sparse_max_weight=3,
        )
        todd_cap = self.int_range(2048, 24000)
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.18))
        self.set_action_pool(ActionPool(final_size=self.int_range(24, 56)))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=samples,
                pool=SourcePool(keep=self.int_range(8, 32), reserve=2),
                z_choices=self.int_range(2, 8),
                target_min_red=TARGET_MIN_RED,
                target_max_red=TARGET_MAX_RED,
            )
        )
        self.set_todd_search(
            ToddSearch(
                sampling=samples,
                pool=SourcePool(keep=self.int_range(6, 20), reserve=2),
                actions_per_bucket=self.int_range(2, 4),
                buckets=ZBucketSearch(min_buckets=64, max_buckets=todd_cap, limit_bucket=todd_cap),
            )
        )

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, SEEDS), dtype=float)
        return float(values.min() + 0.02 * values.std())


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        n = len(evaluator.extract_active())
        super().__init__(
            n_var=n, n_obj=1, xl=np.full(n, LOWER_BOUND), xu=np.full(n, UPPER_BOUND)
        )
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def entrypoint():
    evaluator = Evaluator(path_name="init", max_depth=500)
    algorithm = DE(pop_size=16, variant="DE/rand/1/bin", CR=0.9, F=0.6)
    minimize(Problem(evaluator), algorithm, termination=("n_eval", N_EVAL), seed=31, verbose=False)
    return evaluator.get_best()
