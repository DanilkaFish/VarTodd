"""Score reduction per unit of bucket work, a ratio the weight vector could not express.

The classic score is a weighted sum, so it can reward large reductions and
penalize large buckets, but it cannot ask for reduction *relative to* the bucket
it came from. In the terminal band, where researched z counts grow huge and
acceptance collapses, "which candidate pays for its search cost" is a different
question from "which candidate reduces most".

Cost note: the exploration score reads k.nbucket, so the inner z search cannot
take its reduction-only fast path and must score every touched bucket instead.
That is a real wall-clock cost, paid deliberately here. The finalization score
is where ratios are cheapest -- it runs once per merged candidate -- so the
exploration side stays linear and only finalization uses the ratio.
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

# Guards the ratio denominator. nbucket is bucket_size / max_bucket, so this
# floor corresponds to a bucket a twentieth the size of the largest one.
MIN_NBUCKET = 0.05


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
    efficiency = k.nred / fn.max(k.nbucket, MIN_NBUCKET)
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
