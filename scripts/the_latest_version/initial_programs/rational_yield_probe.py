"""Rational scores: reduction per unit of the work that produced it.

A weighted sum can say "more reduction is good" and "a big binary null space is
good", but it cannot say "reduction *per dimension* is good" -- a quotient is
not in the span of a linear form over the same knobs. This seed ranks by
efficiency ratios rather than by magnitudes.

The ratios are deliberately written on raw knobs where the normalizers would
otherwise cancel: `nred / ndim` carries 1/bn above and 1/dn below and mostly
divides out the very scaling the ratio was meant to expose, while `red / dim`
is the quantity actually wanted.
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

OPTIMIZER_FAMILY = "pymoo_de_rational_yield"
SEEDS = [63, 64]
N_EVAL = 264


@policy.exploration
def explore_score(k, p, fn):
    """Reduction per unit of null-space dimension, minus relative z cost.

    `red / dim` is reduction bought per dimension of the space that had to be
    built to find it -- an efficiency, not a magnitude. Two candidates with the
    same reduction rank differently when one needed a far larger space, and no
    weighting of nred and ndim separately reproduces that ordering.

    The z term is relative too: `zsize / wvwn` is the z vector's length against
    the parity matrix it lives in, so the penalty keeps its meaning as the rank
    falls instead of drifting with the absolute size.
    """
    yield_per_dim = k.red / fn.max(k.dim, 1.0)
    relative_z = k.zsize / fn.max(k.wvwn, 1.0)
    return yield_per_dim * p.w(0) + k.nred * p.w(1) - relative_z * fn.abs(p.w(2))


@policy.final
def final_score(k, p, fn):
    """Reduction weighted by how thin the pool it came from is.

    `pool_size` against the summed per-source counts says whether the merged
    pool is a broad field or a handful of survivors. A reduction that stands
    out in a thin pool is weaker evidence than the same reduction in a crowded
    one, and this is a property of the pool rather than of the candidate -- so
    it can only enter as a factor, never as another additive knob term.
    """
    sources = k.pool_todd + k.pool_tohpe + k.pool_prefix
    crowding = k.pool_size / fn.max(sources, 1.0)
    return (
        k.nred * p.w(0) * fn.clip(crowding, 0.25, 4.0)
        + k.ntohpe * p.w(1)
        + (1.0 - k.nrank_score) * p.w(2)
    )


class Evaluator(BaseEvaluator):
    def float_range(self, low: float, high: float) -> float:
        return self.map_par(
            lambda x: low + (high - low) * np.clip((float(x) + 1.5) / 3.0, 0.0, 1.0)
        )

    def int_range(self, low: int, high: int) -> int:
        return self.map_par(
            lambda x: min(
                high, low + int((high - low + 1) * np.clip((float(x) + 1.5) / 3.0, 0.0, 1.0))
            )
        )

    def policy_mapping(self):
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind(
                    [self.float_range(-4, 4) for _ in range(explore_score.n_params)]
                ),
                final=final_score.bind(
                    [self.float_range(-4, 4) for _ in range(final_score.n_params)]
                ),
            )
        )
        samples = SamplingBudget(
            one_hot="all", sparse=8, dense=self.int_range(6, 32), sparse_max_weight=2
        )
        todd_cap = self.int_range(512, 6000)
        self.set_action_selection(
            ActionSelection(beamwidth=2, mode="softmax", temperature=0.15)
        )
        self.set_action_pool(ActionPool(final_size=self.int_range(14, 32)))
        self.set_tohpe_search(
            TohpeSearch(samples, SourcePool(keep=6, reserve=2), z_choices=4)
        )
        self.set_todd_search(
            ToddSearch(
                SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
                SourcePool(keep=self.int_range(2, 5), reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(
                    min_buckets=16, max_buckets=todd_cap, limit_bucket=todd_cap
                ),
            )
        )

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, SEEDS), dtype=float)
        return float(values.min() + 0.02 * values.std())


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        n = len(evaluator.extract_active())
        super().__init__(n_var=n, n_obj=1, xl=np.full(n, -1.5), xu=np.full(n, 1.5))
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def entrypoint():
    evaluator = Evaluator(path_name="init", max_depth=500)
    algorithm = DE(pop_size=16, variant="DE/rand/1/bin", CR=0.9, F=0.6)
    minimize(
        Problem(evaluator), algorithm, termination=("n_eval", N_EVAL), seed=63, verbose=False
    )
    return evaluator.get_best()
