"""Multiplicative scores: a quality gate that scales the whole score.

Every additive score has the same weakness -- any term can be outvoted by a
large enough value elsewhere, so a candidate that is bad on the decisive axis
still survives on the strength of an irrelevant one. This seed puts the
population-standing terms into a bounded gate that *multiplies* the reduction
payload instead of being added to it, so a candidate the pool already ranks
poorly cannot buy its way back in.

It also reads `max_red`, the bucket's reduction ceiling, as unrealized
headroom: how much of what this bucket could have yielded was left behind.
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

OPTIMIZER_FAMILY = "pymoo_de_multiplicative_gain"
SEEDS = [67, 68]
N_EVAL = 264


@policy.exploration
def explore_score(k, p, fn):
    """Reduction against the reduction this bucket left on the table.

    `max_red - red` is unrealized headroom in exact reduction counts. Passing
    it through a log compresses the long tail so one very rich unexploited
    bucket does not dominate the ordering, while still preferring candidates
    that took most of what their bucket offered. This is a per-candidate
    regret, and it has no additive expression over nred and nbucket alone.
    """
    headroom = fn.log(fn.max(k.max_red - k.red, 0.0) + 1.0)
    return k.nred * p.w(0) - headroom * fn.abs(p.w(1)) + k.ndim * p.w(2)


@policy.final
def final_score(k, p, fn):
    """A bounded standing gate multiplying the reduction payload.

    `1 - nrank_red` and `1 - nrank_score` are 1 for the pool's best candidate
    and fall toward 0 for its worst. Combined inside a sigmoid they form a gate
    in (0, 1) that scales the whole reduction-and-lookahead payload: a
    candidate the pool ranks badly on both axes has its score damped no matter
    how the individual weights came out. An additive form cannot do this -- the
    payload would simply outweigh the standing terms.

    Both gate weights are searched one-sided so the gate cannot invert into
    rewarding poor standing, which would silently turn the score upside down.
    """
    standing = (1.0 - k.nrank_red) * p.w(0) + (1.0 - k.nrank_score) * p.w(1)
    gate = fn.sigmoid(standing)
    payload = k.nred * p.w(2) + k.ntohpe * p.w(3)
    return gate * payload + k.nrank_dim * p.w(4)


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
        explore_w = [self.float_range(-4, 4) for _ in range(explore_score.n_params)]
        final_w = [self.float_range(-4, 4) for _ in range(final_score.n_params)]
        # The two gate weights stay non-negative: a negative one would make the
        # gate reward candidates the pool ranks worst.
        final_w[0] = self.float_range(0.0, 4.0)
        final_w[1] = self.float_range(0.0, 4.0)
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind(explore_w),
                final=final_score.bind(final_w),
            )
        )
        samples = SamplingBudget(
            one_hot="all", sparse=6, dense=self.int_range(4, 24), sparse_max_weight=2
        )
        todd_cap = self.int_range(768, 7000)
        self.set_action_selection(
            ActionSelection(beamwidth=3, mode="softmax", temperature=0.17)
        )
        self.set_action_pool(ActionPool(final_size=self.int_range(20, 44)))
        self.set_tohpe_search(
            TohpeSearch(samples, SourcePool(keep=6, reserve=1), z_choices=4)
        )
        self.set_todd_search(
            ToddSearch(
                SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
                SourcePool(keep=self.int_range(3, 6), reserve=1),
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
    algorithm = DE(pop_size=18, variant="DE/rand/1/bin", CR=0.9, F=0.6)
    minimize(
        Problem(evaluator), algorithm, termination=("n_eval", N_EVAL), seed=67, verbose=False
    )
    return evaluator.get_best()
