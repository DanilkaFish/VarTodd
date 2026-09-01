"""Source-conditioned scores: TOHPE and TODD candidates judged differently.

`k.source` tags each candidate with the search that produced it (1 TOHPE,
2 TOHPE-prefix, 3 TODD), and no existing seed reads it. The two sources are not
interchangeable: TOHPE is cheap and floods the pool with similar mid-band
actions, TODD is expensive and produces the rare late-band actions the descent
eventually depends on. Scoring both with one formula means whichever source is
numerically louder wins the pool regardless of which is actually scarce.

This seed applies a different criterion per source, and lets the *realized*
pool mix modulate how much a reduction is worth -- a feedback term, since
f_todd depends on what the pool selection already did.
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

OPTIMIZER_FAMILY = "pymoo_de_source_aware"
SEEDS = [65, 66]
N_EVAL = 264
TODD_TAG = 3


@policy.exploration
def explore_score(k, p, fn):
    """The scarce feature differs by source, so rank on a different one.

    For a TODD candidate the bucket it opened is the informative part: TODD
    pays for z research, and a candidate from a rich bucket carries more of
    that investment. For a TOHPE candidate the bucket is shared structure
    across the whole z choice, so y sparsity is what actually separates one
    TOHPE action from the next.
    """
    from_todd = k.source >= TODD_TAG
    todd_term = k.nbucket * p.w(1)
    tohpe_term = k.nyw * fn.abs(p.w(2))
    return k.nred * p.w(0) + fn.where(from_todd, todd_term, tohpe_term)


@policy.final
def final_score(k, p, fn):
    """Reduction scaled by the pool's realized TODD share.

    `f_todd` is the fraction of the merged pool that TODD supplied, so it says
    whether the expensive source is currently contributing at all. When TODD
    has gone quiet the surviving reductions are cheap TOHPE ones and are worth
    discounting relative to the lookahead term; when TODD is supplying most of
    the pool the descent is in the band where its actions matter most. The
    scale factor cannot be written additively: it multiplies a knob by a
    property of the pool that knob sits in.
    """
    share = fn.clip(k.f_todd, 0.0, 1.0)
    scaled = k.nred * p.w(0) * (1.0 + share * p.w(1))
    return scaled + k.ntohpe * p.w(2) + (1.0 - k.nrank_score) * p.w(3)


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
        # p.w(1) multiplies a share in [0, 1] and must not invert the reduction
        # term's sign, so it is searched one-sided.
        final_w[1] = self.float_range(0.0, 3.0)
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind(explore_w),
                final=final_score.bind(final_w),
            )
        )
        samples = SamplingBudget(
            one_hot="all", sparse=8, dense=self.int_range(4, 24), sparse_max_weight=2
        )
        todd_cap = self.int_range(1024, 9000)
        self.set_action_selection(
            ActionSelection(beamwidth=3, mode="softmax", temperature=0.2)
        )
        self.set_action_pool(ActionPool(final_size=self.int_range(18, 40)))
        # Both sources must stay alive for a source-conditioned score to have
        # anything to condition on.
        self.set_tohpe_search(
            TohpeSearch(samples, SourcePool(keep=6, reserve=2), z_choices=4)
        )
        self.set_todd_search(
            ToddSearch(
                SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
                SourcePool(keep=self.int_range(4, 8), reserve=2),
                actions_per_bucket=self.int_range(2, 4),
                buckets=ZBucketSearch(
                    min_buckets=24, max_buckets=todd_cap, limit_bucket=todd_cap
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
        Problem(evaluator), algorithm, termination=("n_eval", N_EVAL), seed=65, verbose=False
    )
    return evaluator.get_best()
