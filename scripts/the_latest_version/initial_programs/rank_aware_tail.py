"""Score candidates by their rank within the pool, and by where the pool came from.

Two families of knob are used here that the weight vector never reached:

- k.nrank_red / k.nrank_score: how many candidates in this iteration's
  population beat this one on reduction or pool score, as a fraction of the
  pool. These are scale-free -- they mean the same thing at rank 560 and at
  rank 400, where the raw reduction scale has collapsed from 9 to 1.
- k.f_todd / k.f_tohpe: what fraction of the merged pool each source
  contributed. This lets the tail policy react to TOHPE drying up rather than
  having that transition baked into a rank threshold.

Both are finalization-only: they depend on the settled pool, so they are not
known while source pools are still being filled.

The schedule mirrors the descent's two regimes. The head is reduction-greedy,
which is cheap and right while reductions are still large. The tail switches to
the rank-relative form, where absolute reduction stops discriminating.
"""

from collections.abc import Iterable
from helper import (
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    INITIAL_RANK,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    TARGET_FINAL_RANK,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
    policy,
)
import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_pso"
SEEDS = [37, 38]
N_EVAL = 240
SWITCH_RANK = TARGET_FINAL_RANK + 60
SCHEDULE = [INITIAL_RANK, SWITCH_RANK]
TODD_ROLE = "rank_aware_tail"
LOWER_BOUND = -2.0
UPPER_BOUND = 2.0


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
# [1, 10] here is deliberately permissive: it spans every reduction the
# search realistically produces, so the band imposes no preference and the
# z ranking stays the plain reduction-descending order. This program varies
# its policy by rank instead, and the wide band keeps that the only variable.
# TARGET_MIN_RED must be > 0: a zero reduction is not a usable action.
TARGET_MIN_RED = 1
TARGET_MAX_RED = 10


@policy.exploration
def explore_score(k, p, fn):
    """Linear in the five exploration knobs, shared by both rank bands."""
    return (k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2)
            + k.nyw * p.w(3) + k.nzw * p.w(4))


@policy.final
def final_head(k, p, fn):
    """Head: reduction and basis dimension, while reductions are still large."""
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.ntohpe * p.w(2)


@policy.final
def final_tail(k, p, fn):
    """Tail: rank within the pool, plus a source-composition preference.

    nrank_red is 0 for the best candidate and approaches 1 for the worst, so a
    negative p.w(1) rewards being near the top of the population regardless of
    the absolute reduction scale.
    """
    rank_term = k.nrank_red * p.w(1) + k.nrank_score * p.w(2)
    source_term = k.f_todd * p.w(3) + k.f_tohpe * p.w(4)
    return k.nred * p.w(0) + rank_term + source_term + k.ntohpe * p.w(5)


class Evaluator(BaseEvaluator):
    """Greedy head, rank-relative tail."""

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
        explore_w = [self.float_range(-4, 4) for _ in range(5)]
        head_w = [self.float_range(-4, 4) for _ in range(3)]
        tail_w = [self.float_range(-4, 4) for _ in range(6)]
        # Scheduling stays here, in the existing (ranks, values) setter form.
        self.set_scores(
            SCHEDULE,
            [
                PolicyScores(
                    exploration=explore_score.bind(explore_w),
                    final=final_head.bind(head_w),
                ),
                PolicyScores(
                    exploration=explore_score.bind(explore_w),
                    final=final_tail.bind(tail_w),
                ),
            ],
        )

        head_samples = SamplingBudget(one_hot="all", sparse=8, dense=12, sparse_max_weight=2)
        tail_samples = SamplingBudget(
            one_hot="all",
            sparse=self.int_range(8, 32),
            dense=self.int_range(8, 32),
            sparse_max_weight=3,
        )
        tail_cap = self.int_range(4096, 32000)

        self.set_action_selection(
            SCHEDULE,
            [
                ActionSelection(beamwidth=2, mode="best"),
                ActionSelection(beamwidth=3, mode="softmax", temperature=0.15),
            ],
        )
        self.set_action_pool(
            SCHEDULE,
            [ActionPool(final_size=24), ActionPool(final_size=self.int_range(32, 64))],
        )
        self.set_tohpe_search(
            SCHEDULE,
            [
                TohpeSearch(head_samples, SourcePool(keep=16, reserve=1), z_choices=4, target_min_red=TARGET_MIN_RED, target_max_red=TARGET_MAX_RED),
                TohpeSearch(
                    tail_samples,
                    SourcePool(keep=self.int_range(8, 24), reserve=2),
                    z_choices=self.int_range(2, 6),
                    target_min_red=TARGET_MIN_RED,
                    target_max_red=TARGET_MAX_RED,
                ),
            ],
        )
        self.set_todd_search(
            SCHEDULE,
            [
                ToddSearch(
                    SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
                    SourcePool(keep=4, reserve=1),
                    actions_per_bucket=2,
                    buckets=ZBucketSearch(min_buckets=32, max_buckets=512, limit_bucket=1024),
                ),
                ToddSearch(
                    tail_samples,
                    SourcePool(keep=self.int_range(8, 20), reserve=2),
                    actions_per_bucket=self.int_range(2, 4),
                    buckets=ZBucketSearch(
                        min_buckets=self.int_range(256, 2048),
                        max_buckets=tail_cap,
                        limit_bucket=tail_cap,
                    ),
                ),
            ],
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
    algorithm = PSO(pop_size=16, w=0.7, c1=1.4, c2=1.4, adaptive=True)
    minimize(Problem(evaluator), algorithm, termination=("n_eval", N_EVAL), seed=37, verbose=False)
    return evaluator.get_best()
