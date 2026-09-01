"""Heavy-tail schedule explored by explicit DE, then refined by PSO."""

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
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_de_then_pso_with_restart"
SWITCH_RANK = TARGET_FINAL_RANK + 55
SCHEDULE = [INITIAL_RANK, SWITCH_RANK]
SEEDS = [23, 24]
FIRST_STAGE_EVALS = 64
RESTART_EVALS = 48
REOPEN_MARGIN = 55
TODD_ROLE = "heavy_tail_schedule"
LOWER_BOUND = -2.0
UPPER_BOUND = 2.0
EARLY_TODD_LIMIT = 512


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
    """Linear in the five exploration knobs."""
    return (k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2)
            + k.nyw * p.w(3) + k.nzw * p.w(4))


@policy.final
def final_score(k, p, fn):
    """Linear, with tuned centers on the y weight and the z density."""
    return (k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2)
            + fn.abs(k.nyw - p.w(6)) * p.w(3)
            + fn.abs(k.nzw - p.w(7)) * p.w(4)
            + k.ntohpe * p.w(5))


class Evaluator(BaseEvaluator):
    """Use finite TODD early and full z coverage only near the target."""

    def float_range(self, low: float, high: float, *, group: str = "scores") -> float:
        return self.map_par(
            lambda x: low + (high - low) * (0.5 + 0.5 * np.tanh(float(x) / 2.2)),
            group=group,
        )

    def int_range(self, low: int, high: int, *, group: str = "policy") -> int:
        return self.map_par(
            lambda x: min(
                high,
                low
                + int(
                    (high - low + 1)
                    * (0.5 + 0.5 * np.tanh(float(x) / 2.2))
                ),
            ),
            group=group,
        )

    def policy_mapping(self):
        # map_par order is unchanged: five exploration weights, six final
        # weights, then the y-weight and z-density centers.
        explore_w = [self.float_range(-3.5, 3.5) for _ in range(5)]
        final_w = [self.float_range(-3.5, 3.5) for _ in range(6)]
        final_w.append(self.float_range(0.0, 1.0))
        final_w.append(self.float_range(0.0, 1.0))
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind(explore_w),
                final=final_score.bind(final_w),
            )
        )
        tail_keep = self.int_range(8, 18)
        tail_samples = SamplingBudget(
            one_hot="all", sparse=self.int_range(4, 16), dense=self.int_range(4, 20), sparse_max_weight=3
        )
        terminal_actions_per_bucket = self.int_range(2, 5)
        terminal_min_buckets = self.int_range(512, 2048)
        self.set_action_selection(
            SCHEDULE,
            [ActionSelection(beamwidth=2, mode="best"), ActionSelection(beamwidth=3, mode="softmax", temperature=0.14)],
        )
        self.set_action_pool(SCHEDULE, [ActionPool(final_size=24), ActionPool(final_size=self.int_range(28, 56))])
        self.set_tohpe_search(
            SCHEDULE,
            [
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=16, dense=12, sparse_max_weight=3),
                    SourcePool(keep=self.int_range(12, 32), reserve=1),
                    z_choices=self.int_range(4, 10),
                    target_min_red=TARGET_MIN_RED,
                    target_max_red=TARGET_MAX_RED,
                ),
                TohpeSearch(
                    SamplingBudget(one_hot=24, sparse=8, dense=6, sparse_max_weight=3),
                    SourcePool(keep=8, reserve=0),
                    z_choices=2,
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
                    SourcePool(keep=2, reserve=1),
                    actions_per_bucket=2,
                    buckets=ZBucketSearch(min_buckets=8, max_buckets=128, limit_bucket=EARLY_TODD_LIMIT),
                ),
                ToddSearch(
                    tail_samples,
                    SourcePool(keep=tail_keep, reserve=2),
                    actions_per_bucket=terminal_actions_per_bucket,
                    buckets=ZBucketSearch(min_buckets=terminal_min_buckets, max_buckets=100000, limit_bucket=-1),
                ),
            ],
        )

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, SEEDS), dtype=float)
        return float(values.min() + 0.015 * values.std())


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        active = evaluator.extract_active()
        super().__init__(
            n_var=len(active),
            n_obj=1,
            xl=np.full(len(active), LOWER_BOUND),
            xu=np.full(len(active), UPPER_BOUND),
        )
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def optimize(
    evaluator: Evaluator, algorithm, evaluations: int, seed: int
) -> np.ndarray:
    result = minimize(
        Problem(evaluator),
        algorithm,
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    best = np.asarray(
        result.X if result.X is not None else evaluator.extract_active(),
        dtype=float,
    )
    evaluator.insert(best)
    evaluator.reinit()
    return best


def entrypoint():
    evaluator = Evaluator(path_name="init", max_depth=500)
    params = optimize(
        evaluator,
        DE(pop_size=16, variant="DE/rand/1/bin", CR=0.9, F=0.6),
        FIRST_STAGE_EVALS,
        seed=23,
    )
    restarted = evaluator.set_up_new_init(
        0, rank_thr=evaluator.best_rank + REOPEN_MARGIN, xopt=params
    )
    if restarted is not None:
        optimize(
            evaluator,
            PSO(pop_size=16, w=0.8, c1=0.55, c2=0.55, adaptive=True),
            RESTART_EVALS,
            seed=27,
        )
    return evaluator.get_best()
