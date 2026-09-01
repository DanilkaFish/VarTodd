"""Heavy-after-restart seed: PSO exploration, then PatternSearch refinement."""

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
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_pso_then_pattern_search"
SEEDS = [40, 41]
SCOUT_PARTICLES = 16
SCOUT_ITERS = 4
SCOUT_EVALS = SCOUT_PARTICLES * SCOUT_ITERS
TAIL_EVALS = 96
REOPEN_MARGIN = 48
TODD_ROLE = "heavy_after_restart"
LOWER_BOUND = -1.0
UPPER_BOUND = 1.0


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
TARGET_MIN_RED = 10
TARGET_MAX_RED = 10


@policy.exploration
def explore_score(k, p, fn):
    """Linear in the five exploration knobs."""
    return (k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2)
            + k.nyw * p.w(3) + k.nzw * p.w(4))


@policy.final
def final_score(k, p, fn):
    """Linear, with tuned centers on the basis dimension and the y weight."""
    return (k.nred * p.w(0) + fn.abs(k.ndim - p.w(6)) * p.w(1)
            + k.nbucket * p.w(2) + fn.abs(k.nyw - p.w(7)) * p.w(3)
            + k.nzw * p.w(4) + k.ntohpe * p.w(5))


class Evaluator(BaseEvaluator):
    """Declare both stages permanently; the restart only selects the policy."""

    def __init__(self, *args, **kwargs):
        self.use_heavy_todd = False
        super().__init__(*args, **kwargs)

    def float_range(self, low: float, high: float, *, group: str) -> float:
        return self.map_par(lambda x: low + (high - low) / (1.0 + np.exp(-np.clip(float(x), -20.0, 20.0))), group=group)

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(
            lambda x: min(high, low + int((high - low + 1) / (1.0 + np.exp(-np.clip(float(x), -20.0, 20.0))))),
            group=group,
        )

    def policy_mapping(self):
        # map_par order is unchanged: five exploration weights, six final
        # weights, then the basis-dimension and y-weight centers.
        explore_w = [self.float_range(-4, 4, group="scores") for _ in range(5)]
        final_w = [self.float_range(-4, 4, group="scores") for _ in range(6)]
        final_w.append(self.float_range(0.0, 1.0, group="scores"))
        final_w.append(self.float_range(0.0, 1.0, group="scores"))
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind(explore_w),
                final=final_score.bind(final_w),
            )
        )
        scout_policy = self._build_light_scout_policy()
        tail_policy = self._build_heavy_tail_policy()
        if self.use_heavy_todd:
            self._install_policy(tail_policy)
        else:
            self._install_policy(scout_policy)

    def _build_light_scout_policy(self):
        group = "scout"
        samples = SamplingBudget(one_hot="all", sparse=0, dense=16, sparse_max_weight=2)
        return (
            ActionSelection(beamwidth=2, mode="softmax", temperature=0.22),
            ActionPool(final_size=self.int_range(14, 28, group=group)),
            TohpeSearch(samples, SourcePool(keep=self.int_range(3, 8, group=group), reserve=1), z_choices=3, target_min_red=TARGET_MIN_RED, target_max_red=TARGET_MAX_RED),
            ToddSearch(
                SamplingBudget(one_hot=8, sparse=2, dense=2, sparse_max_weight=2),
                SourcePool(keep=3, reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(
                    min_buckets=8, max_buckets=self.int_range(512, 2000, group=group), limit_bucket=2000
                ),
            ),
        )

    def _build_heavy_tail_policy(self):
        group = "tail"
        samples = SamplingBudget(
            one_hot="all",
            sparse=self.int_range(4, 16, group=group),
            dense=self.int_range(4, 24, group=group),
            sparse_max_weight=3,
        )
        cap = self.int_range(8000, 36000, group=group)
        todd_keep = self.int_range(6, 16, group=group)
        actions_per_bucket = self.int_range(2, 4, group=group)
        min_buckets = self.int_range(512, 4096, group=group)
        return (
            ActionSelection(beamwidth=3, mode="softmax", temperature=0.12),
            ActionPool(final_size=self.int_range(28, 52, group=group)),
            TohpeSearch(samples, SourcePool(keep=8, reserve=2), z_choices=4, target_min_red=TARGET_MIN_RED, target_max_red=TARGET_MAX_RED),
            ToddSearch(
                samples,
                SourcePool(keep=todd_keep, reserve=2),
                actions_per_bucket=actions_per_bucket,
                buckets=ZBucketSearch(min_buckets=min_buckets, max_buckets=cap, limit_bucket=cap),
            ),
        )

    def _install_policy(self, policy):
        selection, action_pool, tohpe, todd = policy
        self.set_action_selection(selection)
        self.set_action_pool(action_pool)
        self.set_tohpe_search(tohpe)
        self.set_todd_search(todd)

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, SEEDS), dtype=float)
        return float(values.min() + 0.02 * values.std())


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
        PSO(pop_size=SCOUT_PARTICLES, w=0.7, c1=1.4, c2=1.4, adaptive=False),
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    best = np.asarray(result.X if result.X is not None else current, dtype=float)
    evaluator.insert(best)
    evaluator.reinit()
    return best


def optimize_pattern_search(evaluator: Evaluator, evaluations: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    if len(current) == 0:
        evaluator([])
        return current
    result = minimize(
        Problem(evaluator),
        PatternSearch(x0=np.clip(current, LOWER_BOUND, UPPER_BOUND), init_delta=0.25, init_rho=0.5),
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
    evaluator.select_parameter_groups("scores", "scout")
    scout_params = optimize_pso(evaluator, SCOUT_EVALS, seed=40)
    evaluator.use_heavy_todd = True
    restarted = evaluator.set_up_new_init(0, rank_thr=evaluator.best_rank + REOPEN_MARGIN, xopt=scout_params)
    if restarted is None:
        return evaluator.get_best()
    evaluator.select_parameter_groups("scores", "tail")
    optimize_pattern_search(evaluator, TAIL_EVALS, seed=44)
    return evaluator.get_best()
