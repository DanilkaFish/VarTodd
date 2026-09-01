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
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_pso"
SEEDS = [23, 24, 25]
SCOUT_EVALS = 144
MID_REFINE_EVALS = 48
MID_REOPEN_MARGIN = 96


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
    """Reduction gated on the basis having room to sample.

    A binary null space of dimension d holds only 2**d - 1 distinct y vectors,
    so a high-reduction candidate from a dim-1 basis is a dead end: nothing
    else can be drawn from it. The gate multiplies the reduction reward by a
    smooth ramp on dim, which a weighted sum cannot do -- it can only add a
    dim term that a large reduction outweighs.
    """
    room = fn.sigmoid((k.ndim - fn.abs(p.w(2))) * 4.0)
    return k.nred * p.w(0) * room + k.ndim * p.w(1) - k.nzw * fn.abs(p.w(3))


@policy.final
def final_score(k, p, fn):
    """Rewards being distinct from the rest of the pool, not just being best.

    With beamwidth 3 at temperature 0.35 the selection is stochastic across the
    top of the pool, so three near-identical actions waste two of the three
    branches. nrank_score is this candidate's position in the population by
    pool score; the log term is flat near the top and falls away slowly, which
    keeps several genuinely different candidates in contention instead of
    concentrating all the mass on one.
    """
    spread = fn.log(k.nrank_score + 0.05) * p.w(1)
    return (k.nred * p.w(0)
            + spread
            + k.ntohpe * p.w(2)
            + fn.where(k.dim > 3, k.ndim * p.w(3), k.nred * p.w(4)))


class Evaluator(BaseEvaluator):
    """Use beam width to compare several low-cap TODD branches."""

    def float_range(self, low: float, high: float) -> float:
        return self.map_par(lambda x: low + (high - low) / (1.0 + np.exp(-np.clip(float(x), -8.0, 8.0) / 2.5)))

    def int_range(self, low: int, high: int) -> int:
        return self.map_par(
            lambda x: min(high, low + int((high - low + 1) / (1.0 + np.exp(-np.clip(float(x), -8.0, 8.0) / 2.5))))
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
        todd_samples = SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2)
        todd_cap = self.int_range(1024, 6000)
        self.set_action_selection(ActionSelection(beamwidth=3, mode="softmax", temperature=0.35))
        self.set_action_pool(ActionPool(final_size=self.int_range(24, 48)))
        self.set_tohpe_search(
            TohpeSearch(
                SamplingBudget(one_hot=14, sparse=0, dense=0, sparse_max_weight=2),
                SourcePool(keep=6, reserve=1),
                z_choices=3,
                target_min_red=TARGET_MIN_RED,
                target_max_red=TARGET_MAX_RED,
            )
        )
        self.set_todd_search(
            ToddSearch(
                todd_samples,
                SourcePool(keep=self.int_range(6, 12), reserve=2),
                actions_per_bucket=3,
                buckets=ZBucketSearch(min_buckets=50, max_buckets=todd_cap, limit_bucket=todd_cap),
            )
        )

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, SEEDS), dtype=float)
        return float(values.min() + 0.02 * values.std())


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        n = len(evaluator.extract_active())
        super().__init__(n_var=n, n_obj=1, xl=np.full(n, -1.0), xu=np.full(n, 1.0))
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def optimize(evaluator: Evaluator, evaluations: int, seed: int) -> np.ndarray:
    algorithm = PSO(pop_size=12, w=0.7, c1=0.4, c2=0.4, adaptive=False)
    result = minimize(Problem(evaluator), algorithm, termination=("n_eval", evaluations), seed=seed, verbose=False)
    return np.asarray(result.X if result.X is not None else evaluator.extract_active(), dtype=float)


def entrypoint():
    evaluator = Evaluator(path_name="init", max_depth=500)
    params = optimize(evaluator, SCOUT_EVALS, seed=23)
    params = evaluator.set_up_new_init(0, rank_thr=evaluator.best_rank + MID_REOPEN_MARGIN, xopt=params)
    if params is not None:
        optimize(evaluator, MID_REFINE_EVALS, seed=26)
    return evaluator.get_best()
