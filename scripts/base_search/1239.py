from typing import Iterable

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from helper import (
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    ExplorationScore,
    FinalizationScore,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
)


INITIAL_RANK = 1701
TAIL_RANK = 1300
SEEDS = [7, 8]
SCHEDULE = [INITIAL_RANK, TAIL_RANK]


def signed(x: float) -> float:
    return float(3.0 * np.tanh(float(x)))


def rank_score(ranks: list[int]) -> float:
    return float(min(ranks)) if ranks else float(INITIAL_RANK)


class Evaluator(BaseEvaluator):
    """Cheap finite-cap scout refiner: load a small-limit saved path and reallocate
    expensive full-tail z-work into more optimizer evals and multi-seed robustness.
    """

    def policy_mapping(self):
        # preserve same number of tunable score entries but schedule a focused tail
        explore = [self.map_par(signed) for _ in range(5)]

        # finalization: keep 6 entries; set tail tohpe weight to 0.0 to avoid TOHPE bias
        final_early = [self.map_par(signed) for _ in range(6)]
        final_tail = [self.map_par(signed) for _ in range(5)] + [0.0]

        # schedule scores: early vs tail
        self.set_scores(
            SCHEDULE,
            [
                PolicyScores(ExplorationScore(explore, pow=1), FinalizationScore(final_early, pow=1)),
                PolicyScores(ExplorationScore(explore, pow=1), FinalizationScore(final_tail, pow=1)),
            ],
        )

        # keep action expansion low for a cheap scout (avoid expensive branching)
        self.set_action_selection(SCHEDULE, [ActionSelection(count=1, mode="best", temperature=0.0), ActionSelection(count=1, mode="best", temperature=0.0)])

        # moderate pool sizes for scout
        self.set_action_pool(SCHEDULE, [ActionPool(final_size=40), ActionPool(final_size=48)])

        # TOHPE: heavy early scouting (cheap) and moderate tail TOHPE reserve
        self.set_tohpe_search(
            SCHEDULE,
            [
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=56, dense=56, sparse_max_weight=4),
                    SourcePool(keep=36, reserve=0),
                    z_choices=12,
                ),
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=16, dense=8, sparse_max_weight=4),
                    SourcePool(keep=36, reserve=6),
                    z_choices=12,
                ),
            ],
        )

        # TODD: light early TODD to seed z diversity; capped finite-tail for cheap scout
        self.set_todd_search(
            SCHEDULE,
            [
                ToddSearch(
                    SamplingBudget(one_hot=16, sparse=4, dense=0, sparse_max_weight=3),
                    # small but nonzero early keep/reserve to seed diverse z without heavy cost
                    SourcePool(keep=4, reserve=1),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=128, max_buckets=1024, limit_bucket=1024),
                ),
                ToddSearch(
                    SamplingBudget(one_hot=56, sparse=24, dense=12, sparse_max_weight=4),
                    # modest tail retention to capture rare TODD actions within the finite cap
                    SourcePool(keep=30, reserve=10),
                    actions_per_bucket=6,
                    # finite positive cap (1024) to keep per-eval cost affordable for a scout
                    buckets=ZBucketSearch(min_buckets=256, max_buckets=1024, limit_bucket=1024),
                ),
            ],
        )

        # modest beam to allow limited branching without huge cost
        self.set_beamsearch_width(SCHEDULE, [1, 2])

    def __call__(self, params: Iterable[float]) -> float:
        # evaluate ensemble across configured SEEDS for robustness (min over seeds)
        ranks = self.run(params, SEEDS)
        return rank_score(ranks)


class Problem(ElementwiseProblem):
    def __init__(self, fun: "Evaluator"):
        n = len(fun.extract_active())
        super().__init__(n_var=n, n_obj=1, xl=np.full(n, -2.0), xu=np.full(n, 2.0))
        self.fun = fun

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fun(np.asarray(x, dtype=float))


def optimize(fun: Evaluator, n_eval: int) -> None:
    if len(fun.extract_active()) == 0:
        fun.run([], SEEDS)
        return
    algorithm = PSO(pop_size=8, w=0.6, c1=0.6, c2=0.4, adaptive=False)
    minimize(Problem(fun), algorithm, termination=("n_eval", n_eval), seed=5, verbose=True)


def entrypoint():
    # load a small-limit scout path and reopen nearer the tail to focus finite-cap refinement
    fun = Evaluator(path_name="i1701_m1701_f1247_bf4868d2", margin=60)
    # cap per-eval z-work (limit_bucket=1024) keeps evals cheap; increase PSO evals for wider mapping
    optimize(fun, n_eval=30)
    return fun.get_best()

entrypoint()