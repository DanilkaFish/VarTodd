from collections.abc import Iterable
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
import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_pso"
SEEDS = [23, 24, 25]
SCOUT_EVALS = 144
MID_REFINE_EVALS = 48
MID_REOPEN_MARGIN = 96


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
                ExplorationScore([self.float_range(-4, 4) for _ in range(5)], centers=[0.0, 0.0, 0.0, 0.0, 0.0], pow=1),
                FinalizationScore(
                    [self.float_range(-4, 4) for _ in range(6)], centers=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], pow=1
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
