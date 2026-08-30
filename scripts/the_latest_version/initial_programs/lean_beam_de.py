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
SEEDS = [17, 18]
N_EVAL = 288


@policy.exploration
def explore_score(k, p, fn):
    """Saturating reduction reward, plus a sparsity preference.

    tanh(nred * gain) flattens once a candidate is clearly good, so the pool
    stops spending its capacity separating two large reductions and keeps
    discriminating among the many small ones -- which is where a beam of two
    actually has a choice to make. p.w(1) is inside the tanh, so the optimizer
    controls where saturation sets in rather than only how much it is worth.

    Sparse y and z vectors are preferred multiplicatively: a dense candidate is
    damped rather than merely penalized by an additive term it can outweigh
    with a large reduction.
    """
    saturating = fn.tanh(k.nred * fn.abs(p.w(1)) + 0.01) * p.w(0)
    sparsity = fn.sigmoid(p.w(2) - (k.nyw + k.nzw) * fn.abs(p.w(3)))
    return saturating * sparsity + k.ndim * p.w(4)


@policy.final
def final_score(k, p, fn):
    """Reduction traded against the size of the bucket that produced it.

    Written on the raw knobs: a ratio of two normalized knobs would carry the
    same 1/bn factor above and below and cancel it, leaving a pointless
    division. red/bucket is the quantity actually wanted -- reduction per unit
    of bucket -- and it is a quotient no weighted sum over the two can express.
    """
    realized = k.red / fn.max(k.bucket, 1.0)
    return (k.nred * p.w(0)
            + realized * p.w(1)
            + fn.sqrt(fn.max(k.ndim, 0.0)) * p.w(2)
            + k.ntohpe * p.w(3)
            - k.nzw * fn.abs(p.w(4)))


class Evaluator(BaseEvaluator):
    def float_range(self, low: float, high: float) -> float:
        return self.map_par(lambda x: low + (high - low) * np.clip((float(x) + 1.5) / 3.0, 0.0, 1.0))

    def int_range(self, low: int, high: int) -> int:
        return self.map_par(
            lambda x: min(high, low + int((high - low + 1) * np.clip((float(x) + 1.5) / 3.0, 0.0, 1.0)))
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
        samples = SamplingBudget(one_hot="all", sparse=8, dense=self.int_range(8, 40), sparse_max_weight=2)
        todd_cap = self.int_range(512, 6000)
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.16))
        self.set_action_pool(ActionPool(final_size=self.int_range(12, 28)))
        self.set_tohpe_search(TohpeSearch(samples, SourcePool(keep=6, reserve=1), z_choices=4))
        self.set_todd_search(
            ToddSearch(
                SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
                SourcePool(keep=self.int_range(2, 5), reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(min_buckets=16, max_buckets=todd_cap, limit_bucket=todd_cap),
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
    minimize(Problem(evaluator), algorithm, termination=("n_eval", N_EVAL), seed=17, verbose=False)
    return evaluator.get_best()
