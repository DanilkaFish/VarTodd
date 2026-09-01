"""Piecewise scores: a comparison chooses which weight vector applies.

Every existing seed scores a candidate with one fixed formula. This one asks a
question about the candidate first -- how large its binary null space is, how
contested its reduction is -- and applies a different weight vector on each
side. The band boundary is itself searched, so the optimizer chooses where the
regime changes rather than inheriting a constant.

Both arms of a `fn.where` are always evaluated (they compile to branchless
selects), so this costs the same as scoring both formulas; it buys the ability
to rank by different criteria in different regions, which no single weighted
sum can express.
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

OPTIMIZER_FAMILY = "pymoo_de_piecewise_gate"
SEEDS = [61, 62]
N_EVAL = 264
DIM_SPLIT = 4


@policy.exploration
def explore_score(k, p, fn):
    """Two weight vectors, selected by the dimension of the binary null space.

    A high-dim band has many y vectors to recall, so sparsity of y is the
    scarce thing worth ranking on. A low-dim band has almost no y choice left
    (dim 1-3 admits only 1-7 nonzero vectors), so the useful signal moves to
    the z side -- which bucket was opened at all. Ranking both regimes with one
    weight on `nyw` forces a compromise that fits neither.
    """
    wide_band = k.dim >= DIM_SPLIT
    wide = k.nred * p.w(0) + k.nyw * fn.abs(p.w(1))
    narrow = k.nred * p.w(2) + k.nbucket * p.w(3)
    return fn.where(wide_band, wide, narrow) + k.ndim * p.w(4)


@policy.final
def final_score(k, p, fn):
    """Different criteria for a contested pool and an uncontested one.

    `nrank_red` is this candidate's reduction standing as a fraction of the
    pool. When many candidates are tied near the top (a small nrank_red for
    several of them), raw reduction has stopped separating them and the
    lookahead term deserves the weight. When the field is spread out, reduction
    is still informative and the dimension standing is what to penalize. The
    threshold p.w(0) is searched, not fixed.
    """
    contested = k.nrank_red <= p.w(0)
    tight = k.nred * p.w(1) + k.ntohpe * p.w(2)
    spread = k.nred * p.w(3) - k.nrank_dim * fn.abs(p.w(4))
    return fn.where(contested, tight, spread)


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
        # p.w(0) of the finalization score is a threshold on nrank_red, which is
        # a fraction of the pool: search it on the scale it is compared against.
        final_w[0] = self.float_range(0.0, 1.0)
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind(explore_w),
                final=final_score.bind(final_w),
            )
        )
        samples = SamplingBudget(
            one_hot="all", sparse=6, dense=self.int_range(4, 28), sparse_max_weight=2
        )
        todd_cap = self.int_range(768, 8000)
        self.set_action_selection(
            ActionSelection(beamwidth=3, mode="softmax", temperature=0.18)
        )
        self.set_action_pool(ActionPool(final_size=self.int_range(16, 36)))
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
        Problem(evaluator), algorithm, termination=("n_eval", N_EVAL), seed=61, verbose=False
    )
    return evaluator.get_best()
