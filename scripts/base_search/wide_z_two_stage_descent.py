"""Frozen two-stage wide-z TOHPE policy for an expensive descent."""

from scripts.optimization_core.helper import (
    TARGET_FINAL_RANK, ActionPool, ActionSelection, BaseEvaluator, Matrix,
    SamplingBudget, SourcePool, ToddSearch, TohpeSearch, ZBucketSearch,
    policy,
)

SEEDS = (37,)


@policy.exploration
def explore_score(k, p, fn):
    return fn.where(k.nred < 0.05, k.nbucket * p.w(0), k.nred * p.w(1)) + k.ndim * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1)


def disabled_todd():
    return ToddSearch(SamplingBudget(0, 0, 0, 0), SourcePool(0, 0), 0, ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0))


class Evaluator(BaseEvaluator):
    def policy_mapping(self):
        self.set_scores({
            "exploration": explore_score.bind([1.0, 2.0, 2.0, 0.0, 0.0]),
            "final": final_score.bind([2.0, 1.5]),
        })
        early = TohpeSearch(SamplingBudget(10, 0, 0, 2), SourcePool(15, 8), 10, 4, 8)
        late = TohpeSearch(SamplingBudget("all", 0, 0, 0), SourcePool(32, 16), 64, 2, 6)
        self.set_action_pool(ActionPool(final_size=8))
        self.set_action_selection(ActionSelection(count=2, mode="softmax", temperature=0.30))
        rank_span = max(1, self.init_rank - self.target_final_rank)
        takeover_rank = self.target_final_rank + round(0.12 * rank_span)
        self.set_tohpe_search([self.init_rank, takeover_rank], [early, late])
        self.set_todd_search(disabled_todd())


def entrypoint(mat: Matrix):
    rank_span = max(1, int(mat.rows) - TARGET_FINAL_RANK)
    max_depth = max(500, rank_span + 64)
    evaluator = Evaluator(mat=mat, max_depth=max_depth)
    evaluator.run(evaluator.extract_active(), SEEDS)
    return evaluator.get_best()
