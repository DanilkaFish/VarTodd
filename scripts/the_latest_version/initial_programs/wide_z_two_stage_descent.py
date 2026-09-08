"""Frozen two-stage wide-z TOHPE policy for an expensive descent."""

from helper import (
    INITIAL_RANK, TARGET_FINAL_RANK, ActionPool, ActionSelection,
    BaseEvaluator, PolicyScores, SamplingBudget, SourcePool, ToddSearch,
    TohpeSearch, ZBucketSearch, policy,
)

SEARCH_STRATEGY = "fixed_agnostic_two_stage_wide_z"
TARGET_POLICY_EVALUATIONS = 1
SEEDS = (37, 41, 53)
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
TAKEOVER_RANK = TARGET_FINAL_RANK + round(0.12 * RANK_SPAN)
MAX_DEPTH = max(500, RANK_SPAN + 64)


@policy.exploration
def explore_score(k, p, fn):
    return fn.where(k.nred < 0.05, k.nbucket * p.w(0), k.nred * p.w(1)) + k.ndim * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.ntohpe * p.w(2)


def disabled_todd():
    return ToddSearch(SamplingBudget(0, 0, 0, 0), SourcePool(0, 0), 0, ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0))


class Evaluator(BaseEvaluator):
    def policy_mapping(self):
        self.set_scores(PolicyScores(
            exploration=explore_score.bind([1.0, 2.0, 2.0, 0.0, 0.0]),
            final=final_score.bind([2.0, 1.5, 1.0]),
        ))
        early = TohpeSearch(SamplingBudget("all", 0, 0, 2), SourcePool(15, 8), 10, 4, 8)
        late = TohpeSearch(SamplingBudget("all", 0, 0, 0), SourcePool(32, 16), 64, 2, 6)
        self.set_action_pool(ActionPool(final_size=45))
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.30))
        self.set_tohpe_search([INITIAL_RANK, TAKEOVER_RANK], [early, late])
        self.set_todd_search(disabled_todd())


def entrypoint():
    evaluator = Evaluator(path_name="init", fin_rank=TARGET_FINAL_RANK, max_depth=MAX_DEPTH)
    evaluator.run(evaluator.extract_active(), SEEDS)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: wide-z two-stage descent. Cost: one three-seed trajectory evaluation with no optimizer. Strategy: remove saturated terminal sampling and spend the late budget on broad TOHPE z coverage."""
