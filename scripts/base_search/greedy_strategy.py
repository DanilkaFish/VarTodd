from typing import Iterable
import numpy as np
from scripts.optimization_core.helper import (
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    ExplorationScore,
    FinalizationScore,
    Matrix,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
    find_rank,
    get_matrix,
)
import random

np.random.seed(42)
random.seed(40)

def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0/beta) * np.log(np.exp(-beta*(xs - m)).sum()))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(4)]

    def policy_mapping(self):
        lr = 0
        ranks = [lr]
        w_pool = [ExplorationScore([1,0,0,0,0],
                                  pow=1) for r in  ranks]
        w_final = [FinalizationScore([1,0,0,0,0,0],
                                  pow=1
                                  ) for r in ranks]
        sampling = SamplingBudget(one_hot=15, sparse=0, dense=15, sparse_max_weight=8)
        self.set_scores(ranks, [PolicyScores(exploration=w_pool[0], final=w_final[0])])
        self.set_action_selection(ActionSelection(count=1, mode="best", temperature=0.0))
        self.set_action_pool(ActionPool(final_size=1))
        self.set_tohpe_search(TohpeSearch(sampling=sampling, pool=SourcePool(keep=1, reserve=0), z_choices=1))
        self.set_todd_search(
            ToddSearch(
                sampling=sampling,
                pool=SourcePool(keep=0, reserve=0),
                actions_per_bucket=1,
                buckets=ZBucketSearch(min_buckets=100000, max_buckets=100000),
            )
        )
        self.set_widths(actions=1, beam=1)

    def __call__(self, params: Iterable):
        tcounts = self.run(params, self.seeds, max_workers=1)
        return min(tcounts) 

def entrypoint(mat: Matrix):
    fun = Evaluator(mat=mat, fin_rank=170, max_depth=200)
    x = fun.extract_active()
    fun(x)
    return fun.get_best(), fun.tcount
