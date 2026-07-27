import numpy as np
import pyswarms as ps
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Iterable
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
    TohpePrefixSearch,
    TohpeSearch,
    ZBucketSearch,
)

np.random.seed(42)
random.seed(40)

PSO_PARTICLES = 36
PSO_WORKERS = 12
PSO_SCORE_SEEDS = 4
ROBUST_SPREAD_WEIGHT = 0.3
PSO_OPTIONS = {"c1": 0.9, "c2": 0.6, "w": 0.9}
# PSO_VELOCITY_CLAMP = (-0.25, 0.25)
PSO_INITIAL_ITERATIONS = 50
PSO_RESTART_ITERATIONS = 50
Z_MAX_BUCKET_CAP = 10_000
Z_LIMIT_BUCKET_CAP = 500_000
Z_MIN_CAP = 200
SAMPLE_COUNT_MAX = 50
TODD_RESERVE_MAX = 4

def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))

def robust_rank_cost(ranks: Iterable[float]) -> float:
    values = np.asarray(list(ranks), dtype=float)
    if values.size == 0:
        raise ValueError("at least one rank is required")
    return float(np.min(values) + ROBUST_SPREAD_WEIGHT * np.std(values))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def z_bucket_ranges(
    z_budget: float,
    z_research_budget: float,
    z_limit_budget: float,
    buckets_space: int,
) -> tuple[int, int, int]:
    space = max(1, int(buckets_space))
    research = float(np.clip(z_research_budget, 0.0, 1.0))
    maximum_cap = min(Z_MAX_BUCKET_CAP, space)
    limit_cap = min(Z_LIMIT_BUCKET_CAP, space)
    minimum = int(Z_MIN_CAP * float(np.clip(z_budget, 0.0, 1.0)))
    maximum = minimum + int((maximum_cap - 1) * research)
    limit = maximum + int((limit_cap - maximum) * z_limit_budget)
    return minimum, maximum, limit

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(PSO_SCORE_SEEDS)]
    validation_seeds = [random.randint(1, 10000) for _ in range(12)]

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float):
        return [int(one_hot_fraction*sample_count), 0, max(0, int(sample_count*(1-one_hot_fraction)))]

    def policy_mapping(self):
        ranks = [0]
        pool_score = ExplorationScore([self.map_par(_w_tanh) for _ in range(5)], pow=1)
        final_weights = [self.map_par(_w_tanh) for _ in range(6)]
        final_centers = [self.map_par(sigmoid) for _ in range(6)]
        final_score = FinalizationScore(final_weights, final_centers, pow=2)
        self.set_scores(ranks, [PolicyScores(exploration=pool_score, final=final_score)])

        z_budget = self.map_par(sigmoid)
        z_research_budget = self.map_par(sigmoid)
        z_limit_budget = self.map_par(sigmoid)
        todd_reserve_budget = self.map_par(sigmoid)
        sample_budget = self.map_par(sigmoid)
        one_hot_fraction = 0.1 + 0.8 * self.map_par(sigmoid)
        sample_count = 1 + int(SAMPLE_COUNT_MAX * sample_budget)
        sample_caps = self._sample_caps(sample_count, one_hot_fraction)
        z_min, z_max, z_limit = z_bucket_ranges(
            z_budget, z_research_budget, z_limit_budget, self.buckets_space
        )
        todd_reserve = min(TODD_RESERVE_MAX, int(TODD_RESERVE_MAX * todd_reserve_budget))

        tohpe_sampling = SamplingBudget(
            one_hot=sample_caps[0], sparse=sample_caps[1], dense=sample_caps[2], sparse_max_weight=2
        )
        todd_sampling = SamplingBudget(
            one_hot=sample_caps[0], sparse=sample_caps[1], dense=sample_caps[2], sparse_max_weight=2
        )
        self.set_action_selection(ActionSelection(count=2, mode="softmax", temperature=0.2))
        self.set_action_pool(ActionPool(final_size=18))
        self.set_tohpe_search(TohpeSearch(pool=SourcePool(keep=4, reserve=0), sampling=SamplingBudget(10, 0, 0), z_choices=4))
        self.set_tohpeprefix_search(
            TohpePrefixSearch(
                sampling=tohpe_sampling,
                pool=SourcePool(keep=12, reserve=todd_reserve),
                actions_per_bucket=2,
                buckets=ZBucketSearch(
                    min_buckets=z_min,
                    max_buckets=z_max,
                    limit_bucket=z_limit,
                ),
            )
        )
        self.set_todd_search(
            ToddSearch(
                sampling=todd_sampling,
                pool=SourcePool(keep=12, reserve=todd_reserve),
                actions_per_bucket=2,
                buckets=ZBucketSearch(
                    min_buckets=z_min,
                    max_buckets=z_max,
                    limit_bucket=z_limit,
                ),
            )
        )

    def evaluate(self, params: Iterable[float], max_workers: int = 1, seeds: Iterable[int] | None = None) -> float:
        seeds = list(self.seeds if seeds is None else seeds)
        tcounts = self.run(params, seeds, max_workers=max_workers)
        return robust_rank_cost(tcounts)
    def __call__(self, params: Iterable):
        return self.evaluate(params, max_workers=1, seeds=self.validation_seeds)
    
def _clone_for_pso_eval(fun: Evaluator) -> Evaluator:
    clone = fun.__class__.__new__(fun.__class__)
    clone.__dict__ = fun.__dict__.copy()
    clone.current_path = fun.current_path
    clone.dao = deepcopy(fun.dao)
    clone.todd = type(fun.todd)(clone.dao, fun.max_depth)
    clone.x0 = list(fun.x0)
    clone.active_params = list(fun.active_params)
    clone.best_paths = []
    clone.best_ranks = []
    clone.best_evals = []
    clone.best_eval = 0
    clone.total_eval = 0
    clone.tcount = []
    clone.best_seen = 0
    clone._best_rank = fun._best_rank
    clone._executor = None
    clone._executor_key = None
    return clone

def _score_position(fun: Evaluator, position, seeds=None) -> tuple[float, Evaluator]:
    local_fun = _clone_for_pso_eval(fun)
    try:
        cost = local_fun.evaluate(np.asarray(position, dtype=float), seeds=seeds)
    finally:
        local_fun.close_workers()
    return float(cost), local_fun

def run_opt(fun: Evaluator, num_eval: int=10, label: str = "pso") -> np.ndarray:
    x = fun.extract_active()
    n_params = len(x)
    bounds = (np.full(n_params, -1.0), np.full(n_params, 1.0))
    print(f"{label}: optimizing {n_params} params for {num_eval} iterations")
    with ThreadPoolExecutor(max_workers=PSO_WORKERS) as executor:
        def objective(positions):
            positions = [np.asarray(pos, dtype=float) for pos in positions]
            results = list(executor.map(lambda pos: _score_position(fun, pos), positions))
            for _, local_fun in results:
                fun.merge_run_state_from(local_fun)
            costs = np.asarray([cost for cost, _ in results], dtype=float)
            return costs

        optimizer = ps.single.GlobalBestPSO(
            n_particles=PSO_PARTICLES,
            dimensions=n_params,
            options=PSO_OPTIONS,
            bounds=bounds,
            # velocity_clamp=PSO_VELOCITY_CLAMP,
        )
        _, best_position = optimizer.optimize(
            objective,
            iters=num_eval,
            verbose=True
        )

    return np.asarray(best_position, dtype=float)
        
def entrypoint(mat: Matrix):
    fun = Evaluator(mat=mat, max_depth=1000)
    x0 = run_opt(fun, PSO_INITIAL_ITERATIONS, label="initial pso")
    print(f"{fun.best_paths[0].final_node.state.rows=}")

    best_rank = fun.best_paths[0].final_node.state.rows
    upper_rank_thr = max(best_rank + 1, best_rank + (fun.init_rank - best_rank) // 3 * 2)
    mid_rank_thr = max(best_rank + 1, (fun.init_rank + best_rank) // 2)
    late_rank_thr = max(best_rank + 1, (mid_rank_thr + best_rank) // 2)

    for stage, rank_thr in enumerate((upper_rank_thr, mid_rank_thr, late_rank_thr), start=1):
        print(f"restart {stage}: setup_new_init at rank_thr={rank_thr}")
        x_active = fun.set_up_new_init(0, rank_thr=rank_thr, xopt=None)
        if x_active is None:
            print(f"restart {stage}: no branch found")
            break
        x0 = run_opt(fun, PSO_RESTART_ITERATIONS, label=f"restart {stage} pso")
    return fun.get_best()
