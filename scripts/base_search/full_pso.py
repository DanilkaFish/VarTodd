import numpy as np
import pyswarms as ps
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Iterable
from tqdm.auto import tqdm
from scripts.optimization_core.helper import (
    BaseEvaluator,
    ExplorationScore,
    FinalizationScore,
    Matrix,
)

np.random.seed(42)
random.seed(40)

def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))

def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0/beta) * np.log(np.exp(-beta*(xs - m)).sum()))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(2)]
    validation_seeds = [random.randint(1, 10000) for _ in range(12)]

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float):
        one_hot = max(0, int(round(sample_count * one_hot_fraction)))
        dense = max(0, int(sample_count) - one_hot)
        return [one_hot, 0, dense]

    def policy_mapping(self):
        ranks = [0]
        self.set_pool_scores(ranks, [ExplorationScore([self.map_par(_w_tanh) for _ in range(5)], pow=1)])
        final_weights = [self.map_par(_w_tanh) for _ in range(6)]
        final_centers = [self.map_par(sigmoid) for _ in range(6)]
        self.set_final_scores(ranks, [FinalizationScore(final_weights, final_centers, pow=1)])

        z_budget = self.map_par(sigmoid)
        sample_budget = self.map_par(sigmoid)
        one_hot_fraction = 0.1 + 0.5 * self.map_par(sigmoid)
        sample_count = 8 + int(48 * sample_budget)
        sample_caps = self._sample_caps(sample_count, one_hot_fraction)

        self.set_min_pool_size(1)
        self.set_min_z_to_research(10 + int(250 * z_budget))
        self.set_temperature(0.2)
        self.set_tohpe_vector_samples(sample_caps)
        self.set_todd_vector_samples(sample_caps)
        self.set_sparse_max_weight(2)
        self.set_max_pool_size(12)
        self.set_tohpe_pool_size(2)
        self.set_todd_pool_size(12)
        self.set_min_tohpe_actions(0)
        self.set_min_todd_actions(0)
        self.set_max_reduction(12)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(1)
        self.set_tohpe_sample(2)
        self.set_bucket_temperature(0.0)
        self.set_bucket_random_fraction(0.0)
        self.set_max_per_signature(0)
        self.set_todd_width(1)
        self.set_beamsearch_width(1)

    def evaluate(self, params: Iterable[float], max_workers: int = 1, seeds: Iterable[int] | None = None) -> float:
        seeds = list(self.seeds if seeds is None else seeds)
        tcounts = self.run(params, seeds, max_workers=max_workers)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.02 * spread
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
    cost = local_fun.evaluate(np.asarray(position, dtype=float), seeds=seeds)
    return float(cost), local_fun

def _unique_positions(items, limit: int):
    out = []
    seen = set()
    for cost, position in items:
        pos = np.asarray(position, dtype=float).copy()
        key = tuple(np.round(pos, 8))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(cost), pos))
        if len(out) >= limit:
            break
    return out

def _validate_top_positions(fun: Evaluator, candidates):
    best_cost = float("inf")
    best_position = None
    for _, position in tqdm(candidates, desc="validating policies", total=len(candidates), leave=True):
        cost, local_fun = _score_position(fun, position, seeds=fun.validation_seeds)
        fun.merge_run_state_from(local_fun)
        if cost < best_cost:
            best_cost = cost
            best_position = np.asarray(position, dtype=float).copy()
    return best_position, best_cost

def run_opt(fun: Evaluator, num_eval: int=10, label: str = "pso") -> np.ndarray:
    x = fun.extract_active()
    n_params = len(x)
    bounds = (np.full(n_params, -1.0), np.full(n_params, 1.0))
    options = {
        'c1': 0.4,
        'c2': 0.4,
        'w': 0.7,
    }

    workers = 16
    search_history = []

    print(f"{label}: optimizing {n_params} params for {num_eval} iterations")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        def objective(positions):
            positions = [np.asarray(pos, dtype=float) for pos in positions]
            costs = np.asarray(
                list(executor.map(lambda pos: _score_position(fun, pos)[0], positions)),
                dtype=float,
            )
            search_history.extend((float(cost), pos.copy()) for cost, pos in zip(costs, positions))
            return costs

        optimizer = ps.single.GlobalBestPSO(
            n_particles=8,
            dimensions=n_params,
            options=options,
            bounds=bounds,
        )
        best_cost, best_position = optimizer.optimize(
            objective,
            iters=num_eval,
            verbose=True
        )

    validation_top_k = 8
    ranked = sorted(search_history + [(float(best_cost), np.asarray(best_position, dtype=float).copy())],
                    key=lambda item: item[0])
    validation_candidates = _unique_positions(ranked, validation_top_k)
    validated_position, validated_cost = _validate_top_positions(fun, validation_candidates)
    print(f"validation best cost={validated_cost:.3f} over {len(validation_candidates)} policies "
          f"and {len(fun.validation_seeds)} seeds")
    return validated_position

def evaluate_final(fun: Evaluator, x: np.ndarray, label: str):
    print(f"{label}: evaluating selected policy on {len(fun.validation_seeds)} validation seeds")
    return fun(x)
        
def entrypoint(mat: Matrix):
    fun = Evaluator(mat=mat, fin_rank=170, max_depth=100)
    num_eval = 50
    x0 = run_opt(fun, num_eval, label="initial pso")
    evaluate_final(fun, x0, "initial final")
    print(f"{fun.best_pathes[0].final_node.state.rows=}")

    best_rank = fun.best_pathes[0].final_node.state.rows
    mid_rank_thr = max(best_rank + 1, (fun.init_rank + best_rank) // 2)
    late_rank_thr = max(best_rank + 1, best_rank + (mid_rank_thr - best_rank) // 2)

    for stage, rank_thr in enumerate((mid_rank_thr, late_rank_thr), start=1):
        print(f"restart {stage}: setup_new_init at rank_thr={rank_thr}")
        x_active = fun.set_up_new_init(0, rank_thr=rank_thr, xopt=None)
        if x_active is None:
            print(f"restart {stage}: no branch found")
            break
        x0 = run_opt(fun, 30, label=f"restart {stage} pso")
        evaluate_final(fun, x0, f"restart {stage} final")
    return fun.get_best()
