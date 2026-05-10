from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Iterable, Optional

import numpy as np
import pyswarms as ps

from scripts.optimization_core.helper import (
    BaseEvaluator,
    ExplorationScore,
    FinalizationScore,
    Matrix,
)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _signed(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return float(np.tanh(x / 1.5))


def _normalize(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-9:
        arr /= norm
    return arr.tolist()


def _make_seeds(count: int, base_seed: int) -> list[int]:
    rng = np.random.default_rng(base_seed)
    return [int(seed) for seed in rng.integers(1, 10000, size=max(1, int(count)))]


class Evaluator(BaseEvaluator):
    def __init__(self, *args, seeds: Iterable[int], enable_todd: bool = True, **kwargs):
        self.seeds = list(seeds)
        self.enable_todd = bool(enable_todd)
        super().__init__(*args, **kwargs)

    def _pool_score(self) -> ExplorationScore:
        weights = _normalize(
            [
                -1.0 + 2 * self.map_par(_sigmoid, 0),
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
                0.0,
            ]
        )
        return ExplorationScore(weights=weights, centers=[0.0] * 5, pow=1)

    def _final_score(self) -> FinalizationScore:
        weights = _normalize(
            [
                -1.0 + 2 * self.map_par(_sigmoid, 0),
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
                0.0,
                0.0,
            ]
        )
        return FinalizationScore(weights=weights, centers=[0.0] * 6, pow=1)

    def policy_mapping(self):
        pool_score = self._pool_score()
        final_score = self._final_score()

        sample_budget = self.map_par(_sigmoid, 0)
        breadth_budget = self.map_par(_sigmoid, 0)
        temp_bias = self.map_par(_sigmoid, 0)
        pool_budget = self.map_par(_sigmoid, 0)
        single_budget = self.map_par(_sigmoid, 0)
        todd_budget = self.map_par(_sigmoid, 0)

        self.set_pool_scores(pool_score)
        self.set_final_scores(final_score)
        self.set_min_pool_size(1)
        self.set_temperature(0.01 + 0.09 * temp_bias)
        self.set_num_samples(8 + int(14 * sample_budget))
        self.set_max_pool_size(2 + int(2 * pool_budget))
        self.set_max_tohpe(5 + int(3*sample_budget))
        self.set_try_only_tohpe(1)
        self.set_enable_tohpe(1)
        self.set_enable_todd(int(self.enable_todd))
        self.set_max_reduction(128)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(4 + int(12 * single_budget))
        self.set_tohpe_num_best(2 + int(2 * breadth_budget))
        self.set_gen_part(0.8 + 0.2 * breadth_budget)
        self.set_todd_width(1)
        self.set_beamsearch_width(1)

        if self.enable_todd:
            self.set_min_z_to_research(10000 + int(30000 * todd_budget))
            self.set_max_z_to_research(1000000 + int(6000000 * todd_budget))
        else:
            self.set_min_z_to_research(0)
            self.set_max_z_to_research(0)

    def evaluate(self, params: Iterable[float], max_workers: int = 1) -> float:
        ranks = self.run(params, self.seeds, max_workers=max_workers)
        if not ranks:
            return float("inf")
        arr = np.asarray(ranks, dtype=float)
        return float(np.quantile(arr, 0.60) + 0.01 * arr.std())

    def __call__(self, params: Iterable[float]):
        return self.evaluate(params, max_workers=1)


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


def _evaluate_pso_particle(fun: Evaluator, position: np.ndarray, seed_workers: int):
    local_fun = _clone_for_pso_eval(fun)
    try:
        cost = local_fun.evaluate(position, max_workers=seed_workers)
        return cost, local_fun
    finally:
        local_fun.close_workers()


def run_pso(
    fun: Evaluator,
    *,
    iters: int,
    particles: int,
    workers: Optional[int] = None,
    seed_workers: int = 1,
) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    if workers is None:
        workers = _int_env("VARTODD_PSO_WORKERS", min(particles, os.cpu_count() or 1))
    workers = max(1, min(int(workers), particles))
    seed_workers = max(1, int(seed_workers))
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def objective(positions):
        if workers == 1:
            results = [_evaluate_pso_particle(fun, np.asarray(pos, dtype=float), seed_workers) for pos in positions]
        else:
            futures = [
                executor.submit(_evaluate_pso_particle, fun, np.asarray(pos, dtype=float), seed_workers)
                for pos in positions
            ]
            results = [future.result() for future in futures]
        for _, local_fun in results:
            fun.merge_run_state_from(local_fun)
        return np.asarray([cost for cost, _ in results], dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.55, "c2": 0.45, "w": 0.72},
        bounds=(np.full(n_params, -1.0), np.full(n_params, 1.0)),
    )

    _, best_position = optimizer.optimize(objective, iters=iters, verbose=True)
    return np.asarray(best_position, dtype=float)


def entrypoint(mat: Matrix):
    fun = Evaluator(
        mat=mat,
        max_depth=_int_env("VARTODD_MAX_DEPTH", 1800),
        seeds=_make_seeds(_int_env("VARTODD_EVAL_SEEDS", 1), _int_env("VARTODD_SEED", 40)),
        enable_todd=0,
    )
    fun = Evaluator(
        path_name="i5103_m5103_f3771_d78840cd52f1", 
        init_rank_thr=3800,
        max_depth=_int_env("VARTODD_MAX_DEPTH", 1800),
        seeds=_make_seeds(_int_env("VARTODD_EVAL_SEEDS", 2), _int_env("VARTODD_SEED", 40)),
        enable_todd=0,
        )
    # xopt = run_pso(
    #     fun,
    #     iters=_int_env("VARTODD_PSO_ITERS", 32),
    #     particles=_int_env("VARTODD_PSO_PARTICLES", 4),
    #     seed_workers=_int_env("VARTODD_SEED_WORKERS", 1),
    # )
    # fun(xopt)

    # if fun.best_paths and _int_env("VARTODD_REFINE_ONCE", 1):
    #     best_rank = fun.best_paths[0].final_node.state.rows
    #     rank_thr = max(best_rank + 1, (fun.init_rank + best_rank) // 2)
    #     x_active = fun.set_up_new_init(0, rank_thr=rank_thr, xopt=xopt)
    fun.enable_todd=1
    # if x_active is not None:
    xopt = run_pso(
        fun,
        iters=_int_env("VARTODD_PSO_REFINE_ITERS", 2),
        particles=_int_env("VARTODD_PSO_REFINE_PARTICLES", 2),
        seed_workers=_int_env("VARTODD_SEED_WORKERS", 2),
    )
    fun(xopt)

    return fun.get_best()
