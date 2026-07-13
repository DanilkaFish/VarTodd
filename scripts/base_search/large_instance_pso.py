from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Iterable, Optional

import numpy as np
import pyswarms as ps

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
)

MAX_DEPTH = 1800
EVAL_SEEDS = 1
BASE_SEED = 40
PSO_ITERS = 32
PSO_PARTICLES = 8
PSO_SEED_WORKERS = 2
PSO_WORKERS: Optional[int] = None
REFINE_ONCE = True
PSO_REFINE_ITERS = 8
PSO_REFINE_PARTICLES = 8
PSO_REFINE_SEED_WORKERS = 2


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
    def __init__(self, *args, seeds: Iterable[int], use_todd_stage: bool = True, **kwargs):
        self.seeds = list(seeds)
        self.use_todd_stage = bool(use_todd_stage)
        super().__init__(*args, **kwargs)

    def _pool_score(self) -> ExplorationScore:
        weights = _normalize(
            [
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
            ]
        )
        return ExplorationScore(weights=weights, centers=[0.0] * 5, pow=1)

    def _final_score(self) -> FinalizationScore:
        weights = _normalize(
            [
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
                self.map_par(_signed, 0),
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
        sparse_budget = self.map_par(_sigmoid, 0)
        dense_budget = self.map_par(_sigmoid, 0)
        bucket_budget = self.map_par(_sigmoid, 0)

        sparse_weight = 4 + int(20 * sparse_budget)
        self.set_scores(PolicyScores(exploration=pool_score, final=final_score))
        self.set_action_selection(ActionSelection(count=1, mode="softmax", temperature=0.01 + 0.09 * temp_bias))
        self.set_action_pool(ActionPool(final_size=10 + int(30 * pool_budget)))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=SamplingBudget(
                    one_hot=80 + int(240 * sample_budget),
                    sparse=20 + int(80 * sparse_budget),
                    dense=0,
                    sparse_max_weight=sparse_weight,
                ),
                pool=SourcePool(keep=5 + int(3 * sample_budget), reserve=1),
                z_choices=4 + int(10 * breadth_budget),
            )
        )
        self.set_todd_search(
            ToddSearch(
                sampling=SamplingBudget(
                    one_hot=10 + int(40 * breadth_budget),
                    sparse=5 + int(35 * sparse_budget),
                    dense=5 + int(45 * dense_budget),
                    sparse_max_weight=sparse_weight,
                ),
                pool=SourcePool(
                    keep=(8 + int(24 * todd_budget)) if self.use_todd_stage else 0,
                    reserve=(1 + int(3 * todd_budget)) if self.use_todd_stage else 0,
                ),
                actions_per_bucket=4 + int(12 * single_budget),
                buckets=ZBucketSearch(
                    min_buckets=(10000 + int(30000 * todd_budget)) if self.use_todd_stage else 0,
                    max_buckets=(1000000 + int(6000000 * todd_budget)) if self.use_todd_stage else 0,
                    temperature=0.02 + 0.28 * bucket_budget,
                    random_fraction=0.05 + 0.20 * bucket_budget,
                ),
            )
        )

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
        workers = PSO_WORKERS if PSO_WORKERS is not None else min(particles, os.cpu_count() or 1)
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
        bounds=(np.full(n_params, -2.0), np.full(n_params, 2.0)),
    )

    _, best_position = optimizer.optimize(objective, iters=iters, verbose=True)
    return np.asarray(best_position, dtype=float)


def entrypoint(mat: Matrix):
    fun = Evaluator(
        mat=mat,
        max_depth=MAX_DEPTH,
        seeds=_make_seeds(EVAL_SEEDS, BASE_SEED),
        use_todd_stage=0,
    )

    xopt = run_pso(
        fun,
        iters=PSO_ITERS,
        particles=PSO_PARTICLES,
        seed_workers=PSO_SEED_WORKERS,
    )

    if fun.best_paths and REFINE_ONCE:
        best_rank = fun.best_paths[0].final_node.state.rows
        rank_thr = best_rank + 30
        x_active = fun.set_up_new_init(0, rank_thr=rank_thr, xopt=xopt)
        fun.use_todd_stage = 1
        if x_active is not None:
            xopt = run_pso(
                fun,
                iters=PSO_REFINE_ITERS,
                particles=PSO_REFINE_PARTICLES,
                seed_workers=PSO_REFINE_SEED_WORKERS,
            )

    return fun.get_best()
