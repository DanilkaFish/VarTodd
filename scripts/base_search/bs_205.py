from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from typing import Iterable, Optional
import random

import numpy as np
import pyswarms as ps

from helper import BaseEvaluator, ExplorationScore, FinalizationScore

np.random.seed(42)
random.seed(40)


def _w_tanh(z: float, scale: float = 2.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(float(z) / sharp))


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-x)))


class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(3)]
    deep_search = False
    deep_max_z = 1600

    def policy_mapping(self):
        ranks = [max(1, int(self.init_rank * 0.82)), 0]

        wred = self.map_par(_w_tanh, 0)
        wdim = self.map_par(_w_tanh, 0)
        wbucket = self.map_par(_w_tanh, 0)
        fwred = self.map_par(_w_tanh, 0)
        fwdim = self.map_par(_w_tanh, 0)
        budget = self.map_par(sigmoid, 0)

        wvw = self.map_par(_w_tanh, 0)
        fwvw = self.map_par(_w_tanh, 0)
        fcenter = self.map_par(_w_tanh, 0)


        self.set_pool_scores(
            ranks,
            [
                ExplorationScore(weights=[wred, wdim, wbucket, wvw, 0.0], pow=1),
                ExplorationScore(weights=[wred, 0.5 * wdim, 0.5 * wbucket, wvw, 0.0], pow=1),
            ],
        )
        self.set_final_scores(
            ranks,
            [
                FinalizationScore(weights=[fwred, fwdim, wbucket, fwvw, 0.0, 0.0], centers=[0, 0, 0, 0, 0, 0], pow=2),
                FinalizationScore(weights=[fwred, 0.5 * fwdim, 0.4 * wbucket, fwvw, 0.0, 0.0], centers=[fcenter, 0, 0, 0, 0, 0], pow=2),
            ],
        )

        beam = 1
        scout_max_z = 0 if not self.deep_search else 10000
        self.set_min_pool_size(1)
        self.set_min_z_to_research(ranks, [220 + 480 * budget, 140 + 300 * budget])
        self.set_max_z_to_research(ranks, [scout_max_z, self.deep_max_z if self.deep_search else scout_max_z])
        self.set_temperature(0.24 + 0.16 * budget)
        self.set_num_samples(ranks, [30, 20])
        self.set_max_pool_size(ranks, [6 + int(5 * budget), 2 + int(2 * budget)])
        self.set_max_tohpe(ranks, [5, 5])
        self.set_try_only_tohpe(ranks, [1, 1])
        self.set_max_reduction(30)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(40)
        self.set_tohpe_num_best(ranks, [10, 5])
        self.set_gen_part(ranks, [0.3 + 0.30 * budget, 0.5 + 0.2 * budget])
        self.set_todd_width(ranks, [beam, 1])
        self.set_beamsearch_width(ranks, [beam, 1])

    def evaluate(self, params: Iterable, max_workers: Optional[int] = None):
        if self.deep_search:
            seeds = self.seeds[:1]
        else:
            seeds = self.seeds
        if max_workers is None:
            ranks = self.run(params, seeds)
        else:
            ranks = self.run(params, seeds, max_workers=max_workers)
        return float(np.quantile(ranks, 0.65) + 0.01 * np.std(ranks))

    def __call__(self, params: Iterable):
        return self.evaluate(params)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


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
    iters: int = 2,
    particles: int = 3,
    workers: Optional[int] = None,
    seed_workers: Optional[int] = None,
) -> np.ndarray:
    n_params = len(fun.extract_active())
    low, high = -1.0, 1.0
    if n_params == 0:
        return np.asarray([], dtype=float)
    if workers is None:
        workers = _int_env("VARTODD_PSO_WORKERS", min(particles, os.cpu_count() or 1))
    if seed_workers is None:
        seed_workers = 1
    workers = max(1, min(int(workers), particles/2))
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
        options={"c1": 0.6, "c2": 0.5, "w": 0.75},
        bounds=(np.full(n_params, low), np.full(n_params, high)),
    )
    try:
        _, best_position = optimizer.optimize(objective, iters=iters, verbose=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return np.asarray(best_position, dtype=float)


def entrypoint(mat):
    fun = Evaluator(path_name="init", max_depth=1535)
    xopt = run_pso(fun, iters=8, particles=8)
    fun(xopt)
    if fun.best_paths:
        x_active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 150), xopt=xopt)
        xopt = run_pso(fun, iters=8, particles=8)
        fun.deep_search = True
        x_active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 10), xopt=xopt)
        if x_active is not None and len(x_active) > 0:
            xopt = run_pso(fun, iters=4, particles=4)
            fun(np.asarray(x_active, dtype=float))
    return fun.get_best()
