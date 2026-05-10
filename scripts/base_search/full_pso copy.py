import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pyswarms as ps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimization_core"))

from helper import BaseEvaluator, ExplorationScore, FinalizationScore, Matrix


def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(float(z) / sharp))


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _make_seeds(count: int, base_seed: int) -> list[int]:
    rng = np.random.default_rng(base_seed)
    return [int(seed) for seed in rng.integers(1, 10000, size=max(1, count))]


def _scaled(values: list[float], scales: list[float]) -> list[float]:
    return [value * scale for value, scale in zip(values, scales)]


def _score_pow(z: float) -> float:
    return 0.5 + 2.5 * _sigmoid(z)


class Evaluator(BaseEvaluator):
    def __init__(
        self,
        *args,
        seeds: Iterable[int],
        deep_max_z: int = 160000,
        **kwargs,
    ):
        self.seeds = list(seeds)
        self.deep_search = False
        self.deep_max_z = int(deep_max_z)
        super().__init__(*args, **kwargs)

    def _score_weights(self, count: int) -> list[float]:
        return [self.map_par(_w_tanh, 0) for _ in range(count)]

    def _score_centers(self, count: int) -> list[float]:
        return [self.map_par(_w_tanh, 0, scale=1.0) for _ in range(count)]

    def policy_mapping(self):
        ranks = [max(1, int(self.init_rank * 0.82)), 0]
        pool_weights = self._score_weights(5)
        pool_centers = [0 for i in range(5)]
        pool_pow = self.map_par(_score_pow, 0)
        final_weights = self._score_weights(6)
        final_centers = [0 for i in range(6)]
        final_pow = self.map_par(_score_pow, 0)
        budget = self.map_par(_sigmoid, 0)

        self.set_pool_scores(
            ranks,
            [
                ExplorationScore(weights=pool_weights, centers=pool_centers, pow=pool_pow),
                ExplorationScore(
                    weights=_scaled(pool_weights, [1.0, 0.5, 0.5, 1.0, 0.5]),
                    centers=pool_centers,
                    pow=pool_pow,
                ),
            ],
        )
        self.set_final_scores(
            ranks,
            [
                FinalizationScore(weights=final_weights, centers=final_centers, pow=final_pow),
                FinalizationScore(
                    weights=_scaled(final_weights, [1.0, 0.5, 0.4, 1.0, 0.5, 1.0]),
                    centers=final_centers,
                    pow=final_pow,
                ),
            ],
        )

        scout_max_z = 1000000
        self.set_min_pool_size(4)
        self.set_min_z_to_research(ranks, [2 + 1000 * budget, 1 + 1000 * budget])
        self.set_max_z_to_research(ranks, [scout_max_z, self.deep_max_z if self.deep_search else scout_max_z])
        self.set_temperature(0.24 + 0.7 * budget)
        self.set_num_samples(ranks, [30, 20])
        self.set_max_pool_size(ranks, [30 + int(5 * budget), 20 + int(2 * budget)])
        self.set_max_tohpe(ranks, [5, 5])
        self.set_try_only_tohpe(ranks, [1, 0])
        self.set_enable_tohpe(1)
        self.set_enable_todd(1)
        self.set_max_reduction(30)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(40)
        self.set_tohpe_num_best(ranks, [10, 5])
        self.set_gen_part(ranks, [0.1 + 0.9*budget, 0.5 + 0.5 * budget])
        self.set_todd_width(ranks, [1, 1])
        self.set_beamsearch_width(ranks, [1, 1])

    def evaluate(self, params: Iterable[float]) -> float:
        seeds = self.seeds[:1] if self.deep_search else self.seeds
        ranks = self.run(params, seeds, max_workers=None)
        return float(np.quantile(ranks, 0.65) + 0.01 * np.std(ranks))

    def __call__(self, params: Iterable[float]) -> float:
        return self.evaluate(params)


def run_pso(fun: Evaluator, iters: int, particles: int) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    def objective(positions):
        return np.asarray([fun(position) for position in positions], dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.6, "c2": 0.5, "w": 0.75},
        bounds=(np.full(n_params, -1.0), np.full(n_params, 1.0)),
    )
    _, best_position = optimizer.optimize(objective, iters=iters, verbose=True)
    return np.asarray(best_position, dtype=float)


def entrypoint(mat: Matrix):
    fun = Evaluator(
        mat=mat,
        max_depth=_env_int("VARTODD_MAX_DEPTH", 400),
        seeds=_make_seeds(
            _env_int("VARTODD_EVAL_SEEDS", 3),
            _env_int("VARTODD_SEED", 40),
        ),
        deep_max_z=_env_int("VARTODD_DEEP_MAX_Z", 160000),
    )

    particles = _env_int("VARTODD_PSO_PARTICLES", 48)
    xopt = run_pso(fun, iters=_env_int("VARTODD_PSO_ITERS", 70), particles=particles)

    best_rank = fun.best_pathes[0].final_node.state.rows
    print(f"{best_rank=}")
    mid_rank_thr = max(best_rank + 1, 2*(fun.init_rank + best_rank) // 3)
    late_rank_thr = max(best_rank + 1, best_rank + (mid_rank_thr - best_rank) // 3)

    for rank_thr in (mid_rank_thr, late_rank_thr):
        x_active = fun.set_up_new_init(0, rank_thr=rank_thr, xopt=None)
        if x_active is None:
            break
        xopt = run_pso(fun, iters=_env_int("VARTODD_PSO_REFINE_ITERS", 30), particles=particles)

    return fun.get_best()
