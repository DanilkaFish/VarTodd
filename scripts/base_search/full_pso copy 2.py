from typing import Iterable
import random

import numpy as np
import pyswarms as ps

from helper import BaseEvaluator, ExplorationScore, FinalizationScore

np.random.seed(131)
random.seed(131)


def unit(x: float) -> float:
    return float(np.clip(x, 0.0, 0.999))


def softmin(xs: Iterable[float], beta: float = 5.0) -> float:
    values = np.asarray(list(xs), dtype=float)
    m = values.min()
    return float(m - (1.0 / beta) * np.log(np.exp(-beta * (values - m)).sum()))


class Evaluator(BaseEvaluator):
    """Optimize score centers while all score weights are fixed to -1."""

    seeds = [random.randint(1, 10000) for _ in range(2)]
    validation_seeds = [random.randint(1, 10000) for _ in range(10)]
    phase = "scout"

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float, sparse_fraction: float):
        total = max(1, int(round(sample_count)))
        one_hot = max(0, int(round(total * one_hot_fraction)))
        sparse = max(0, int(round(total * sparse_fraction)))
        dense = max(0, total - one_hot - sparse)
        return [one_hot, sparse, dense]

    def policy_mapping(self):
        ranks = [520, 465, 380]

        pool_centers = [self.map_par(unit, 0) for _ in range(5)]
        final_centers = [self.map_par(unit, 0) for _ in range(6)]
        z_budget = self.map_par(unit, 0)
        sample_budget = self.map_par(unit, 0)
        hot_fraction = self.map_par(unit, 0)
        width_budget = self.map_par(unit, 0)

        self.set_pool_scores(
            ExplorationScore(
                weights=[-1.0, -1.0, -1.0, -1.0, -1.0],
                centers=pool_centers,
                pow=2,
            )
        )
        self.set_final_scores(
            FinalizationScore(
                weights=[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                centers=final_centers,
                pow=2,
            )
        )

        deep = self.phase == "deep"
        sample_count = 8 + int(52 * sample_budget)
        one_hot = 0.10 + 0.62 * hot_fraction
        sample_caps = [
            self._sample_caps(sample_count + 6, one_hot, 0.08),
            self._sample_caps(sample_count, one_hot + 0.10, 0.04),
            self._sample_caps(max(8, sample_count - 4), one_hot + 0.18, 0.00),
        ]
        self.set_tohpe_vector_samples(ranks, sample_caps)
        self.set_todd_vector_samples(ranks, sample_caps)

        beam = 1 + int(width_budget > 0.72)
        self.set_beamsearch_width(ranks, [beam, 1, 1])
        self.set_todd_width(ranks, [1, 1, 1])
        self.set_min_pool_size(1)
        self.set_max_pool_size(ranks, [12, 12, 10])
        self.set_tohpe_pool_size(ranks, [3, 2, 2])
        self.set_todd_pool_size(ranks, [8 if deep else 4, 12 if deep else 6, 12])
        self.set_min_tohpe_actions(1)
        self.set_min_todd_actions(0)
        self.set_tohpe_sample(ranks, [3, 2, 2])
        self.set_temperature(ranks, [0.18, 0.11, 0.07])
        self.set_min_z_to_research(
            ranks,
            [
                20 + int(280 * z_budget),
                160 + int(1200 * z_budget),
                280 + int(3000 * z_budget),
            ],
        )
        self.set_max_z_to_research(
            ranks,
            [28_000, 300_000],
        )
        self.set_sparse_max_weight(2)
        self.set_min_reduction(1)
        self.set_max_reduction(12)
        self.set_max_from_single_ns(1)
        self.set_bucket_temperature(0.0)
        self.set_bucket_random_fraction(0.0)

    def evaluate(self, params: Iterable[float], seeds: Iterable[int] | None = None) -> float:
        seed_list = list(self.seeds if seeds is None else seeds)
        ranks = self.run(params, seed_list, max_workers=5)
        spread = float(np.std(ranks)) if len(ranks) > 1 else 0.0
        return softmin(ranks, beta=5.0) + 0.015 * spread

    def __call__(self, params: Iterable):
        return self.evaluate(params, seeds=self.seeds)

    def validate_policy(self, params: Iterable[float]) -> float:
        return self.evaluate(params, seeds=self.validation_seeds)


def run_pso(
    fun: Evaluator,
    iters: int,
    particles: int,
    validate_top_k: int,
) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    history: list[tuple[float, np.ndarray]] = []

    def objective(positions):
        clipped = np.clip(np.asarray(positions, dtype=float), 0.0, 0.999)
        costs = []
        for pos in clipped:
            cost = float(fun.evaluate(pos, seeds=fun.seeds))
            costs.append(cost)
            history.append((cost, pos.copy()))
        return np.asarray(costs, dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.55, "c2": 0.45, "w": 0.74},
        bounds=(np.zeros(n_params), np.full(n_params, 0.999)),
    )
    best_cost, best_position = optimizer.optimize(objective, iters=iters, verbose=True)

    ranked = sorted(
        history + [(float(best_cost), np.asarray(best_position, dtype=float).copy())],
        key=lambda item: item[0],
    )
    best_valid = np.asarray(best_position, dtype=float)
    best_valid_cost = float("inf")
    seen = set()
    validated = 0
    for _, pos in ranked:
        key = tuple(np.round(pos, 5))
        if key in seen:
            continue
        seen.add(key)
        cost = float(fun.validate_policy(pos))
        if cost < best_valid_cost:
            best_valid_cost = cost
            best_valid = pos.copy()
        validated += 1
        if validated >= validate_top_k:
            break
    return np.asarray(np.clip(best_valid, 0.0, 0.999), dtype=float)


def entrypoint(mat):
    fun = Evaluator(mat=mat, fin_rank=170, max_depth=200)
    # fun = Evaluator(path_name="init", max_depth=390)
    xopt = run_pso(fun, iters=22, particles=7, validate_top_k=5)

    if fun.best_paths:
        fun.phase = "deep"
        active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 40), xopt=xopt)
        if active is not None and len(active):
            run_pso(fun, iters=10, particles=5, validate_top_k=3)

    return fun.get_best()
