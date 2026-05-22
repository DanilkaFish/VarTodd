from copy import deepcopy
from typing import Iterable
import random

import numpy as np
import pyswarms as ps

from helper import BaseEvaluator, ExplorationScore, FinalizationScore

np.random.seed(42)
random.seed(40)


def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(float(z) / sharp))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def softmin(xs: Iterable[float], beta: float = 6.0) -> float:
    values = np.asarray(list(xs), dtype=float)
    m = values.min()
    return float(m - (1.0 / beta) * np.log(np.exp(-beta * (values - m)).sum()))


class Evaluator(BaseEvaluator):
    """Validated full-score PSO seed copied from the strong full_pso setting."""

    seeds = [random.randint(1, 10000) for _ in range(2)]
    validation_seeds = [random.randint(1, 10000) for _ in range(12)]

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float):
        one_hot = max(0, int(round(sample_count * one_hot_fraction)))
        dense = max(0, int(sample_count) - one_hot)
        return [one_hot, 0, dense]

    def policy_mapping(self):
        ranks = [0]
        self.set_pool_scores(
            ranks,
            [ExplorationScore([self.map_par(_w_tanh) for _ in range(5)], pow=1)],
        )
        self.set_final_scores(
            ranks,
            [FinalizationScore([self.map_par(_w_tanh) for _ in range(6)], pow=1)],
        )

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

    def evaluate(
        self,
        params: Iterable[float],
        seeds: Iterable[int] | None = None,
    ) -> float:
        seed_list = list(self.seeds if seeds is None else seeds)
        ranks = self.run(params, seed_list, max_workers=4)
        spread = float(np.std(ranks)) if len(ranks) > 1 else 0.0
        return softmin(ranks, beta=6.0) + 0.02 * spread

    def __call__(self, params: Iterable):
        return self.evaluate(params, seeds=self.validation_seeds)

def _score_position(
    fun: Evaluator,
    position: Iterable[float],
    seeds: Iterable[int] | None = None,
) -> tuple[float, Evaluator]:
    cost = fun.evaluate(np.asarray(position, dtype=float), seeds=seeds)
    return float(cost), fun

def run_opt(fun: Evaluator, num_eval: int = 10) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)

    search_history = []

    def objective(positions):
        costs = []
        for position in positions:
            pos = np.asarray(position, dtype=float)
            cost, local_fun = _score_position(fun, pos)
            local_fun.close_workers()
            costs.append(cost)
            search_history.append((float(cost), pos.copy()))
        return np.asarray(costs, dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=8,
        dimensions=n_params,
        options={"c1": 0.4, "c2": 0.4, "w": 0.7},
        bounds=(np.full(n_params, -1.0), np.full(n_params, 1.0)),
    )
    best_cost, best_position = optimizer.optimize(
        objective,
        iters=num_eval,
        verbose=True,
    )
    return best_position


def entrypoint(mat):
    fun = Evaluator(mat=mat, fin_rank=170, max_depth=200)
    x0 = run_opt(fun, 4)
    return fun.get_best()
    best_rank = int(fun.best_paths[0].final_node.state.rows)
    late_rank_thr = max(best_rank + 1, best_rank + (fun.init_rank - best_rank) // 4)

    for rank_thr in (late_rank_thr,):
        active = fun.set_up_new_init(0, rank_thr=rank_thr, xopt=None)
        x0 = run_opt(fun, 20)
    return fun.get_best()
