from __future__ import annotations

import os
from typing import Iterable

import numpy as np

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


def _make_seeds(count: int, base_seed: int) -> list[int]:
    rng = np.random.default_rng(base_seed)
    size = max(1, int(count))
    return [int(seed) for seed in rng.integers(1, 10000, size=size)]


def _escore(weights: list[float]) -> ExplorationScore:
    return ExplorationScore(weights=weights, centers=[0.0] * 5, pow=1)


def _fscore(weights: list[float]) -> FinalizationScore:
    return FinalizationScore(weights=weights, centers=[0.0] * 6, pow=1)


def _stage_thresholds(init_rank: int, best_rank: int) -> tuple[int, int]:
    handoff = max(best_rank + 1, (2 * init_rank + best_rank) // 3)
    finish = max(best_rank + 1, (handoff + best_rank) // 2)
    return handoff, finish


class Evaluator(BaseEvaluator):
    def __init__(
        self,
        *args,
        scout_seeds: Iterable[int],
        refine_seeds: Iterable[int],
        finish_seeds: Iterable[int],
        **kwargs,
    ):
        self.phase = "scout"
        self.scout_seeds = list(scout_seeds)
        self.refine_seeds = list(refine_seeds)
        self.finish_seeds = list(finish_seeds)
        super().__init__(*args, **kwargs)

    def _run_for_phase(self, seeds: list[int]) -> float:
        ranks = self.run([], seeds, max_workers=1)
        if not ranks:
            return float("inf")
        return float(np.quantile(ranks, 0.5))

    def policy_mapping(self):
        early_pool = _escore([1.00, 0.40, 0.16, 0.08, 0.00])
        mid_pool = _escore([1.00, 0.32, 0.22, 0.08, 0.00])
        late_pool = _escore([1.00, 0.22, 0.32, 0.06, 0.00])

        early_final = _fscore([1.00, 0.35, 0.16, 0.08, 0.00, 0.00])
        mid_final = _fscore([1.00, 0.28, 0.22, 0.08, 0.00, 0.00])
        late_final = _fscore([1.00, 0.18, 0.34, 0.06, 0.00, 0.00])

        self.set_min_pool_size(1)
        self.set_enable_tohpe(1)
        self.set_min_reduction(1)
        self.set_max_reduction(128)
        self.set_todd_width(1)
        self.set_beamsearch_width(1)

        if self.phase == "scout":
            ranks = [max(1, int(self.init_rank * 0.84)), max(1, int(self.init_rank * 0.67)), 0]
            self.set_pool_scores(ranks, [early_pool, mid_pool, late_pool])
            self.set_final_scores(ranks, [early_final, mid_final, late_final])
            self.set_temperature(ranks, [0.10, 0.08, 0.05])
            self.set_num_samples(ranks, [18, 14, 10])
            self.set_max_pool_size(ranks, [4, 3, 2])
            self.set_max_tohpe(ranks, [4, 3, 2])
            self.set_tohpe_num_best(ranks, [6, 5, 3])
            self.set_max_from_single_ns(ranks, [20, 16, 10])
            self.set_gen_part(ranks, [0.18, 0.28, 0.42])
            self.set_enable_todd(ranks, [0, 1, 1])
            self.set_try_only_tohpe(ranks, [1, 1, 0])
            self.set_min_z_to_research(ranks, [0, 48, 96])
            self.set_max_z_to_research(ranks, [0, 256, 768])
            return

        if self.phase == "refine":
            ranks = [max(1, int(self.init_rank * 0.58)), 0]
            self.set_pool_scores(ranks, [mid_pool, late_pool])
            self.set_final_scores(ranks, [mid_final, late_final])
            self.set_temperature(ranks, [0.06, 0.03])
            self.set_num_samples(ranks, [14, 8])
            self.set_max_pool_size(ranks, [3, 2])
            self.set_max_tohpe(ranks, [3, 2])
            self.set_tohpe_num_best(ranks, [4, 2])
            self.set_max_from_single_ns(ranks, [14, 8])
            self.set_gen_part(ranks, [0.34, 0.50])
            self.set_enable_todd(1)
            self.set_try_only_tohpe(ranks, [1, 0])
            self.set_min_z_to_research(ranks, [72, 128])
            self.set_max_z_to_research(ranks, [640, 1800])
            return

        ranks = [max(1, int(self.init_rank * 0.46)), 0]
        self.set_pool_scores(ranks, [mid_pool, late_pool])
        self.set_final_scores(ranks, [mid_final, late_final])
        self.set_temperature(ranks, [0.03, 0.00])
        self.set_num_samples(ranks, [10, 6])
        self.set_max_pool_size(ranks, [2, 2])
        self.set_max_tohpe(ranks, [2, 1])
        self.set_tohpe_num_best(ranks, [3, 1])
        self.set_max_from_single_ns(ranks, [10, 6])
        self.set_gen_part(ranks, [0.42, 0.58])
        self.set_enable_todd(1)
        self.set_try_only_tohpe(0)
        self.set_min_z_to_research(ranks, [128, 192])
        self.set_max_z_to_research(ranks, [1200, 2600])

    def __call__(self, params: Iterable[float]):
        if list(params):
            raise ValueError("large_instance_light uses a fixed policy and expects no parameters")
        if self.phase == "scout":
            return self._run_for_phase(self.scout_seeds)
        if self.phase == "refine":
            return self._run_for_phase(self.refine_seeds)
        return self._run_for_phase(self.finish_seeds)


def entrypoint(mat: Matrix):
    fun = Evaluator(
        mat=mat,
        max_depth=_int_env("VARTODD_MAX_DEPTH", 1800),
        scout_seeds=_make_seeds(_int_env("VARTODD_SCOUT_SEEDS", 2), _int_env("VARTODD_SEED", 40)),
        refine_seeds=_make_seeds(_int_env("VARTODD_REFINE_SEEDS", 1), _int_env("VARTODD_SEED", 41)),
        finish_seeds=_make_seeds(_int_env("VARTODD_FINISH_SEEDS", 1), _int_env("VARTODD_SEED", 42)),
    )

    fun([])
    if not fun.best_paths:
        return fun.get_best()

    best_rank = fun.best_paths[0].final_node.state.rows
    handoff_rank, finish_rank = _stage_thresholds(fun.init_rank, best_rank)

    fun.phase = "refine"
    x_active = fun.set_up_new_init(0, rank_thr=handoff_rank, xopt=None)
    if x_active is not None:
        fun(np.asarray(x_active, dtype=float))

    if _int_env("VARTODD_SKIP_FINISH", 0) == 0 and fun.best_paths:
        best_rank = fun.best_paths[0].final_node.state.rows
        _, finish_rank = _stage_thresholds(fun.init_rank, best_rank)
        fun.phase = "finish"
        x_active = fun.set_up_new_init(0, rank_thr=finish_rank, xopt=None)
        if x_active is not None:
            fun(np.asarray(x_active, dtype=float))

    return fun.get_best()
