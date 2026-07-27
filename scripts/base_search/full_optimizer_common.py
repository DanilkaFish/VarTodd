from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import qmc

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

NATIVE_LOWER = -1.0
NATIVE_UPPER = 1.0
ELITE_COUNT = 6
ELITE_MIN_DISTANCE = 0.05
OPTIMIZER_WORKERS = 8
SCORE_SEED_COUNT = 4
ROBUST_SPREAD_WEIGHT = 0.3
PHASE_EVALUATIONS = 3_600
TOTAL_EVALUATIONS = 2 * PHASE_EVALUATIONS
Z_MAX_BUCKET_CAP = 10_000
Z_LIMIT_BUCKET_CAP = 50_000
SAMPLE_COUNT_MAX = 100
TODD_RESERVE_MAX = 6
Z_MIN_CAP = 500

_seed_rng = random.Random(40)


def to_unit(params: Iterable[float]) -> np.ndarray:
    values = np.asarray(params, dtype=float)
    return (values - NATIVE_LOWER) / (NATIVE_UPPER - NATIVE_LOWER)


def from_unit(positions: Iterable[float]) -> np.ndarray:
    values = np.asarray(positions, dtype=float)
    return NATIVE_LOWER + values * (NATIVE_UPPER - NATIVE_LOWER)


@dataclass(slots=True)
class _Elite:
    cost: float
    full_params: np.ndarray


class EliteArchive:
    def __init__(
        self,
        capacity: int = ELITE_COUNT,
        min_distance: float = ELITE_MIN_DISTANCE,
    ):
        if capacity < 1:
            raise ValueError("elite archive capacity must be positive")
        if min_distance < 0:
            raise ValueError("elite archive distance must be non-negative")
        self.capacity = int(capacity)
        self.min_distance = float(min_distance)
        self._entries: list[_Elite] = []

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _active_unit(full_params: np.ndarray, active_params: Sequence[int]) -> np.ndarray:
        return to_unit(full_params[np.asarray(active_params, dtype=int)])

    def add(
        self,
        cost: float,
        full_params: Iterable[float],
        active_params: Sequence[int],
    ) -> None:
        params = np.asarray(full_params, dtype=float).copy()
        active = list(active_params)
        candidate = self._active_unit(params, active)

        for index, entry in enumerate(self._entries):
            existing = self._active_unit(entry.full_params, active)
            rms_distance = float(
                np.linalg.norm(candidate - existing) / np.sqrt(candidate.size)
            )
            if rms_distance < self.min_distance:
                if float(cost) < entry.cost:
                    self._entries[index] = _Elite(float(cost), params)
                    self._entries.sort(key=lambda item: item.cost)
                return

        self._entries.append(_Elite(float(cost), params))
        self._entries.sort(key=lambda item: item.cost)
        del self._entries[self.capacity :]

    def projected(self, active_params: Sequence[int]) -> list[np.ndarray]:
        active = np.asarray(active_params, dtype=int)
        return [to_unit(entry.full_params[active]) for entry in self._entries]


def _is_distinct(position: np.ndarray, positions: Sequence[np.ndarray]) -> bool:
    return all(
        not np.allclose(position, existing, rtol=0.0, atol=1e-12)
        for existing in positions
    )


def initial_population(
    fun,
    archive: EliteArchive,
    size: int,
    seed: int,
) -> np.ndarray:
    current = np.clip(to_unit(fun.extract_active()), 0.0, 1.0)
    dimensions = int(current.size)
    if dimensions == 0:
        raise ValueError("optimizer requires at least one active parameter")
    if size < 1:
        raise ValueError("population size must be positive")

    positions = [current]
    for elite in archive.projected(fun.active_params):
        elite = np.clip(elite, 0.0, 1.0)
        if len(positions) >= size:
            break
        if _is_distinct(elite, positions):
            positions.append(elite)

    remaining = size - len(positions)
    if remaining:
        lhs = qmc.LatinHypercube(d=dimensions, seed=seed).random(remaining)
        positions.extend(lhs)
    return np.asarray(positions, dtype=float)


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
    seeds = [_seed_rng.randint(1, 10_000) for _ in range(SCORE_SEED_COUNT)]
    validation_seeds = [_seed_rng.randint(1, 10_000) for _ in range(12)]

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float) -> list[int]:
        return [
            int(one_hot_fraction * sample_count),
            0,
            max(0, int(sample_count * (1 - one_hot_fraction))),
        ]

    def policy_mapping(self):
        ranks = [0]
        pool_score = ExplorationScore([self.map_par(_w_tanh) for _ in range(5)], pow=1)
        final_weights = [self.map_par(_w_tanh) for _ in range(6)]
        final_centers = [self.map_par(sigmoid) for _ in range(6)]
        final_score = FinalizationScore(final_weights, final_centers, pow=2)
        self.set_scores(
            ranks,
            [PolicyScores(exploration=pool_score, final=final_score)],
        )

        z_budget = self.map_par(sigmoid)
        z_research_budget = self.map_par(sigmoid)
        z_limit_budget = self.map_par(sigmoid)
        todd_reserve_budget = self.map_par(sigmoid)
        sample_budget = self.map_par(sigmoid)
        one_hot_fraction = 0.1 + 0.8 * self.map_par(sigmoid)
        sample_count = 1 + int(SAMPLE_COUNT_MAX * sample_budget)
        sample_caps = self._sample_caps(sample_count, one_hot_fraction)
        z_min, z_max, z_limit = z_bucket_ranges(
            z_budget,
            z_research_budget,
            z_limit_budget,
            self.buckets_space,
        )
        todd_reserve = min(
            TODD_RESERVE_MAX,
            int(TODD_RESERVE_MAX * todd_reserve_budget),
        )

        sampling = SamplingBudget(
            one_hot=sample_caps[0],
            sparse=sample_caps[1],
            dense=sample_caps[2],
            sparse_max_weight=2,
        )
        self.set_action_selection(
            ActionSelection(count=2, mode="softmax", temperature=0.2)
        )
        self.set_action_pool(ActionPool(final_size=18))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=sampling,
                pool=SourcePool(keep=12, reserve=0),
                z_choices=4,
            )
        )
        self.set_todd_search(
            ToddSearch(
                sampling=sampling,
                pool=SourcePool(keep=12, reserve=todd_reserve),
                actions_per_bucket=2,
                buckets=ZBucketSearch(
                    min_buckets=z_min,
                    max_buckets=z_max,
                    limit_bucket=z_limit,
                ),
            )
        )

    def evaluate(
        self,
        params: Iterable[float],
        max_workers: int = 1,
        seeds: Iterable[int] | None = None,
    ) -> float:
        selected_seeds = list(self.seeds if seeds is None else seeds)
        tcounts = self.run(params, selected_seeds, max_workers=max_workers)
        return robust_rank_cost(tcounts)

    def __call__(self, params: Iterable[float]) -> float:
        return self.evaluate(params, max_workers=1, seeds=self.validation_seeds)


def _clone_for_evaluation(fun: Evaluator) -> Evaluator:
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
    clone.best_params = []
    clone.best_eval = 0
    clone.total_eval = 0
    clone.tcount = []
    clone.best_seen = 0
    clone.best_seed = None
    clone._best_rank = fun._best_rank
    clone.init_events = list(fun.init_events)
    clone.time_to_final_rank_seconds = None
    clone._executor = None
    clone._executor_key = None
    return clone


def _score_position(
    fun: Evaluator,
    native_position: Iterable[float],
) -> tuple[float, Evaluator]:
    local_fun = _clone_for_evaluation(fun)
    try:
        cost = local_fun.evaluate(np.asarray(native_position, dtype=float))
    finally:
        local_fun.close_workers()
    return float(cost), local_fun


class BatchObjective:
    def __init__(
        self,
        fun: Evaluator,
        archive: EliteArchive,
        workers: int = OPTIMIZER_WORKERS,
    ):
        self.fun = fun
        self.archive = archive
        self.workers = max(1, int(workers))
        self._executor: ThreadPoolExecutor | None = None

    def __enter__(self) -> "BatchObjective":
        self._executor = ThreadPoolExecutor(max_workers=self.workers)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._executor is not None
        self._executor.shutdown()
        self._executor = None

    def _evaluate_one(self, unit_position: np.ndarray) -> tuple[float, Evaluator]:
        return _score_position(self.fun, from_unit(unit_position))

    def __call__(
        self,
        unit_positions: Iterable[Iterable[float]],
    ) -> np.ndarray:
        if self._executor is None:
            raise RuntimeError("BatchObjective must be used as a context manager")
        positions = np.asarray(unit_positions, dtype=float)
        if positions.ndim == 1:
            positions = positions[np.newaxis, :]
        results = list(self._executor.map(self._evaluate_one, positions))
        for cost, local_fun in results:
            self.fun.merge_run_state_from(local_fun)
            self.archive.add(cost, local_fun.x0, local_fun.active_params)
        return np.asarray([cost for cost, _ in results], dtype=float)


OptimizePhase = Callable[[Evaluator, int, EliteArchive, int], np.ndarray]


def run_two_phase(fun: Evaluator, optimize_phase: OptimizePhase):
    archive = EliteArchive()
    initial_rank = int(fun.init_rank)

    optimize_phase(fun, PHASE_EVALUATIONS, archive, 0)
    if not fun.best_paths:
        raise RuntimeError("the initial optimizer phase produced no paths")

    best_rank = int(fun.best_paths[0].final_node.state.rows)
    midpoint = max(best_rank + 1, (initial_rank + best_rank) // 2)
    print(f"path restart: set_up_new_init at rank_thr={midpoint}")
    active = fun.set_up_new_init(0, rank_thr=midpoint, xopt=None)
    if active is None:
        print("path restart: no branch found")
        return fun.get_best()

    optimize_phase(fun, PHASE_EVALUATIONS, archive, 1)
    return fun.get_best()
