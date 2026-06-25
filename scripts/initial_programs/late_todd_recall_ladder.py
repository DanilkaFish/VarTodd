from typing import Iterable

import cma
import numpy as np
import pyswarms as ps

from helper import (
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    PARAM_ALWAYS_ACTIVE,
    ExplorationScore,
    FinalizationScore,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
)


np.random.seed(307)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20, 20)))))


def signed(x: float, scale: float = 5.0, sharp: float = 1.45) -> float:
    return float(scale * np.tanh(float(x) / sharp))


def tail_score(ranks: Iterable[float]) -> float:
    values = np.asarray(list(ranks), dtype=float)
    if values.size == 0:
        return 10_000.0
    return float(np.quantile(values, 0.75) + 0.012 * np.std(values))


class Evaluator(BaseEvaluator):
    """Late TODD recall ladder with compact active parameters after each restart."""

    seeds = [307, 1301, 2459]
    phase = "scout"

    def policy_mapping(self):
        ranks = [10_000, 470, 436, 421, 408, 405]
        recall = self.map_par(sigmoid, 408)
        z_budget = self.map_par(sigmoid, 436)
        sample_bias = self.map_par(sigmoid, 470)
        center_bias = self.map_par(sigmoid, 408)
        temp_bias = self.map_par(sigmoid, 408)

        pool_red = 0.7 + self.map_par(signed, PARAM_ALWAYS_ACTIVE)
        pool_dim = -0.3 + 0.5 * self.map_par(signed, 436)
        pool_bucket = 0.8 + self.map_par(signed, 436)
        pool_vw = self.map_par(signed, 421)
        pool_z = self.map_par(signed, 421)

        final_red = 0.9 + self.map_par(signed, PARAM_ALWAYS_ACTIVE)
        final_dim = -0.2 + 0.5 * self.map_par(signed, 436)
        final_bucket = 0.7 + self.map_par(signed, 421)
        final_vw = self.map_par(signed, 408)
        final_z = self.map_par(signed, 408)
        final_tohpe = -0.25 + 0.7 * self.map_par(sigmoid, 408)

        dim_center = 0.10 + 0.20 * center_bias
        weight_center = 0.30 + 0.30 * center_bias
        self.set_scores(
            PolicyScores(
                exploration=ExplorationScore(
                    weights=[pool_red, pool_dim, pool_bucket, pool_vw, pool_z],
                    centers=[0.0, dim_center, 0.45, weight_center, weight_center],
                    pow=1,
                ),
                final=FinalizationScore(
                    weights=[final_red, final_dim, final_bucket, final_vw, final_z, final_tohpe],
                    centers=[0.15, dim_center, 0.55, weight_center, weight_center, 0.0],
                    pow=2,
                ),
            )
        )

        early_sampling = SamplingBudget(
            one_hot=10 + int(10 * sample_bias),
            sparse=1,
            dense=8 + int(14 * sample_bias),
            sparse_max_weight=3,
        )
        mid_sampling = SamplingBudget(
            one_hot=12 + int(14 * sample_bias),
            sparse=1,
            dense=10 + int(18 * sample_bias),
            sparse_max_weight=3,
        )
        late_sampling = SamplingBudget(
            one_hot="all",
            sparse=1 + int(2 * sample_bias),
            dense=12 + int(18 * sample_bias),
            sparse_max_weight=4,
        )
        tail_sampling = SamplingBudget(
            one_hot="all",
            sparse=0,
            dense=8 + int(12 * sample_bias),
            sparse_max_weight=3,
        )
        samplings = [early_sampling, early_sampling, mid_sampling, late_sampling, late_sampling, tail_sampling]

        deep_phase = self.phase == "recall"
        action_counts = [1, 1, 1, 2 if deep_phase else 1, 2 if deep_phase else 1, 1]
        beam_widths = [2, 2, 1, 1, 1, 1]
        temperatures = [
            0.18,
            0.14,
            0.10,
            0.06 + 0.06 * temp_bias,
            0.04 + 0.05 * temp_bias,
            0.03 + 0.04 * temp_bias,
        ]
        self.set_action_selection(
            ranks,
            [
                ActionSelection(count=count, mode="softmax", temperature=temp)
                for count, temp in zip(action_counts, temperatures)
            ],
        )
        self.set_widths(beam=list(zip(ranks, beam_widths)), actions=list(zip(ranks, action_counts)))
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in [10, 12, 16, 20, 22, 18]])

        tohpe_keep = [8, 7, 5, 3, 2, 1]
        todd_keep = [
            1,
            2 + int(2 * recall),
            6 + int(4 * recall),
            12 + int(8 * recall),
            16 + int(8 * recall),
            18 + int(6 * recall),
        ]
        min_buckets = [
            80 + int(240 * z_budget),
            180 + int(700 * z_budget),
            460 + int(1800 * z_budget),
            1000 + int(5200 * z_budget),
            1500 + int(9000 * z_budget),
            1800 + int(10000 * z_budget),
        ]
        max_buckets = [180_000, 360_000, 700_000, 1_400_000, 2_200_000, 2_600_000]
        if deep_phase:
            min_buckets = [int(v * 1.45) for v in min_buckets]

        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(
                    sampling=samplings[i],
                    pool=SourcePool(keep=tohpe_keep[i], reserve=1 if i < 4 else 0),
                    z_choices=6 if i < 2 else 3,
                )
                for i in range(len(ranks))
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=samplings[i],
                    pool=SourcePool(keep=todd_keep[i], reserve=0 if i < 3 else 2),
                    actions_per_bucket=2 if i < 3 else 3,
                    buckets=ZBucketSearch(min_buckets=min_buckets[i], max_buckets=max_buckets[i]),
                )
                for i in range(len(ranks))
            ],
        )

    def __call__(self, params: Iterable[float]) -> float:
        ranks = self.run(params, self.seeds)
        return tail_score(ranks)


def run_pso(fun: Evaluator, n_eval: int, low: float = -1.6, high: float = 1.6) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)
    particles = 4
    iters = max(1, int(np.ceil(n_eval / particles)))

    def objective(positions):
        return np.asarray([fun(position) for position in positions], dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.65, "c2": 0.55, "w": 0.82},
        bounds=(np.full(n_params, low), np.full(n_params, high)),
    )
    _, best_position = optimizer.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best_position, dtype=float)


def run_cma(
    fun: Evaluator,
    x0: np.ndarray | None,
    n_eval: int,
    sigma: float,
    low: float = -1.2,
    high: float = 1.2,
) -> np.ndarray:
    current = np.asarray(fun.extract_active() if x0 is None else x0, dtype=float)
    if current.size == 0:
        return current
    if current.size != len(fun.extract_active()):
        current = np.asarray(fun.extract_active(), dtype=float)
    current = np.clip(current, low + 1e-6, high - 1e-6)
    def objective(x):
        return float(fun(np.asarray(x, dtype=float)))

    xopt, _ = cma.fmin2(
        objective,
        current,
        sigma,
        options={
            "bounds": [low, high],
            "maxfevals": n_eval,
            "popsize": 7,
            "seed": 308,
            "verbose": -9,
        },
        restarts=0,
        bipop=False,
    )
    return np.asarray(xopt, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=250)
    xopt = run_pso(fun, n_eval=84)

    active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 42), xopt=xopt)
    if active is not None and len(active):
        fun.phase = "recall"
        xopt = run_cma(fun, np.asarray(active, dtype=float), n_eval=104, sigma=0.30)

    active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 18), xopt=xopt)
    if active is not None and len(active):
        run_cma(fun, np.asarray(active, dtype=float), n_eval=50, sigma=0.14, low=-0.65, high=0.65)

    return fun.get_best()
