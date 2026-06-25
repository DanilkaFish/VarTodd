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


np.random.seed(211)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20, 20)))))


def signed(x: float, scale: float = 4.8, sharp: float = 1.35) -> float:
    return float(scale * np.tanh(float(x) / sharp))


def robust_rank_score(ranks: Iterable[float]) -> float:
    values = np.asarray(list(ranks), dtype=float)
    if values.size == 0:
        return 10_000.0
    return float(np.quantile(values, 0.65) + 0.018 * np.std(values))


class Evaluator(BaseEvaluator):
    """Balanced source-mix scout: TOHPE path building, then TODD recall refinement."""

    seeds = [211, 1217, 3221, 4651]

    @staticmethod
    def _sampling(total: int, hot: float, sparse: float, weight: int) -> SamplingBudget:
        one_hot = max(1, int(round(total * hot)))
        sparse_count = max(0, int(round(total * sparse)))
        dense = max(0, int(total) - one_hot - sparse_count)
        return SamplingBudget(one_hot=one_hot, sparse=sparse_count, dense=dense, sparse_max_weight=weight)

    def policy_mapping(self):
        ranks = [10_000, 468, 430, 421, 405]

        source_bias = self.map_par(sigmoid, 430)
        sample_bias = self.map_par(sigmoid, 468)
        z_bias = self.map_par(sigmoid, 430)
        temp_bias = self.map_par(sigmoid, 421)

        pool_red = 0.6 + self.map_par(signed, PARAM_ALWAYS_ACTIVE)
        pool_dim = -0.25 + 0.5 * self.map_par(signed, 468)
        pool_bucket = 0.7 + self.map_par(signed, 430)
        pool_vw = self.map_par(signed, 421)
        pool_z = self.map_par(signed, 421)

        final_red = 0.8 + self.map_par(signed, PARAM_ALWAYS_ACTIVE)
        final_dim = -0.15 + 0.4 * self.map_par(signed, 468)
        final_bucket = 0.8 + self.map_par(signed, 430)
        final_vw = self.map_par(signed, 421)
        final_z = self.map_par(signed, 421)
        final_tohpe = 0.45 * self.map_par(signed, 408)

        exploration = ExplorationScore(
            weights=[pool_red, pool_dim, pool_bucket, pool_vw, pool_z],
            centers=[0.0, 0.15, 0.35, 0.45, 0.35],
            pow=1,
        )
        final = FinalizationScore(
            weights=[final_red, final_dim, final_bucket, final_vw, final_z, final_tohpe],
            centers=[0.2, 0.15, 0.4, 0.45, 0.35, 0.0],
            pow=2,
        )
        self.set_scores(PolicyScores(exploration=exploration, final=final))

        totals = [
            14 + int(18 * sample_bias),
            18 + int(22 * sample_bias),
            20 + int(28 * sample_bias),
            18 + int(28 * sample_bias),
            14 + int(20 * sample_bias),
        ]
        hot = [0.30, 0.40, 0.50, 0.60, 0.70]
        sparse = [0.12, 0.10, 0.08, 0.04, 0.0]
        samplings = [self._sampling(t, h, s, 2 + int(3 * sample_bias)) for t, h, s in zip(totals, hot, sparse)]
        late_all = SamplingBudget(one_hot="all", sparse=1, dense=8 + int(12 * sample_bias), sparse_max_weight=3)
        todd_samplings = [samplings[0], samplings[1], samplings[2], late_all, late_all]

        action_counts = [1, 1, 2, 2, 2]
        temperatures = [0.22, 0.18, 0.14, 0.08 + 0.08 * temp_bias, 0.05 + 0.06 * temp_bias]
        final_pools = [12, 14, 18, 20, 18]

        self.set_action_selection(
            ranks,
            [
                ActionSelection(count=count, mode="softmax", temperature=temp)
                for count, temp in zip(action_counts, temperatures)
            ],
        )
        self.set_widths(beam=list(zip(ranks, [2, 2, 2, 1, 1])), actions=list(zip(ranks, action_counts)))
        self.set_action_pool(ranks, [ActionPool(final_size=size) for size in final_pools])

        tohpe_keep = [8, 7, 5, 3, 2]
        todd_keep = [
            2 + int(3 * source_bias),
            4 + int(4 * source_bias),
            8 + int(5 * source_bias),
            12 + int(6 * source_bias),
            14 + int(6 * source_bias),
        ]
        min_z = [
            60 + int(260 * z_bias),
            220 + int(900 * z_bias),
            520 + int(2200 * z_bias),
            900 + int(4600 * z_bias),
            1200 + int(6500 * z_bias),
        ]
        max_z = [220_000, 420_000, 780_000, 1_250_000, 1_800_000]

        self.set_tohpe_search(
            ranks,
            [
                TohpeSearch(
                    sampling=samplings[i],
                    pool=SourcePool(keep=tohpe_keep[i], reserve=1 if i < 4 else 0),
                    z_choices=6 if i < 2 else 4,
                )
                for i in range(len(ranks))
            ],
        )
        self.set_todd_search(
            ranks,
            [
                ToddSearch(
                    sampling=todd_samplings[i],
                    pool=SourcePool(keep=todd_keep[i], reserve=0 if i < 2 else 1),
                    actions_per_bucket=2 if i < 3 else 3,
                    buckets=ZBucketSearch(min_buckets=min_z[i], max_buckets=max_z[i]),
                )
                for i in range(len(ranks))
            ],
        )

    def __call__(self, params: Iterable[float]) -> float:
        first = self.run(params, self.seeds[:2])
        if self.best_rank < 100000 and float(np.median(first)) > self.best_rank + 18:
            return robust_rank_score(first) + 5.0
        return robust_rank_score(first + self.run(params, self.seeds[2:]))


def run_pso(fun: Evaluator, n_eval: int, low: float = -1.8, high: float = 1.8) -> np.ndarray:
    n_params = len(fun.extract_active())
    if n_params == 0:
        return np.asarray([], dtype=float)
    particles = 5
    iters = max(1, int(np.ceil(n_eval / particles)))

    def objective(positions):
        return np.asarray([fun(position) for position in positions], dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options={"c1": 0.75, "c2": 0.55, "w": 0.78},
        bounds=(np.full(n_params, low), np.full(n_params, high)),
    )
    _, best_position = optimizer.optimize(objective, iters=iters, verbose=False)
    return np.asarray(best_position, dtype=float)


def run_cma(
    fun: Evaluator,
    x0: np.ndarray | None,
    n_eval: int,
    sigma: float,
    low: float = -1.6,
    high: float = 1.6,
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
            "seed": 212,
            "verbose": -9,
        },
        restarts=0,
        bipop=False,
    )
    return np.asarray(xopt, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=240)
    xopt = run_pso(fun, n_eval=74)

    active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 34), xopt=xopt)
    if active is not None and len(active):
        xopt = run_cma(fun, np.asarray(active, dtype=float), n_eval=88, sigma=0.32, low=-1.1, high=1.1)

    active = fun.set_up_new_init(0, rank_thr=int(fun.best_rank + 14), xopt=xopt)
    if active is not None and len(active):
        run_cma(fun, np.asarray(active, dtype=float), n_eval=58, sigma=0.16, low=-0.65, high=0.65)

    return fun.get_best()
