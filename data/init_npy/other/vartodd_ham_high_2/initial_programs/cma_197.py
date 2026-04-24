from typing import Iterable
import cma
import numpy as np
import random
from helper import BaseEvaluator, ExplorationScore, FinalizationScore

np.random.seed(42)
random.seed(40)


def _w_tanh(z: float, scale: float = 2.4, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(3)]

    def policy_mapping(self):
        split = max(1, int(self.init_rank * 0.80))
        ranks = [split, 0]

        wred = self.map_par(_w_tanh, 0)
        wdim = self.map_par(_w_tanh, 0)
        fwred = self.map_par(_w_tanh, 0)
        fwdim = self.map_par(_w_tanh, 0)
        fwtohpe = self.map_par(_w_tanh, 0)
        budget = self.map_par(sigmoid, 0)

        self.set_pool_scores(
            ranks,
            [
                ExplorationScore(weights=[wred, wdim, 0.0, 0.0, 0.0], pow=1),
                ExplorationScore(weights=[wred, 0.5 * wdim, 0.0, 0.0, 0.0], pow=1),
            ],
        )
        self.set_final_scores(
            ranks,
            [
                FinalizationScore(
                    weights=[fwred, fwdim, 0.0, 0.0, 0.0, fwtohpe],
                    centers=[1, 0, 0, 0, 0, 0],
                    pow=2,
                ),
                FinalizationScore(
                    weights=[fwred, 0.5 * fwdim, 0.0, 0.0, 0.0, 0.0],
                    centers=[1, 0, 0, 0, 0, 0],
                    pow=2,
                ),
            ],
        )

        beam = int(1 + budget)
        width = int(1 + budget)
        self.set_min_pool_size(1)
        self.set_min_z_to_research(ranks, [24 + 70 * budget, 14 + 36 * budget])
        self.set_max_z_to_research(20000)
        self.set_temperature(0.4)
        self.set_num_samples(ranks, [5, 4])
        self.set_max_pool_size(ranks, [5 + beam, 4 + beam])
        self.set_max_tohpe(ranks, [3, 2])
        self.set_try_only_tohpe(ranks, [1, 1])
        self.set_max_reduction(7)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(2)
        self.set_tohpe_num_best(ranks, [4, 3])
        self.set_gen_part(ranks, [0.16 + 0.22 * budget, 0.12 + 0.10 * budget])
        self.set_todd_width(ranks, [width, 1])
        self.set_beamsearch_width(ranks, [beam, 1])

    def __call__(self, params: Iterable):
        first = self.run(params, self.seeds[:3])
        if self.best_rank < 100000 and first[0] > self.best_rank + 18:
            return float(first[0] + 6.0)
        rest = self.run(params, self.seeds[3:])
        tcounts = first + rest
        return float(np.quantile(tcounts, 0.7) + 0.02 * np.std(tcounts))


def run_cma(fun: Evaluator, maxfevals: int = 45, sigma: float = 0.35) -> np.ndarray:
    x0 = np.zeros_like(fun.extract_active(), dtype=float)

    def objective(x):
        if x.ndim == 1:
            return float(fun(x))
        return np.asarray([fun(xi) for xi in x], dtype=float)

    xopt, _ = cma.fmin2(
        objective,
        x0,
        sigma,
        options={
            "maxfevals": maxfevals,
            "popsize": 5,
            "bounds": [-1.2, 1.2],
            "verbose": 0,
            "seed": 42,
            "tolfun": 1e-4,
            "tolx": 1e-4,
        },
        restarts=0,
        bipop=False,
    )
    return np.asarray(xopt, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=170)
    xopt = run_cma(fun, maxfevals=20, sigma=0.35)
    fun(xopt)
    return fun.get_best()
