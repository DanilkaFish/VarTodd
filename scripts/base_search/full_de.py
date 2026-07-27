import numpy as np
from scipy.optimize import differential_evolution

from scripts.base_search.full_optimizer_common import (
    BatchObjective,
    EliteArchive,
    Evaluator,
    Matrix,
    OPTIMIZER_WORKERS,
    initial_population,
    reset_optimizer_random_state,
    run_two_phase,
)


DE_POPULATION = 48
DE_RANDOM_SEED = 142
DE_STRATEGY = "rand1bin"
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.8


def generations_for_budget(budget: int) -> int:
    population_batches, remainder = divmod(int(budget), DE_POPULATION)
    if remainder or population_batches < 1:
        raise ValueError(
            "DE budget must be a positive multiple of the population size"
        )
    return population_batches - 1


def run_opt(
    fun: Evaluator,
    budget: int,
    archive: EliteArchive,
    phase: int,
) -> np.ndarray:
    dimensions = len(fun.active_params)
    generations = generations_for_budget(budget)
    init = initial_population(
        fun,
        archive,
        size=DE_POPULATION,
        seed=DE_RANDOM_SEED + phase,
    )
    print(
        f"de phase {phase + 1}: optimizing {dimensions} parameters "
        f"with {DE_POPULATION} candidates for {generations} generations"
    )

    reset_optimizer_random_state(DE_RANDOM_SEED + phase)
    with BatchObjective(fun, archive, workers=OPTIMIZER_WORKERS) as objective:

        def vectorized_objective(positions):
            return objective(np.asarray(positions, dtype=float).T)

        result = differential_evolution(
            vectorized_objective,
            bounds=[(0.0, 1.0)] * dimensions,
            strategy=DE_STRATEGY,
            maxiter=generations,
            mutation=DE_MUTATION,
            recombination=DE_RECOMBINATION,
            rng=np.random.default_rng(DE_RANDOM_SEED + phase),
            disp=True,
            polish=False,
            init=init,
            tol=0.0,
            atol=-1.0,
            updating="deferred",
            workers=1,
            vectorized=True,
        )
    return np.asarray(result.x, dtype=float)


def entrypoint(mat: Matrix):
    return run_two_phase(Evaluator(mat=mat, max_depth=1000), run_opt)
