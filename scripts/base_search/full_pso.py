import numpy as np
import pyswarms as ps

from scripts.base_search.full_optimizer_common import (
    BatchObjective,
    EliteArchive,
    Evaluator,
    Matrix,
    OPTIMIZER_WORKERS,
    Z_LIMIT_BUCKET_CAP,
    initial_population,
    run_two_phase,
)


PSO_PARTICLES = 48
PSO_WORKERS = OPTIMIZER_WORKERS
PSO_RANDOM_SEED = 42
PSO_OPTIONS = {
    "w": 0.7298,
    "c1": 1.49618,
    "c2": 1.49618,
    "k": 3,
    "p": 2,
}
PSO_VELOCITY_CLAMP = (-0.20, 0.20)


def iterations_for_budget(budget: int) -> int:
    iterations, remainder = divmod(int(budget), PSO_PARTICLES)
    if remainder:
        raise ValueError("PSO budget must be divisible by the particle count")
    return iterations


def run_opt(
    fun: Evaluator,
    budget: int,
    archive: EliteArchive,
    phase: int,
) -> np.ndarray:
    dimensions = len(fun.active_params)
    iterations = iterations_for_budget(budget)
    bounds = (np.zeros(dimensions), np.ones(dimensions))
    init_pos = initial_population(
        fun,
        archive,
        size=PSO_PARTICLES,
        seed=PSO_RANDOM_SEED + phase,
    )
    print(
        f"pso phase {phase + 1}: optimizing {dimensions} parameters "
        f"with {PSO_PARTICLES} particles for {iterations} iterations"
    )

    with BatchObjective(fun, archive, workers=PSO_WORKERS) as objective:
        optimizer = ps.single.LocalBestPSO(
            n_particles=PSO_PARTICLES,
            dimensions=dimensions,
            options=PSO_OPTIONS,
            bounds=bounds,
            velocity_clamp=PSO_VELOCITY_CLAMP,
            bh_strategy="reflective",
            init_pos=init_pos,
        )
        _, best_position = optimizer.optimize(
            objective,
            iters=iterations,
            verbose=True,
        )
    return np.asarray(best_position, dtype=float)


def entrypoint(mat: Matrix):
    return run_two_phase(Evaluator(mat=mat, max_depth=1000), run_opt)
