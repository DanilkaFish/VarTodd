import cma
import numpy as np

from scripts.base_search.full_optimizer_common import (
    BatchObjective,
    ELITE_COUNT,
    EliteArchive,
    Evaluator,
    Matrix,
    OPTIMIZER_WORKERS,
    PHASE_EVALUATIONS,
    initial_population,
    reset_optimizer_random_state,
    run_two_phase,
)


CMA_POPULATION = 16
CMA_SIGMA = 0.25
CMA_ACTIVE = True
CMA_BIPOP = True
CMA_RESTARTS = 9
CMA_RANDOM_SEED = 242


def run_opt(
    fun: Evaluator,
    budget: int,
    archive: EliteArchive,
    phase: int,
) -> np.ndarray:
    dimensions = len(fun.active_params)
    seed_count = max(CMA_RESTARTS + 1, ELITE_COUNT + 1)
    restart_seeds = initial_population(
        fun,
        archive,
        size=seed_count,
        seed=CMA_RANDOM_SEED + phase,
    )
    current = restart_seeds[0]
    elite_seeds = [
        position
        for position in archive.projected(fun.active_params)
        if not np.allclose(position, current, rtol=0.0, atol=1e-12)
    ][: CMA_POPULATION - 1]
    center_index = 0

    def next_center():
        nonlocal center_index
        center = restart_seeds[center_index % len(restart_seeds)]
        center_index += 1
        return center.copy()

    def inject_elites(strategy):
        if len(elite_seeds):
            strategy.inject(elite_seeds, force=True)

    print(
        f"cma-es phase {phase + 1}: optimizing {dimensions} parameters "
        f"with an approximately {budget}-evaluation BIPOP budget"
    )
    options = {
        "bounds": [0.0, 1.0],
        "popsize": CMA_POPULATION,
        "CMA_active": CMA_ACTIVE,
        "maxfevals": int(budget),
        "seed": CMA_RANDOM_SEED + phase,
        "verbose": 1,
        "verb_disp": 1,
        "verb_log": 0,
    }

    reset_optimizer_random_state(CMA_RANDOM_SEED + phase)
    with BatchObjective(fun, archive, workers=OPTIMIZER_WORKERS) as objective:
        best_position, _ = cma.fmin2(
            None,
            next_center,
            CMA_SIGMA,
            options=options,
            restarts=CMA_RESTARTS,
            bipop=CMA_BIPOP,
            parallel_objective=lambda positions: objective(positions).tolist(),
            init_callback=inject_elites,
        )
    return np.asarray(best_position, dtype=float)


def entrypoint(mat: Matrix):
    return run_two_phase(Evaluator(mat=mat, max_depth=1000), run_opt)
