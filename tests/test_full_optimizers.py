import unittest

import numpy as np

from scripts.base_search.full_optimizer_common import (
    EliteArchive,
    PHASE_EVALUATIONS,
    from_unit,
    initial_population,
    to_unit,
    run_two_phase,
)


class CoordinateConversionTests(unittest.TestCase):
    def test_native_unit_round_trip(self):
        native = np.asarray([-1.0, -0.25, 0.0, 0.75, 1.0])

        np.testing.assert_allclose(from_unit(to_unit(native)), native)


class EliteArchiveTests(unittest.TestCase):
    def test_nearby_candidate_is_replaced_only_by_lower_cost(self):
        archive = EliteArchive(capacity=2, min_distance=0.05)
        active = [0, 2]
        archive.add(10.0, [-1.0, 0.0, -0.5, 0.0], active)
        archive.add(12.0, [-0.98, 0.0, -0.48, 0.0], active)
        archive.add(5.0, [-0.96, 0.0, -0.46, 0.0], active)
        archive.add(7.0, [1.0, 0.0, 1.0, 0.0], active)

        projected = archive.projected([2, 0])

        self.assertEqual(len(projected), 2)
        np.testing.assert_allclose(projected[0], to_unit([-0.46, -0.96]))
        np.testing.assert_allclose(projected[1], [1.0, 1.0])

    def test_projection_uses_the_new_active_parameter_order(self):
        archive = EliteArchive(capacity=2, min_distance=0.05)
        archive.add(1.0, [-1.0, -0.5, 0.0, 0.5, 1.0], [0, 1])

        projected = archive.projected([4, 2, 0])

        np.testing.assert_allclose(projected, [[1.0, 0.5, 0.0]])


class InitialPopulationTests(unittest.TestCase):
    def test_current_position_and_elites_precede_latin_hypercube_fill(self):
        fun = type(
            "FakeEvaluator",
            (),
            {
                "x0": [-1.0, 0.0, 1.0, -0.5],
                "active_params": [0, 2],
                "extract_active": lambda self: np.asarray(
                    [self.x0[index] for index in self.active_params]
                ),
            },
        )()
        archive = EliteArchive(capacity=2, min_distance=0.05)
        archive.add(2.0, [0.0, 0.0, -1.0, 0.0], [0, 2])

        population = initial_population(fun, archive, size=5, seed=123)

        self.assertEqual(population.shape, (5, 2))
        np.testing.assert_allclose(population[0], [0.0, 1.0])
        np.testing.assert_allclose(population[1], [0.5, 0.0])
        self.assertTrue(np.all((population >= 0.0) & (population <= 1.0)))


class TwoPhaseTests(unittest.TestCase):
    def test_runs_equal_phases_around_one_midpoint_path_restart(self):
        final_node = type("Node", (), {"state": type("State", (), {"rows": 40})()})()
        best_path = type("Path", (), {"final_node": final_node})()

        class FakeEvaluator:
            def __init__(self):
                self.init_rank = 100
                self.best_paths = []
                self.set_up_calls = []

            def set_up_new_init(self, path_num, rank_thr, xopt=None):
                self.set_up_calls.append((path_num, rank_thr, xopt))
                self.init_rank = rank_thr + 1
                return np.asarray([0.0])

            def get_best(self):
                return "best-result"

        fun = FakeEvaluator()
        phase_calls = []

        def optimize_phase(evaluator, budget, archive, phase):
            phase_calls.append((budget, archive, phase))
            if phase == 0:
                evaluator.best_paths = [best_path]
            return np.asarray([0.5])

        result = run_two_phase(fun, optimize_phase)

        self.assertEqual(result, "best-result")
        self.assertEqual([budget for budget, _, _ in phase_calls], [3600, 3600])
        self.assertIs(phase_calls[0][1], phase_calls[1][1])
        self.assertEqual([phase for _, _, phase in phase_calls], [0, 1])
        self.assertEqual(fun.set_up_calls, [(0, 70, None)])
        self.assertEqual(PHASE_EVALUATIONS, 3600)


class PsoConfigurationTests(unittest.TestCase):
    def test_local_ring_configuration_and_budget(self):
        from scripts.base_search import full_pso

        self.assertEqual(full_pso.PSO_PARTICLES, 48)
        self.assertEqual(full_pso.PSO_OPTIONS["k"], 3)
        self.assertEqual(full_pso.PSO_OPTIONS["p"], 2)
        self.assertAlmostEqual(full_pso.PSO_OPTIONS["w"], 0.7298)
        self.assertAlmostEqual(full_pso.PSO_OPTIONS["c1"], 1.49618)
        self.assertAlmostEqual(full_pso.PSO_OPTIONS["c2"], 1.49618)
        self.assertEqual(full_pso.PSO_VELOCITY_CLAMP, (-0.20, 0.20))
        self.assertEqual(full_pso.iterations_for_budget(3_600), 75)


class DeConfigurationTests(unittest.TestCase):
    def test_rugged_search_configuration_and_budget(self):
        from scripts.base_search import full_de

        self.assertEqual(full_de.DE_POPULATION, 48)
        self.assertEqual(full_de.DE_STRATEGY, "rand1bin")
        self.assertEqual(full_de.DE_MUTATION, (0.5, 1.0))
        self.assertAlmostEqual(full_de.DE_RECOMBINATION, 0.8)
        self.assertEqual(full_de.generations_for_budget(3_600), 74)


class CmaConfigurationTests(unittest.TestCase):
    def test_bipop_active_configuration_and_budget(self):
        from scripts.base_search import full_cmaes

        self.assertEqual(full_cmaes.CMA_POPULATION, 16)
        self.assertAlmostEqual(full_cmaes.CMA_SIGMA, 0.25)
        self.assertTrue(full_cmaes.CMA_ACTIVE)
        self.assertTrue(full_cmaes.CMA_BIPOP)
        self.assertEqual(full_cmaes.CMA_RESTARTS, 9)
        self.assertEqual(full_cmaes.PHASE_EVALUATIONS, 3_600)


if __name__ == "__main__":
    unittest.main()
