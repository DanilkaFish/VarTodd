import random
import unittest

import numpy as np
import pyswarms as ps

from scripts.base_search import full_optimizer_common


class OptimizerRepeatabilityTests(unittest.TestCase):
    def test_reset_optimizer_random_state_repeats_python_and_numpy_streams(self):
        full_optimizer_common.reset_optimizer_random_state(712)
        expected = (random.random(), np.random.random(4))

        random.random()
        np.random.random(17)

        full_optimizer_common.reset_optimizer_random_state(712)
        actual = (random.random(), np.random.random(4))

        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_pyswarms_velocity_repeats_after_phase_seed_reset(self):
        init = np.linspace(0.0, 1.0, 12, dtype=float).reshape(4, 3)
        kwargs = {
            "n_particles": 4,
            "dimensions": 3,
            "options": {"w": 0.7, "c1": 1.4, "c2": 1.4, "k": 2, "p": 2},
            "bounds": (np.zeros(3), np.ones(3)),
            "velocity_clamp": (-0.2, 0.2),
            "bh_strategy": "reflective",
            "init_pos": init,
        }

        full_optimizer_common.reset_optimizer_random_state(42)
        first = ps.single.LocalBestPSO(**kwargs)
        np.random.random(31)
        full_optimizer_common.reset_optimizer_random_state(42)
        second = ps.single.LocalBestPSO(**kwargs)

        np.testing.assert_array_equal(
            first.swarm.velocity,
            second.swarm.velocity,
        )


if __name__ == "__main__":
    unittest.main()
