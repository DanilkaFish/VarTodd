import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.base_search import full_pso
from scripts.optimization_core.mcts_dao import Path as OptimizationPath


class _ImmediateExecutor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def map(self, function, values):
        return [function(value) for value in values]


class _Optimizer:
    def __init__(self, **unused):
        pass

    def optimize(self, objective, **unused):
        costs = objective(np.asarray([[0.25]], dtype=float))
        return float(costs[0]), np.asarray([0.75], dtype=float)


class _Evaluator:
    def __init__(self):
        self.merged = []

    def extract_active(self):
        return np.asarray([0.0], dtype=float)

    def merge_run_state_from(self, other):
        self.merged.append(other)


class FullPsoTests(unittest.TestCase):
    def test_tohpe_and_prefix_contracts(self):
        tohpe = full_pso.TohpeSearch(z_choices=7)
        prefix = full_pso.TohpePrefixSearch(
            actions_per_bucket=2,
            buckets=full_pso.ZBucketSearch(min_buckets=5, max_buckets=9, limit_bucket=12),
        )

        self.assertEqual(tohpe.z_choices, 7)
        self.assertFalse(hasattr(tohpe, "actions_per_bucket"))
        self.assertEqual(prefix.actions_per_bucket, 2)
        self.assertEqual(prefix.buckets.min_buckets, 5)
        self.assertEqual(prefix.buckets.max_buckets, 9)
        self.assertFalse(hasattr(full_pso, "TohpeEffSearch"))

    def test_tohpeprefix_search_is_disabled_by_default(self):
        search = full_pso.TohpePrefixSearch()

        self.assertEqual(search.pool.keep, 0)
        self.assertEqual(search.pool.reserve, 0)

    def test_todd_search_has_no_embedded_tohpe_sampling(self):
        self.assertFalse(hasattr(full_pso.ToddSearch(), "tohpe_sampling"))

    def test_full_pso_configures_opt_in_tohpeprefix(self):
        source = Path(full_pso.__file__).read_text()
        self.assertIn("TohpePrefixSearch(", source)
        self.assertIn("set_tohpeprefix_search(", source)
        self.assertNotIn("TohpeEffSearch", source)
        self.assertIn("actions_per_bucket=2", source)
        self.assertNotIn("tohpe_sampling=", source)

    def test_path_band_reports_three_source_acceptance_and_pool_counts(self):
        group = {
            "start_rank": 10,
            "end_rank": 9,
            "steps": 1,
            "profile_id": "P1",
            "split_reasons": [],
            "red": [1.0],
            "red_max": [1.0],
            "basis_dim": [2.0],
            "accepted_tohpe": [3.0],
            "accepted_tohpeprefix": [5.0],
            "accepted_todd": [7.0],
            "researched_z": [11.0],
            "pool_size": [6.0],
            "pool_tohpe_size": [1.0],
            "pool_tohpeprefix_size": [2.0],
            "pool_todd_size": [3.0],
        }

        text = OptimizationPath._format_band([group], 1)

        self.assertIn("src=H:3/P:5/T:7", text)
        self.assertIn("pool:6(H1/P2/T3)", text)

    def test_robust_rank_cost_uses_best_rank_and_spread_penalty(self):
        ranks = [20, 20, 20, 20, 20, 40]

        self.assertAlmostEqual(
            full_pso.robust_rank_cost(ranks),
            20.0 + 0.3 * np.std(ranks),
        )

    def test_evaluator_default_scoring_uses_four_seeds(self):
        self.assertEqual(len(full_pso.Evaluator.seeds), 4)

    def test_z_bucket_ranges_are_ordered_at_opposing_budget_extremes(self):
        for z_budget, z_research_budget, z_limit_budget in ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)):
            minimum, maximum, limit = full_pso.z_bucket_ranges(
                z_budget, z_research_budget, z_limit_budget, buckets_space=1_000
            )

            self.assertLessEqual(minimum, maximum)
            self.assertLessEqual(maximum, limit)
            self.assertLessEqual(limit, 1_000)

    def test_z_bucket_ranges_cap_large_searches(self):
        minimum, maximum, limit = full_pso.z_bucket_ranges(
            1.0, 1.0, 1.0, buckets_space=1_000_000
        )

        self.assertEqual(
            (minimum, maximum, limit),
            (
                full_pso.Z_MIN_CAP,
                full_pso.Z_MIN_CAP + full_pso.Z_MAX_BUCKET_CAP - 1,
                full_pso.Z_LIMIT_BUCKET_CAP,
            ),
        )

    def test_run_opt_uses_heavy_swarm_settings(self):
        captured = {}

        class CapturingOptimizer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def optimize(self, objective, **unused):
                costs = objective(np.asarray([[0.25]], dtype=float))
                return float(costs[0]), np.asarray([0.75], dtype=float)

        evaluator = _Evaluator()
        local_evaluator = object()
        with patch.object(full_pso, "ThreadPoolExecutor", return_value=_ImmediateExecutor()), \
             patch.object(full_pso.ps.single, "GlobalBestPSO", CapturingOptimizer), \
             patch.object(full_pso, "_score_position", return_value=(3.0, local_evaluator)):
            full_pso.run_opt(evaluator, num_eval=1)

        self.assertEqual(captured["n_particles"], 36)
        self.assertEqual(captured["options"], {"c1": 0.9, "c2": 0.6, "w": 0.9})
        self.assertNotIn("velocity_clamp", captured)

    def test_entrypoint_uses_heavy_restart_budget(self):
        calls = []
        path = SimpleNamespace(final_node=SimpleNamespace(state=SimpleNamespace(rows=80)))

        class EntrypointEvaluator:
            init_rank = 100
            best_paths = [path]

            def set_up_new_init(self, *unused, **unused_kw):
                return object()

            def get_best(self):
                return "best"

        def fake_run_opt(fun, num_eval, label):
            calls.append((num_eval, label))
            return np.asarray([0.0], dtype=float)

        with patch.object(full_pso, "Evaluator", return_value=EntrypointEvaluator()), \
             patch.object(full_pso, "run_opt", side_effect=fake_run_opt):
            self.assertEqual(full_pso.entrypoint(object()), "best")

        self.assertEqual([num_eval for num_eval, _ in calls], [50, 50, 50, 50])

    def test_evaluator_call_retains_its_seed_set(self):
        evaluator = full_pso.Evaluator.__new__(full_pso.Evaluator)
        evaluator.evaluate = Mock(return_value=3.0)

        self.assertEqual(evaluator(np.asarray([0.0], dtype=float)), 3.0)
        evaluator.evaluate.assert_called_once_with(
            np.asarray([0.0], dtype=float),
            max_workers=1,
            seeds=evaluator.validation_seeds,
        )

    def test_run_opt_returns_pso_best_without_validation_candidates(self):
        evaluator = _Evaluator()
        local_evaluator = object()

        with patch.object(full_pso, "ThreadPoolExecutor", return_value=_ImmediateExecutor()), \
             patch.object(full_pso.ps.single, "GlobalBestPSO", _Optimizer), \
             patch.object(full_pso, "_score_position", return_value=(3.0, local_evaluator)):
            result = full_pso.run_opt(evaluator, num_eval=1)

        np.testing.assert_array_equal(result, np.asarray([0.75], dtype=float))
        self.assertEqual(evaluator.merged, [local_evaluator])


if __name__ == "__main__":
    unittest.main()
