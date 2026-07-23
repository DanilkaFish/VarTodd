import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.optimization_core.helper import BaseEvaluator, _worker_run_one_from_template
from scripts.optimization_core.todd import Todd


class FakePath:
    def branch_path(self, *unused):
        return object()


class ToddPath:
    def __init__(self, root):
        self.final_node = root


def make_evaluator(best_rank=1000):
    evaluator = BaseEvaluator.__new__(BaseEvaluator)
    evaluator.total_eval = 0
    evaluator.tcount = []
    evaluator._best_rank = best_rank
    evaluator.best_paths = []
    evaluator.best_ranks = []
    evaluator.best_evals = []
    evaluator.best_eval = 0
    evaluator.best_seen = 0
    evaluator.best_seed = None
    evaluator.current_path = FakePath()
    evaluator.dao = object()
    evaluator.x0 = []
    evaluator._search_started_at = 100.0
    evaluator.time_to_final_rank_seconds = None
    return evaluator


class EvaluatorTimingTests(unittest.TestCase):
    def test_todd_report_without_timing_keeps_legacy_tuple_shape(self):
        root = SimpleNamespace(state=SimpleNamespace(rows=10))
        dao = SimpleNamespace()

        result = Todd(dao, depth=0).run(ToddPath(root), True, 7)

        self.assertEqual(result, (root, (0, 0)))

    def test_todd_timing_reports_the_child_improvement_instant(self):
        root = SimpleNamespace(state=SimpleNamespace(rows=10))
        first_child = SimpleNamespace(state=SimpleNamespace(rows=8))
        root.add_child = lambda *, state, incoming: first_child
        first_child.incoming = SimpleNamespace(cand=SimpleNamespace(final_score=0.0))
        dao = SimpleNamespace(
            policy_config_at=lambda **unused: SimpleNamespace(
                selection=SimpleNamespace(count=1)
            )
        )
        improvement = SimpleNamespace(
            chosen=[SimpleNamespace(final_score=0.0)],
            states=[first_child.state],
            stats=object(),
        )
        exhausted = SimpleNamespace(chosen=[], states=[], stats=object())

        perf_counter = Mock(side_effect=[100.0, 101.5])
        with (
            patch(
                "scripts.optimization_core.todd.policy_iteration",
                side_effect=[improvement, exhausted],
            ) as policy_iteration,
            patch(
                "scripts.optimization_core.todd.time",
                SimpleNamespace(perf_counter=perf_counter),
                create=True,
            ),
        ):
            node, counters, discovered_at = Todd(dao, depth=2).run(
                ToddPath(root),
                with_report=True,
                with_timing=True,
                seed=7,
            )

        self.assertIs(node, first_child)
        self.assertEqual(counters, (1, 1))
        self.assertEqual(discovered_at, 101.5)
        self.assertEqual(policy_iteration.call_count, 2)
        self.assertEqual(perf_counter.call_count, 2)

    def test_worker_propagates_todd_discovery_timestamp(self):
        todd = SimpleNamespace()
        node = SimpleNamespace(state=SimpleNamespace(rows=400))
        path = object()
        todd.run = Mock(return_value=(node, (3, 1), 112.5))

        with patch(
            "scripts.optimization_core.helper.time.perf_counter",
            side_effect=AssertionError("worker must not replace TODD's timestamp"),
        ):
            result = _worker_run_one_from_template(7, path, todd)

        self.assertEqual(result, (7, node, (3, 1), 112.5))
        todd.run.assert_called_once_with(
            path,
            with_report=True,
            with_timing=True,
            seed=7,
        )

    def test_lower_rank_records_elapsed_time(self):
        evaluator = make_evaluator()
        node = SimpleNamespace(state=SimpleNamespace(rows=400))
        with patch("scripts.optimization_core.helper.time.perf_counter", return_value=117.25):
            evaluator._record_run_result(7, node, (3, 1), [])
        self.assertEqual(evaluator.best_rank, 400)
        self.assertEqual(evaluator.time_to_final_rank_seconds, 17.25)

    def test_equal_rank_merge_keeps_earliest_discovery_time(self):
        evaluator = make_evaluator(400)
        evaluator.time_to_final_rank_seconds = 17.25
        other = make_evaluator(400)
        other.time_to_final_rank_seconds = 11.5
        other.total_eval = 2
        other.tcount = [400]
        other.best_paths = [object()]
        other.best_seen = 1
        with patch("scripts.optimization_core.helper._copy_path_headers", side_effect=list):
            evaluator.merge_run_state_from(other)
        self.assertEqual(evaluator.time_to_final_rank_seconds, 11.5)

    def test_run_keeps_earliest_discovery_for_equal_final_ranks(self):
        evaluator = make_evaluator()
        evaluator.active_params = []
        evaluator.insert = lambda unused: None
        evaluator.reinit = lambda: None
        evaluator.todd = object()
        node = SimpleNamespace(state=SimpleNamespace(rows=400))
        worker_results = [
            # Seed-order processing sees this later discovery first.
            (1, node, (3, 1), 150.0),
            # This result was discovered earlier, but is processed second after sorting.
            (2, node, (3, 1), 125.0),
        ]

        with patch(
            "scripts.optimization_core.helper._worker_run_one_from_template",
            side_effect=worker_results,
        ):
            self.assertEqual(evaluator.run([], [1, 2], max_workers=1), [400, 400])

        self.assertEqual(evaluator.time_to_final_rank_seconds, 25.0)


if __name__ == "__main__":
    unittest.main()
