import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.optimization_core.helper import BaseEvaluator


class FakePath:
    def branch_path(self, *unused):
        return object()


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


if __name__ == "__main__":
    unittest.main()
