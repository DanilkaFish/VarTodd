import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.optimization_core import run_standard
from scripts.optimization_core.helper import Matrix


class RunStandardTests(unittest.TestCase):
    def test_default_discovery_is_degree_ordered_and_stops_before_32(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            circuit = root / "gf_mult_Vandaele_wo_ancilla"
            circuit.mkdir()
            for name in ("gf2^16_1612310", "gf2^3_310", "gf2^32_3226310"):
                np.save(circuit / f"{name}.npy", np.zeros((2, 2), dtype=bool))
            with patch.object(run_standard, "DATA_ROOT", root):
                names = run_standard.discover_names("gf_mult_Vandaele_wo_ancilla", 32)
        self.assertEqual(names, ["gf2^3_310", "gf2^16_1612310"])

    def test_output_is_required(self):
        with self.assertRaises(SystemExit):
            run_standard._parse_args(["scripts/base_search/full_pso.py"])

    def test_default_optimizer_is_full_pso(self):
        args = run_standard._parse_args(["--output", "report.json"])
        self.assertEqual(args.module_path, "scripts/base_search/full_pso.py")

    def test_record_has_ranks_durations_paths_and_result_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "run.json"
            matrix = Matrix.from_numpy(np.zeros((4, 2), dtype=bool))
            result = np.zeros((3, 2), dtype=bool)
            entry_result = (
                result,
                "distinctive optimizer report fragment\n",
                "time_to_final_rank_seconds: 1.25",
            )
            module = type("Module", (), {"entrypoint": staticmethod(lambda _: entry_result)})
            with patch.object(run_standard, "import_module", return_value=module):
                with patch.object(run_standard, "get_matrix", return_value=matrix):
                    with patch.object(run_standard.time, "perf_counter", side_effect=[10.0, 13.5]):
                        record, _ = run_standard._run_one(
                            "gf2^3_310",
                            module_path="fake.module",
                            last_name="full_pso",
                            init_circuit="gf_mult_Vandaele_wo_ancilla",
                            output_path=output_path,
                            initial_rank=4,
                        )
                    run_standard._write_json_report(output_path, [record])
                    saved = json.loads(output_path.read_text())
                    result_exists = Path(saved[0]["result_path"]).is_file()
        self.assertEqual(saved[0]["problem_name"], "gf_mult_Vandaele_wo_ancilla/gf2^3_310")
        self.assertEqual(saved[0]["initial_rank"], 4)
        self.assertEqual(saved[0]["final_rank"], 3)
        self.assertEqual(saved[0]["execution_seconds"], 3.5)
        self.assertEqual(saved[0]["time_to_final_rank_seconds"], 1.25)
        self.assertIn("distinctive optimizer report fragment", saved[0]["paths"])
        self.assertIn("time_to_final_rank_seconds", saved[0]["paths"])
        self.assertTrue(result_exists)


if __name__ == "__main__":
    unittest.main()
