# Standard Runner JSON Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Emit one JSON report from `run_standard.py` containing the rank and timing information for every selected problem.

**Architecture:** `BaseEvaluator` measures elapsed monotonic time at which it first discovers a better rank and exposes the final value in its report. The runner extracts the value without changing the optimizer tuple API and publishes ordered JSON records after all jobs complete.

**Tech Stack:** Python 3, NumPy, `argparse`, `json`, `re`, `time`, and `unittest`.

## Global Constraints

- `--output PATH` is required and points directly to the JSON report.
- The optional optimizer module defaults to `scripts/base_search/full_pso.py`.
- Default discovery selects `gf2^<degree>` names with degree below 32, ordered by degree then name.
- Matrices remain `.npy` files beside the report and records use an absolute `result_path`.
- `execution_seconds` covers the entrypoint. `time_to_final_rank_seconds` is the first discovery time for the final rank.
- Optimizer entrypoints continue returning `(result, report, best_path)`.

---

### Task 1: Track time to the final rank in `BaseEvaluator`

**Files:**

- Create: `tests/test_evaluator_timing.py`
- Modify: `scripts/optimization_core/helper.py:734-743,932-982,1108-1119`

**Interfaces:**

- Produces `BaseEvaluator.time_to_final_rank_seconds: float | None`.
- Includes `time_to_final_rank_seconds: <seconds>` in the existing `get_best()` text.

- [ ] **Step 1: Write the failing regression tests**

```python
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
```

- [ ] **Step 2: Verify the test is red**

Run: `python -m unittest tests.test_evaluator_timing -v`.

Expected: FAIL because the implementation does not initialize or merge the timing field.

- [ ] **Step 3: Add the minimal timing implementation**

Initialize the state in `BaseEvaluator.__init__` immediately after `best_seen`:

```python
self._search_started_at = time.perf_counter()
self.time_to_final_rank_seconds: Optional[float] = None
```

When `rank < self._best_rank` in `_record_run_result`, save this before modifying the rank:

```python
self.time_to_final_rank_seconds = time.perf_counter() - self._search_started_at
```

When merging equal ranks, retain the earlier non-null time:

```python
other_time = other.time_to_final_rank_seconds
if other_time is not None and (
    self.time_to_final_rank_seconds is None
    or other_time < self.time_to_final_rank_seconds
):
    self.time_to_final_rank_seconds = other_time
```

On a strictly lower merged rank, copy `other.time_to_final_rank_seconds`. Add a `time_to_final_rank_seconds: <seconds>` line to `get_best()` after the current search statistics; use `n/a` only when the value is absent.

- [ ] **Step 4: Verify the test is green**

Run: `python -m unittest tests.test_evaluator_timing -v`.

Expected: PASS with two tests.

- [ ] **Step 5: Commit this change**

Stage `scripts/optimization_core/helper.py` and `tests/test_evaluator_timing.py`, then commit with message `Track time to final evaluator rank`.

### Task 2: Add direct JSON output to the standard runner

**Files:**

- Create: `tests/test_run_standard.py`
- Modify: `scripts/optimization_core/run_standard.py:1-198`

**Interfaces:**

- Produces `discover_names(init_circuit, stop_before_gf_degree) -> list[str]`.
- Produces `_extract_time_to_final_rank(paths) -> float | None`.
- Produces `_write_json_report(output_path, records) -> None`.
- Changes `_run_one` to return `(record, tcount)`.

- [ ] **Step 1: Write the failing runner tests**

```python
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
            entry_result = (result, "report", "time_to_final_rank_seconds: 1.25")
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
        self.assertIn("time_to_final_rank_seconds", saved[0]["paths"])
        self.assertTrue(result_exists)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify the runner tests are red**

Run: `python -m unittest tests.test_run_standard -v`.

Expected: FAIL because discovery uses dimensions, `--output` is optional, the optimizer has no default, and report helpers are absent.

- [ ] **Step 3: Implement selection, record construction, and JSON publication**

Add `json`, `re`, and `time` imports plus constants for the GF-name expression, the timing expression, and the default stop degree of 32. Replace dimension filtering with this behavior:

```python
def discover_names(init_circuit: str, stop_before_gf_degree: int) -> list[str]:
    root = DATA_ROOT / init_circuit
    records = []
    for path in root.rglob("*.npy"):
        name = path.relative_to(root).with_suffix("").as_posix()
        match = GF_DEGREE_RE.match(Path(name).name)
        if match is None:
            raise ValueError(f"cannot determine GF degree from problem name: {name}")
        degree = int(match.group(1))
        if degree < stop_before_gf_degree:
            records.append((degree, name))
    return [name for _, name in sorted(records)]
```

Make the `module_path` positional argument optional with default `scripts/base_search/full_pso.py`. Replace `--m-init`, `--n-init`, and `--output-root` with required `--output` and optional `--stop-before-gf-degree` arguments. Implement report helpers:

```python
def _extract_time_to_final_rank(paths: str) -> float | None:
    match = TIME_TO_FINAL_RANK_RE.search(paths)
    return float(match.group(1)) if match else None


def _write_json_report(output_path: Path, records: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
```

Time `entrypoint` with `time.perf_counter()` in `_run_one`; change `validate` to expose `paths`; save `.npy` to `output_path.parent`; return a record with `problem_name`, `initial_rank`, `final_rank`, `execution_seconds`, `time_to_final_rank_seconds`, `paths`, and absolute `result_path`. In `main`, calculate initial row counts before submitting jobs, retain each future's input index, and call `_write_json_report` with records in original selection order only after all jobs succeed.

- [ ] **Step 4: Verify the runner tests are green**

Run: `python -m unittest tests.test_run_standard -v`.

Expected: PASS with four tests.

- [ ] **Step 5: Commit this change**

Stage `scripts/optimization_core/run_standard.py` and `tests/test_run_standard.py`, then commit with message `Add standard-run JSON reports`.

### Task 3: Verify the completed behavior

**Files:**

- Test: `tests/test_evaluator_timing.py`
- Test: `tests/test_run_standard.py`

**Interfaces:**

- Consumes the evaluator timing and JSON report APIs.
- Produces verification evidence for every requested behavior.

- [ ] **Step 1: Run all focused tests**

Run: `python -m unittest tests.test_evaluator_timing tests.test_run_standard -v`.

Expected: PASS with six tests and no failures.

- [ ] **Step 2: Validate default selection without invoking an optimizer**

Run: `python scripts/optimization_core/run_standard.py scripts/base_search/full_pso.py --output /tmp/run-standard-check.json --list-only`.

Expected: names from GF(2^3) through GF(2^16) appear in numeric degree order, no GF(2^32) name appears, and list-only mode does not create the report.

- [ ] **Step 3: Compile and inspect changed source**

Run: `python -m py_compile scripts/optimization_core/helper.py scripts/optimization_core/run_standard.py tests/test_evaluator_timing.py tests/test_run_standard.py`.

Expected: exit status 0.

Run: `git diff --check`.

Expected: no whitespace errors.

- [ ] **Step 4: Commit final verification state**

Stage the four source and test files, then commit with message `Verify standard runner reporting` if any uncommitted implementation changes remain.
