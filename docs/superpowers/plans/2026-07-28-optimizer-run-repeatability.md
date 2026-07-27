# Optimizer Run Repeatability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make repeated PSO, CMA-ES, and differential-evolution runs produce the same optimizer trajectory and result when inputs, configuration, and dependency versions are unchanged.

**Architecture:** Add one shared phase-seeding function for Python's and NumPy's process-global RNGs. Invoke it immediately before each third-party optimizer starts, while retaining the existing private LHS and differential-evolution generators and all current optimizer settings.

**Tech Stack:** Python 3.12, NumPy, SciPy, PySwarms, CMA, `unittest`, `unittest.mock`.

## Global Constraints

- Preserve optimizer hyperparameters, budgets, population sizes, score formulas, and thread counts.
- Preserve CMA's explicit seed and differential evolution's private `numpy.random.default_rng`.
- Cross-version and cross-platform floating-point differences remain allowed.
- Wall-clock timing and soft-deadline interruption remain timing-dependent.
- Work in the current workspace; do not create an isolated worktree.

---

### Task 1: Shared RNG reset and PSO repeatability

**Files:**
- Modify: `scripts/base_search/full_optimizer_common.py:42-44`
- Modify: `scripts/base_search/full_pso.py:4-13,45-65`
- Create: `tests/test_optimizer_repeatability.py`

**Interfaces:**
- Produces: `reset_optimizer_random_state(seed: int) -> None`
- Consumes: `PSO_RANDOM_SEED + phase`

- [ ] **Step 1: Write failing tests for the shared RNG contract and PSO velocity initialization**

```python
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

        np.testing.assert_array_equal(first.swarm.velocity, second.swarm.velocity)
```

- [ ] **Step 2: Run the two tests and verify the missing helper fails**

Run:

```bash
.venv/bin/python -B -m unittest \
  tests.test_optimizer_repeatability.OptimizerRepeatabilityTests.test_reset_optimizer_random_state_repeats_python_and_numpy_streams \
  tests.test_optimizer_repeatability.OptimizerRepeatabilityTests.test_pyswarms_velocity_repeats_after_phase_seed_reset -v
```

Expected: both tests error with `AttributeError` because
`reset_optimizer_random_state` does not exist.

- [ ] **Step 3: Add the minimal shared RNG reset**

```python
def reset_optimizer_random_state(seed: int) -> None:
    phase_seed = int(seed)
    random.seed(phase_seed)
    np.random.seed(phase_seed % (2**32))
```

- [ ] **Step 4: Import and invoke the helper in `full_pso.run_opt`**

Add `reset_optimizer_random_state` to the import from
`full_optimizer_common`. Immediately before constructing `LocalBestPSO`, add:

```python
reset_optimizer_random_state(PSO_RANDOM_SEED + phase)
optimizer = ps.single.LocalBestPSO(
    # existing arguments unchanged
)
```

- [ ] **Step 5: Run the two tests and verify they pass**

Run the command from Step 2.

Expected: both tests pass.

- [ ] **Step 6: Commit the first deterministic boundary**

```bash
git add \
  scripts/base_search/full_optimizer_common.py \
  scripts/base_search/full_pso.py \
  tests/test_optimizer_repeatability.py
git commit -m "fix: seed full pso phase randomness"
```

### Task 2: Apply the phase reset to CMA-ES and differential evolution

**Files:**
- Modify: `scripts/base_search/full_cmaes.py:4-14,61-82`
- Modify: `scripts/base_search/full_de.py:4-12,50-71`
- Modify: `tests/test_optimizer_repeatability.py`

**Interfaces:**
- Consumes: `reset_optimizer_random_state(seed: int) -> None`
- Uses: `CMA_RANDOM_SEED + phase`
- Uses: `DE_RANDOM_SEED + phase`

- [ ] **Step 1: Add failing source-boundary tests for all three optimizers**

```python
from pathlib import Path


def _source(module) -> str:
    return Path(module.__file__).read_text()


class OptimizerRepeatabilityTests(unittest.TestCase):
    # Keep the Task 1 tests.

    def test_every_optimizer_resets_its_phase_random_state(self):
        from scripts.base_search import full_cmaes, full_de, full_pso

        expectations = (
            (full_pso, "reset_optimizer_random_state(PSO_RANDOM_SEED + phase)"),
            (full_cmaes, "reset_optimizer_random_state(CMA_RANDOM_SEED + phase)"),
            (full_de, "reset_optimizer_random_state(DE_RANDOM_SEED + phase)"),
        )
        for module, call in expectations:
            with self.subTest(module=module.__name__):
                self.assertIn(call, _source(module))
```

- [ ] **Step 2: Run the boundary test and verify CMA-ES and DE fail**

Run:

```bash
.venv/bin/python -B -m unittest \
  tests.test_optimizer_repeatability.OptimizerRepeatabilityTests.test_every_optimizer_resets_its_phase_random_state -v
```

Expected: failure because the reset call is absent from `full_cmaes.py` and
`full_de.py`.

- [ ] **Step 3: Reset CMA-ES phase randomness**

Import `reset_optimizer_random_state` from `full_optimizer_common`. Immediately
before entering the optimizer call, add:

```python
reset_optimizer_random_state(CMA_RANDOM_SEED + phase)
with BatchObjective(fun, archive, workers=OPTIMIZER_WORKERS) as objective:
    # existing cma.fmin2 call unchanged
```

- [ ] **Step 4: Reset differential-evolution phase randomness**

Import `reset_optimizer_random_state` from `full_optimizer_common`. Immediately
before entering the optimizer call, add:

```python
reset_optimizer_random_state(DE_RANDOM_SEED + phase)
with BatchObjective(fun, archive, workers=OPTIMIZER_WORKERS) as objective:
    # existing differential_evolution call unchanged
```

- [ ] **Step 5: Run the new repeatability test module**

Run:

```bash
.venv/bin/python -B -m unittest tests.test_optimizer_repeatability -v
```

Expected: all repeatability tests pass.

- [ ] **Step 6: Run unaffected current tests**

Run:

```bash
.venv/bin/python -B -m unittest \
  tests.test_run_standard \
  tests.test_evaluator_timing -v
```

Expected: all listed tests pass. `tests.test_full_pso` is excluded because it
already targets the optimizer API from before the shared-common refactor and
fails on the unchanged baseline.

- [ ] **Step 7: Commit CMA-ES and DE phase isolation**

```bash
git add \
  scripts/base_search/full_cmaes.py \
  scripts/base_search/full_de.py \
  tests/test_optimizer_repeatability.py
git commit -m "fix: isolate optimizer phase random state"
```

### Task 3: Final repeatability verification

**Files:**
- Verify: `scripts/base_search/full_optimizer_common.py`
- Verify: `scripts/base_search/full_pso.py`
- Verify: `scripts/base_search/full_cmaes.py`
- Verify: `scripts/base_search/full_de.py`
- Verify: `tests/test_optimizer_repeatability.py`

**Interfaces:**
- Verifies: identical process-global Python and NumPy streams for equal phase seeds
- Verifies: identical PySwarms initial velocities for equal positions and phase seeds

- [ ] **Step 1: Run repeatability tests twice in fresh processes**

Run:

```bash
.venv/bin/python -B -m unittest tests.test_optimizer_repeatability -v
.venv/bin/python -B -m unittest tests.test_optimizer_repeatability -v
```

Expected: both fresh-process runs pass with identical test results.

- [ ] **Step 2: Run diff and whitespace validation**

Run:

```bash
git diff --check
git diff -- \
  scripts/base_search/full_optimizer_common.py \
  scripts/base_search/full_pso.py \
  scripts/base_search/full_cmaes.py \
  scripts/base_search/full_de.py \
  tests/test_optimizer_repeatability.py
```

Expected: no whitespace errors; diff contains only the shared helper, the three
phase-reset calls, imports, and repeatability tests.

- [ ] **Step 3: Inspect repository status without touching unrelated files**

Run:

```bash
git status --short
```

Expected: existing unrelated user files remain unchanged.
