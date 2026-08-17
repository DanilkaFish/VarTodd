# VarTODD FastTODD Mimic Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Vandaele-aligned sampling defaults and live T-count progress to the Python FastTODD mimic.

**Architecture:** Keep all changes inside `scripts/benchmarks/vartodd_fasttodd_mimic.py`. Named constants document the distinct TOHPE and FastTODD sampling defaults, while a small stateful reporter writes distinct T-count events to stderr.

**Tech Stack:** Python 3, argparse, existing pyvartodd bindings.

## Global Constraints

- Use TOHPE sampling `[1, 0, 0, 2]` as the closest current approximation to Vandaele's single dependency.
- Use FastTODD sampling `["all", 0, 0, 2]` for exhaustive individual basis solutions.
- Preserve command-line overrides, action selection, bucket search, seeding, timing, output matrices, and CSV fields.
- Write progress only to stderr and flush immediately.
- Do not add or run tests, per the user's explicit instruction.

---

### Task 1: Sampling defaults and live progress

**Files:**
- Modify: `scripts/benchmarks/vartodd_fasttodd_mimic.py`

**Interfaces:**
- Produces: `VANDAELE_TOHPE_SAMPLING`, `VANDAELE_FASTTODD_SAMPLING`, and `_TCountReporter`.
- Preserves: `run_mimic(matrix_path, args) -> dict[str, object]`, existing CLI flags, and CSV schema.

- [ ] **Step 1: Define named Vandaele sampling constants**

Add immutable string tuples near `CSV_FIELDS`:

```python
VANDAELE_TOHPE_SAMPLING = ("1", "0", "0", "2")
VANDAELE_FASTTODD_SAMPLING = ("all", "0", "0", "2")
```

- [ ] **Step 2: Add a distinct-count stderr reporter**

Add a helper that remembers the most recent count and prints only changes:

```python
class _TCountReporter:
    def __init__(self) -> None:
        self._last: int | None = None

    def emit(self, t_count: int, label: str) -> None:
        if self._last == t_count:
            return
        self._last = t_count
        print(f"T-count {t_count}: {label}", file=sys.stderr, flush=True)
```

- [ ] **Step 3: Emit progress at action boundaries**

Create the reporter after matrix loading and emit `initial`. After each accepted TOHPE state, increment `action_in_stage` and emit `TOHPE stage {stages}, action {action_in_stage}`. After each accepted full-TODD state, increment `todd_actions` and emit `FastTODD stage {stages}, action {todd_actions}`.

- [ ] **Step 4: Connect argparse defaults to the named constants**

Set `--tohpe-sampling` default to `list(VANDAELE_TOHPE_SAMPLING)` and `--todd-sampling` default to `list(VANDAELE_FASTTODD_SAMPLING)`. Leave `_sampling_budget` conversion and explicit CLI overrides unchanged.

- [ ] **Step 5: Inspect and commit the scoped change**

Run `git diff --check -- scripts/benchmarks/vartodd_fasttodd_mimic.py` and inspect the patch. Do not execute tests or benchmark scripts.

```bash
git add scripts/benchmarks/vartodd_fasttodd_mimic.py docs/superpowers/plans/2026-08-17-vartodd-fasttodd-mimic-progress.md
git commit -m "feat: report FastTODD mimic T-count progress"
```
