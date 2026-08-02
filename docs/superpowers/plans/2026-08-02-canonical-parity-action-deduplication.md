# Canonical Parity Action Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Canonicalize applied parity matrices and merge final candidates that produce the same canonical state, retaining the highest-final-score representative.

**Architecture:** Add one exact matrix-level canonicalization operation that sorts row indices, removes zero rows, and cancels equal rows modulo two without copying inside the comparator. Change policy finalization to carry each applied canonical state beside its candidate, deduplicate by exact matrix equality after scoring, recompute unique-pool metadata/statistics, and select directly from those finalized actions.

**Tech Stack:** C++20, existing compact `Matrix`/`RowCView` types, CMake/CTest core regression executable.

## Global Constraints

- Equality is exact after zero removal, duplicate-pair cancellation, and lexicographic row sorting.
- The highest finalization score wins each state-equivalence group; existing candidate ordering breaks equal-score ties.
- Returned policy states are canonical and are not recomputed after selection.
- Exploration statistics remain based on all evaluated candidates; final-pool metadata and finalization statistics describe unique representatives.
- Canonicalization preserves matrix column width, including for an empty result.
- Sorting must use row indices or views and must not copy full rows in its comparator.

---

### Task 1: Exact parity-matrix canonicalization

**Files:**
- Modify: `include/matrix.hpp:356-396`
- Modify: `src/core/matrix.cpp:35-105`
- Test: `tests/core_regression.cpp:40-110`

**Interfaces:**
- Consumes: `Matrix::rows()`, `Matrix::cols()`, `Matrix::operator[]`, `RowCView::none`, and row three-way comparison.
- Produces: `Matrix canonical_parity_matrix(const Matrix& matrix)` in namespace `todd`; later finalization code uses this exact function.

- [ ] **Step 1: Write a failing canonicalization regression**

Add a test that constructs the literal rows `000`, `101`, `011`, `101`, `110`, `110`, `110`, calls the wished-for `canonical_parity_matrix`, and asserts that the exact result is the two-row matrix `011`, `110`. Construct a second matrix with the same parity rows in a different order and assert that its canonical result is exactly equal. Also assert that a matrix containing only zero/even-duplicate rows canonicalizes to `0 x original_cols`, not `0 x 0`.

The production mutation this catches is omission of zero filtering, modulo-two cancellation, stable lexicographic ordering, or width preservation.

- [ ] **Step 2: Run the focused test target and verify RED**

Run:

```bash
cmake --build build --target vartodd_core_regression -j2
```

Expected: compilation fails because `canonical_parity_matrix` is not declared. This is the intended missing-feature failure.

- [ ] **Step 3: Declare and implement the minimal canonicalization operation**

Declare in `include/matrix.hpp`:

```cpp
Matrix canonical_parity_matrix(const Matrix& matrix);
```

Implement in `src/core/matrix.cpp` by collecting indices of nonzero rows, sorting those indices using the complete row comparison, scanning adjacent equal runs, and pushing one representative only for odd run lengths:

```cpp
Matrix canonical_parity_matrix(const Matrix& matrix) {
    std::vector<index_t> order;
    order.reserve(static_cast<std::size_t>(matrix.rows()));
    for (index_t row = 0; row < matrix.rows(); ++row) {
        if (!matrix[row].none())
            order.push_back(row);
    }
    std::ranges::sort(order, [&](index_t lhs, index_t rhs) {
        return (matrix[lhs] <=> matrix[rhs]) < 0;
    });

    Matrix out(0, matrix.cols());
    out.reserve_rows(static_cast<index_t>(order.size()));
    for (std::size_t begin = 0; begin < order.size();) {
        std::size_t end = begin + 1;
        while (end < order.size() && matrix[order[begin]] == matrix[order[end]])
            ++end;
        if (((end - begin) & 1U) != 0)
            out.push_back(matrix[order[begin]]);
        begin = end;
    }
    return out;
}
```

- [ ] **Step 4: Build and run the core regression to verify GREEN**

Run:

```bash
cmake --build build --target vartodd_core_regression -j2
ctest --test-dir build -C Debug -R vartodd_core_regression --output-on-failure
```

Expected: build succeeds and the core regression passes.

- [ ] **Step 5: Commit canonicalization**

```bash
git add include/matrix.hpp src/core/matrix.cpp tests/core_regression.cpp
git commit -m "feat: canonicalize parity matrix rows"
```

### Task 2: Finalize and merge actions by canonical state

**Files:**
- Modify: `src/core/todd_generator.cpp:137-138,489-599,848-932`
- Test: `tests/core_regression.cpp:427-560`

**Interfaces:**
- Consumes: `canonical_parity_matrix(const Matrix&)`, existing `Candidate`, `FinalizationScore`, `candidate_tie_preferred`, `SeenValues`, and deterministic `PyRNG` selection.
- Produces: internal `FinalizedAction { Candidate candidate; Matrix state; }`; unique-state finalization and selection returning the stored canonical state.

- [ ] **Step 1: Write a failing duplicate-action integration regression**

Construct the literal four-row, three-column input `001`, `100`, `001`, `101`. Configure policy iteration with:

```cpp
cfg.selection = ActionSelection{16, "best", 0.0f};
cfg.pool = ActionPool{16};
cfg.tohpe = TohpeSearch{SamplingBudget{k_all_one_hot_samples, 32, 64, 3}, SourcePool{16, 0}, 8};
cfg.tohpeprefix = TohpePrefixSearch{
    SamplingBudget{k_all_one_hot_samples, 32, 64, 3}, SourcePool{16, 0}, 4,
    ZBucketSearch{32, 0, 0.0f, 0.0f, 128}};
cfg.todd = ToddSearch{
    SamplingBudget{k_all_one_hot_samples, 32, 64, 3}, SourcePool{16, 0}, 4,
    ZBucketSearch{32, 0, 0.0f, 0.0f, 128}};
```

Call `policy_iteration_impl(data, cfg, 123, 0)` and assert:

- one chosen action and one state remain from the reproduced equivalence class;
- the state rows are exactly canonical `100`, `101`;
- the winning action has reduction `4` and the greatest observed final score (`2.0f / 3.0f`, within a small float tolerance);
- `pool_size == 1`, `pool_tohpe_size == 1`, and the other source sizes are zero;
- a repeated fixed-seed call returns the identical candidate and state.

The production mutation this catches is deduplication after selection, exact-order-only equality, choosing by pool score instead of final score, stale pool metadata, or reapplying a selected action into a noncanonical state.

- [ ] **Step 2: Run the core regression and verify RED**

Run:

```bash
cmake --build build --target vartodd_core_regression -j2
ctest --test-dir build -C Debug -R vartodd_core_regression --output-on-failure
```

Expected: the new assertion fails because current final selection exports multiple actions whose states differ only by row order.

- [ ] **Step 3: Replace the candidate-only final pool with finalized actions**

Add the internal type:

```cpp
struct FinalizedAction {
    Candidate candidate;
    Matrix state;
};

using FinalizedActions = std::vector<FinalizedAction>;
```

Update best and softmax selection helpers to move/sort `FinalizedAction` values while comparing `action.candidate.final_score` through `scored_candidate_preferred`. Preserve the current draw order by sorting unique finalized actions canonically before assigning Gumbel draws.

- [ ] **Step 4: Apply, canonicalize, score, and deduplicate every merged candidate**

In `merge_and_finalize_candidates`, for each merged candidate:

1. Build its nullspace and apply it once.
2. Replace the state with `canonical_parity_matrix(state)`.
3. Compute `tohpe_dim` from this state when required.
4. Invoke `config.fscore(candidate)` to populate `final_score`.
5. Find an existing finalized action with an exactly equal canonical state.
6. Insert when absent; otherwise replace it only when `scored_candidate_preferred` prefers the new candidate by final score/reduction/canonical tie-break.

Do not push into the old `TopKPool<FinalizationScore>`: source merging already limits input to `config.pool.final_size`, and deduplication can only shrink it.

- [ ] **Step 5: Recompute unique final-pool metadata and statistics**

After deduplication, count retained representatives by source and assign the unique total and per-source counts to every survivor. Reset/update `max_final_score`, `mean_final_score`, `max_final_tohpe_dim`, and `mean_final_tohpe_dim` by iterating only over retained actions in deterministic order.

Keep exploration counters and `SeenValues` unchanged because they describe all evaluated candidates.

- [ ] **Step 6: Return stored states without a second application**

Change `build_result` to receive selected `FinalizedAction` values. Populate `num_better_*` on `action.candidate`, export that candidate, and move `action.state` into `Result::states`. Remove its nullspace providers and `scratch_killed` application path. Simplify `select_and_apply_candidates` accordingly while retaining `ctx.base_seed` as `Result::seed`.

- [ ] **Step 7: Build and run the core regression to verify GREEN**

Run:

```bash
cmake --build build --target vartodd_core_regression -j2
ctest --test-dir build -C Debug -R vartodd_core_regression --output-on-failure
```

Expected: the duplicate-action regression and all existing core regressions pass.

- [ ] **Step 8: Commit finalization deduplication**

```bash
git add src/core/todd_generator.cpp tests/core_regression.cpp
git commit -m "fix: merge actions with equivalent parity states"
```

### Task 3: Full verification and performance guard

**Files:**
- Modify if needed: `tests/core_regression.cpp`
- Verify: all changed C++ sources and tests

**Interfaces:**
- Consumes: completed canonicalization and unique finalization behavior.
- Produces: verified Debug/Release behavior and a measured 5,000-row canonicalization result.

- [ ] **Step 1: Run Debug and Release core regressions**

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
ctest --test-dir build -C Debug -R vartodd_core_regression --output-on-failure
cmake --build build --config Release --target vartodd_core_regression -j2
ctest --test-dir build -C Release -R vartodd_core_regression --output-on-failure
```

Expected: both configurations report 100% passing tests.

- [ ] **Step 2: Run relevant Python regression coverage**

```bash
python3 -m unittest tests.test_evaluator_timing tests.test_full_pso
```

Expected: all selected Python tests pass. If the active interpreter cannot import the built extension, report the environment mismatch explicitly and run with the interpreter matching the extension ABI.

- [ ] **Step 3: Measure 5,000-row canonicalization**

Add a temporary benchmark invocation outside the repository that constructs a deterministic 5,000-row narrow matrix, calls `canonical_parity_matrix` repeatedly in a Release-linked executable, and records median wall time. Do not commit the benchmark. Confirm that canonicalization does not dominate one policy finalization on the same target-shaped input; report both timings rather than imposing an arbitrary absolute threshold.

- [ ] **Step 4: Inspect the final diff and repository status**

```bash
git diff --check
git status --short
git log -3 --oneline
```

Expected: no whitespace errors; only intended files/commits belong to this work; pre-existing untracked experiment artifacts remain untouched.

- [ ] **Step 5: Commit any verification-only test adjustment**

Only if Step 1 or Step 2 required a legitimate test correction:

```bash
git add tests/core_regression.cpp
git commit -m "test: cover canonical policy states"
```
