# Full TODD Kernel Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated transformed-column loads in `full_todd_kernel` with an exact pair-centric generated-row builder and in-place solution-basis solve.

**Architecture:** Add one `todd::detail` function that materializes the generated system by visiting each relevant triangular column once. `FullToddGenerator::full_todd_kernel` will pass that matrix to the existing in-place `solve_and_build_solution_basis`, preserving the generated rows, solver row order, solution transform, and returned basis.

**Tech Stack:** C++20, contiguous `Matrix`/`RowView` GF(2) storage, CMake/Ninja, existing `vartodd_core_regression`, Linux `perf`.

## Global Constraints

- Optimize the exact benchmark `build/bin/RelWithDebInfo/bench ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy 25 8 1 0`.
- Preserve the generated linear system and its row order exactly.
- Preserve bucket order, sampling order, seeds, candidate scoring, and deterministic behavior.
- Do not retain the optimization unless repeated benchmark medians improve reliably.
- Do not modify or stage unrelated workspace files.

---

### Task 1: Specify Pair-Centric Generated Rows

**Files:**
- Modify: `include/nullspace.hpp`
- Modify: `tests/core_regression.cpp`
- Modify: `src/core/nullspace.cpp`

**Interfaces:**
- Consumes: `RowCView`, `MatrixWithData::FullToddData`, and the solution-column count.
- Produces: `Matrix todd::detail::build_full_todd_generated_rows(RowCView z, const MatrixWithData::FullToddData& full_todd, index_t solution_cols)`.

- [ ] **Step 1: Write the failing generated-row equivalence test**

Add a test-only streaming reference to `tests/core_regression.cpp`:

```cpp
Matrix reference_full_todd_generated_rows(
    RowCView z,
    const MatrixWithData::FullToddData& ft,
    index_t solution_cols) {
    const index_t total_cols = ft.nonpiv_cols + solution_cols;
    Matrix out(z.size() + 1, total_cols);

    std::vector<index_t> set_bits;
    for (auto bit = z.find_first(); bit != RowCView::npos; bit = z.find_next(bit))
        set_bits.push_back(bit);

    auto add_col = [&](RowView dst, index_t col) {
        const auto prow = ft.pivot_row_of_col[col];
        if (prow != MatrixWithData::FullToddData::npos)
            dst ^= ft.LY_nonpivot[prow];
        else
            dst.flip(ft.nonpivot_index[col]);
    };

    for (index_t gamma = 0; gamma < z.size(); ++gamma) {
        for (index_t s : set_bits) {
            if (s == gamma)
                continue;
            const index_t col =
                s > gamma ? ft.offset[s] + gamma : ft.offset[gamma] + s;
            add_col(out[gamma], col);
        }
    }

    for (std::size_t bi = 0; bi < set_bits.size(); ++bi) {
        const index_t b = set_bits[bi];
        for (std::size_t ai = 0; ai <= bi; ++ai)
            add_col(out[z.size()], ft.offset[b] + set_bits[ai]);
    }
    return out;
}
```

Construct a small `MatrixWithData`, and compare the reference against the
wished-for `detail::build_full_todd_generated_rows` for empty, one-bit, and
multi-bit `z` values:

```cpp
require(actual == expected, "pair-centric full TODD rows differ from streaming reference");
```

- [ ] **Step 2: Run the regression target and verify RED**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
```

Expected: compilation fails because
`todd::detail::build_full_todd_generated_rows` is not declared.

- [ ] **Step 3: Declare and minimally implement the pair-centric builder**

Declare the function after `MatrixWithData` in `include/nullspace.hpp`.

In `src/core/nullspace.cpp`, allocate `Matrix rows(z.size() + 1, total_cols)`
and build it as follows:

```cpp
for (index_t gamma = 0; gamma < n; ++gamma) {
    if (z.test(gamma))
        continue;
    for (index_t s : set_bits)
        add_col_one(rows[gamma], pair_col(gamma, s));
}

for (std::size_t bi = 0; bi < set_bits.size(); ++bi) {
    const index_t b = set_bits[bi];
    add_col_one(rows[n], ft.offset[b] + b);
    for (std::size_t ai = 0; ai < bi; ++ai) {
        const index_t a = set_bits[ai];
        add_col_three(rows[a], rows[b], rows[n], ft.offset[b] + a);
    }
}
```

For pivot columns, `add_col_three` must load each source word once:

```cpp
const RowCView src = ft.LY_nonpivot[prow];
for (index_t word = 0; word < src.blocks(); ++word) {
    const std::uint64_t value = src.data()[word];
    first.data()[word] ^= value;
    second.data()[word] ^= value;
    final.data()[word] ^= value;
}
```

For a non-pivot column, flip the mapped bit in the corresponding destination
rows.

- [ ] **Step 4: Run the focused regression test and verify GREEN**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
```

Expected: `vartodd_core_regression ok`.

- [ ] **Step 5: Commit the generated-row builder**

```bash
git add include/nullspace.hpp src/core/nullspace.cpp tests/core_regression.cpp
git commit -m "Build full TODD rows pairwise"
```

---

### Task 2: Refactor `full_todd_kernel` to Use In-Place Elimination

**Files:**
- Modify: `src/core/nullspace.cpp`
- Modify: `include/todd_index.hpp`
- Modify: `tests/core_regression.cpp`

**Interfaces:**
- Consumes: `detail::build_full_todd_generated_rows` from Task 1 and existing `solve_and_build_solution_basis(Matrix&, index_t, const Matrix&, const SumEntry*, index_t)`.
- Produces: `FullToddGenerator::full_todd_kernel` with unchanged public signature and output semantics.

- [ ] **Step 1: Extend the green characterization test across the public kernel**

For a small `MatrixWithData` and several valid bucket keys:

1. Build reference generated rows with
   `reference_full_todd_generated_rows`.
2. Resolve `ptr` and `len` through `ToddIndex::sum_bucket`.
3. Compute the expected basis with
   `solve_and_build_solution_basis(reference_rows, ft.nonpiv_cols,
   data->tohpe_basis(), ptr, len)`.
4. Compare it to `FullToddGenerator(data).full_todd_kernel(z, ptr, len)`:

```cpp
require(actual_basis == expected_basis,
        "full_todd_kernel basis differs from materialized reference solve");
```

This is the characterization coverage for the refactor step. The new
pair-centric behavior already completed its red-green cycle in Task 1.

- [ ] **Step 2: Run the characterization test before refactoring**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
```

Expected: `vartodd_core_regression ok`.

- [ ] **Step 3: Integrate the pair-centric matrix and in-place solver**

Replace the generated callback call in `full_todd_kernel` with:

```cpp
Matrix generated =
    detail::build_full_todd_generated_rows(z, ft, M_->P().rows());
return solve_and_build_solution_basis(
    generated, ft.nonpiv_cols, M_->tohpe_basis(), ptr, len);
```

Remove `solve_and_build_solution_basis_generated` after confirming it has no
remaining callers.

- [ ] **Step 4: Run Debug and RelWithDebInfo regressions**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
cmake --build build --config RelWithDebInfo --target vartodd_core_regression bench -j2
build/bin/RelWithDebInfo/vartodd_core_regression
```

Expected: both regression runs print `vartodd_core_regression ok`.

- [ ] **Step 5: Commit the solver integration**

```bash
git add include/todd_index.hpp src/core/nullspace.cpp tests/core_regression.cpp
git commit -m "Solve materialized full TODD rows in place"
```

---

### Task 3: Validate Performance and Retain Only a Real Improvement

**Files:**
- No source files are planned for this task.
- Record benchmark results in the final handoff; do not commit generated profile data.

**Interfaces:**
- Consumes: optimized RelWithDebInfo `bench`.
- Produces: verified median runtime and an after-profile for comparison with the 34.34-second baseline.

- [ ] **Step 1: Run three target benchmark samples**

Run three times:

```bash
build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  25 8 1 0
```

Expected: `iterations_run=25` every time. Compute the median `total_ms`.

- [ ] **Step 2: Capture and inspect the after-profile**

Run:

```bash
perf record -F 499 -g --call-graph fp \
  -o /tmp/vartodd-full-kernel-pairwise.data -- \
  build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  25 8 1 0
perf report -i /tmp/vartodd-full-kernel-pairwise.data \
  --stdio --no-children --call-graph none \
  --percent-limit 0.15 --sort overhead,symbol
```

Expected: lower `full_todd_kernel` self time and no replacement hotspot that
eliminates the wall-clock improvement.

- [ ] **Step 3: Apply the acceptance rule**

Keep the pair-centric implementation only if its three-run median is reliably
below the 34.34-second baseline and repeated generated-row/basis tests remain
exact. If it regresses, revert only the two implementation commits while
retaining the design and measurements.

- [ ] **Step 4: Run final verification**

Run:

```bash
git diff --check
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
cmake --build build --config RelWithDebInfo --target vartodd_core_regression bench -j2
build/bin/RelWithDebInfo/vartodd_core_regression
```

Expected: no whitespace errors and both regression binaries pass.
