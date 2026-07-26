# Full TODD Kernel Acceleration Design

## Objective

Accelerate `FullToddGenerator::full_todd_kernel` for:

```text
build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  25 8 1 0
```

The implementation must preserve the generated linear system, its row order,
the resulting solution basis, candidate generation, and deterministic behavior.

## Baseline and profile

The corrected RelWithDebInfo build completes the target workload in 34.34
seconds. A second run under `perf record` completes in 34.68 seconds.

`perf` attributes:

- 68.01% self time to `FullToddGenerator::full_todd_kernel`;
- 1.54% to `detail::insert_into_y_basis`;
- the dominant instructions inside the kernel to full-width XORs performed by
  `add_col` while generating rows;
- a secondary concentration to full-width XORs during left-side elimination.

Consequently, this work targets generated-row construction and left-side
elimination. It does not initially alter transformed-solution basis insertion
or coefficient-vector sampling.

## Current construction

For each `gamma` row, the current builder walks every set index in `z` and
applies the corresponding transformed triangular column. It then walks all
pairs of set indices again to build the final row.

When both endpoints of an off-diagonal pair are set, the same transformed
column is therefore loaded three times:

1. for the first endpoint's gamma row;
2. for the second endpoint's gamma row;
3. for the final row.

The streaming solver also copies every newly discovered left-pivot row into a
separate `pivot_rows` matrix.

## Proposed construction

Materialize the complete `(n + 1) x total_cols` generated matrix, but construct
it by unique relevant triangular columns:

1. Build `S`, the set-bit indices of `z`.
2. For each pair with exactly one endpoint in `S`, apply its transformed column
   once to the gamma row belonging to the endpoint outside `S`.
3. For each off-diagonal pair with both endpoints in `S`, load its transformed
   column once and XOR it into:
   - the first endpoint's gamma row;
   - the second endpoint's gamma row;
   - the final row.
4. For each diagonal whose endpoint is in `S`, apply its transformed column to
   the final row.

For pivot columns, the multi-destination operation reads each
`LY_nonpivot` word once and updates one or three destination rows. For
non-pivot columns, it flips the mapped bit in the same destinations.

The constructed rows and row order are exactly the same as in the current
implementation.

## Solver

Run left-side elimination directly in the materialized matrix. A pivot map
refers to the original row containing each pivot, so no separate full-width
copy into `pivot_rows` is required.

The right-side extraction, solution transform, TOHPE rows, and
`insert_into_y_basis` behavior remain unchanged in the first implementation.

The materialized matrix occupies approximately the storage that the streaming
solver currently reserves for `pivot_rows`, so this is not intended to
increase the asymptotic peak memory requirement.

## Correctness

Tests will compare the optimized kernel against a retained test-only reference
builder on several small matrices and bucket keys. They will verify:

- every generated row is identical before elimination;
- returned bases span the same subspace;
- basis dimensions agree;
- existing policy-iteration and tensor-preservation regression tests pass.

The optimized code must not change bucket order, sampling order, seeds, or
candidate scoring.

## Performance validation

Validation uses a fresh RelWithDebInfo build and the exact target command.
Measure at least three runs before and after the change and compare medians.

Also capture an after-profile to confirm that:

- `full_todd_kernel` self time decreases;
- repeated transformed-column loads decrease;
- no replacement hotspot offsets the improvement.

If the pair-centric implementation does not improve the median reliably, it
will not be kept. Scratch reuse and touched-only pivot reset remain possible
follow-up experiments, but are outside this first change.
