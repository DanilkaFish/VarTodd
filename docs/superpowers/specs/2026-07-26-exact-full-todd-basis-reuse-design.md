# Exact Full-TODD Basis Reuse and Dense-Support Design

## Objective

Accelerate exact full-TODD basis construction for:

```text
build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  25 8 1 0
```

Every `full_todd_kernel` result must remain a complete independent basis for
exactly the same solution space. The implementation may choose different basis
vectors or a different row order. Rank cutoffs, projected-rank filters, and
approximate bases are out of scope.

## Current evidence

A fresh unprofiled run of the current RelWithDebInfo build completes the target
workload in 6.87 seconds. A `perf` run completes in 7.31 seconds and attributes:

- 3.02% self time to `FullToddGenerator::full_todd_kernel`;
- 1.69 percentage points inside the kernel to generated-row and elimination
  XORs;
- 0.63% to exact output-basis insertion.

With `min_buckets=0` and Todd pool reserve equal to one, the target workload
normally constructs a full-TODD nullspace during candidate generation and
constructs the same nullspace again if one of its candidates is selected.
The second construction is redundant.

`MatrixWithData` is recreated after each accepted policy iteration, so any new
`FullToddData` precomputation must amortize over only a small number of kernel
calls. Large Four-Russians tables are therefore deferred.

## Part 1: Reuse the exact generated nullspace

Candidate generation will create one shared immutable `NullSpace` for each
researched Todd bucket. Every accepted candidate produced from that bucket
will retain the same shared handle.

The candidate-generation loop will use the shared nullspace for:

- the full basis dimension;
- sampled linear combinations;
- rank-divergence calculations.

Finalization and application will use the retained nullspace when it is
present. This reuses the exact basis and the exact `ToddWitness` metadata that
were used during exploration. Candidates created through other APIs, tests, or
older construction paths may have no retained handle; those candidates keep
the current reconstruction fallback.

The retained object is immutable and shared only within one policy iteration.
It cannot be reused with a different `MatrixWithData`.

This change does not alter candidate ordering, scores, sampling, seeds, or the
chosen vector. Its only behavioral effect is to remove duplicate exact work.

## Part 2: Exact complement aggregates in `FullToddData`

Let `C(a,b)` denote the current pre-reduced contribution for triangular column
`(min(a,b), max(a,b))`. It contains both the non-pivot coordinates and the
right-hand solution coordinates used by the streaming solver.

Precompute the following packed rows:

```text
incident[g] = XOR C(g,s) for every s != g
all_pairs   = XOR C(a,b) for every a <= b
```

For the benchmark's initial 1701-by-96 matrix, these 97 packed rows occupy
approximately 57 KiB. Construction requires one pass over the triangular
columns and does not create the much larger per-block subset-XOR table.

For support `S` and complement `T`, a generated gamma row is:

```text
R(g,S) = XOR C(g,s), s in S and s != g
```

It may equivalently be constructed as:

```text
R(g,S) = incident[g] XOR XOR C(g,t), t in T and t != g
```

The builder will choose the representation requiring fewer column
contributions for each gamma row.

The final generated row is:

```text
F(S) = XOR C(a,b), a <= b and a,b in S
```

For dense `S`, it may equivalently be constructed as:

```text
F(S) = all_pairs XOR F(T) XOR XOR incident[t], t in T
```

The equality is exact over GF(2): off-diagonal pairs wholly inside `T` occur
once in `all_pairs` and once in `F(T)`, cross pairs occur once in
`all_pairs` and once through `incident[t]`, and pairs wholly inside `S`
remain. The builder will select the direct or complement expression according
to their operation counts.

The existing streaming construction and elimination order remain in place.
This retains the cache locality that the rejected pair-centric materialization
experiment lost.

## Data structures

`MatrixWithData::FullToddData` gains:

```text
Matrix incident_xor;  // P.cols() rows, total_cols columns
Row    all_pairs_xor; // total_cols columns
```

`Candidate` gains an internal shared handle to the immutable full-TODD
`NullSpace` used to create it. The handle is not included in `CandidateExport`
or the Python result representation.

The current `pivot_row_of_col`, `nonpivot_index`, `LY_nonpivot`, and offsets
remain the source of individual `C(a,b)` contributions.

## Edge cases

- Empty and zero-width matrices preserve the current empty result.
- Empty support uses the direct path and generates zero rows.
- Full support uses the complement path with an empty complement.
- Diagonal columns occur only in `all_pairs` and `F(S)`/`F(T)`, never in
  `incident`.
- Half-density support uses whichever exact expression has the lower measured
  operation count; ties use the direct path.
- A candidate without a cached nullspace reconstructs it using the current
  code path.
- The existing maximum-rank stop at `P.rows()` remains valid because reaching
  the ambient dimension proves that the complete space has already been
  obtained. No smaller cutoff is introduced.

## Testing

Add focused tests that:

1. Compare every precomputed `incident[g]` and `all_pairs` row with a
   test-only reference assembled from individual triangular contributions.
2. Compare adaptive generated rows with the current direct formula for empty,
   singleton, half-density, almost-full, and full supports.
3. Compare the returned row spaces, rather than row order, for representative
   small matrices and bucket transforms.
4. Confirm a cached Todd candidate and the reconstruction fallback produce
   identical applied matrices for the same sampled vector.
5. Confirm TOHPE candidates and externally constructed candidates still use
   the fallback correctly.
6. Run the existing Debug and RelWithDebInfo regression suites.

## Performance validation

Implement and measure the two parts separately:

1. exact nullspace reuse;
2. complement aggregates on top of reuse.

For each stage:

- rebuild the RelWithDebInfo benchmark;
- run the exact target command at least three times;
- compare medians with the current build;
- capture a `perf` profile if the median changes enough to retain the stage.

Keep each stage only if it improves the target median without changing
row-space tests or policy outputs. Because the current kernel accounts for
about 3% of total runtime, the expected whole-benchmark gain is modest; a
larger claimed improvement requires profile evidence.

Four-Russians subset tables and block Gaussian elimination are explicitly
deferred. They may be reconsidered only if the post-change profile still shows
generated-row XORs as a meaningful hotspot.
