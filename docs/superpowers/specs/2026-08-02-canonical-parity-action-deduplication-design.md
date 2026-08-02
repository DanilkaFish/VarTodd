# Canonical parity action deduplication

## Goal

For one policy iteration, distinct candidates that produce the same parity
matrix must occupy one final action slot. Applied matrices must have a canonical
row order so exact `Matrix` equality represents equality up to row reordering
after parity cancellation.

## Parity-matrix canonical form

Canonicalization operates on the rows of an applied matrix:

1. Remove zero rows.
2. Sort nonzero rows lexicographically by their complete bit-vector value.
3. For each run of equal rows, retain one row when the multiplicity is odd and
   retain none when it is even.
4. Emit retained rows in lexicographic order with the original matrix width.

Consequently, matrices that differ only by row ordering, zero rows, or pairs of
equal rows have identical canonical matrices. This ordering becomes part of the
public action-result behavior: every state returned by policy iteration is in
canonical parity form.

Canonicalization must be an exact operation. It must not use a probabilistic
hash as the equality criterion.

## Finalization and deduplication flow

The source pools continue to generate and merge candidates as they do now.
Finalization then processes every merged candidate in this order:

1. Reconstruct the candidate nullspace.
2. Apply the candidate and canonicalize the resulting parity matrix.
3. Compute the optional post-action TOHPE dimension from that canonical state.
4. Compute the candidate finalization score.
5. Group candidates by exact equality of their canonical applied matrices.

Within each group, retain the candidate with the greatest finalization score.
If finalization scores are equal, use the existing deterministic candidate
tie-break. Greedy or softmax action selection runs only over the retained unique
representatives, so equivalent actions cannot consume multiple selection
slots.

The retained action carries its already-computed canonical state through
selection. Selected actions are not applied a second time.

## Candidate metadata and statistics

Exploration statistics and `SeenValues` continue to describe all evaluated
source candidates.

After deduplication, final-pool metadata is recomputed from the retained
representatives:

- `pool_size` is the number of unique canonical states;
- per-source pool sizes count the surviving representative candidates by their
  source;
- final-score and final-TOHPE-dimension summary statistics describe unique
  representatives only.

The representative candidate retains its own source, reduction, pool score,
and other action metadata. Merged-away candidates are not exported.

If fewer unique states exist than `selection.count`, policy iteration returns
all unique states without padding or duplicate actions.

## Performance

Canonicalization sorts row views or row indices and copies only retained rows
into the canonical output. For `r` rows and `b` 64-bit blocks per row, expected
work is `O(r log r)` row comparisons and `O(r * b)` output scanning/copying.

At 5,000 narrow rows, one canonicalization performs roughly 61,000 short row
comparisons. The merged final pool is bounded by `ActionPool.final_size`, so
this cost is expected to remain small relative to candidate generation and
full-TODD application. The implementation must avoid copying full rows inside
the sort comparator.

## Error handling and determinism

Existing matrix-size and allocation overflow checks remain in force.
Canonicalization must preserve the input column count, including an empty
canonical result, so keys from one policy iteration always share a width.

Final-score comparison and representative tie-breaking retain the existing
deterministic ordering. This change introduces no unseeded randomness.

## Tests

Core regression coverage must demonstrate:

1. Canonicalization removes zero rows, cancels even duplicate multiplicities,
   retains odd multiplicities, and sorts surviving rows.
2. Two matrices containing the same parity rows in different orders canonicalize
   to exactly equal matrices.
3. For the input rows `001, 100, 001, 101`, current policy generation produces
   candidates leading to both row orders `101, 100` and `100, 101`; finalization
   returns one canonical state for that equivalence class.
4. When equivalent candidates have different finalization scores, the exported
   representative has the greatest score.
5. Returned states are canonical, unique, and deterministic across repeated
   fixed-seed calls.
6. Existing policy smoke, source-composition, and repeatability tests continue
   to pass.
