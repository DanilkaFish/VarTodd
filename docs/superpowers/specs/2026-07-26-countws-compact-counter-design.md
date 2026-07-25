# Compact CountWS Design

## Goal

Reduce the time and memory used by `CountWS` while preserving selection by
descending count. Equal-count buckets may use deterministic first-touch order
instead of ascending bucket ID.

The optimization must work well for both balanced input vectors, which touch a
large fraction of the buckets, and almost-uniform vectors, which touch only a
small fraction.

## Baseline

The reference workload is:

```text
build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  12 40 8 0
```

The verified pre-change run completed 12 iterations in 9.118 seconds, or
759.8 milliseconds per iteration. A `perf` profile attributed 37.26% of sampled
cycles to `CountWS::argmax_n_into` and 23.26% to `CountWS::add`.

For this matrix, the Todd index has 1,407,877 buckets and a maximum bucket size
of 9. Each bucket entry contributes at most 2, so every counter is bounded by
18.

## Representation

`CountWS` will store touched buckets by compact position rather than storing the
count at the bucket ID:

- `id_to_pos` maps a bucket ID to a compact touched position. Each mapping word
  also contains the current epoch, so reset normally remains O(1).
- `counts[pos]` stores the count for a touched position.
- `pos_to_id[pos]` maps the compact position back to its bucket ID.
- `touched_size` is the number of valid compact positions in the current epoch.

The mapping word uses the smallest safe unsigned integer type. Its low bits
hold a position in `[0, K)`, and its remaining bits hold the epoch. The
benchmark needs 21 position bits, leaving 11 epoch bits in a 32-bit word.
When the epoch wraps, `id_to_pos` is cleared before the next epoch begins.
A wider mapping word is used when the bucket count leaves too few epoch bits in
32 bits.

The count array uses the smallest unsigned integer type that can represent
`2 * max_bucket_size`. The benchmark therefore uses 8-bit counts. Wider count
types are selected for matrices with larger buckets. The bound calculation
must be checked for arithmetic overflow.

The hot accumulation loop is instantiated for the selected mapping and count
types. Runtime representation dispatch occurs once outside the pair loop, not
once per update.

## Operations

### Reset

Configuration records the bucket count and maximum possible counter value,
allocates or grows storage, advances the epoch, and sets `touched_size` to
zero. Storage is reused by subsequent calls.

On epoch wrap, the ID-to-position mapping is cleared and the epoch restarts at
one. Compact count and ID arrays do not need clearing because only positions
below `touched_size` are valid.

### Add

An update reads the packed mapping for its bucket ID:

1. If the stored epoch is current, decode the position and increment its
   contiguous counter.
2. Otherwise, allocate the next compact position, write its ID and initial
   count, and update the packed mapping.

The initial count remains `delta - parity`, matching existing parity behavior.
Debug checks verify the configured count bound and mapping capacity.

### Selection

Top-N selection scans `counts[0:touched_size]` sequentially. It retains compact
positions together with cached counts, so comparisons never reload counts
through bucket IDs.

For the common `n <= 8` case, selection uses a fixed-size bounded set. It
recomputes the current worst retained entry only after a replacement. Larger
requests use cached scored positions in a heap. The final result is ordered by
descending count. Equal counts retain first-touch order, which is deterministic
for a fixed input and traversal order.

The selected result contains both bucket ID and count. This removes the need
for callers to access the internal count representation after selection.
`argmax` uses the same sequential representation and returns the first touched
position among equal maxima.

## Interfaces

The internal result type is a scored bucket containing:

```text
bucket_id
count
```

`TohpeGenerator::best_z_n_details_into` consumes scored buckets directly when
constructing `TohpeZInfo`. Raw `cnt`, `tag`, and `used` vectors are no longer
part of the usable `CountWS` interface.

`CountWS` remains an internal, reusable workspace. It is not thread-safe, which
matches its current use as mutable state owned by one generator.

## Correctness and Error Handling

- Bucket IDs must remain below the configured bucket count and fit the Todd
  index's 32-bit bucket-ID contract.
- The maximum counter bound is computed from the index's maximum bucket size
  with checked multiplication.
- Narrow counter selection is permitted only when that proven bound fits.
- Mapping packing must preserve every position value and reserve a usable,
  nonzero epoch.
- Epoch wrap clears the mapping before reuse.
- `n == 0` or an empty touched set returns no candidates.
- `n` larger than the touched set returns every touched bucket, ordered by
  descending count and first-touch order for ties.

## Testing

Regression tests will cover:

- first insertion and repeated updates to the same bucket;
- parity-zero and parity-one initialization;
- reset reuse and forced epoch wrap;
- empty and zero-length selections;
- `n == 1`, `n <= 8`, `n > 8`, and `n >= touched_size`;
- descending count ordering;
- deterministic first-touch ordering for equal counts;
- counts at each supported storage-width boundary;
- a balanced synthetic vector and almost-all-zero/almost-all-one vectors;
- equivalence of selected bucket scores with a simple wide reference counter.

Tests will be written and observed failing before production changes.

## Performance Validation

The exact reference workload will be run repeatedly before and after the
change. Comparison will use the median total time rather than a single run.
The result must show a meaningful median improvement and must not regress the
synthetic almost-uniform cases.

A fresh `perf` profile will verify that time in `CountWS::add` and
`CountWS::argmax_n_into` decreased rather than merely moving into helper
functions. Full regression tests and a RelWithDebInfo build must pass before
completion is reported.

Potential follow-up work such as software prefetching, buffered radix
aggregation, or dense full-array scans is explicitly out of scope unless the
new profile shows that ID-to-position mapping latency remains a dominant
hotspot.
