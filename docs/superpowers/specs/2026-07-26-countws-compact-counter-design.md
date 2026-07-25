# Compact CountWS Design

## Goal

Reduce the time and memory used by `CountWS` while preserving selection of the
requested number of greatest-count buckets. The returned buckets have no
ordering guarantee.

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

The five-run pre-change median completed 12 iterations in 6.820 seconds, or
568.3 milliseconds per iteration. An earlier `perf` profile identified both
`CountWS::argmax_n_into` and `CountWS::add` as dominant costs.

For this matrix, the Todd index has 1,407,877 buckets and a maximum bucket size
of 9. Each bucket entry contributes at most 2, so every counter is bounded by
18.

## Representation

`CountWS` stores a packed state directly at each bucket ID:

- `state_by_id[id]` combines the bucket's current epoch and count.
- `touched_ids` contains each bucket first touched in the current epoch.
- `touched_size` is the valid prefix of `touched_ids`.

The count uses the smallest safe unsigned integer type for
`2 * max_bucket_size`. The state word is twice that width, with the count in
the low half and epoch in the high half. The benchmark therefore uses an
8-bit count and 8-bit epoch in one 16-bit state cell. Wider 32-bit and 64-bit
state cells support 16-bit and 32-bit counts. Exceptionally large counts use a
wide fallback.

This direct state layout replaced an experimentally implemented
ID-to-compact-position map. Profiling that representation showed that the
dependent map lookup followed by `counts[pos]` raised accumulation to 49.2% of
cycles and produced no median speedup. The direct packed state requires one
random load/update and makes the benchmark state array about 2.7 MiB.

Runtime representation dispatch occurs once outside the pair loop, not once
per update.

## Operations

### Reset

Configuration records the bucket count and maximum possible counter value,
allocates or grows storage, advances the epoch, and sets `touched_size` to
zero. Storage is reused by subsequent calls.

On epoch wrap, `state_by_id` is cleared and the epoch restarts at one.
`touched_ids` does not need clearing because only positions below
`touched_size` are valid.

### Add

An update reads the packed state for its bucket ID:

1. If the stored epoch is current, increment the count in the same state word.
2. Otherwise, append the ID to `touched_ids` and write the current epoch with
   the initial count.

The initial count remains `delta - parity`, matching existing parity behavior.
Debug checks verify the configured count bound and bucket-ID capacity.

### Selection

If fewer than all touched buckets are requested,
`std::ranges::nth_element` partitions `touched_ids[0:touched_size]` at
`min(n, touched_size)`. Its comparator extracts the narrow count from each
bucket's packed state.

The selected prefix is converted to scored bucket results without sorting it.
The result therefore contains the requested greatest-count buckets in
unspecified order. When equal counts straddle the partition boundary, any
corresponding buckets are valid.

The selected result contains both bucket ID and count. This removes the need
for callers to access the internal count representation after selection.
`argmax` uses `std::ranges::max_element` over the same scored range.

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
- State packing must preserve the configured count bound and a nonzero epoch.
- Epoch wrap clears the packed state array before reuse.
- `n == 0` or an empty touched set returns no candidates.
- `n` larger than the touched set returns every touched bucket in unspecified
  order.

## Testing

Regression tests will cover:

- first insertion and repeated updates to the same bucket;
- parity-zero and parity-one initialization;
- reset reuse and forced epoch wrap;
- empty and zero-length selections;
- several output sizes, including `n == 1` and `n >= touched_size`;
- membership in the greatest-count set without asserting output order;
- membership in the valid maximal set when equal counts straddle the partition
  boundary;
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
