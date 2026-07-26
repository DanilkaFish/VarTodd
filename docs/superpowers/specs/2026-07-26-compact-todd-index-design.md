# Compact Purpose-Built ToddIndex Design

## Objective

Replace the general-purpose bucket map in `ToddIndex` with a representation
matched to how policy iteration uses it. The new implementation must:

- build faster for the target workload;
- materially reduce retained and peak memory;
- preserve exact bucket membership and member order;
- preserve the current deterministic bucket-ID assignment and policy behavior;
- keep arbitrary vector lookup exact despite hash collisions;
- leave `FullToddData` unchanged.

The primary benchmark is:

```text
build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  25 8 1 0
```

## Evidence and workload shape

The target's initial matrix has 1,701 rows and 96 columns. Its index contains:

- 1,447,551 row and row-pair sums;
- 1,407,877 distinct buckets;
- only 39,674 entries beyond the first entry of each bucket;
- a maximum bucket size of 9.

Nearly every sum is unique. The current representation nevertheless retains a
full key, two 64-bit range values, collision metadata, and general-purpose hash
table storage for every bucket. Its retained footprint is approximately
100 MiB.

The current profile attributes about 30.8% of total runtime to bucket lookup
and insertion and another 8.3% to bucket assembly.

On the development machine, which has one shared 16 MiB L3 cache:

- one pinned-equivalent benchmark completes in approximately 6.94 seconds;
- two processes on separate physical cores complete in 7.55--7.61 seconds
  each;
- four processes on separate physical cores complete in 8.95--9.23 seconds
  each.

This demonstrates concurrent throughput loss, but it does not prove that
`ToddIndex` is its only cause. Single-process construction, retained size, and
concurrent throughput will therefore be measured separately.

## Usage-driven requirements

The frequent operations are:

1. map a row index to its bucket ID;
2. map a row pair to its bucket ID;
3. read a bucket size while ranking buckets;
4. update `CountWS` using those IDs.

The policy path only needs a complete key and complete member list for the
small number of buckets selected for TOHPE or full-TODD nullspace
construction.

All members of a selected bucket are still required:

- `apply_solution_transform` consumes them;
- `Witness::init_from_entries_` extracts the special single and pair list;
- `rank_divergence` depends on that exact witness metadata;
- full-TODD basis construction applies the corresponding solution transform.

The original member order is significant and will be preserved: all single
entries precede pair entries, and pair entries retain triangular row-major
order.

## Construction table

Use a purpose-built, open-addressed flat table whose slots are 32-bit words.
The table capacity is a checked power of two sized for a maximum load factor
of 0.7.

A nonempty slot packs:

- `bucket_id + 1` in the low bits;
- a fingerprint derived from the mixed 64-bit hash in the remaining bits.

The number of ID bits is computed from the maximum possible number of sums.
If no fingerprint bits remain, the implementation performs the exact
representative comparison for every occupied probe.

Lookup proceeds as follows:

1. compute the existing deterministic XOR hash;
2. mix it to obtain the initial slot and fingerprint;
3. probe occupied slots;
4. skip slots with a different fingerprint;
5. reconstruct the slot's representative sum from `P`;
6. compare every vector block exactly;
7. return the bucket only after exact equality.

The hash and fingerprint are accelerators only. Hash collisions cannot merge
different vectors.

New bucket IDs are assigned on first encounter while iterating singles and
pairs in the current order. This preserves deterministic insertion-ordered
IDs and keeps the mostly unique pair IDs correlated with triangular traversal,
which is preferable for `CountWS` locality.

## Source-position encoding

Every input sum receives a checked 32-bit source ordinal:

```text
0 .. P.rows()-1                  single rows
P.rows() .. total_sum_count-1    pairs in triangular row-major order
```

`ToddIndex` will explicitly reject a total sum count that cannot be represented
by this encoding. Such an index would already require more than 16 GiB for the
current `pair_id_` and `sum_entries_` arrays alone.

Decoding an ordinal yields either `SumEntry{i}` or `SumEntry{i,j}`. Pair
decoding uses checked triangular row offsets and binary search. Decoding is
not part of the hot `pair_bucket_id` path.

## Retained representation

Keep:

- `single_id_`, one `uint32_t` per matrix row;
- `pair_id_`, one `uint32_t` per row pair in triangular row-major order;
- the flat lookup table;
- `representative_pos_`, one source ordinal per bucket;
- adaptive bucket lengths;
- sparse duplicate-bucket metadata and duplicate source ordinals;
- the small row-hash masks required for arbitrary vector lookup.

Per-row hashes are construction scratch. Release them after the index has been
built; no policy hot path consumes them.

Remove:

- `ankerl::unordered_dense::map head_`;
- `bucket_next_`;
- `bucket_key_blocks_`;
- `bucket_off_`;
- the full `sum_entries_` array.

Bucket lengths are built temporarily as `uint32_t`, then stored in the
narrowest exact unsigned type selected from `uint8_t`, `uint16_t`, and
`uint32_t`. Operations that scan all lengths dispatch on the storage type once
outside their loop.

For each duplicate occurrence, construction records `(bucket_id,
source_ordinal)`. After construction these records are grouped by bucket ID
while preserving ascending source ordinal. Only nonsingleton buckets receive
sparse range metadata.

For the target's initial matrix, the estimated persistent footprint is:

```text
flat lookup slots          about 8.0 MiB
representative positions   about 5.4 MiB
pair IDs                   about 5.5 MiB
byte bucket lengths        about 1.4 MiB
sparse duplicates          below 0.6 MiB
other fixed data           below 0.1 MiB
```

The target is at most 24 MiB retained, excluding `P`, `FullToddData`, and
explicit caller-owned scratch buffers.

## Selected-bucket materialization

Replace stable raw pointers into `sum_entries_` with explicit materialization
into caller-owned scratch storage:

```text
bool materialize_bucket(uint32_t id, vector<SumEntry>& out) const
bool materialize_bucket(RowCView key, vector<SumEntry>& out) const
```

Materialization:

1. clears the destination;
2. decodes the representative ordinal;
3. finds sparse duplicate metadata if the bucket length exceeds one;
4. decodes duplicate ordinals in original order.

Generators will keep scratch vectors outside bucket-search loops and reuse
their capacity. Algorithms that need pointer-and-length inputs may use
`out.data()` and `out.size()` for the duration of the call.

ID-based paths will be used wherever the ID is already known:

- `TohpeGenerator::make(z, bucket_id)`;
- full-TODD bucket exploration;
- `FullToddGenerator::make(row)`;
- `FullToddGenerator::make(row1, row2)`.

Arbitrary `RowCView` lookup remains available through the retained flat table.

`key_of(id)` will return an owning `Row` reconstructed from the representative
ordinal. Internal callers will move or retain that row for as long as a view
is required.

Remove the unused `sum_key_sizes()` and `sum_key_sizes_scratch()` view-based
bulk key APIs; they cannot provide stable views after persistent key storage is
removed. These are C++ source-compatibility changes only. No Python result
schema or policy configuration changes.

## Construction data flow

Construction performs one primary pass:

1. allocate `single_id_`, `pair_id_`, and the flat table;
2. iterate singles in row order;
3. iterate pairs in triangular row-major order;
4. compute each sum and its hash;
5. find or insert its exact bucket;
6. write its direct ID mapping;
7. create a representative or record a sparse duplicate;
8. increment the temporary bucket length.

After the pass:

1. determine `max_bucket_`;
2. compact bucket lengths;
3. group sparse duplicate records;
4. release temporary construction buffers.

There is no second full pass over all rows and pairs and no full-key copy for
new buckets.

## Error handling and edge cases

- Empty matrices produce empty mappings and lookup tables.
- Zero-column rows still compare exactly and form the appropriate bucket.
- Flat-table capacity, sum counts, source ordinals, and packed slot fields use
  checked arithmetic.
- A full probe sequence is treated as an internal construction error.
- Invalid bucket IDs return an empty materialization.
- Missing arbitrary keys return `false` and an empty destination.
- Extremely large bucket lengths select wider exact storage.
- Hash and fingerprint collisions are covered by complete vector comparison.

## Testing

Add a test-only reference index using straightforward owning keys and entries.
For empty, duplicate-heavy, collision-prone, random, and target-shaped small
matrices, compare:

- bucket count and maximum bucket size;
- every single and pair mapping;
- every reconstructed key;
- every materialized member list and its order;
- arbitrary present and absent key lookup;
- top-bucket selection;
- TOHPE candidate details;
- full-TODD and TOHPE nullspace row spaces;
- `rank_divergence` and applied matrices.

Add synthetic boundary tests for every adaptive bucket-length transition.
Exercise packed-slot arithmetic and source-limit overflow through factored
checked helpers, without allocating boundary-sized indexes.

Add `ToddIndex::storage_bytes()` to report retained container capacities. A
target-data diagnostic must confirm no more than 24 MiB retained by
`ToddIndex`.

Run the existing Debug and RelWithDebInfo regression suites.

## Performance validation

Measure the following against a fresh build of the current implementation:

1. ToddIndex-only construction time on the target matrix;
2. exact 25-iteration benchmark median over at least three runs;
3. `perf` attribution for construction and TOHPE counting;
4. retained bytes from `storage_bytes()`;
5. two-process and four-process pinned throughput using the same core layout
   as the baseline.

The change is retained only if:

- all exactness tests pass;
- target retained `ToddIndex` storage is at most 24 MiB;
- the single-process benchmark median improves;
- `pair_bucket_id` and `CountWS` do not regress enough to offset construction
  savings.

Concurrent scaling is an evaluation metric, not a correctness gate, because
frequency, memory bandwidth, and unrelated data structures also affect it.

## Deferred work

The previously committed exact full-TODD basis-reuse and complement design
remains deferred. This change will not add fields to or modify
`MatrixWithData::FullToddData`.

Bit-packed `pair_id_`, hash-sorted bucket IDs, Four-Russians tables, and
additional `CountWS` changes are also deferred. They risk slowing the hottest
random-access path and should only be reconsidered using the post-change
profile.
