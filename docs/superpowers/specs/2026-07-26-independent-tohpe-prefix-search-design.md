# Independent TOHPE Prefix Search

## Goal

Replace the standalone `TohpeGenerator` search with two independent searches over Todd-index buckets:

- `TohpeSearch` samples the transformed TOHPE-prefix subspace only.
- `ToddSearch` samples the complete transformed Todd basis.

The sources keep separate pools and are merged only for final ranking and action selection.

## Public policy interface

`TohpeSearch` has the same search controls as `ToddSearch`:

```cpp
struct TohpeSearch {
    SamplingBudget sampling{};
    SourcePool     pool{2, 0};
    Int            actions_per_bucket = 4;
    ZBucketSearch  buckets{};
};
```

`ToddSearch` retains `sampling`, `pool`, `actions_per_bucket`, and `buckets`. Its temporary
`tohpe_sampling` field is removed. `TohpeSearch.z_choices` is removed.

The Python binding, mapping adapters, reporting, strategy scripts, and benchmark callers use these
fields. Existing three-item `TohpeSearch` pickle state is accepted by mapping its sampling and pool
to the new type and using default per-bucket and bucket-search values. Existing five-item `ToddSearch`
pickle state is accepted while ignoring the removed prefix-sampling item.

## Prefix basis generation

`FullToddGenerator` gains a prefix-only kernel for a `(z, bucket)` pair. It:

1. starts from `MatrixWithData::tohpe_basis()`;
2. applies `detail::apply_solution_transform` to each TOHPE row;
3. inserts each transformed row into a pivoted basis only when it is independent.

The resulting matrix contains only independent transformed TOHPE rows. Its storage is reserved only
for the possible prefix size and does not allocate full-basis rows, pivot rows for Todd completion, or
`FullToddData`. `FullToddGenerator` becomes lazy so this prefix operation also does not initialize the
full-Todd cache.

Each prefix sample is a coefficient vector over that compact matrix. Independence makes coefficient-to-
candidate mapping injective; the existing capped sampler avoids duplicate coefficient vectors.

## Independent searches and finalization

Both source searches independently:

1. select/diversify Todd-index buckets from their own `ZBucketSearch`;
2. stop using their own minimum, maximum, and source-pool reserve policy;
3. generate up to their own `actions_per_bucket` candidates per bucket;
4. retain candidates in their own exploration pool and update source-specific statistics.

Prefix candidates retain the corresponding full-Todd bucket identity (`k`, `l`, and `z`) and are tagged
as `CandidateSourceTohpe`. They do not use the former direct-TOHPE candidate representation.

At finalization, both sources merge with their independent reserve guarantees. Applying or scoring a
surviving candidate builds the complete Todd nullspace from its bucket identity, which is required for
the resulting matrix state. Full bases are therefore built only by the Todd search and for the small
final merged candidate set.

## Verification

Regression tests will cover:

- transformed-prefix basis construction without full-Todd initialization;
- prefix candidates using a separate bucket traversal and pool;
- no duplicate samples within a prefix search;
- final application of a prefix-source candidate through the complete Todd nullspace;
- Python API/config compatibility and existing policy-iteration smoke coverage.
