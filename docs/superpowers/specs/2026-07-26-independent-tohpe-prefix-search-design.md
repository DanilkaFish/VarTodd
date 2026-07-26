# Shared-Traversal TOHPE Prefix Search

## Goal

Replace the standalone `TohpeGenerator` search with TOHPE-prefix and complete-Todd sampling over one
Todd-index bucket traversal:

- `TohpeSearch` samples the transformed TOHPE-prefix subspace only.
- `ToddSearch` samples the complete transformed Todd basis.

The sources keep separate per-bucket and global pools, share coefficient collision tracking whenever
they sample the same basis, and are merged only for final ranking and action selection.

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

## Shared bucket traversal

The standard top-bucket ordering is traversed once. Current policies use zero temperature and random
fraction, so no source-specific diversified order is needed.

At each bucket, TOHPE and Todd independently apply the existing Todd stop rule using their own
`ZBucketSearch` and source-pool state:

1. a source remains active through its `min_buckets`;
2. afterwards it stops when its reserve is met, or after its maximum once its pool is nonempty;
3. its `limit_bucket` bounds how far it can search.

If one source stops first, the traversal continues in the other source's only mode. It ends only when
both sources are inactive or the shared bucket ordering is exhausted. Each active source contributes up
to its own `actions_per_bucket` candidates to its own local and global exploration pool.

For a bucket, `FullToddGenerator` accepts a basis request:

- TOHPE only: construct only the transformed TOHPE-prefix basis;
- Todd only: construct the complete solution basis;
- both: construct the complete basis once and expose its transformed prefix.

In the both mode, prefix sampling fills the TOHPE per-bucket pool first. Full-basis sampling fills the
Todd per-bucket pool afterward using the same coefficient `seen` set, so a full-space coefficient that
duplicates an already sampled prefix coefficient is not evaluated twice.

## Prefix basis generation

The TOHPE-only request starts from `MatrixWithData::tohpe_basis()` and applies the solution transform
to each row before incremental pivot insertion. Its matrix reserves exactly the known TOHPE basis size;
no full-basis rows or Todd-completion pivot rows are allocated. It also does not initialize the
full-Todd cache.

The transform remains exact: a bucket can contain multiple transform entries, so the implementation does
not assume a global two-bit update bound. It reuses the compact transformed row and incrementally removes
dependencies against the prefix pivot basis.

## Finalization

Prefix candidates retain the corresponding full-Todd bucket identity (`k`, `l`, and `z`) and are tagged
as `CandidateSourceTohpe`. At finalization, both sources merge with their independent reserve guarantees.
Applying or scoring a surviving prefix candidate builds the complete Todd nullspace from that identity,
which is required for the resulting matrix state.

## Verification

Regression tests will cover:

- transformed-prefix basis construction without full-Todd initialization;
- each source continuing after the other source's stop condition;
- prefix and full samples sharing collision tracking in both mode;
- final application of a prefix-source candidate through the complete Todd nullspace;
- Python API/config compatibility and existing policy-iteration smoke coverage.
