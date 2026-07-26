# Legacy and Efficient TOHPE Search Design

## Goal

Restore the original `TohpeSearch` interface and standalone TOHPE search while
preserving the current shared, prefix-only implementation under the explicit
name `TohpeEffSearch`.

## Public configuration

`TohpeSearch` again has the legacy fields `sampling`, `pool`, and `z_choices`.
Its defaults remain `SamplingBudget()`, `SourcePool{2, 0}`, and `z_choices=8`.
It controls the legacy TOHPE generator.

`TohpeEffSearch` owns the newer fields `sampling`, `pool`,
`actions_per_bucket`, and `buckets`. Its default pool is `SourcePool{0, 0}` so
the efficient search is disabled unless a caller explicitly configures it.

`PolicyConfig` holds both searches. Python bindings, pickle support, schedule
adapters, summaries, and search scripts expose the two names without silently
mapping one configuration into the other.

## Search behavior

The legacy `TohpeSearch` uses the original standalone TOHPE generator and its
`z_choices` limit. `TohpeEffSearch` retains the current shared bucket traversal
with Todd: it may request a TOHPE-only, Todd-only, or combined basis, uses a
shared collision set for combined sampling, and keeps source pools separate
until final selection. Enabling both TOHPE modes is valid; their candidates are
merged with Todd candidates at finalization.

## Compatibility and verification

Existing Python code that constructs, reads, represents, or pickles
`TohpeSearch(..., z_choices=...)` continues to work. The full PSO configuration
uses `TohpeEffSearch` explicitly to preserve its current behavior. Regression
tests cover the restored legacy API, disabled-by-default efficient config, and
the existing shared-traversal/lazy-full-basis behavior.
