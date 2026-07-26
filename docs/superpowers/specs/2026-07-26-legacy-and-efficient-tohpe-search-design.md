# TOHPE Prefix Search Design

## Goal

Preserve the original `TohpeSearch` interface and standalone TOHPE search while
exposing the shared, prefix-only implementation as `TohpePrefixSearch`.

## Public configuration

`TohpeSearch` has the fields `sampling`, `pool`, and `z_choices`.
Its defaults remain `SamplingBudget()`, `SourcePool{2, 0}`, and `z_choices=8`.
It controls the standalone TOHPE generator.

`TohpePrefixSearch` owns the newer fields `sampling`, `pool`,
`actions_per_bucket`, and `buckets`. Its default pool is `SourcePool{0, 0}` so
the prefix search is disabled unless a caller explicitly configures it.

`PolicyConfig` holds `tohpe`, `tohpe_prefix`, and `todd`. Python bindings,
pickle support, schedule adapters, summaries, and search scripts expose the
three names without silently mapping one configuration into another.

## Search behavior

`TohpeSearch` uses the original standalone TOHPE generator and its
`z_choices` limit. `TohpePrefixSearch` retains the current shared bucket traversal
with Todd: it may request a TOHPE-only, Todd-only, or combined basis, uses a
shared collision set for combined sampling. The prefix-search name makes its
TOHPE-basis-prefix sampling explicit.

The `tohpe`, `tohpe_prefix`, and `todd` sources each retain an independent
candidate pool and reserve quota. Result statistics report accepted actions and
final-pool composition for all three sources; compact path formatting renders
them as `H`, `P`, and `T` respectively.

## Compatibility and verification

Existing Python code that constructs, reads, represents, or pickles
`TohpeSearch(..., z_choices=...)` continues to work. The full PSO configuration
uses `TohpePrefixSearch` explicitly to preserve its current behavior. Regression
tests cover the restored API, disabled-by-default prefix config, separate
three-source accounting, and the existing shared-traversal/lazy-full-basis
behavior.
