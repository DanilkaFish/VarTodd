# TODD TOHPE-prefix sampling design

## Goal

Allow each TODD nullspace search to draw two collision-free groups of coefficient
vectors: one from the transformed TOHPE-derived prefix of the TODD basis and one
from the complete TODD basis.

## Basis metadata

`solve_and_build_solution_basis_generated` already attempts transformed TOHPE
basis rows before it constructs TODD-only rows.  It will return a small result
object containing the completed basis and `tohpe_prefix_size`, the number of
independent transformed TOHPE rows actually retained.  The rows in
`[0, tohpe_prefix_size)` are therefore exactly the TOHPE-derived prefix.

`FullToddGenerator` passes this metadata into the returned `NullSpace`, which
exposes `tohpe_prefix_size()`.  A count is sufficient because the rows are
contiguous by construction; no redundant index vector is stored.

## Search API

`ToddSearch` gains a trailing `tohpe_sampling` field of type `SamplingBudget`.
Its default is the all-zero budget, preserving existing C++ aggregates, Python
calls, serialized policy data, and caller behavior.  The Python binding accepts
it as the fifth optional constructor argument and accepts both four-field legacy
and five-field pickle tuples.

Python policy adapters accept `tohpe_sampling` in mapping-based configuration.
`full_pso.py` supplies the same explicit sampling budget for both the prefix and
full-basis passes, enabling the new behavior for that optimizer.

## Collision-free sampling

`PyRNG` gains a two-region sampling routine.  It emits prefix samples first,
embedding their local coefficients in a full-dimension coefficient row.  It then
emits full-basis samples.  Both passes insert into one shared seen set keyed by
the full coefficient row; an already-seen full-space sample is retried, with the
existing finite-space and small-space exhaustion behavior retained.

No two emitted coefficients collide.  Since the nullspace basis is linearly
independent, distinct coefficient rows also produce distinct candidate vectors.
If the requested unique budget exceeds the finite coefficient universe, the
routine emits every available non-zero vector once.

## Tests

Core regression tests will verify that generated TODD bases report a valid TOHPE
prefix, the two-region sampler emits no duplicate rows while preserving all
available requested samples, and a zero prefix or zero prefix budget is safe.
Python-facing tests will verify the new `ToddSearch` argument and legacy pickle
compatibility.  Existing policy-iteration coverage will run with explicit
prefix sampling enabled.
