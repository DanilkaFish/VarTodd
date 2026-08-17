# TOHPE Score-Aware Z Ranking Design

## Goal

Rank the standalone TOHPE source's positive-reduction `z` candidates with the
same normalized `PolicyScores.exploration` function that ranks candidates
across sampled `y` vectors. This lets evolution tune reduction, collision
bucket size, and `z` density at both ranking levels without adding another
configuration object.

The default reduction-only exploration score must retain the current selected
`z` values, deterministic tie order, and optimized counting path.

## Scope

This change applies only to `generate_tohpe_candidates` and the score-aware
overload it uses. The public mathematical helpers
`TohpeGenerator::best_z`, `best_z_n`, and the existing
`best_z_n_details_into` remain reduction-greedy for compatibility.

`TohpePrefixSearch` and `ToddSearch` already rank their generated actions with
the exploration score and are unchanged. No new Python configuration fields or
pickle formats are introduced.

## Candidate features

For every touched `z` bucket and a fixed sampled kernel vector `y`, the inner
ranker uses the existing exploration feature order and normalization:

1. exact reduction: `reduction / bn / 2`;
2. TOHPE basis dimension: `basis_dim / dn`;
3. collision bucket size: `bucket_size / bn`;
4. `y` weight: `y.count() / wvwn`;
5. `z` density: `z.count() / z.size()`.

The basis dimension and `y` weight are constant while ranking `z` for one
`y`, but they are still supplied so the inner score is exactly the same score
as the outer candidate pool. Reduction, bucket size, and `z` density vary
between `z` candidates.

Candidates with exact reduction less than or equal to zero are discarded
before scoring. Therefore a score that penalizes reduction cannot make a
zero-reduction action eligible.

The implementation reuses all current `ScoringFunction` semantics. In
particular, polynomial scoring with a positive weight rewards distance from
the corresponding center; changing that behavior is outside this feature.

## Architecture

### Shared feature evaluation

Add a const feature-level evaluation method to `ExplorationScore`. It accepts
the five already-normalized candidate inputs, or the raw values needed to
produce them, and returns the scalar score. `ExplorationScore::operator()`
delegates to this method and stores the returned value in
`Candidate::pool_score`.

This prevents the inner TOHPE ranker from constructing a full `Candidate` only
to evaluate its score and guarantees that inner and outer scoring use one
formula.

### Score-aware TOHPE overload

Add a score-aware overload of `TohpeGenerator::best_z_n_details_into`. The
overload receives a lightweight callback with the exact reduction, bucket
size, and `z` row for one touched bucket. The callback captures the normalized
exploration score plus the fixed basis dimension and `y` weight.

`TohpeGenerator` remains independent of policy and scoring classes. The
existing reduction-only overload and Python-facing helpers do not change.

### Ranked count selection

Extend `CountWS` with a score-aware top-N operation. It evaluates the supplied
rank function once for each touched positive-reduction bucket, then keeps the
best `tohpe.z_choices` entries. It does not first truncate by reduction, since
that could discard a lower-reduction candidate preferred by the exploration
score.

The deterministic comparison order is:

1. exploration score descending;
2. exact reduction descending;
3. bucket ID ascending.

Only the selected bucket keys are retained in the output. A bucket key may be
reconstructed transiently to calculate `z` density, but all `z` rows are not
stored simultaneously.

### Reduction-only fast path

`ExplorationScore` exposes a predicate indicating when, for a fixed `y`, its
linear score and the deterministic tie rules produce the same order as exact
reduction. At minimum this covers the default positive reduction-only score.

When the predicate is true, standalone TOHPE calls the existing packed
`argmax_n_into` path. Other score shapes use the exact score-aware traversal.
This preserves default performance while allowing arbitrary configured
weights and centers to affect `z` ranking correctly.

## Data flow

For each sampled nonzero coefficient vector:

1. construct its kernel vector `y`;
2. count exact reductions for every touched `z` bucket as today;
3. remove nonpositive reductions;
4. choose either the reduction-only fast path or exact score-aware path;
5. return at most `tohpe.z_choices` ordered `TohpeZInfo` values;
6. build normal TOHPE `Candidate` objects and pass them through the existing
   source pool and finalization stages.

The source pool evaluates the same exploration score again when inserting the
completed candidate. This repeated evaluation is intentional: it keeps the
pool implementation uniform and provides the persisted `pool_score` without
duplicating state between layers.

## Statistics and compatibility

`evaluated` continues to count sampled `y` vectors. `accepted`,
`accepted_tohpe`, and score/reduction aggregates continue to count the
positive candidates returned to the source pool.

Nonpositive touched buckets are filtered internally and are not added to
`rejected`; historically only a nonpositive bucket that survived the top-N
reduction truncation could increment that field. The new behavior makes the
positive-only contract explicit and avoids making statistics depend on how
many discarded buckets were enumerated.

With the default exploration score, output ordering and candidate states must
be byte-for-byte deterministic relative to the current reduction-only path.

## Error handling

Existing integer-overflow checks for reductions, dimensions, and bucket sizes
remain in force. The scoring callback follows the existing assumption that
configured scoring functions produce comparable floating-point values; this
feature does not introduce separate score validation or alter non-finite-score
policy globally.

If no touched bucket has positive exact reduction, the score-aware method
returns an empty result and no candidate is emitted for that `y`.

## Verification

Regression tests will cover:

- default exploration settings return the same `z` IDs, reductions, and order
  as reduction-only `best_z_n_details_into`;
- a configured bucket-size or `z`-density weight can select a lower-reduction
  positive candidate over the maximum-reduction candidate;
- zero-reduction candidates are never selected, including when the reduction
  weight is negative;
- equal exploration scores break ties by reduction and then bucket ID;
- inner score evaluation equals the `pool_score` assigned to the completed
  candidate;
- selected reported reductions equal the actual row-count change after
  applying and canonicalizing the TOHPE action;
- the existing C++ regression suite and Python policy/configuration tests pass.

A focused timing comparison on a representative GF input will confirm that
the default reduction-only path does not materialize or score every touched
`z` key.
