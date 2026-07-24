# Full PSO: heavy, robust optimizer

## Goal

Make `scripts/base_search/full_pso.py` a heavier optimizer that chooses policies
which perform consistently across random search seeds, without losing its compact
single-file structure or restoring the removed candidate-validation stage.

## Search budget

The optimizer will evaluate 48 particles for 100 initial iterations and two
80-iteration restarts.  Each particle uses six fixed scoring seeds.  Relative to
the current 32-particle, 100 + 60 + 60 iteration, two-seed setup, this is
approximately 5.3 times the core search work:

```
48 * (100 + 80 + 80) * 6 / (32 * (100 + 60 + 60) * 2) = 5.32
```

## Objective

Each particle evaluation returns the median rank over its six scoring seeds plus
a non-trivial dispersion penalty.  Lower is better.  This removes the current
soft-min bias toward policies that succeed only on one lucky seed while retaining
a scalar objective required by global-best PSO.

`Evaluator.__call__` remains available for callers that need the independent
validation seed set, but `run_opt` does not add a separate validation phase.

## PSO configuration

Use a 48-particle global-best swarm with `w=0.72`, `c1=0.9`, `c2=1.3`, and a
symmetric velocity clamp.  The coefficients provide stronger attraction than the
previous `0.4/0.4` setup while the clamp keeps bounded parameters from repeatedly
crossing the search domain.

## Policy ranges

Retain the existing policy mapping and its compact parameter vector.  Change the
Z-bucket mapping so that the generated values always satisfy:

```
min_buckets <= max_buckets <= limit_bucket <= buckets_space
```

The three values remain controlled by the existing two Z budget parameters:
one chooses the minimum fraction of the selected search cap, and the other
chooses the cap/limit scale.  No independent range parameter or helper module is
introduced.

## Error handling and tests

The optimizer continues to propagate worker evaluation failures.  Tests will
cover the robust score aggregation, the valid Z-bucket ordering at opposing
parameter extremes, the PSO configuration passed to `GlobalBestPSO`, and the
updated restart budget.  Existing tests continue to assert that no candidate
validation stage is present.
