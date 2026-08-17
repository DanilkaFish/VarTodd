# VarTODD FastTODD Mimic Progress Design

## Goal

Make `scripts/benchmarks/vartodd_fasttodd_mimic.py` report every new T-count during long parity-matrix runs and change its sampling defaults to the closest configuration exposed by VarTODD for the Vandaele algorithm.

## Vandaele-aligned sampling defaults

Vandaele's TOHPE phase obtains one dependency vector from its incremental kernel, chooses the best `z` for that dependency, applies it, and repeats. VarTODD's current sampler cannot request the deterministic first dependency directly. Its closest configuration is therefore:

```text
TOHPE: one_hot=1, sparse=0, dense=0, sparse_max_weight=2
```

The fixed seed keeps this approximation reproducible, but it is not strict dependency-order equivalence with Vandaele.

Vandaele's FastTODD phase evaluates the individual dependency solutions generated while scanning every pair-derived `z`. The corresponding VarTODD configuration is:

```text
FastTODD: one_hot=all, sparse=0, dense=0, sparse_max_weight=2
```

The existing effectively exhaustive bucket defaults and one retained action per bucket remain unchanged. Both sampling defaults will be named constants so their different purposes are explicit. Command-line overrides remain supported.

## Progress output

The script will write progress to stderr and flush every line immediately. CSV output remains isolated on stdout or in the requested CSV file.

Output has the same shape as the Rust benchmark:

```text
T-count 5103: initial
T-count 5099: TOHPE stage 1, action 1
T-count 5089: FastTODD stage 1, action 1
```

The initial count is printed once after loading the matrix. Every successful TOHPE action is printed after replacing the state and incrementing the stage-local action number. Every successful FastTODD action is printed after replacing the state and incrementing the global FastTODD action count.

`_apply_one` already rejects states whose row count does not strictly decrease. Consequently, every emitted action count is new and adjacent duplicates cannot occur. The reporting code will nevertheless centralize last-count tracking to preserve that output contract if action acceptance changes later.

## Behavioral constraints

This change must not alter policy scores, source pools, bucket traversal, action selection, seeding, timing counters, output matrices, or CSV fields. Only the two command-line sampling defaults and progress observability change.

The optional time limit remains supported. If it expires, the script keeps the already emitted progress and writes the existing aggregate result for the state reached before expiration.

## Testing

Tests will first fail against the current behavior and then verify:

- the default TOHPE sampling budget is `[1, 0, 0, 2]`;
- the default FastTODD sampling budget is `["all", 0, 0, 2]`;
- explicit command-line sampling overrides still work;
- progress starts with the initial count;
- progress includes labeled TOHPE and FastTODD reductions;
- emitted T-counts never repeat adjacently;
- progress stays on stderr while CSV stays on stdout.

Verification will use only a small parity-matrix `.npy` fixture. The GF(2^64) input will not be launched.
