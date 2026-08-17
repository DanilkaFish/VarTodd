# Deterministic Parallel FastTODD Z Search

## Goal

Reduce the wall time of the FastTODD z-candidate search by evaluating it on 12 CPU threads while preserving the completed serial algorithm's exact table output and candidate tie-breaking. Remove deadline-based partial execution, report the current T-count before every FastTODD stage, and leave the GF(2^64) run for the user to launch after verification.

## Scope

The change applies to both FastTODD entry points in `quantum-circuit-optimization`:

- the normal circuit optimizer path through `fast_todd`;
- the locally added NPY/timing path through `fast_todd_timed` and `bench_fasttodd_stages`.

TOHPE, matrix construction, kernel elimination, the selected transformation, and `proper` remain serial. Only the complete `(i, j)` z-candidate scan within one outer FastTODD stage becomes parallel.

## Architecture

### One FastTODD implementation

Replace the duplicated normal and timed FastTODD loops with one internal implementation. `fast_todd` returns only its resulting table; `fast_todd_timed` additionally returns the existing stage/action timing counters. Both therefore use the same candidate search, transformations, progress output, and termination logic.

The deadline-aware `fast_todd_timed_until` entry point is removed. The benchmark binary no longer parses or advertises `--time-limit`; every started stage runs to a complete candidate-search result.

### Stage progress

Immediately before each outer FastTODD stage, emit this progress shape to standard error:

```text
FastTODD stage <number>: current T-count <count>
```

The stage number is one-based. Standard error keeps the benchmark's CSV standard output machine-readable. A stage consists of TOHPE followed by one complete z-candidate scan and, when an improvement exists, application of its transformation.

### Twelve-worker z search

Use `std::thread::scope` with a fixed maximum of 12 workers. A normal FastTODD scan with at least 12 useful outer-loop indices uses all 12; smaller tables avoid spawning workers that cannot receive work. An atomic next-index counter dynamically assigns outer `i` values because the triangular `j = i + 1 .. n` workload becomes smaller as `i` increases.

All stage inputs are shared read-only: the table, constructed matrix, augmented matrix, inverted pivot map, and table-key map. Each worker owns all candidate scratch data and its local best result. No worker modifies the shared FastTODD table.

For every assigned `i`, a worker visits `j` and then the eliminated row index `k` in the same ascending order as the current serial code. Each improving candidate records:

- score;
- serial order key `(i, j, k)`;
- `z`;
- `y`.

After workers finish, the caller reduces their local results by highest score and then smallest serial order key. The current serial implementation replaces its maximum only for a strictly larger score, so this ordering reproduces its first-maximum tie behavior independently of worker scheduling.

Candidate scoring computes the key for `table[l] XOR z` from worker-local scratch rather than temporarily XORing the shared table. This is algebraically identical to the current mutate-then-restore sequence and makes candidate evaluation race-free.

The winning transformation is applied once, serially, only after the complete parallel scan. The next stage therefore receives exactly the table that the completed serial algorithm would have produced.

## Exactness Contract

For the same input table and build target, a full 12-thread run must match a 1-thread reference run in:

- output table length and column order;
- every output column's bits;
- selected candidate sequence and number of FastTODD actions;
- TOHPE and FastTODD stage/action counters.

Elapsed timing fields and progress-output timing are observational and are not required to match. There is no deadline or partial-stage result after this change.

## Failure Handling

A worker panic propagates to the caller; FastTODD does not silently accept a partial scan or fall back to a potentially different result. Existing input assumptions and malformed-NPY errors are unchanged. The fixed worker count requires no CLI configuration and introduces no new crate dependency.

## Verification

Implementation follows test-driven development:

1. Add a deterministic selector test proving equal-score candidates choose the earliest `(i, j, k)`.
2. Add whole-algorithm regression cases that run the same small tables with one and 12 workers and compare exact ordered output columns and action counters.
3. Add a test covering a table with fewer useful outer indices than workers.
4. Verify the benchmark CLI no longer accepts or advertises deadline execution and still produces valid CSV while progress is written to standard error.
5. Run all Rust tests in debug and release/native-CPU configurations.
6. Benchmark one or more smaller GF matrices to confirm that the parallel region is active and results remain identical before handing off GF(2^64).

## GF(2^64) Handoff

Codex will not launch the full GF(2^64) optimization. After verification, the handoff will provide a release/native-CPU command for:

```text
data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^64_644310.npy
```

The command will run to completion, show per-stage T-count progress on standard error, and print the timing/result-count CSV row on standard output.
