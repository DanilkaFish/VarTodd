# `run_standard` JSON report

## Goal

Make every standard optimizer batch write one explicitly named JSON report
containing a record for each selected GF multiplication problem.  The report
captures the input and output ranks, complete runtime, the time at which the
final best rank was first found, and the optimizer's saved path text.

## Command-line interface

`run_standard.py` accepts a required `--output PATH` option.  `PATH` is the
single JSON report file, rather than a root directory inferred by the runner.
Optimized matrices remain separate `.npy` files in the report's parent
directory; each JSON record stores its path.

`--names` remains the opt-in way to run an explicit set of input matrices.
Without it, the default input circuit is `gf_mult_Vandaele_wo_ancilla` and the
runner selects every problem whose parsed GF degree is less than 32.  It orders
them by degree and then name, so all GF(2^3) through GF(2^16) problems are run
before any GF(2^32) problem.  Existing dimensional filter options are removed
because they do not represent this boundary reliably.

## JSON schema

The report is a JSON array, in selection order.  Every record contains:

```json
{
  "problem_name": "gf_mult_Vandaele_wo_ancilla/gf2^16_1612310",
  "initial_rank": 1231,
  "final_rank": 401,
  "execution_seconds": 123.456,
  "time_to_final_rank_seconds": 98.765,
  "paths": "... optimizer path text ...",
  "result_path": "/absolute/or/relative/result.npy"
}
```

`execution_seconds` measures the full entrypoint execution.  `paths` stores
the same complete search and best-path text currently printed by the runner.
`time_to_final_rank_seconds` is elapsed monotonic time from evaluator creation
to the first discovery of the eventual lowest rank.  It is `null` for an
optimizer that does not expose this metric.

The runner only publishes the JSON file after every selected job has completed,
so a successful file is always valid JSON.  The parent creates the output
directory when necessary.

## Evaluator timing

`BaseEvaluator` starts a `perf_counter` timer at construction.  When a strictly
better rank is recorded, it stores the elapsed time for that rank.  During PSO
merges, a newly better rank transfers its discovery time; equal final ranks
retain the earlier timestamp.  This gives the time of actual discovery in a
worker, rather than the later time at which futures happen to be merged.

`get_best()` emits the metric in its search report.  The runner reads that
structured line without changing the existing optimizer entrypoint tuple API.

## Tests

Tests cover GF-degree default discovery and ordering, report generation with a
mock optimizer result, and timing propagation in `BaseEvaluator` for a lower
rank and an equal-rank merge.  They also verify that `--output` is required and
that records appear in selection order despite parallel completion order.
