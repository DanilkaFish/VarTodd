# Deterministic Parallel FastTODD Z Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run each complete FastTODD z-candidate scan on up to 12 CPU threads while producing the exact candidate and ordered table selected by the serial scan, removing partial deadline execution, and reporting each stage's current T-count.

**Architecture:** Add a focused internal `fast_todd_search` module that owns immutable candidate scoring, serial-order tie-breaking, and scoped-worker scheduling. Keep TOHPE and stage mutation in `t_opt.rs`, but consolidate its duplicated normal/timed FastTODD loops around the shared search API. The NPY benchmark becomes a full-run-only client and sends stage progress to stderr while keeping CSV on stdout.

**Tech Stack:** Rust 2021 standard library (`std::thread::scope`, `AtomicUsize`), existing `hashbrown::HashMap`, Cargo unit/integration tests; no new crate dependencies.

## Global Constraints

- Production FastTODD uses a fixed maximum of 12 workers; small scans spawn only useful workers.
- A completed 12-worker scan must choose the same maximum as the serial `(i, j, k)` traversal: highest score, then smallest `(i, j, k)`.
- Shared stage inputs are immutable during search; only the chosen transformation mutates the table after all workers join.
- TOHPE, matrix construction, kernel elimination, transformation application, and `proper` remain serial.
- Remove `fast_todd_timed_until`, duration parsing, and `--time-limit`; no partial-stage result remains.
- Print `FastTODD stage <number>: current T-count <count>` to stderr before every outer stage.
- Preserve the pre-existing local timing work in `quantum-circuit-optimization/src/t_opt.rs` and the untracked NPY benchmark source; never stage `target/` or unrelated outer-repository files.
- Do not launch GF(2^64); provide its release/native-CPU command only after verification.

## File Structure

- Create `quantum-circuit-optimization/src/fast_todd_search.rs`: private candidate type, deterministic reduction, pure scoring, pair evaluation, and 12-worker scoped scheduling.
- Modify `quantum-circuit-optimization/src/lib.rs`: register the private `fast_todd_search` module.
- Modify `quantum-circuit-optimization/src/t_opt.rs`: prepare immutable search inputs, call the search module, consolidate the FastTODD loop, report progress, and retain timing counters.
- Modify `quantum-circuit-optimization/src/bin/bench_fasttodd_stages.rs`: call `fast_todd_timed`, delete deadline handling, and reject the removed option.
- Create `quantum-circuit-optimization/tests/bench_fasttodd_cli.rs`: black-box progress/CSV and removed-option tests.

---

### Task 1: Deterministic candidate ordering

**Files:**
- Create: `quantum-circuit-optimization/src/fast_todd_search.rs`
- Modify: `quantum-circuit-optimization/src/lib.rs:8`

**Interfaces:**
- Produces: `Candidate { score: i32, order: (usize, usize, usize), z: BitVector, y: BitVector }`.
- Produces: `choose_better(Option<Candidate>, Option<Candidate>) -> Option<Candidate>`, selecting highest score and then smallest serial order.

- [ ] **Step 1: Register the module and write failing ordering tests**

Add `mod fast_todd_search;` to `src/lib.rs`. Create `src/fast_todd_search.rs` with tests that express the wished-for API before defining it:

```rust
#[cfg(test)]
mod tests {
    use super::{choose_better, Candidate};
    use crate::bit_vector::BitVector;

    fn candidate(score: i32, order: (usize, usize, usize)) -> Candidate {
        Candidate {
            score,
            order,
            z: BitVector::new(1),
            y: BitVector::new(1),
        }
    }

    #[test]
    fn higher_score_wins() {
        let winner = choose_better(
            Some(candidate(3, (0, 1, 0))),
            Some(candidate(5, (8, 9, 4))),
        )
        .unwrap();
        assert_eq!(winner.score, 5);
        assert_eq!(winner.order, (8, 9, 4));
    }

    #[test]
    fn equal_score_keeps_earliest_serial_candidate() {
        let winner = choose_better(
            Some(candidate(7, (3, 8, 2))),
            Some(candidate(7, (2, 9, 4))),
        )
        .unwrap();
        assert_eq!(winner.order, (2, 9, 4));
    }
}
```

- [ ] **Step 2: Run the tests and verify RED**

Run from `quantum-circuit-optimization`:

```bash
cargo test fast_todd_search::tests -- --nocapture
```

Expected: compilation fails because `Candidate` and `choose_better` do not exist.

- [ ] **Step 3: Implement the minimal deterministic reducer**

Add above the tests:

```rust
use crate::bit_vector::BitVector;

#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub(crate) score: i32,
    pub(crate) order: (usize, usize, usize),
    pub(crate) z: BitVector,
    pub(crate) y: BitVector,
}

pub(crate) fn choose_better(
    left: Option<Candidate>,
    right: Option<Candidate>,
) -> Option<Candidate> {
    match (left, right) {
        (None, candidate) | (candidate, None) => candidate,
        (Some(left), Some(right)) => {
            if right.score > left.score
                || (right.score == left.score && right.order < left.order)
            {
                Some(right)
            } else {
                Some(left)
            }
        }
    }
}
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

```bash
cargo test fast_todd_search::tests -- --nocapture
```

Expected: both ordering tests pass.

- [ ] **Step 5: Commit the isolated module**

```bash
git add src/lib.rs src/fast_todd_search.rs
git commit -m "test: define deterministic FastTODD candidate ordering"
```

### Task 2: Pure candidate scoring

**Files:**
- Modify: `quantum-circuit-optimization/src/fast_todd_search.rs`

**Interfaces:**
- Consumes: current table columns, table-key map, candidate `z`, and dependency vector `y`.
- Produces: `score_candidate(table, key_to_index, z, y) -> i32`, without mutating `table`.

- [ ] **Step 1: Write a failing pure-versus-mutate/restore scoring test**

Add test helpers and the following test inside the module's existing test block:

```rust
use hashbrown::HashMap;

fn bits(width: usize, ones: &[usize]) -> BitVector {
    let mut value = BitVector::new(width);
    for &bit in ones {
        value.xor_bit(bit);
    }
    value
}

fn mutating_reference_score(
    mut table: Vec<BitVector>,
    key_to_index: &HashMap<Vec<i128>, usize>,
    z: &BitVector,
    y: &BitVector,
) -> i32 {
    let mut score = 0;
    for index in 0..table.len() {
        if y.get(index) {
            table[index].xor(z);
            let key = table[index].get_integer_vec();
            if key_to_index.contains_key(&key)
                && !y.get(*key_to_index.get(&key).unwrap())
            {
                score += 2;
            }
            table[index].xor(z);
        }
    }
    if y.popcount() & 1 == 1 {
        if key_to_index.contains_key(&z.get_integer_vec()) {
            score += 1;
        } else {
            score -= 1;
        }
    }
    score
}

#[test]
fn pure_score_matches_existing_mutate_restore_score() {
    let table = vec![
        bits(8, &[0]),
        bits(8, &[1]),
        bits(8, &[0, 1]),
        bits(8, &[2]),
    ];
    let key_to_index = table
        .iter()
        .enumerate()
        .map(|(index, column)| (column.get_integer_vec(), index))
        .collect::<HashMap<_, _>>();
    let z = bits(8, &[0, 1]);
    let y = bits(table.len(), &[0, 1, 3]);

    assert_eq!(mutating_reference_score(table.clone(), &key_to_index, &z, &y), 1);
    assert_eq!(super::score_candidate(&table, &key_to_index, &z, &y), 1);
}
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
cargo test pure_score_matches_existing_mutate_restore_score -- --nocapture
```

Expected: compilation fails because `score_candidate` is missing.

- [ ] **Step 3: Implement immutable XOR-key scoring**

Add `use hashbrown::HashMap;` and implement:

```rust
fn xored_key(column: &BitVector, z_key: &[i128]) -> Vec<i128> {
    column
        .get_integer_vec()
        .into_iter()
        .zip(z_key.iter().copied())
        .map(|(left, right)| left ^ right)
        .collect()
}

fn score_candidate(
    table: &[BitVector],
    key_to_index: &HashMap<Vec<i128>, usize>,
    z: &BitVector,
    y: &BitVector,
) -> i32 {
    let z_key = z.get_integer_vec();
    let mut score = 0;
    for index in 0..table.len() {
        if !y.get(index) {
            continue;
        }
        let key = xored_key(&table[index], &z_key);
        if let Some(other_index) = key_to_index.get(&key) {
            if !y.get(*other_index) {
                score += 2;
            }
        }
    }
    if y.popcount() & 1 == 1 {
        score += if key_to_index.contains_key(&z_key) { 1 } else { -1 };
    }
    score
}
```

- [ ] **Step 4: Verify scoring and ordering tests stay GREEN**

```bash
cargo test fast_todd_search::tests -- --nocapture
```

Expected: all search-module tests pass.

- [ ] **Step 5: Commit pure scoring**

```bash
git add src/fast_todd_search.rs
git commit -m "refactor: score FastTODD candidates without table mutation"
```

### Task 3: Scoped 12-worker search

**Files:**
- Modify: `quantum-circuit-optimization/src/fast_todd_search.rs`

**Interfaces:**
- Produces: `SearchContext<'a>` containing immutable table/matrix/kernel data.
- Produces: `find_best_candidate(&SearchContext, max_workers: usize) -> Option<Candidate>`.
- Produces: `effective_worker_count(table_len, max_workers) -> usize`, capped at `table_len - 1` and at least one.

- [ ] **Step 1: Write failing scheduler and worker-count tests**

Add tests that compare the deterministic reducer under serial and parallel scheduling independently of FastTODD algebra:

```rust
#[test]
fn worker_count_uses_twelve_only_when_work_exists() {
    assert_eq!(super::effective_worker_count(40, 12), 12);
    assert_eq!(super::effective_worker_count(4, 12), 3);
    assert_eq!(super::effective_worker_count(1, 12), 1);
}

#[test]
fn parallel_index_reduction_matches_serial_order() {
    let evaluate = |i| {
        let order = if i == 4 { (4, 7, 3) } else { (i, i + 1, 0) };
        let score = if i == 2 || i == 4 { 9 } else { i as i32 };
        Some(candidate(score, order))
    };

    let serial = super::parallel_best_by_i(10, 1, &evaluate).unwrap();
    let parallel = super::parallel_best_by_i(10, 12, &evaluate).unwrap();
    assert_eq!(parallel.score, serial.score);
    assert_eq!(parallel.order, serial.order);
    assert_eq!(parallel.order, (2, 3, 0));
}
```

- [ ] **Step 2: Run the scheduler tests and verify RED**

```bash
cargo test fast_todd_search::tests -- --nocapture
```

Expected: compilation fails because the scheduler functions are missing.

- [ ] **Step 3: Implement dynamic scoped scheduling**

Add imports for `AtomicUsize`, `Ordering`, and `thread`. Implement this scheduler; every joined local result must pass through `choose_better`:

```rust
fn effective_worker_count(table_len: usize, max_workers: usize) -> usize {
    max_workers.max(1).min(table_len.saturating_sub(1).max(1))
}

fn parallel_best_by_i<F>(
    outer_len: usize,
    max_workers: usize,
    evaluate_i: &F,
) -> Option<Candidate>
where
    F: Fn(usize) -> Option<Candidate> + Sync,
{
    let next_i = AtomicUsize::new(0);
    let worker_count = effective_worker_count(outer_len + 1, max_workers);
    thread::scope(|scope| {
        let handles = (0..worker_count)
            .map(|_| {
                scope.spawn(|| {
                    let mut local_best = None;
                    loop {
                        let i = next_i.fetch_add(1, Ordering::Relaxed);
                        if i >= outer_len {
                            break;
                        }
                        local_best = choose_better(local_best, evaluate_i(i));
                    }
                    local_best
                })
            })
            .collect::<Vec<_>>();

        handles.into_iter().fold(None, |best, handle| {
            choose_better(best, handle.join().expect("FastTODD z-search worker panicked"))
        })
    })
}
```

- [ ] **Step 4: Add the immutable FastTODD search context and exact pair evaluator**

Define:

```rust
pub(crate) struct SearchContext<'a> {
    pub(crate) table: &'a [BitVector],
    pub(crate) nb_qubits: usize,
    pub(crate) matrix: &'a [BitVector],
    pub(crate) augmented_matrix: &'a [BitVector],
    pub(crate) pivots: &'a HashMap<usize, usize>,
    pub(crate) key_to_index: &'a HashMap<Vec<i128>, usize>,
}
```

Implement `evaluate_i(context, i)` by moving the current `src/t_opt.rs:282-375` candidate body into the new module with these exact semantic changes:

```rust
fn evaluate_i(context: &SearchContext<'_>, i: usize) -> Option<Candidate> {
    let mut best = None;
    for j in i + 1..context.table.len() {
        let mut z = context.table[i].clone();
        z.xor(&context.table[j]);

        let (mut r_mat, mut augmented_r_mat) = build_r_matrices(context, &z);

        for k in 0..r_mat.len() {
            let pivot_index = r_mat[k].get_first_one();
            if r_mat[k].get(pivot_index) {
                let pivot = r_mat[k].clone();
                let augmented_pivot = augmented_r_mat[k].clone();
                for row in k + 1..r_mat.len() {
                    if r_mat[row].get(pivot_index) {
                        r_mat[row].xor(&pivot);
                        augmented_r_mat[row].xor(&augmented_pivot);
                    }
                }
            } else if augmented_r_mat[k].get(i) ^ augmented_r_mat[k].get(j) {
                let y = augmented_r_mat[k].clone();
                let score = score_candidate(context.table, context.key_to_index, &z, &y);
                if score > 0 {
                    best = choose_better(
                        best,
                        Some(Candidate {
                            score,
                            order: (i, j, k),
                            z: z.clone(),
                            y,
                        }),
                    );
                }
            }
        }
    }
    best
}

pub(crate) fn find_best_candidate(
    context: &SearchContext<'_>,
    max_workers: usize,
) -> Option<Candidate> {
    parallel_best_by_i(
        context.table.len().saturating_sub(1),
        max_workers,
        &|i| evaluate_i(context, i),
    )
}
```

Implement `build_r_matrices` with the current loop order exactly: `k = 0..nb_qubits`, then `a` descending, then `b` ascending; append the final z-quadratic/linear row last:

```rust
fn build_r_matrices(
    context: &SearchContext<'_>,
    z: &BitVector,
) -> (Vec<BitVector>, Vec<BitVector>) {
    let z_vec = z.get_boolean_vec();
    let mut r_mat = Vec::with_capacity(context.nb_qubits + 1);
    let mut augmented_r_mat = Vec::with_capacity(context.nb_qubits + 1);

    for k in 0..context.nb_qubits {
        let mut col = BitVector::new_block_size(context.matrix[0].blocks.len());
        let mut augmented_col =
            BitVector::new_block_size(context.augmented_matrix[0].blocks.len());
        let mut pair_index = 0;
        for a in (0..context.nb_qubits).rev() {
            for b in 0..a {
                if (a == k && z_vec[b]) || (b == k && z_vec[a]) {
                    col.xor_bit(context.nb_qubits + pair_index);
                    if let Some(row) = context.pivots.get(&(context.nb_qubits + pair_index)) {
                        col.xor(&context.matrix[*row]);
                        augmented_col.xor(&context.augmented_matrix[*row]);
                    }
                }
                pair_index += 1;
            }
        }
        r_mat.push(col);
        augmented_r_mat.push(augmented_col);
    }

    let mut col = BitVector::new_block_size(context.matrix[0].blocks.len());
    let mut augmented_col =
        BitVector::new_block_size(context.augmented_matrix[0].blocks.len());
    let mut pair_index = 0;
    for a in (0..context.nb_qubits).rev() {
        for b in 0..a {
            if z_vec[a] && z_vec[b] {
                col.xor_bit(context.nb_qubits + pair_index);
                if let Some(row) = context.pivots.get(&(context.nb_qubits + pair_index)) {
                    col.xor(&context.matrix[*row]);
                    augmented_col.xor(&context.augmented_matrix[*row]);
                }
            }
            pair_index += 1;
        }
        if z_vec[a] {
            col.xor_bit(a);
            if let Some(row) = context.pivots.get(&a) {
                col.xor(&context.matrix[*row]);
                augmented_col.xor(&context.augmented_matrix[*row]);
            }
        }
    }
    r_mat.push(col);
    augmented_r_mat.push(augmented_col);
    (r_mat, augmented_r_mat)
}
```

Do not iterate any hash map while choosing a result.

- [ ] **Step 5: Run module tests and the compiler**

```bash
cargo test fast_todd_search::tests -- --nocapture
cargo check --all-targets
```

Expected: all tests pass and the module is `Send + Sync` under both scalar and native-CPU builds.

- [ ] **Step 6: Commit the parallel search engine**

```bash
git add src/fast_todd_search.rs
git commit -m "feat: parallelize deterministic FastTODD z search"
```

### Task 4: One FastTODD stage loop with progress

**Files:**
- Modify: `quantum-circuit-optimization/src/t_opt.rs:246-579`

**Interfaces:**
- Consumes: `find_best_candidate` and `SearchContext` from Task 3.
- Produces: `fast_todd_impl(table, nb_qubits, max_workers, report_progress) -> (Vec<BitVector>, FastToddTiming)`.
- Keeps public: `fast_todd(table, nb_qubits) -> Vec<BitVector>` and `fast_todd_timed(table, nb_qubits) -> (Vec<BitVector>, FastToddTiming)`.

- [ ] **Step 1: Write failing one-versus-12 whole-algorithm tests**

Add a `#[cfg(test)] mod tests` at the bottom of `t_opt.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::fast_todd_impl;
    use crate::circuit::{Circuit, SlicedCircuit};
    use crate::fast_todd_search::{find_best_candidate, SearchContext};
    use std::path::PathBuf;

    fn gf4_phase_table() -> (Vec<crate::bit_vector::BitVector>, usize) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("circuits/inputs/gf2^4_mult.qc");
        let (circuit, _, _) = Circuit::from_qc(path.to_str().unwrap());
        let gadgetized = circuit.hadamard_gadgetization();
        let sliced = SlicedCircuit::from_circ(&gadgetized);
        (sliced.phase_polynomials[0].table.clone(), sliced.nb_qubits)
    }

    fn ordered_keys(table: &[crate::bit_vector::BitVector]) -> Vec<Vec<i128>> {
        table.iter().map(|column| column.get_integer_vec()).collect()
    }

    #[test]
    fn twelve_workers_match_one_worker_exactly() {
        let (table, nb_qubits) = gf4_phase_table();
        let (serial, serial_timing) = fast_todd_impl(table.clone(), nb_qubits, 1, false);
        let (parallel, parallel_timing) = fast_todd_impl(table, nb_qubits, 12, false);

        assert_eq!(ordered_keys(&parallel), ordered_keys(&serial));
        assert_eq!(parallel_timing.stages, serial_timing.stages);
        assert_eq!(parallel_timing.tohpe_stages, serial_timing.tohpe_stages);
        assert_eq!(parallel_timing.todd_stages, serial_timing.todd_stages);
        assert_eq!(parallel_timing.tohpe_actions, serial_timing.tohpe_actions);
        assert_eq!(parallel_timing.todd_actions, serial_timing.todd_actions);
    }

    #[test]
    fn twelve_workers_select_the_same_ordered_candidate() {
        let (table, nb_qubits) = gf4_phase_table();
        let table = super::tohpe(table, nb_qubits);
        let (matrix, augmented_matrix, pivots, key_to_index) =
            super::prepare_fast_todd_search(&table, nb_qubits);
        let context = SearchContext {
            table: &table,
            nb_qubits,
            matrix: &matrix,
            augmented_matrix: &augmented_matrix,
            pivots: &pivots,
            key_to_index: &key_to_index,
        };

        let serial = find_best_candidate(&context, 1).unwrap();
        let parallel = find_best_candidate(&context, 12).unwrap();
        assert_eq!(parallel.score, serial.score);
        assert_eq!(parallel.order, serial.order);
        assert_eq!(parallel.z.get_integer_vec(), serial.z.get_integer_vec());
        assert_eq!(parallel.y.get_integer_vec(), serial.y.get_integer_vec());
    }

    #[test]
    fn progress_line_has_stable_shape() {
        assert_eq!(
            super::stage_progress(3, 1275),
            "FastTODD stage 3: current T-count 1275"
        );
    }
}
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
cargo test t_opt::tests -- --nocapture
```

Expected: compilation fails because `fast_todd_impl`, `prepare_fast_todd_search`, and `stage_progress` do not exist.

- [ ] **Step 3: Consolidate the two loops and connect the search context**

Import `find_best_candidate` and `SearchContext`; remove the duplicated candidate scan from both existing loops. Use this structure:

```rust
const FAST_TODD_WORKERS: usize = 12;

fn stage_progress(stage: usize, t_count: usize) -> String {
    format!("FastTODD stage {stage}: current T-count {t_count}")
}

fn fast_todd_impl(
    mut table: Vec<BitVector>,
    nb_qubits: usize,
    max_workers: usize,
    report_progress: bool,
) -> (Vec<BitVector>, FastToddTiming) {
    let mut timing = FastToddTiming::default();
    loop {
        timing.stages += 1;
        if report_progress {
            eprintln!("{}", stage_progress(timing.stages, table.len()));
        }

        let tohpe_start = Instant::now();
        let (next_table, actions) = tohpe_with_action_count(table.clone(), nb_qubits);
        table = next_table;
        timing.tohpe_seconds += tohpe_start.elapsed().as_secs_f64();
        timing.tohpe_stages += 1;
        timing.tohpe_actions += actions;

        let todd_start = Instant::now();
        let (matrix, augmented_matrix, pivots, key_to_index) =
            prepare_fast_todd_search(&table, nb_qubits);
        let context = SearchContext {
            table: &table,
            nb_qubits,
            matrix: &matrix,
            augmented_matrix: &augmented_matrix,
            pivots: &pivots,
            key_to_index: &key_to_index,
        };
        let candidate = find_best_candidate(&context, max_workers);
        timing.todd_seconds += todd_start.elapsed().as_secs_f64();
        timing.todd_stages += 1;

        let Some(candidate) = candidate else { break };
        for index in 0..table.len() {
            if candidate.y.get(index) {
                table[index].xor(&candidate.z);
            }
        }
        if candidate.y.popcount() & 1 == 1 {
            table.push(candidate.z);
        }
        table = proper(table);
        timing.todd_actions += 1;
    }
    (table, timing)
}

pub fn fast_todd(table: Vec<BitVector>, nb_qubits: usize) -> Vec<BitVector> {
    fast_todd_impl(table, nb_qubits, FAST_TODD_WORKERS, true).0
}

pub fn fast_todd_timed(
    table: Vec<BitVector>,
    nb_qubits: usize,
) -> (Vec<BitVector>, FastToddTiming) {
    fast_todd_impl(table, nb_qubits, FAST_TODD_WORKERS, true)
}
```

`prepare_fast_todd_search` must contain the current serial matrix extension, identity augmented matrix, `kernel` call, pivot inversion, and table-key map construction in their existing order.

Use this concrete preparation helper:

```rust
fn prepare_fast_todd_search(
    table: &[BitVector],
    nb_qubits: usize,
) -> (
    Vec<BitVector>,
    Vec<BitVector>,
    HashMap<usize, usize>,
    HashMap<Vec<i128>, usize>,
) {
    let mut matrix = table.to_vec();
    for index in 0..table.len() {
        let mut t_vec = table[index].get_boolean_vec();
        t_vec.truncate(nb_qubits);
        let mut quadratic = Vec::<bool>::new();
        for _ in 0..nb_qubits {
            if t_vec.pop().unwrap() {
                quadratic.append(&mut t_vec.clone());
            } else {
                quadratic.append(&mut vec![false; t_vec.len()]);
            }
        }
        matrix[index].extend_vec(quadratic, nb_qubits);
    }

    let mut augmented_matrix = Vec::with_capacity(table.len());
    for index in 0..table.len() {
        let mut basis = BitVector::new(table.len());
        basis.xor_bit(index);
        augmented_matrix.push(basis);
    }

    let mut row_to_pivot = HashMap::<usize, usize>::new();
    kernel(&mut matrix, &mut augmented_matrix, &mut row_to_pivot);
    let pivots = row_to_pivot
        .iter()
        .map(|(row, pivot)| (*pivot, *row))
        .collect::<HashMap<_, _>>();
    let key_to_index = table
        .iter()
        .enumerate()
        .map(|(index, column)| (column.get_integer_vec(), index))
        .collect::<HashMap<_, _>>();

    (matrix, augmented_matrix, pivots, key_to_index)
}
```

Delete the old normal loop only after the one-worker and 12-worker paths share this implementation.

- [ ] **Step 4: Run exactness tests and verify GREEN repeatedly**

```bash
cargo test t_opt::tests -- --nocapture
cargo test twelve_workers_match_one_worker_exactly -- --nocapture
cargo test twelve_workers_match_one_worker_exactly -- --nocapture
```

Expected: all three runs pass with identical ordered output and counters; repeated execution checks schedule independence.

- [ ] **Step 5: Run all library tests**

```bash
cargo test --lib -- --nocapture
```

Expected: all library tests pass.

- [ ] **Step 6: Commit the common FastTODD loop**

Because `src/t_opt.rs` contained pre-existing timing work, inspect `git diff -- src/t_opt.rs` before staging and confirm those counters remain present. Then:

```bash
git add src/t_opt.rs
git commit -m "refactor: unify FastTODD execution and report stage progress"
```

### Task 5: Remove partial deadline execution from the NPY runner

**Files:**
- Modify: `quantum-circuit-optimization/src/t_opt.rs`
- Modify: `quantum-circuit-optimization/src/bin/bench_fasttodd_stages.rs`
- Create: `quantum-circuit-optimization/tests/bench_fasttodd_cli.rs`

**Interfaces:**
- Removes: `fast_todd_timed_until(table, nb_qubits, deadline)`.
- Benchmark consumes only: `fast_todd_timed(table, nb_qubits)`.
- CLI accepts one input plus `--header` and `--csv <path>`; `--time-limit` is an error.

- [ ] **Step 1: Write failing black-box CLI tests**

Create `tests/bench_fasttodd_cli.rs`:

```rust
use std::path::PathBuf;
use std::process::Command;

fn input() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("circuits/inputs/tof_3.qc")
}

#[test]
fn benchmark_keeps_csv_on_stdout_and_progress_on_stderr() {
    let output = Command::new(env!("CARGO_BIN_EXE_bench_fasttodd_stages"))
        .arg(input())
        .arg("--header")
        .output()
        .unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stdout.starts_with("algorithm,circuit,initial_t_count"));
    assert!(stdout.lines().last().unwrap().starts_with("fasttodd,"));
    assert!(!stdout.contains("current T-count"));
    assert!(stderr.contains("FastTODD stage 1: current T-count"));
}

#[test]
fn removed_time_limit_is_rejected() {
    let output = Command::new(env!("CARGO_BIN_EXE_bench_fasttodd_stages"))
        .arg(input())
        .args(["--time-limit", "1s"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("unknown option: --time-limit"));
}
```

- [ ] **Step 2: Run the integration tests and verify RED**

```bash
cargo test --test bench_fasttodd_cli -- --nocapture
```

Expected: the progress/CSV test passes through the common runner from Task 4, while the removed-option test fails because the current binary still accepts `--time-limit`.

- [ ] **Step 3: Delete deadline APIs and parsing**

In `t_opt.rs`, remove `fast_todd_timed_until` completely and keep only the common `fast_todd_timed`. In the benchmark:

- change the import to `use quantum_circuit_optimization::t_opt::fast_todd_timed;`;
- remove `Duration`, `parse_duration`, `time_limit`, and `deadline`;
- replace both calls with `fast_todd_timed(table, nb_qubits)`;
- validate options after the input so only `--header` and `--csv <path>` are accepted;
- print `unknown option: <flag>` and exit with status 2 for `--time-limit` or any other unknown flag.

Use a single index-based parser so `--csv` consumes its path:

```rust
let mut csv_path = None;
let mut emit_header = false;
let mut index = 2;
while index < args.len() {
    match args[index].as_str() {
        "--header" => emit_header = true,
        "--csv" => {
            index += 1;
            if index >= args.len() {
                eprintln!("missing value for --csv");
                std::process::exit(2);
            }
            csv_path = Some(args[index].clone());
        }
        option => {
            eprintln!("unknown option: {option}");
            std::process::exit(2);
        }
    }
    index += 1;
}
```

- [ ] **Step 4: Verify the CLI tests are GREEN**

```bash
cargo test --test bench_fasttodd_cli -- --nocapture
```

Expected: both tests pass; CSV is stdout-only and progress is stderr-only.

- [ ] **Step 5: Verify no deadline code remains**

```bash
rg -n "fast_todd_timed_until|time_limit|time-limit|parse_duration|deadline|Duration" src tests
```

Expected: no matches.

- [ ] **Step 6: Commit the full-run-only benchmark path**

```bash
git add src/t_opt.rs src/bin/bench_fasttodd_stages.rs tests/bench_fasttodd_cli.rs
git commit -m "feat: make FastTODD benchmark full-run only"
```

### Task 6: Formatting, complete verification, and GF handoff

**Files:**
- Modify mechanically if needed: Rust files touched above.
- Do not create GF(2^64) output or launch the GF(2^64) process.

**Interfaces:**
- Verifies the fixed production command and exactness contract.
- Produces the user-facing GF(2^64) command.

- [ ] **Step 1: Format and inspect only intended changes**

```bash
cargo fmt --all -- --check
git status --short
git diff --check
```

If the format check fails, run `cargo fmt --all`, then repeat all three commands. Confirm `target/` and the pre-existing `Cargo.lock` remain unstaged.

- [ ] **Step 2: Run the complete debug test suite**

```bash
cargo test --all-targets -- --nocapture
```

Expected: all unit, binary, and integration tests pass.

- [ ] **Step 3: Run native-CPU release verification**

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --release --all-targets -- --nocapture
```

Expected: all tests pass with AVX2/native code generation.

- [ ] **Step 4: Smoke-test a smaller NPY GF matrix**

From `quantum-circuit-optimization` run:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin bench_fasttodd_stages -- ../data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^8_84320.npy --header
```

Expected: stderr shows one or more `FastTODD stage N: current T-count X` lines; stdout contains exactly the CSV header and one `fasttodd,gf2^8_84320,...` row.

- [ ] **Step 5: Review the final diff and run focused exactness once more**

```bash
git diff --check
cargo test twelve_workers_match_one_worker_exactly -- --nocapture
git status --short
```

Expected: exactness passes; only intended source/test changes and known pre-existing files are present.

- [ ] **Step 6: Commit any formatter-only changes**

If `cargo fmt` changed tracked implementation files after the prior commits:

```bash
git add src/lib.rs src/fast_todd_search.rs src/t_opt.rs src/bin/bench_fasttodd_stages.rs tests/bench_fasttodd_cli.rs
git commit -m "style: format parallel FastTODD implementation"
```

If there are no formatter changes, do not create an empty commit.

- [ ] **Step 7: Hand off the GF(2^64) command without running it**

From `/home/danilkafish/Projects/VarTodd`, provide:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release \
  --manifest-path quantum-circuit-optimization/Cargo.toml \
  --bin bench_fasttodd_stages -- \
  'data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^64_644310.npy' \
  --header
```

Explain that stage progress goes to stderr and the final CSV result goes to stdout.
