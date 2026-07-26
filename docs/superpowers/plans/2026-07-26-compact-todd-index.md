# Compact Purpose-Built ToddIndex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the general-purpose `ToddIndex` bucket map with an exact, insertion-ordered flat index that retains no full bucket keys and stores only sparse duplicate members.

**Architecture:** Build buckets in the existing single/pair traversal order with a packed open-addressing table. Keep the hot `single_id_` and row-major `pair_id_` arrays, reconstruct keys from one source ordinal per bucket, compact bucket lengths by maximum value, and materialize complete ordered member lists only for selected buckets.

**Tech Stack:** C++20, packed `std::uint32_t` metadata, `std::variant`, existing `Matrix`/`Row` primitives, CMake, `perf`.

## Global Constraints

- Bucket membership and `SumEntry` order must remain exact.
- Hash and fingerprint collisions must be resolved by complete vector comparison.
- New bucket IDs must retain current first-encounter order.
- The target retained `ToddIndex` size must be at most 24 MiB.
- The exact 25-iteration benchmark median must improve.
- `pair_bucket_id` and `CountWS` must not incur an offsetting regression.
- `FullToddData` must not be modified.
- Python result schemas and policy configuration must not change.
- Preserve the user's existing uncommitted changes in `demos/bench.cpp` and `include/todd_generator.hpp`.

---

## File Structure

- `include/todd_index.hpp`: public materialization/key API, adaptive bucket-length storage, compact retained fields.
- `src/core/todd_index.cpp`: checked sizing, flat probing, representative comparison, source decoding, sparse duplicate grouping, and selection methods.
- `include/nullspace.hpp`: reusable bucket-entry scratch owned by generators where needed.
- `src/core/nullspace.cpp`: replace stable range access with explicit materialization and use direct IDs for row/pair overloads.
- `src/core/todd_generator.cpp`: materialize selected Todd buckets into loop-owned scratch.
- `tests/core_regression.cpp`: exact reference-index, adaptive-width, materialization, nullspace, and policy regression coverage.
- `demos/bench_todd_index.cpp`: dedicated construction-time and retained-size benchmark.
- `demos/CMakeLists.txt`: build the dedicated index benchmark.

---

### Task 0: Capture a Reproducible Baseline

**Files:**
- No source changes.
- Store transient logs only under `/tmp`.

**Interfaces:**
- Produces: baseline build time, end-to-end median, profile, and concurrent timings used by Task 6.

- [ ] **Step 1: Configure an isolated multi-config build**

From the implementation worktree, run:

```bash
cmake -S . -B build -G "Ninja Multi-Config" \
  -DBUILD_TESTING=ON -DVARTODD_ENABLE_NATIVE_ARCH=ON
cmake --build build --config RelWithDebInfo \
  --target bench vartodd_core_regression -j2
```

Expected: both binaries are rebuilt from the worktree's baseline source.

- [ ] **Step 2: Verify the baseline regression binary**

Run:

```bash
build/bin/RelWithDebInfo/vartodd_core_regression
```

Expected: `vartodd_core_regression ok`.

- [ ] **Step 3: Record three exact baseline samples**

Run three times:

```bash
build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  25 8 1 0
```

Record all `total_ms` values and their median in `/tmp`, not in the
repository.

- [ ] **Step 4: Capture the baseline profile**

Run:

```bash
perf record -q -g --call-graph dwarf -o /tmp/vartodd_index_baseline.data \
  build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  25 8 1 0
perf report --stdio --no-children \
  -i /tmp/vartodd_index_baseline.data --percent-limit 0.4
```

Record the `ToddIndex` construction and TOHPE counting percentages.

- [ ] **Step 5: Record pinned two-process and four-process baselines**

Run the exact benchmark concurrently on physical cores `0,2`, then
`0,2,4,6`. Redirect each process's normal output to a distinct `/tmp` log and
record every `total_ms`. Expected values should be consistent with the
previous observations:

```text
two processes: approximately 7.55--7.61 seconds each
four processes: approximately 8.95--9.23 seconds each
```

If materially different, use the newly measured values as the baseline and
report the discrepancy.

---

### Task 1: Freeze Bucket Semantics Behind a Materialization API

**Files:**
- Modify: `include/todd_index.hpp:201-276`
- Modify: `src/core/todd_index.cpp:170-223`
- Modify: `tests/core_regression.cpp:85-105`

**Interfaces:**
- Produces: `bool ToddIndex::materialize_bucket(std::uint32_t id, std::vector<SumEntry>& out) const`
- Produces: `bool ToddIndex::materialize_bucket(RowCView key, std::vector<SumEntry>& out) const`
- Preserves temporarily: existing `sum_bucket(..., const SumEntry*&, index_t&)` methods for downstream migration.

- [ ] **Step 1: Add a failing exact materialization test**

Add helpers that compare `SumEntry` values:

```cpp
bool same_entry(const SumEntry& a, const SumEntry& b) {
    return a.a == b.a && a.b == b.b;
}

void require_same_entries(const std::vector<SumEntry>& actual,
                          const std::vector<SumEntry>& expected,
                          const char* message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t i = 0; i < actual.size(); ++i)
        require(same_entry(actual[i], expected[i]), message);
}
```

In `check_todd_index_basics`, materialize the pair bucket both by ID and by
key, then compare the values:

```cpp
std::vector<SumEntry> by_id;
std::vector<SumEntry> by_key;
require(index.materialize_bucket(bucket_id, by_id), "pair materialization by id failed");
require(index.materialize_bucket(pair, by_key), "pair materialization by key failed");
require_same_entries(by_id, by_key, "id/key materialization order differs");
```

Add a duplicate-heavy matrix whose rows are `000`, `000`, `001`, `001`,
`010`, `010`. Build a reference vector by visiting singles first and pairs in
`i,j` order. Verify every `single_id`, every `pair_bucket_id`, every
`key_of(id)`, and every materialized list against the reference.

- [ ] **Step 2: Build to verify the new API test fails**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
```

Expected: compilation fails because `ToddIndex::materialize_bucket` does not
exist.

- [ ] **Step 3: Implement materialization using the current stable ranges**

Declare the two methods in `ToddIndex`. Implement the ID overload as:

```cpp
bool ToddIndex::materialize_bucket(std::uint32_t id, std::vector<SumEntry>& out) const {
    out.clear();
    const SumEntry* ptr = nullptr;
    index_t len = 0;
    if (!sum_bucket(id, ptr, len))
        return false;
    out.assign(ptr, ptr + len);
    return true;
}
```

The key overload resolves the existing stable range and copies it in the same
way. Invalid IDs and missing keys must clear `out` and return `false`.

- [ ] **Step 4: Run the characterization tests**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
```

Expected: `vartodd_core_regression ok`.

- [ ] **Step 5: Commit the materialization seam**

```bash
git add include/todd_index.hpp src/core/todd_index.cpp tests/core_regression.cpp
git commit -m "Characterize ToddIndex bucket materialization"
```

---

### Task 2: Add Exact Adaptive Bucket-Length Storage

**Files:**
- Modify: `include/todd_index.hpp:190-276`
- Modify: `tests/core_regression.cpp:135-216`

**Interfaces:**
- Produces: `todd::detail::BucketLengths`
- Produces: `void BucketLengths::assign(const std::vector<std::uint32_t>& counts, std::uint32_t max_count)`
- Produces: `index_t BucketLengths::get(std::size_t id) const noexcept`
- Produces: `template<class F> decltype(auto) BucketLengths::with_values(F&& f) const`
- Produces: `std::size_t BucketLengths::count_bytes() const noexcept`
- Produces: `std::size_t BucketLengths::storage_bytes() const noexcept`

- [ ] **Step 1: Write failing adaptive-width tests**

Add:

```cpp
void check_bucket_lengths() {
    detail::BucketLengths lengths;

    lengths.assign({0, 1, 9, 255}, 255);
    require(lengths.count_bytes() == 1, "byte bucket lengths not selected");
    require(lengths.get(3) == 255, "byte bucket length changed");

    lengths.assign({0, 256, 65535}, 65535);
    require(lengths.count_bytes() == 2, "16-bit bucket lengths not selected");
    require(lengths.get(1) == 256, "16-bit bucket length changed");

    lengths.assign({0, 65536}, 65536);
    require(lengths.count_bytes() == 4, "32-bit bucket lengths not selected");
    require(lengths.get(1) == 65536, "32-bit bucket length changed");
}
```

Call it from `main`.

- [ ] **Step 2: Build to verify the helper is missing**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
```

Expected: compilation fails because `detail::BucketLengths` is undefined.

- [ ] **Step 3: Implement `BucketLengths`**

Use:

```cpp
using Storage = std::variant<
    std::vector<std::uint8_t>,
    std::vector<std::uint16_t>,
    std::vector<std::uint32_t>>;
```

`assign` selects one byte for `max_count <= UINT8_MAX`, two for
`max_count <= UINT16_MAX`, and four otherwise. Convert every count with an
asserted/exact cast. `with_values` performs one `std::visit`; callers scanning
all buckets must dispatch once rather than once per element.

- [ ] **Step 4: Run the regression binary**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
```

Expected: `vartodd_core_regression ok`.

- [ ] **Step 5: Commit adaptive bucket lengths**

```bash
git add include/todd_index.hpp tests/core_regression.cpp
git commit -m "Add adaptive ToddIndex bucket lengths"
```

---

### Task 3: Replace Generic Hashing and Stored Keys with a Flat Exact Index

**Files:**
- Modify: `include/todd_index.hpp:1-15,201-276`
- Modify: `src/core/todd_index.cpp:1-223`
- Modify: `src/core/nullspace.cpp:500-560`
- Modify: `src/core/todd_generator.cpp:680-705`
- Modify: `tests/core_regression.cpp:85-135`

**Interfaces:**
- Changes: `Row ToddIndex::key_of(std::uint32_t id) const`
- Produces: exact internal `find_bucket_(HashKey hash, RowCView key) const noexcept`
- Produces: internal `find_or_insert_bucket_(HashKey hash, RowCView key, std::uint32_t source_pos)`
- Produces: internal source decoding and representative comparison helpers.
- Removes: `sum_key_sizes()`, `sum_key_sizes_scratch()`, and `row_hashes()`.

- [ ] **Step 1: Extend tests for owning keys, missing keys, and stable IDs**

Update the characterization test so `Row key = index.key_of(id)` owns its
storage. Add an absent vector and verify:

```cpp
std::vector<SumEntry> absent_entries{{0}};
require(!index.materialize_bucket(absent.cview(), absent_entries),
        "absent key unexpectedly found");
require(absent_entries.empty(), "missing lookup did not clear output");
```

For the duplicate-heavy matrix, independently assign reference IDs in first
encounter order and compare every ID. This will catch a hash-sorted or
collision-chain ordering change.

- [ ] **Step 2: Run the tests before the rewrite**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
```

Expected: pass. This is the behavioral baseline for the representation
replacement.

- [ ] **Step 3: Add checked flat-table sizing and slot packing**

In `todd_index.cpp`, add helpers with these contracts:

```cpp
std::size_t checked_sum_count(std::size_t rows);
std::size_t checked_table_capacity(std::size_t max_buckets);
unsigned id_bits_for(std::size_t max_buckets);
std::uint32_t pack_slot(std::uint32_t id, std::uint32_t fingerprint,
                        unsigned id_bits, std::uint32_t id_mask);
```

`checked_sum_count` must reject counts greater than
`UINT32_MAX`. `checked_table_capacity` chooses a power of two satisfying a
0.7 maximum load and rejects overflow. Zero sums use an empty table.

- [ ] **Step 4: Implement representative reconstruction and exact comparison**

Store one source ordinal per bucket in `representative_pos_`. Add:

```cpp
SumEntry decode_source_position_(std::uint32_t pos) const;
bool representative_equals_(std::uint32_t id, RowCView key) const noexcept;
Row row_from_source_position_(std::uint32_t pos) const;
```

Singles decode directly. Pairs subtract `m_`, locate the triangular row with
`std::upper_bound(pair_row_off_, pair_pos) - 1`, and recover `j`. Exact
comparison checks all logical blocks against `P[i]` or `P[i] ^ P[j]`.

- [ ] **Step 5: Replace `head_`, collision links, and full keys**

Remove the `ankerl/unordered_dense.h` include and these fields:

```text
head_
bucket_next_
bucket_key_blocks_
bucket_key_blocks_per_row_
scratch_sum_key_sizes_
```

Add:

```cpp
std::vector<std::uint32_t> lookup_slots_;
std::vector<std::uint32_t> representative_pos_;
std::vector<std::size_t>   pair_row_off_;
std::uint32_t              slot_id_mask_{};
unsigned                   slot_id_bits_{};
```

Probe linearly from `HashKeyHash::mix(hash) & (capacity - 1)`. Compare the
packed fingerprint before reconstructing the representative. An empty slot
creates the next insertion-ordered bucket. A matching representative returns
the existing ID.

- [ ] **Step 6: Build buckets with the flat table**

Pass a monotonically increasing source ordinal to the insertion routine:

```cpp
std::uint32_t source_pos = 0;
for (index_t i = 0; i < m_; ++i, ++source_pos) {
    const auto id = find_or_insert_bucket_(hP_[i], P_[i], source_pos);
    ++bucket_len_[id];
    single_id_[i] = id;
}
for (index_t i = 0; i < m_; ++i) {
    for (index_t j = i + 1; j < m_; ++j, ++source_pos) {
        // Build the current exact XOR row and use hP_[i] ^ hP_[j].
    }
}
```

Keep the current `bucket_off_` and `sum_entries_` second pass temporarily so
the stable-pointer API continues to function during this task. Release
`hP_` after construction.

- [ ] **Step 7: Migrate owning `key_of` call sites**

Use an owning row before taking views:

```cpp
Row key = index.key_of(bucket_id);
auto ns = local_gen.make(key.cview(), ptr, len);
```

In TOHPE details, move the returned `Row` directly into `TohpeZInfo::z`.

- [ ] **Step 8: Run exact regression tests**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
cmake --build build --config RelWithDebInfo --target vartodd_core_regression bench -j2
build/bin/RelWithDebInfo/vartodd_core_regression
```

Expected: both configurations print `vartodd_core_regression ok`.

- [ ] **Step 9: Capture the partial benchmark**

Run the exact benchmark three times. Record the median, but do not apply the
24 MiB gate yet because stable range storage is intentionally retained in
this intermediate commit.

- [ ] **Step 10: Commit the flat exact index**

```bash
git add include/todd_index.hpp src/core/todd_index.cpp src/core/nullspace.cpp \
        src/core/todd_generator.cpp tests/core_regression.cpp
git commit -m "Build ToddIndex with a flat exact table"
```

---

### Task 4: Store Only Sparse Duplicate Members and Migrate Consumers

**Files:**
- Modify: `include/todd_index.hpp:201-276`
- Modify: `include/nullspace.hpp:405-520`
- Modify: `src/core/todd_index.cpp:56-267`
- Modify: `src/core/nullspace.cpp:299-365,450-650`
- Modify: `src/core/todd_generator.cpp:660-710`
- Modify: `tests/core_regression.cpp:85-135,218-260`

**Interfaces:**
- Retains: the two `materialize_bucket` overloads from Task 1.
- Changes: `BucketIdSize` to `std::pair<std::uint32_t, std::uint32_t>`.
- Removes: both raw-pointer `sum_bucket` overloads.
- Removes: `bucket_off_`, `std::vector<index_t> bucket_len_`, and `sum_entries_`.
- Uses: `detail::BucketLengths` from Task 2.

- [ ] **Step 1: Add a failing sparse-storage and width assertion**

Extend the duplicate-heavy test:

```cpp
require(index.bucket_length_bytes() == 1,
        "small duplicate-heavy index should use byte lengths");
```

Add a policy smoke comparison using the same seed/config twice and require
identical chosen candidate exports and output matrices. This locks down member
ordering after materialization.

- [ ] **Step 2: Build to verify the width API is missing**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
```

Expected: compilation fails because `bucket_length_bytes()` is undefined.

- [ ] **Step 3: Collect sparse duplicates during the primary pass**

Make insertion return `{id, inserted}`. On an existing bucket, append:

```cpp
duplicate_build.emplace_back(id, source_pos);
```

The first occurrence remains in `representative_pos_[id]`. After the pass,
sort `duplicate_build` by `(bucket_id, source_pos)` and produce:

```cpp
struct DuplicateBucket {
    std::uint32_t id;
    std::uint32_t begin;
    std::uint32_t end;
};

std::vector<DuplicateBucket> duplicate_buckets_;
std::vector<std::uint32_t>   duplicate_positions_;
```

Only buckets with at least one extra member receive metadata.

- [ ] **Step 4: Compact bucket lengths**

Use a temporary `std::vector<std::uint32_t> build_lengths`. Compute
`max_bucket_`, then call:

```cpp
bucket_len_.assign(build_lengths, static_cast<std::uint32_t>(max_bucket_));
```

Implement `bucket_size`, both bucket-selection methods, and
`bucket_length_bytes` using `BucketLengths`. Dispatch once around full scans.

- [ ] **Step 5: Materialize representative plus sparse duplicates**

Implement:

```cpp
bool ToddIndex::materialize_bucket(std::uint32_t id,
                                   std::vector<SumEntry>& out) const {
    out.clear();
    if (id >= buckets_num())
        return false;
    out.reserve(static_cast<std::size_t>(bucket_size(id)));
    out.push_back(decode_source_position_(representative_pos_[id]));
    if (bucket_size(id) > 1) {
        auto it = std::ranges::lower_bound(
            duplicate_buckets_, id, {}, &DuplicateBucket::id);
        assert(it != duplicate_buckets_.end() && it->id == id);
        for (std::uint32_t p = it->begin; p < it->end; ++p)
            out.push_back(decode_source_position_(duplicate_positions_[p]));
    }
    return true;
}
```

The key overload first performs exact flat lookup, then calls the ID overload.
Delete the old range and full-entry fields and the second full construction
pass.

- [ ] **Step 6: Migrate witness and generator consumers**

Replace every raw pointer lookup with caller-owned storage:

```cpp
std::vector<SumEntry> entries;
if (M_->index().materialize_bucket(bucket_id, entries)) {
    const SumEntry* ptr = entries.data();
    const index_t len = static_cast<index_t>(entries.size());
    // Existing immediate consumer.
}
```

Add reusable scratch vectors to `TohpeGenerator` and `FullToddGenerator` where
calls repeat. In `generate_todd_candidates`, allocate one scratch vector
outside the bucket loop and reuse it.

Change row and pair overloads to use `single_id()[row]` and
`pair_bucket_id(row1,row2)` directly. Arbitrary-vector overloads continue to
use exact flat lookup.

- [ ] **Step 7: Remove obsolete APIs and narrow bucket-size scratch**

Delete the two raw `sum_bucket` overloads, `sum_key_sizes`,
`sum_key_sizes_scratch`, and `row_hashes`. Change:

```cpp
using BucketIdSize = std::pair<std::uint32_t, std::uint32_t>;
```

Use checked conversion from adaptive lengths. Structured-binding callers
continue to promote the size as needed.

- [ ] **Step 8: Run all exact regression tests**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
cmake --build build --config RelWithDebInfo --target vartodd_core_regression bench -j2
build/bin/RelWithDebInfo/vartodd_core_regression
```

Expected: both binaries print `vartodd_core_regression ok`.

- [ ] **Step 9: Commit sparse bucket storage**

```bash
git add include/todd_index.hpp include/nullspace.hpp src/core/todd_index.cpp \
        src/core/nullspace.cpp src/core/todd_generator.cpp tests/core_regression.cpp
git commit -m "Store sparse ToddIndex bucket members"
```

---

### Task 5: Add Direct Size and Construction Benchmarks

**Files:**
- Create: `demos/bench_todd_index.cpp`
- Modify: `demos/CMakeLists.txt`
- Modify: `include/todd_index.hpp:201-276`
- Modify: `src/core/todd_index.cpp`
- Modify: `tests/core_regression.cpp`

**Interfaces:**
- Produces: `std::size_t ToddIndex::storage_bytes() const noexcept`
- Produces: executable `build/bin/<config>/bench_todd_index`

- [ ] **Step 1: Add a failing storage-accounting test**

For a small index, sum the public component expectations indirectly:

```cpp
require(index.storage_bytes() > 0, "ToddIndex storage accounting is empty");
require(index.storage_bytes() < 4096,
        "small ToddIndex retained unexpectedly large storage");
```

Use a dedicated small matrix so the bound is stable.

- [ ] **Step 2: Build to verify `storage_bytes` is missing**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
```

Expected: compilation fails because `ToddIndex::storage_bytes` is undefined.

- [ ] **Step 3: Implement exact retained-capacity accounting**

Sum `capacity() * sizeof(value_type)` for:

```text
masks_
single_id_
pair_id_
lookup_slots_
representative_pos_
pair_row_off_
adaptive bucket lengths
duplicate_buckets_
duplicate_positions_
scratch_bucket_id_sizes_
```

Use checked/saturating addition only if required by the existing size types;
the method is diagnostic and must not allocate.

- [ ] **Step 4: Create `bench_todd_index`**

Implement a CLI:

```text
bench_todd_index <npy_path> <repetitions>
```

Load `P` once. For each repetition, time only `ToddIndex index(P)`. Report:

```text
rows=<...> cols=<...> buckets=<...> max_bucket=<...>
storage_bytes=<...> storage_mib=<...>
total_ms=<...> ms_per_build=<...>
```

Prevent dead-code elimination by accumulating bucket counts and return zero
after printing the checksum.

- [ ] **Step 5: Add and build the benchmark target**

Add `bench_todd_index` beside `bench`, with the same RelWithDebInfo frame
pointer and native-architecture options and a link to `vartodd`.

Run:

```bash
cmake --build build --config RelWithDebInfo --target bench_todd_index -j2
build/bin/RelWithDebInfo/bench_todd_index \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy 5
```

Expected: `storage_mib` is no greater than `24.0`.

- [ ] **Step 6: Run regression tests**

Run:

```bash
build/bin/Debug/vartodd_core_regression
build/bin/RelWithDebInfo/vartodd_core_regression
```

Expected: both print `vartodd_core_regression ok`.

- [ ] **Step 7: Commit measurement support**

```bash
git add demos/bench_todd_index.cpp demos/CMakeLists.txt include/todd_index.hpp \
        src/core/todd_index.cpp tests/core_regression.cpp
git commit -m "Measure ToddIndex construction and storage"
```

---

### Task 6: Validate Single-Process and Concurrent Performance

**Files:**
- Modify only if profiling reveals a localized regression in files already changed by Tasks 1--5.
- Record results in the implementation handoff; do not commit generated `perf` data or logs.

**Interfaces:**
- Consumes: `bench`, `bench_todd_index`, `vartodd_core_regression`.
- Produces: verified acceptance or targeted revert of a regressing stage.

- [ ] **Step 1: Rebuild every relevant RelWithDebInfo binary**

Run:

```bash
cmake --build build --config RelWithDebInfo \
  --target bench bench_todd_index vartodd_core_regression -j2
```

Expected: successful build from the feature worktree's current sources.

- [ ] **Step 2: Run final exactness verification**

Run:

```bash
build/bin/Debug/vartodd_core_regression
build/bin/RelWithDebInfo/vartodd_core_regression
```

Expected: both print `vartodd_core_regression ok`.

- [ ] **Step 3: Measure construction and size**

Run at least three `bench_todd_index` samples with five internal repetitions.
Record the median construction time and retained bytes. Required:

```text
storage_bytes <= 25165824
```

- [ ] **Step 4: Measure the exact end-to-end benchmark**

Run at least three samples:

```bash
build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  25 8 1 0
```

Compare medians against the freshly rebuilt baseline. The new median must be
lower.

- [ ] **Step 5: Capture an after-profile**

Run:

```bash
perf record -q -g --call-graph dwarf -o /tmp/vartodd_compact_index.data \
  build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  25 8 1 0
perf report --stdio --no-children \
  -i /tmp/vartodd_compact_index.data --percent-limit 0.4
```

Confirm the old `ToddIndex::get_bucket_id_`/general hash-map hotspot is gone
and `pair_bucket_id`/`CountWS` has no offsetting regression.

- [ ] **Step 6: Repeat pinned concurrent measurements**

Use physical cores `0`, `2`, `4`, and `6`, matching the baseline. Run two
processes, then four, each with the exact 25-iteration command. Record each
process's `total_ms` and compare the median per process with:

```text
two processes: 7.55--7.61 seconds baseline
four processes: 8.95--9.23 seconds baseline
```

This is diagnostic rather than a correctness gate.

- [ ] **Step 7: Apply the acceptance rule**

Keep the redesign only if:

- exactness tests pass;
- retained index bytes are at most 24 MiB;
- the end-to-end single-process median improves;
- no downstream hot-path regression offsets construction savings.

If one isolated component regresses, revert only that component and rerun
Steps 1--6. Do not retain a noise-level or negative end-to-end result merely
because construction or memory improved.

- [ ] **Step 8: Commit any profile-driven correction**

If no correction was needed, do not create an empty commit. Otherwise:

```bash
git add <only the corrected source and test files>
git commit -m "Refine compact ToddIndex hot paths"
```
