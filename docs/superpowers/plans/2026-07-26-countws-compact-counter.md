# Compact CountWS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ID-indexed 64-bit `CountWS` counters with compact touched-position counters and use `std::ranges::nth_element` to return an unsorted top-K set.

**Architecture:** A generation-stamped ID-to-position map resolves bucket IDs to compact positions. Narrow counts and position-to-ID values are stored contiguously by touched position; selection partitions reusable compact-position scratch space. Runtime storage-width dispatch happens once around the accumulation loops.

**Tech Stack:** C++20, CMake/Ninja multi-config build, the existing `vartodd_core_regression` executable, `std::variant`, `std::ranges::nth_element`, and Linux `perf`.

## Global Constraints

- The returned candidates must be members of the greatest-count top-K set, but their order is unspecified.
- Equal counts that cross the partition boundary may select any tied buckets.
- Reset must remain O(1) except when a packed epoch wraps.
- Work after reset must remain proportional to touched buckets, including almost-all-zero and almost-all-one inputs.
- Bucket IDs remain 32-bit.
- Counts use the narrowest unsigned type that can represent `2 * ToddIndex::max_bucket()`.
- The exact performance workload is `build/bin/RelWithDebInfo/bench ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy 12 40 8 0`.
- Preserve all unrelated uncommitted workspace changes.

---

### Task 1: Record a Repeatable Pre-Change Baseline

**Files:**
- Read: `demos/bench.cpp`
- Read: `src/core/nullspace.cpp:278-372`
- Create outside repository: `/tmp/vartodd-countws-baseline.txt`

**Interfaces:**
- Consumes: existing `build/bin/RelWithDebInfo/bench`
- Produces: five wall-clock samples and their median for final comparison

- [ ] **Step 1: Verify the current benchmark target is fresh**

Run:

```bash
cmake --build build --config RelWithDebInfo --target bench -j2
```

Expected: successful build or `ninja: no work to do`.

- [ ] **Step 2: Run five baseline samples**

Run the following command five times and retain each `total_ms`:

```bash
build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  12 40 8 0
```

Expected: every run reports `iterations_run=12`.

- [ ] **Step 3: Record the median**

Write the five `total_ms` values and their numeric median to
`/tmp/vartodd-countws-baseline.txt`. Do not add benchmark output to the
repository.

### Task 2: Build and Test Compact Touched-Position Storage

**Files:**
- Modify: `include/nullspace.hpp:7-35`
- Modify: `tests/core_regression.cpp:5-12,104-150`

**Interfaces:**
- Produces: `todd::CountWSScore { std::uint32_t bucket_id; index_t count; }`
- Produces: `todd::detail::CompactCountStorage<Count, MapWord>`
- `CompactCountStorage::reset(std::size_t buckets, index_t max_count, bool parity) -> void`
- `CompactCountStorage::add(index_t id, index_t delta) -> void`
- `CompactCountStorage::argmax_n(std::size_t n) -> std::vector<CountWSScore>`
- `CompactCountStorage::argmax_n_into(std::size_t n, std::vector<CountWSScore>& out) -> void`
- `CompactCountStorage::argmax() const -> CountWSScore`

- [ ] **Step 1: Add a reference-based failing regression test**

Include `nullspace.hpp`, `<algorithm>`, and `<map>` in
`tests/core_regression.cpp`. Add `check_compact_count_storage()` and call it
from `main()`. The test constructs
`detail::CompactCountStorage<std::uint8_t, std::uint8_t>`, resets eight
buckets with maximum count 20 and both parity values, applies repeated updates,
and compares the unordered top-K `(bucket_id, count)` pairs against a
`std::map<std::uint32_t, index_t>` reference.

Use this top-set assertion:

```cpp
auto by_score = [](const CountWSScore& a, const CountWSScore& b) {
    return a.count > b.count;
};
std::ranges::sort(actual, by_score);
std::ranges::sort(expected, by_score);
require(actual.size() == expected.size(), "compact CountWS result size mismatch");
for (std::size_t i = 0; i < actual.size(); ++i)
    require(actual[i].count == expected[i].count,
            "compact CountWS must return the greatest counts");
```

Also verify:

```cpp
require(ws.argmax_n(0).empty(), "zero CountWS sample count should be empty");
require(ws.argmax_n(100).size() == touched,
        "oversized CountWS sample count should return every touched bucket");
```

Run enough resets on the 8-bit mapping word to force epoch wrap, then confirm
that old buckets are absent and new updates retain correct counts.

Exercise both density extremes with fixed loops: touch every even ID in a
64-bucket workspace for the balanced case, then reset and touch only IDs 0 and
63 for the almost-uniform case. In both cases, compare all returned scores with
the wide reference map.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
cmake --build build --config RelWithDebInfo --target vartodd_core_regression -j2
```

Expected: compilation fails because `CountWSScore` and
`detail::CompactCountStorage` do not exist.

- [ ] **Step 3: Implement compact storage**

In `include/nullspace.hpp`, add the necessary C++20 headers:

```cpp
#include <algorithm>
#include <bit>
#include <cassert>
#include <numeric>
#include <ranges>
#include <type_traits>
```

Define:

```cpp
struct CountWSScore {
    std::uint32_t bucket_id = 0;
    index_t       count     = 0;
};
```

Implement `detail::CompactCountStorage<Count, MapWord>` with:

```cpp
std::unique_ptr<MapWord[]>       id_to_pos_;
std::unique_ptr<Count[]>         counts_;
std::unique_ptr<std::uint32_t[]> pos_to_id_;
std::unique_ptr<std::uint32_t[]> selection_positions_;
std::size_t                      capacity_       = 0;
std::size_t                      buckets_        = 0;
std::size_t                      touched_size_   = 0;
unsigned                         position_bits_  = 0;
MapWord                          position_mask_  = 0;
MapWord                          epoch_          = 0;
MapWord                          max_epoch_      = 0;
index_t                          max_count_      = 0;
bool                             parity_         = false;
```

`reset` must clear `id_to_pos_` when storage grows, the bucket count changes,
or the packed epoch wraps. Allocate `counts_`, `pos_to_id_`, and
`selection_positions_` without value-initializing their primitive elements.

`add` must decode a current-epoch position or append a new touched position:

```cpp
const MapWord state = id_to_pos_[id];
if ((state >> position_bits_) == epoch_) {
    const auto pos = static_cast<std::size_t>(state & position_mask_);
    counts_[pos] = static_cast<Count>(counts_[pos] + delta);
} else {
    const auto pos = touched_size_++;
    counts_[pos] = static_cast<Count>(delta - parity_);
    pos_to_id_[pos] = static_cast<std::uint32_t>(id);
    id_to_pos_[id] = static_cast<MapWord>((epoch_ << position_bits_) | pos);
}
```

Before narrowing, debug-assert that the new count is at most `max_count_` and
fits `Count`.

`argmax_n_into` must fill positions with `std::iota`, partition only when
`0 < n < touched_size_`, and return the unsorted selected prefix:

```cpp
std::iota(first, last, std::uint32_t{0});
std::ranges::nth_element(first, first + n, last, [this](auto a, auto b) {
    return counts_[a] > counts_[b];
});
```

Do not add a small-N path or sort the selected prefix.

- [ ] **Step 4: Run the regression test to verify GREEN**

Run:

```bash
cmake --build build --config RelWithDebInfo --target vartodd_core_regression -j2
ctest --test-dir build -C RelWithDebInfo --output-on-failure -R vartodd_core_regression
```

Expected: build succeeds and `1/1` matching test passes.

- [ ] **Step 5: Commit the compact storage and tests**

```bash
git add include/nullspace.hpp tests/core_regression.cpp
git commit -m "Add compact touched-position counters"
```

### Task 3: Select Storage Width Once and Integrate Tohpe Generation

**Files:**
- Modify: `include/nullspace.hpp:23-35,153-160`
- Modify: `src/core/nullspace.cpp:278-372,604-657`
- Modify: `tests/core_regression.cpp`

**Interfaces:**
- Consumes: `detail::CompactCountStorage<Count, MapWord>` from Task 2
- Produces: `CountWS::reset(std::size_t buckets, index_t max_bucket, bool parity)`
- Produces: `CountWS::with_storage(F&& function)`
- Produces: `CountWS::argmax_n(std::size_t n) -> std::vector<CountWSScore>`
- Produces: `CountWS::count_bytes() const -> std::size_t`
- Produces: `CountWS::mapping_bytes() const -> std::size_t`
- Changes: `TohpeGenerator::scratch_candidates_` to `std::vector<CountWSScore>`

- [ ] **Step 1: Add failing width-dispatch and public-workspace tests**

Extend `check_compact_count_storage()` with:

```cpp
CountWS narrow;
narrow.reset(32, 9, false);
require(narrow.count_bytes() == 1, "max bucket 9 should select byte counters");
require(narrow.mapping_bytes() == 4, "32 buckets should select 32-bit mappings");

CountWS medium;
medium.reset(32, 128, false);
require(medium.count_bytes() == 2, "max bucket 128 should select 16-bit counters");
```

Use `CountWS::add` and `CountWS::argmax_n` to repeat the reference top-K test.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
cmake --build build --config RelWithDebInfo --target vartodd_core_regression -j2
```

Expected: compilation fails because the legacy `CountWS::reset` signature and
result types do not match.

- [ ] **Step 3: Implement one-time storage dispatch**

Replace the legacy public vectors with a `std::variant` over
`CompactCountStorage` specializations for:

```text
Count = uint8_t, uint16_t, uint32_t, index_t
MapWord = uint32_t, uint64_t
```

Use a 32-bit mapping when `std::bit_width(buckets - 1) <= 24`, leaving at least
eight epoch bits; otherwise use a 64-bit mapping. Choose the count type using
the checked bound `2 * max_bucket`.

`CountWS::with_storage` performs `std::visit` once. Convenience `add`,
`argmax_n`, and `argmax_n_into` wrappers may visit per call for tests, but the
production accumulation loop must not use those wrappers.

- [ ] **Step 4: Integrate the scored result and outer dispatch**

In `TohpeGenerator::best_z_n_details_into`, replace:

```cpp
ws_.reset(idx.buckets_num());
ws_.parity = y_count % 2;
```

with:

```cpp
ws_.reset(idx.buckets_num(), idx.max_bucket(), (y_count & 1U) != 0);
```

Wrap all single and pair additions plus `argmax_n_into` in one visit:

```cpp
ws_.with_storage([&](auto& counts) {
    for (std::size_t oi = 0; oi < ones_size; ++oi) {
        const index_t i = scratch_ones_[oi];
        counts.add(idx.single_id()[static_cast<std::size_t>(i)], 1);

        for (std::size_t zi = 0; zi < zeros_size; ++zi) {
            const index_t j = scratch_zeros_[zi];
            counts.add(idx.pair_bucket_id(i, j), 2);
        }
    }
    if ((y_count & 1U) != 0) {
        for (std::size_t zi = 0; zi < zeros_size; ++zi) {
            const index_t i = scratch_zeros_[zi];
            counts.add(idx.single_id()[static_cast<std::size_t>(i)], 2);
        }
    }
    counts.argmax_n_into(num_samples, scratch_candidates_);
});
```

Build each `TohpeZInfo` from `CountWSScore::bucket_id` and
`CountWSScore::count`; do not perform a post-selection lookup through the
workspace.

Delete the old `CountWS` implementations from `src/core/nullspace.cpp` and
change `scratch_candidates_` to `std::vector<CountWSScore>`.

- [ ] **Step 5: Run focused and full regression tests**

Run:

```bash
cmake --build build --config RelWithDebInfo --target vartodd_core_regression bench -j2
ctest --test-dir build -C RelWithDebInfo --output-on-failure
```

Expected: build succeeds and all tests pass.

- [ ] **Step 6: Commit integration**

```bash
git add include/nullspace.hpp src/core/nullspace.cpp tests/core_regression.cpp
git commit -m "Use compact counters for Tohpe selection"
```

### Task 4: Verify Correctness and Performance

**Files:**
- Inspect: `include/nullspace.hpp`
- Inspect: `src/core/nullspace.cpp`
- Inspect: `tests/core_regression.cpp`
- Create outside repository: `/tmp/vartodd-countws-after.txt`
- Create outside repository: `/tmp/vartodd-countws-after.perf`

**Interfaces:**
- Consumes: optimized `bench` and `vartodd_core_regression`
- Produces: fresh test, build, median benchmark, and profile evidence

- [ ] **Step 1: Run fresh correctness verification**

```bash
cmake --build build --config RelWithDebInfo --target vartodd_core_regression bench -j2
ctest --test-dir build -C RelWithDebInfo --output-on-failure
```

Expected: build succeeds and all tests pass.

- [ ] **Step 2: Run five post-change benchmark samples**

Run the exact benchmark five times:

```bash
build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  12 40 8 0
```

Expected: every run reports `iterations_run=12`. Record all `total_ms` values
and their median in `/tmp/vartodd-countws-after.txt`.

- [ ] **Step 3: Re-profile the exact workload**

```bash
perf record -g --call-graph dwarf \
  -o /tmp/vartodd-countws-after.perf -- \
  build/bin/RelWithDebInfo/bench \
  ./data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy \
  12 40 8 0

perf report --stdio --no-children --percent-limit 0.25 \
  -i /tmp/vartodd-countws-after.perf
```

Expected: no lost samples; compare compact `add` and `nth_element` selection
costs with the 23.26% and 37.26% pre-change hotspots.

- [ ] **Step 4: Audit the final diff**

```bash
git diff HEAD~2 --check
git diff HEAD~2 -- include/nullspace.hpp src/core/nullspace.cpp tests/core_regression.cpp
git status --short
```

Confirm that only the planned counter files were changed by the implementation
commits and that unrelated pre-existing modifications remain unstaged.
