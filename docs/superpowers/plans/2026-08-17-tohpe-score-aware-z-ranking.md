# TOHPE Score-Aware Z Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make standalone TOHPE rank every positive-reduction `z` bucket with the normalized global exploration score, while preserving the existing reduction-only API and default fast path.

**Architecture:** Put the shared five-feature formula and fixed-`y` fast-path predicate on `ExplorationScore`. Add an exact callback-ranked top-N operation to both `CountWS` storage layouts, use it from a score-aware `TohpeGenerator::best_z_n_details_into` overload, and select that overload only when the configured score can change fixed-`y` reduction order. The existing mathematical `best_z*` methods remain reduction-greedy.

**Tech Stack:** C++20, `Matrix`/`RowCView`, `ToddIndex`, packed and compact `CountWS` storage, CMake/Ninja multi-config builds, `vartodd_core_regression`, Python `unittest` regression suite.

## Global Constraints

- Follow the approved design in `docs/superpowers/specs/2026-08-17-tohpe-score-aware-z-ranking-design.md`.
- Change only standalone `generate_tohpe_candidates`; do not alter `TohpePrefixSearch` or `ToddSearch` ranking.
- Preserve `best_z`, `best_z_n`, `best_z_n_into`, and the existing three-argument `best_z_n_details_into` as reduction-greedy public behavior.
- Filter `reduction <= 0` before invoking the score callback.
- Score every touched positive bucket before top-N truncation; never preselect by reduction for a non-default score.
- Break inner ties by score descending, reduction descending, then bucket ID ascending.
- Do not add policy configuration fields, Python bindings, or pickle fields.
- Preserve unrelated dirty-worktree changes. Stage only files named in each task.
- Use `apply_patch` for edits and run the stated failing test before implementation in each task.

---

## Task 0: Record the default standalone-TOHPE baseline

**Files:**

- Verify only; write timing output under `/tmp`

- [ ] **Step 1: Build the current Release extension**

```bash
cmake --build build --config Release --target pyvartodd -j2
```

- [ ] **Step 2: Record repeated default-policy output and timing**

Run this from the repository root and retain `/tmp/vartodd_tohpe_z_baseline.txt` through Task 5:

```bash
.venv/bin/python - <<'PY' | tee /tmp/vartodd_tohpe_z_baseline.txt
import statistics
import time

import numpy as np
from pyvartodd.Release.pyvartodd import (
    ActionPool,
    ActionSelection,
    Matrix,
    MatrixWithData,
    PolicyConfig,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpePrefixSearch,
    TohpeSearch,
    ZBucketSearch,
    policy_iteration,
)

raw = np.ascontiguousarray(
    np.load("data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^8_84320.npy") != 0,
    dtype=np.bool_,
)
data = MatrixWithData(Matrix.from_numpy(raw), False)
off_prefix = TohpePrefixSearch(pool=SourcePool(keep=0, reserve=0))
off_todd = ToddSearch(
    pool=SourcePool(keep=0, reserve=0),
    buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
)
cfg = PolicyConfig(
    selection=ActionSelection(count=1, mode="best", temperature=0.0),
    pool=ActionPool(final_size=8),
    tohpe=TohpeSearch(
        sampling=SamplingBudget(one_hot="all", sparse=32, dense=32, sparse_max_weight=2),
        pool=SourcePool(keep=8, reserve=0),
        z_choices=8,
    ),
    tohpeprefix=off_prefix,
    todd=off_todd,
)
policy_iteration(data, cfg, 123, 0)
elapsed = []
signature = None
for add_seed in range(7):
    start = time.perf_counter()
    result = policy_iteration(data, cfg, 123, add_seed)
    elapsed.append(time.perf_counter() - start)
    current = [(c.reduction, c.bucket_size, c.pool_score) for c in result.chosen]
    if add_seed == 0:
        signature = current
print(f"median_seconds={statistics.median(elapsed):.9f}")
print(f"seed0_signature={signature!r}")
PY
```

Expected: seven iterations complete and print a nonempty seed-0 signature. This is a comparison guard, not a timing unit test.

---

## Task 1: Share exploration feature evaluation and identify the fast path

**Files:**

- Modify: `include/todd_generator.hpp:269-290`
- Modify: `tests/core_regression.cpp` near `check_candidate_tie_order`

- [ ] **Step 1: Add failing scoring-contract tests**

Add this regression function and call it from `main()` immediately after `check_candidate_tie_order()`:

```cpp
void check_exploration_score_feature_contract() {
    Candidate cand;
    cand.reduction  = 6;
    cand.basis_dim  = 3;
    cand.bucket_size = 4;
    cand.vec_weight = 5;
    cand.z_weight   = 2;
    cand.z_size     = 8;

    ExplorationScore score({2.0f, -1.0f, 0.5f, 3.0f, -4.0f},
                           {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, ScoringFunction{});
    score.bn = 10.0f;
    score.dn = 12.0f;
    score.wvwn = 20.0f;

    const float direct = score.evaluate_features(6, 3, 4, 5, 2, 8);
    const auto [pooled, unused] = score(cand);
    (void)unused;
    require(std::abs(direct - pooled) < 1e-7f && std::abs(direct - cand.pool_score) < 1e-7f,
            "exploration feature scoring diverged from candidate scoring");

    ExplorationScore default_score(1.0f, 7.0f, 0.0f, -3.0f, 0.0f);
    require(default_score.ranks_fixed_y_by_reduction(),
            "fixed-y constants should not disable the reduction fast path");
    default_score.weights[2] = 0.25f;
    require(!default_score.ranks_fixed_y_by_reduction(),
            "bucket scoring must disable the reduction fast path");
    default_score.weights[2] = 0.0f;
    default_score.weights[4] = -1.0f;
    require(!default_score.ranks_fixed_y_by_reduction(),
            "z-density scoring must disable the reduction fast path");
    default_score.weights[4] = 0.0f;
    default_score.sc.type = ScoringFunction::POLYNOM;
    require(!default_score.ranks_fixed_y_by_reduction(),
            "nonlinear scoring must use exact fixed-y ranking");
}
```

- [ ] **Step 2: Build and confirm the test fails to compile**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
```

Expected: compilation fails because `evaluate_features` and `ranks_fixed_y_by_reduction` do not exist.

- [ ] **Step 3: Implement the shared formula and predicate**

Replace the current `ExplorationScore::operator()` body with this API and delegation:

```cpp
float evaluate_features(Int reduction, Int basis_dim, Int bucket_size, Int vec_weight,
                        Int z_weight, Int z_size) const {
    const std::array<float, 5> x = {
        reduction / bn / 2.0f,
        basis_dim / dn,
        bucket_size / bn,
        vec_weight / wvwn,
        z_weight / static_cast<float>(std::max<Int>(1, z_size)),
    };
    return sc.evaluate(weights, centers, x);
}

bool ranks_fixed_y_by_reduction() const noexcept {
    const auto weight = [this](std::size_t i) {
        return i < weights.size() ? weights[i] : 0.0f;
    };
    return sc.type == ScoringFunction::LINEAR && bn > 0.0f &&
           weight(0) >= 0.0f && weight(2) == 0.0f && weight(4) == 0.0f;
}

auto operator()(Candidate& cand) const {
    cand.pool_score = evaluate_features(cand.reduction, cand.basis_dim, cand.bucket_size,
                                        cand.vec_weight, cand.z_weight, cand.z_size);
    return std::make_pair(cand.pool_score, 0);
}
```

The predicate deliberately ignores basis-dimension and `y`-weight coefficients: those features are constant while ranking `z` values for one fixed `y`. A zero reduction coefficient is eligible because the specified tie fallback still orders by reduction.

- [ ] **Step 4: Run the focused regression**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
```

Expected: `vartodd_core_regression ok`.

- [ ] **Step 5: Commit Task 1**

```bash
git add include/todd_generator.hpp tests/core_regression.cpp
git commit -m "refactor: share TOHPE exploration feature scoring"
```

---

## Task 2: Add exact score-ranked selection to every CountWS layout

**Files:**

- Modify: `include/nullspace.hpp:20-382`
- Modify: `tests/core_regression.cpp` near the existing CountWS tests

- [ ] **Step 1: Add failing exact-ranking tests**

Add a result type expectation and a reusable test helper. Run it once through the narrow packed layout and once through the wide compact layout:

```cpp
void require_scored_count_selection(index_t max_bucket) {
    CountWS ws;
    ws.reset(16, max_bucket, false);
    ws.add(5, 4);
    ws.add(2, 4);
    ws.add(7, 2);
    ws.add(1, 0); // touched but nonpositive: the callback must never see it

    std::array<int, 16> calls{};
    std::vector<RankedCountWSScore> selected;
    ws.argmax_n_by_into(3, selected, [&](std::uint32_t id, index_t reduction) {
        ++calls[id];
        return id == 7 ? 10.0f : 0.0f;
    });

    require(selected.size() == 3, "score-ranked CountWS lost a positive bucket");
    require(selected[0].bucket_id == 7 && selected[0].count == 2,
            "score-ranked CountWS must allow a lower reduction to win");
    require(selected[1].bucket_id == 2 && selected[2].bucket_id == 5,
            "score-ranked CountWS tie order must be reduction then bucket ID");
    require(calls[1] == 0, "score-ranked CountWS evaluated a zero reduction");
    require(calls[2] == 1 && calls[5] == 1 && calls[7] == 1,
            "score-ranked CountWS callback must run once per positive bucket");

    ws.argmax_n_by_into(1, selected,
                        [](std::uint32_t, index_t reduction) { return -static_cast<float>(reduction); });
    require(selected.size() == 1 && selected[0].bucket_id == 7,
            "negative reduction weight must not make a zero reduction eligible");
}

void check_scored_count_selection() {
    require_scored_count_selection(8);
    require_scored_count_selection(std::numeric_limits<std::uint32_t>::max());
}
```

Call `check_scored_count_selection()` after `check_count_selection_tie_order()` in `main()`.

- [ ] **Step 2: Build and confirm the test fails to compile**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
```

Expected: compilation fails because `RankedCountWSScore` and `argmax_n_by_into` do not exist.

- [ ] **Step 3: Add the ranked result and storage-local algorithms**

Next to `CountWSScore`, add:

```cpp
struct RankedCountWSScore {
    std::uint32_t bucket_id = 0;
    index_t       count     = 0;
    float         score     = 0.0f;
};
```

Add this public template to `PackedCountStorage`:

```cpp
template <class ScoreFn>
void argmax_n_by_into(std::size_t n, std::vector<RankedCountWSScore>& out,
                      ScoreFn&& score_fn) const {
    out.clear();
    if (n == 0 || touched_size_ == 0)
        return;

    out.reserve(touched_size_);
    for (std::size_t i = 0; i < touched_size_; ++i) {
        const std::uint32_t id = touched_ids_[i];
        const index_t count = count_of_(id);
        if (count == 0)
            continue;
        out.push_back(RankedCountWSScore{
            .bucket_id = id,
            .count = count,
            .score = score_fn(id, count),
        });
    }

    const auto better = [](const RankedCountWSScore& a, const RankedCountWSScore& b) {
        if (a.score != b.score)
            return a.score > b.score;
        if (a.count != b.count)
            return a.count > b.count;
        return a.bucket_id < b.bucket_id;
    };
    n = std::min(n, out.size());
    if (n < out.size()) {
        std::ranges::nth_element(out, out.begin() + static_cast<std::ptrdiff_t>(n), better);
        out.resize(n);
    }
    std::ranges::sort(out, better);
}
```

Add the corresponding concrete implementation to `CompactCountStorage`:

```cpp
template <class ScoreFn>
void argmax_n_by_into(std::size_t n, std::vector<RankedCountWSScore>& out,
                      ScoreFn&& score_fn) const {
    out.clear();
    if (n == 0 || touched_size_ == 0)
        return;

    out.reserve(touched_size_);
    for (std::size_t pos = 0; pos < touched_size_; ++pos) {
        const std::uint32_t id = pos_to_id_[pos];
        const index_t count = static_cast<index_t>(counts_[pos]);
        if (count == 0)
            continue;
        out.push_back(RankedCountWSScore{
            .bucket_id = id,
            .count = count,
            .score = score_fn(id, count),
        });
    }

    const auto better = [](const RankedCountWSScore& a, const RankedCountWSScore& b) {
        if (a.score != b.score)
            return a.score > b.score;
        if (a.count != b.count)
            return a.count > b.count;
        return a.bucket_id < b.bucket_id;
    };
    n = std::min(n, out.size());
    if (n < out.size()) {
        std::ranges::nth_element(out, out.begin() + static_cast<std::ptrdiff_t>(n), better);
        out.resize(n);
    }
    std::ranges::sort(out, better);
}
```

Keep the two small layout-specific loops instead of adding a virtual/interface layer.

- [ ] **Step 4: Expose the operation through CountWS**

Add this wrapper beside `argmax_n_into`:

```cpp
template <class ScoreFn>
void argmax_n_by_into(std::size_t n, std::vector<RankedCountWSScore>& out,
                      ScoreFn&& score_fn) const {
    with_storage([&](const auto& storage) {
        storage.argmax_n_by_into(n, out, std::forward<ScoreFn>(score_fn));
    });
}
```

This operation must not mutate the packed touched-ID order, so later reduction-only selection remains deterministic.

- [ ] **Step 5: Run focused regressions**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
```

Expected: `vartodd_core_regression ok`; both packed and wide callback-ranking branches execute.

- [ ] **Step 6: Commit Task 2**

```bash
git add include/nullspace.hpp tests/core_regression.cpp
git commit -m "feat: rank CountWS buckets by callback score"
```

---

## Task 3: Add the score-aware TohpeGenerator overload

**Files:**

- Modify: `include/nullspace.hpp:498-522`
- Modify: `src/core/nullspace.cpp:508-561`
- Modify: `tests/core_regression.cpp` near `check_tohpe_policy`

- [ ] **Step 1: Add a deterministic TOHPE fixture helper**

Add this helper near the other test helpers:

```cpp
Matrix matrix_from_words(std::initializer_list<std::uint64_t> words, index_t cols) {
    Matrix out(static_cast<index_t>(words.size()), cols);
    index_t row = 0;
    for (const std::uint64_t word : words) {
        for (index_t bit = 0; bit < cols; ++bit) {
            if ((word >> bit) & 1ULL)
                out[row].set(bit);
        }
        ++row;
    }
    return out;
}
```

The fixture `matrix_from_words({1, 4, 11, 14, 16, 21, 26, 28, 31}, 5)` has a one-dimensional TOHPE kernel. Its reduction-greedy `z` has reduction 2 and weight 4, while the sparsest positive `z` has reduction 1 and weight 1.

- [ ] **Step 2: Add failing overload, equivalence, and exact-reduction tests**

Add this test and call it from `main()` before the policy tests:

```cpp
void check_score_aware_tohpe_z_ranking() {
    Matrix P = matrix_from_words({1, 4, 11, 14, 16, 21, 26, 28, 31}, 5);
    auto data = std::make_shared<MatrixWithData>(P, false);
    require(data->tohpe_basis().rows() == 1, "score-aware TOHPE fixture kernel changed");
    const Row y(data->tohpe_basis()[0]);
    TohpeGenerator generator(data);

    std::vector<TohpeZInfo> reduction_ranked;
    generator.best_z_n_details_into(y, 8, reduction_ranked);
    require(!reduction_ranked.empty() && reduction_ranked[0].reduction == 2 &&
                reduction_ranked[0].z.count() == 4,
            "score-aware TOHPE fixture greedy candidate changed");

    ExplorationScore default_score(1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    default_score.bn = static_cast<float>(data->index().max_bucket());
    default_score.dn = static_cast<float>(data->tohpe_basis().rows() + 5);
    default_score.wvwn = static_cast<float>(P.rows());

    std::vector<TohpeZInfo> scored_like_default;
    generator.best_z_n_details_into(
        y, 8,
        [&](index_t reduction, index_t bucket_size, RowCView z) {
            return default_score.evaluate_features(
                static_cast<Int>(reduction), 1, static_cast<Int>(bucket_size),
                static_cast<Int>(y.count()), static_cast<Int>(z.count()), static_cast<Int>(z.size()));
        },
        scored_like_default);
    require(scored_like_default.size() == reduction_ranked.size(),
            "default score-aware TOHPE result size changed");
    for (std::size_t i = 0; i < reduction_ranked.size(); ++i) {
        require(scored_like_default[i].bucket_id == reduction_ranked[i].bucket_id &&
                    scored_like_default[i].reduction == reduction_ranked[i].reduction &&
                    scored_like_default[i].bucket_size == reduction_ranked[i].bucket_size &&
                    scored_like_default[i].z == reduction_ranked[i].z,
                "default score-aware TOHPE order differs from reduction order");
    }

    ExplorationScore sparse_score(0.0f, 0.0f, 0.0f, 0.0f, -1.0f);
    sparse_score.bn = default_score.bn;
    sparse_score.dn = default_score.dn;
    sparse_score.wvwn = default_score.wvwn;
    std::vector<TohpeZInfo> sparse_ranked;
    generator.best_z_n_details_into(
        y, 1,
        [&](index_t reduction, index_t bucket_size, RowCView z) {
            return sparse_score.evaluate_features(
                static_cast<Int>(reduction), 1, static_cast<Int>(bucket_size),
                static_cast<Int>(y.count()), static_cast<Int>(z.count()), static_cast<Int>(z.size()));
        },
        sparse_ranked);
    require(sparse_ranked.size() == 1 && sparse_ranked[0].reduction == 1 &&
                sparse_ranked[0].z.count() == 1,
            "z-density score should select the lower-reduction sparse candidate");

    Candidate candidate(0, static_cast<Int>(sparse_ranked[0].reduction),
                        k_single_sentinel<Int>(), k_single_sentinel<Int>(), Row(y),
                        Row(sparse_ranked[0].z), 1, static_cast<Int>(sparse_ranked[0].bucket_size),
                        sparse_ranked[0].bucket_id, CandidateSourceTohpe);
    const float inner_score = sparse_score.evaluate_features(
        candidate.reduction, candidate.basis_dim, candidate.bucket_size,
        candidate.vec_weight, candidate.z_weight, candidate.z_size);
    const auto [outer_score, unused] = sparse_score(candidate);
    (void)unused;
    require(std::abs(inner_score - outer_score) < 1e-7f,
            "inner TOHPE z score differs from candidate pool score");

    const Matrix state = canonical_parity_matrix(
        generator.make(sparse_ranked[0].z, sparse_ranked[0].bucket_id).apply(y));
    require(P.rows() - state.rows() == sparse_ranked[0].reduction,
            "score-aware TOHPE reported reduction differs from applied reduction");
}
```

- [ ] **Step 3: Build and confirm the test fails to compile**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
```

Expected: overload resolution fails because no callback-taking `best_z_n_details_into` exists.

- [ ] **Step 4: Declare the callback overload and reusable scratch**

Add `<functional>` and declare:

```cpp
using TohpeZScoreFunction = std::function<float(index_t, index_t, RowCView)>;

void best_z_n_details_into(RowCView y, index_t num_samples,
                           const TohpeZScoreFunction& score_fn,
                           std::vector<TohpeZInfo>& scratch_out) const;
```

Add these private members:

```cpp
void count_z_reductions_(RowCView y) const;

mutable std::vector<RankedCountWSScore> scratch_ranked_candidates_;
```

- [ ] **Step 5: Refactor counting once and implement exact scored selection**

Move the common partition-and-count portion of the old overload into `count_z_reductions_`. It must stop after the `counts.add(...)` loops and must not call either top-N method.

The old overload becomes:

```cpp
void TohpeGenerator::best_z_n_details_into(RowCView y, index_t num_samples,
                                           std::vector<TohpeZInfo>& scratch_out) const {
    count_z_reductions_(y);
    const ToddIndex& idx = M_->index();
    ws_.argmax_n_into(num_samples, scratch_candidates_);

    scratch_out.clear();
    scratch_out.reserve(scratch_candidates_.size());
    for (const auto& candidate : scratch_candidates_) {
        scratch_out.push_back(TohpeZInfo{
            .z = idx.key_of(candidate.bucket_id),
            .reduction = candidate.count,
            .bucket_size = idx.bucket_size(candidate.bucket_id),
            .bucket_id = candidate.bucket_id,
        });
    }
}
```

Implement the new overload without a reduction shortlist:

```cpp
void TohpeGenerator::best_z_n_details_into(RowCView y, index_t num_samples,
                                           const TohpeZScoreFunction& score_fn,
                                           std::vector<TohpeZInfo>& scratch_out) const {
    count_z_reductions_(y);
    const ToddIndex& idx = M_->index();
    ws_.argmax_n_by_into(num_samples, scratch_ranked_candidates_,
                         [&](std::uint32_t bucket_id, index_t reduction) {
        const Row z = idx.key_of(bucket_id);
        return score_fn(reduction, idx.bucket_size(bucket_id), z.cview());
    });

    scratch_out.clear();
    scratch_out.reserve(scratch_ranked_candidates_.size());
    for (const auto& candidate : scratch_ranked_candidates_) {
        scratch_out.push_back(TohpeZInfo{
            .z = idx.key_of(candidate.bucket_id),
            .reduction = candidate.count,
            .bucket_size = idx.bucket_size(candidate.bucket_id),
            .bucket_id = candidate.bucket_id,
        });
    }
}
```

Do not route `best_z`, `best_z_n`, or `best_z_n_into` through the callback overload.

- [ ] **Step 6: Run focused regressions**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
```

Expected: `vartodd_core_regression ok`; the direct scored overload selects reduction 1 while the mathematical greedy API still selects reduction 2.

- [ ] **Step 7: Commit Task 3**

```bash
git add include/nullspace.hpp src/core/nullspace.cpp tests/core_regression.cpp
git commit -m "feat: add score-aware TOHPE z selection"
```

---

## Task 4: Wire standalone policy generation to exploration-ranked z choices

**Files:**

- Modify: `src/core/todd_generator.cpp:597-644`
- Modify: `tests/core_regression.cpp` near `check_tohpe_policy`

- [ ] **Step 1: Add a failing end-to-end policy test**

Add this test and call it after `check_tohpe_policy()`:

```cpp
void check_tohpe_policy_scores_inner_z_choices() {
    const Matrix P = matrix_from_words({1, 4, 11, 14, 16, 21, 26, 28, 31}, 5);
    auto data = std::make_shared<MatrixWithData>(P, false);

    const auto run = [&](ExplorationScore exploration) {
        PolicyConfig cfg;
        cfg.scores.exploration = std::move(exploration);
        cfg.selection = ActionSelection{1, "best", 0.0f};
        cfg.pool = ActionPool{4};
        cfg.tohpe = TohpeSearch{
            SamplingBudget{k_all_one_hot_samples, 0, 0, 2}, SourcePool{4, 0}, 1};
        cfg.tohpeprefix = TohpePrefixSearch{
            SamplingBudget{}, SourcePool{0, 0}, 1, ZBucketSearch{0, 0, 0.0f, 0.0f, 0}};
        cfg.todd = ToddSearch{
            SamplingBudget{}, SourcePool{0, 0}, 1, ZBucketSearch{0, 0, 0.0f, 0.0f, 0}};
        return policy_iteration_impl(data, cfg, 123, 0);
    };

    const auto greedy = run(ExplorationScore(1.0f, 0.0f, 0.0f, 0.0f, 0.0f));
    require(greedy.chosen.size() == 1 && greedy.chosen[0].reduction == 2,
            "default standalone TOHPE choice changed");

    const auto sparse = run(ExplorationScore(0.0f, 0.0f, 0.0f, 0.0f, -1.0f));
    require(sparse.chosen.size() == 1 && sparse.chosen[0].reduction == 1,
            "standalone TOHPE did not use exploration score for inner z choice");
    require(std::abs(sparse.chosen[0].pool_score + 0.2f) < 1e-6f,
            "standalone TOHPE persisted a different outer score than its inner z score");
    require(sparse.stats.accepted_tohpe == 1 && sparse.stats.rejected == 0,
            "standalone TOHPE positive-only accounting changed");
}
```

- [ ] **Step 2: Build and run the test to confirm the behavioral failure**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
```

Expected: the new test fails at `standalone TOHPE did not use exploration score for inner z choice`; current code still chooses reduction 2 before applying the density score in the outer pool.

- [ ] **Step 3: Select the fast or exact path in generate_tohpe_candidates**

Compute the path once outside the sampled-vector callback:

```cpp
const bool reduction_fast_path = config.escore.ranks_fixed_y_by_reduction();
```

Replace the unconditional generator call with:

```cpp
const Int vec_weight = detail::checked_int_from_index(vec.count(), "Candidate vector weight overflow");
if (reduction_fast_path) {
    ctx.get_tohpe_gen().best_z_n_details_into(vec, config.tohpe.z_choices, scratch_best_z);
    std::erase_if(scratch_best_z, [](const TohpeZInfo& info) { return info.reduction <= 0; });
} else {
    ctx.get_tohpe_gen().best_z_n_details_into(
        vec, config.tohpe.z_choices,
        [&](index_t reduction, index_t bucket_size, RowCView z) {
            return config.escore.evaluate_features(
                detail::checked_int_from_index(reduction, "Candidate reduction overflow"), dim_int,
                detail::checked_int_from_index(bucket_size, "Candidate bucket size overflow"), vec_weight,
                detail::checked_int_from_index(z.count(), "Candidate z weight overflow"),
                detail::checked_int_from_index(std::max<index_t>(1, z.size()), "Candidate z size overflow"));
        },
        scratch_best_z);
}
```

In the subsequent loop, replace the old rejection branch with `assert(red > 0);`. Keep candidate construction and the outer `TopKPool<ExplorationScore>` insertion unchanged so `Candidate::pool_score` is still populated by the shared evaluator.

The fast path uses the unchanged packed `argmax_n_into`; filtering its returned zero suffix preserves all positive top-N choices and avoids incrementing `rejected` for internally discarded buckets.

- [ ] **Step 4: Run focused regressions**

Run:

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
build/bin/Debug/vartodd_core_regression
```

Expected: `vartodd_core_regression ok`; default policy reduction remains 2 and density-scored policy reduction becomes 1.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/core/todd_generator.cpp tests/core_regression.cpp
git commit -m "feat: score standalone TOHPE z choices by exploration"
```

---

## Task 5: Full verification and compatibility audit

**Files:**

- Verify only; modify earlier files only if a regression exposes a defect

- [ ] **Step 1: Run Debug and Release C++ regressions**

```bash
cmake --build build --config Debug --target vartodd_core_regression -j2
ctest --test-dir build -C Debug -R vartodd_core_regression --output-on-failure
cmake --build build --config Release --target vartodd_core_regression -j2
ctest --test-dir build -C Release -R vartodd_core_regression --output-on-failure
```

Expected: both configurations pass.

- [ ] **Step 2: Rebuild the Python extension and run the repository Python suite**

```bash
cmake --build build --config Release --target pyvartodd vartodd_core_regression -j2
.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: all tests pass; no binding or pickle change is needed because the existing `PolicyScores.exploration` object supplies the knobs.

- [ ] **Step 3: Audit API and fast-path scope**

Run:

```bash
rg -n "best_z_n_details_into|ranks_fixed_y_by_reduction|evaluate_features|argmax_n_by_into" include src tests
git diff --check
git status --short
```

Confirm from the matches that:

- only `generate_tohpe_candidates` calls the callback overload in production;
- `best_z`, `best_z_n`, and `best_z_n_into` still call the reduction-only overload;
- `TohpePrefixSearch` and `ToddSearch` are unchanged;
- the default predicate routes positive linear reduction-only scoring to `argmax_n_into`;
- no unrelated dirty-worktree files are staged.

- [ ] **Step 4: Repeat the default-path timing comparison**

Rerun the exact Python command from Task 0, changing only the final `tee` path to `/tmp/vartodd_tohpe_z_after.txt`. Compare the two files:

```bash
sed -n '1,2p' /tmp/vartodd_tohpe_z_baseline.txt
sed -n '1,2p' /tmp/vartodd_tohpe_z_after.txt
```

Expected: the seed-0 signature is identical. The median should remain in the same range; investigate a repeatable regression above 20%. Code inspection must also show that the default predicate calls the old `argmax_n_into` path and never constructs the score callback.

- [ ] **Step 5: Request code review**

Invoke `superpowers:requesting-code-review` and address any correctness findings, especially callback evaluation count, tie ordering, and fixed-`y` predicate soundness.

- [ ] **Step 6: Commit verification fixes only if needed**

If verification required changes:

```bash
git add include/nullspace.hpp include/todd_generator.hpp src/core/nullspace.cpp src/core/todd_generator.cpp tests/core_regression.cpp
git commit -m "fix: harden score-aware TOHPE z ranking"
```

If no fixes were needed, do not create an empty commit.
