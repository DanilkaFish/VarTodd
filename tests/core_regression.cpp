#include "matrix.hpp"
#include "nullspace.hpp"
#include "random.hpp"
#include "todd_index.hpp"
#include "todd_generator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace {

using namespace todd;

static_assert(std::is_same_v<decltype(std::declval<const ToddIndex&>().key_of(0)), Row>,
              "ToddIndex keys must own their reconstructed storage");

void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

bool same_entry(const SumEntry& lhs, const SumEntry& rhs) {
    return lhs.a == rhs.a && lhs.b == rhs.b;
}

void require_same_entries(const std::vector<SumEntry>& actual, const std::vector<SumEntry>& expected,
                          const char* message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t i = 0; i < actual.size(); ++i)
        require(same_entry(actual[i], expected[i]), message);
}

std::filesystem::path data_path(const char* relative) {
    return std::filesystem::path(VARTODD_TEST_DATA_DIR) / relative;
}

std::filesystem::path tmp_path(const char* filename) {
    auto dir = std::filesystem::path(VARTODD_TEST_TMP_DIR);
    std::filesystem::create_directories(dir);
    return dir / filename;
}

Matrix matrix_from_words(std::initializer_list<std::uint64_t> words, index_t cols) {
    Matrix  out(static_cast<index_t>(words.size()), cols);
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

void check_row_views() {
    std::array<std::uint64_t, 3> raw{0, 1ULL << 5, ~0ULL};
    RowCView                    wide(raw.data(), 70, 3);
    Row                         owned(wide);

    require(owned.blocks() == ceil_div64(70), "Row copy should canonicalize block count");
    require(owned.count() == 1, "Row copy should ignore padding bits");
    require(owned.cview().find_first() == 69, "Row scan should ignore padding bits");

    Row assigned(70);
    assign(assigned.view(), wide);
    require(assigned == owned, "assign should copy logical row content");
    require((assigned.cview() <=> wide) == std::strong_ordering::equal, "row ordering should use logical bits");

    std::unordered_set<Row, RowHash, RowEq> seen;
    seen.emplace(owned.cview());
    require(seen.find(wide) != seen.end(), "RowHash/RowEq should support heterogeneous logical lookup");

    Row zero = owned ^ wide;
    require(zero.none(), "row xor should use logical row content");
}

void check_matrix_basics() {
    Matrix loaded = Matrix::from_npy(data_path("init_npy/other/adder_8.qc.matrix.npy").string());
    require(loaded.rows() == 173 && loaded.cols() == 61, "adder_8 fixture shape changed");

    const auto roundtrip = tmp_path("core_regression_roundtrip.npy");
    loaded.save_npy(roundtrip.string());
    require(Matrix::from_npy(roundtrip.string()) == loaded, "Matrix npy roundtrip failed");

    Matrix zero_width = Matrix::zeros(3, 0);
    require(zero_width.transpose().rows() == 0 && zero_width.transpose().cols() == 3,
            "zero-width transpose shape mismatch");
    require((zero_width * Row(0)).none(), "zero-width matvec should produce zero row");

    Row row(5);
    row.set(2);
    Matrix rows(0, 5);
    rows.push_back(row);
    rows.push_back(rows[0]);
    require(rows.rows() == 2 && rows[1].test(2), "Matrix::push_back should tolerate self-view source");

    Matrix duplicated = rows;
    duplicated.append_down_inplace(duplicated);
    require(duplicated.rows() == 4 && duplicated[2].test(2) && duplicated[3].test(2),
            "append_down_inplace should tolerate self append");

    require((Matrix::identity(5) * row.cview()) == row, "identity matvec failed");
}

void check_canonical_parity_matrix() {
    Matrix input(7, 3);
    input[1].set(0);
    input[1].set(2); // 101
    input[2].set(1);
    input[2].set(2); // 011
    assign(input[3], input[1]);
    input[4].set(0);
    input[4].set(1); // 110
    assign(input[5], input[4]);
    assign(input[6], input[4]);

    Matrix expected(2, 3);
    assign(expected[0], input[4]);
    assign(expected[1], input[2]);
    require(canonical_parity_matrix(input) == expected,
            "parity canonicalization should remove zero/even rows and sort odd rows");

    Matrix reordered(7, 3);
    assign(reordered[0], input[6]);
    assign(reordered[1], input[3]);
    assign(reordered[2], input[0]);
    assign(reordered[3], input[5]);
    assign(reordered[4], input[2]);
    assign(reordered[5], input[1]);
    assign(reordered[6], input[4]);
    require(canonical_parity_matrix(reordered) == expected,
            "parity canonicalization should ignore input row order");

    Matrix empty_parity(3, 3);
    assign(empty_parity[1], input[1]);
    assign(empty_parity[2], input[1]);
    const auto empty = canonical_parity_matrix(empty_parity);
    require(empty.rows() == 0 && empty.cols() == 3,
            "empty canonical parity matrix should preserve its column width");
}

void check_todd_index_basics() {
    Matrix    loaded = Matrix::from_npy(data_path("init_npy/other/adder_8.qc.matrix.npy").string());
    ToddIndex index(loaded);

    require(index.rows() == loaded.rows() && index.cols() == loaded.cols(), "ToddIndex shape mismatch");
    require(index.max_bucket() > 0 && index.buckets_num() > 0, "ToddIndex should build nonempty buckets");

    std::vector<SumEntry> single_entries;
    require(index.materialize_bucket(loaded[0], single_entries) && !single_entries.empty(),
            "single-row bucket lookup failed");

    Row pair = loaded[0] ^ loaded[1];
    const auto bucket_id = index.pair_bucket_id(0, 1);
    require(index.key_of(bucket_id) == pair, "pair bucket key mismatch");

    std::vector<SumEntry> by_id;
    std::vector<SumEntry> by_key;
    require(index.materialize_bucket(bucket_id, by_id), "pair materialization by id failed");
    require(index.materialize_bucket(pair, by_key), "pair materialization by key failed");
    require_same_entries(by_id, by_key, "id/key materialization order differs");
}

void check_todd_index_duplicate_order() {
    Matrix P(6, 3);
    P[2].set(0);
    P[3].set(0);
    P[4].set(1);
    P[5].set(1);

    ToddIndex index(P);
    require(index.buckets_num() == 4, "duplicate-heavy bucket count changed");
    require(index.max_bucket() == 6, "duplicate-heavy maximum bucket changed");
    require(index.bucket_length_bytes() == 1, "small ToddIndex should use byte bucket lengths");
    require(index.storage_bytes() > 0, "ToddIndex storage accounting is empty");
    require(index.storage_bytes() < 4096, "small ToddIndex retained unexpectedly large storage");

    const std::array<std::uint32_t, 6> expected_single_ids{0, 0, 1, 1, 2, 2};
    require(index.single_id() == std::vector<std::uint32_t>(expected_single_ids.begin(), expected_single_ids.end()),
            "single bucket IDs no longer follow first-encounter order");

    const std::array<std::uint32_t, 15> expected_pair_ids{0, 1, 1, 2, 2, 1, 1, 2, 2, 0, 3, 3, 3, 3, 0};
    std::size_t                         pair_pos = 0;
    for (index_t i = 0; i < P.rows(); ++i) {
        for (index_t j = i + 1; j < P.rows(); ++j) {
            require(index.pair_bucket_id(i, j) == expected_pair_ids[pair_pos++],
                    "pair bucket IDs no longer follow first-encounter order");
        }
    }

    const std::array<std::vector<SumEntry>, 4> expected_entries{
        std::vector<SumEntry>{SumEntry{0}, SumEntry{1}, SumEntry{0, 1}, SumEntry{2, 3}, SumEntry{4, 5}},
        std::vector<SumEntry>{SumEntry{2}, SumEntry{3}, SumEntry{0, 2}, SumEntry{0, 3}, SumEntry{1, 2},
                              SumEntry{1, 3}},
        std::vector<SumEntry>{SumEntry{4}, SumEntry{5}, SumEntry{0, 4}, SumEntry{0, 5}, SumEntry{1, 4},
                              SumEntry{1, 5}},
        std::vector<SumEntry>{SumEntry{2, 4}, SumEntry{2, 5}, SumEntry{3, 4}, SumEntry{3, 5}},
    };

    std::vector<SumEntry> actual;
    for (std::uint32_t id = 0; id < expected_entries.size(); ++id) {
        require(index.materialize_bucket(id, actual), "duplicate-heavy bucket materialization failed");
        require_same_entries(actual, expected_entries[id], "duplicate-heavy member order changed");
    }

    Row expected_key(3);
    for (std::uint32_t id = 0; id < expected_entries.size(); ++id) {
        expected_key.reset();
        if (id == 1 || id == 3)
            expected_key.set(0);
        if (id == 2 || id == 3)
            expected_key.set(1);
        require(index.key_of(id) == expected_key, "duplicate-heavy representative key changed");
    }

    Row absent(3);
    absent.set(2);
    actual.emplace_back(0);
    require(!index.materialize_bucket(absent, actual), "absent ToddIndex key unexpectedly found");
    require(actual.empty(), "absent ToddIndex lookup did not clear output");
}

void require_top_count_set(const std::vector<CountWSScore>& actual,
                           const std::map<std::uint32_t, index_t>& reference, std::size_t requested) {
    const std::size_t expected_size = std::min(requested, reference.size());
    require(actual.size() == expected_size, "compact CountWS result size mismatch");

    std::vector<index_t> expected_counts;
    expected_counts.reserve(reference.size());
    for (const auto& [id, count] : reference) {
        (void)id;
        expected_counts.push_back(count);
    }
    std::ranges::sort(expected_counts, std::greater{});
    expected_counts.resize(expected_size);

    std::vector<index_t> actual_counts;
    actual_counts.reserve(actual.size());
    std::unordered_set<std::uint32_t> selected_ids;
    for (const auto& score : actual) {
        const auto it = reference.find(score.bucket_id);
        require(it != reference.end(), "compact CountWS returned an untouched bucket");
        require(it->second == score.count, "compact CountWS returned a wrong bucket count");
        require(selected_ids.insert(score.bucket_id).second, "compact CountWS returned a duplicate bucket");
        actual_counts.push_back(score.count);
    }
    std::ranges::sort(actual_counts, std::greater{});
    require(actual_counts == expected_counts, "compact CountWS must return the greatest counts");
}

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

void check_packed_count_storage() {
    detail::PackedCountStorage<std::uint8_t, std::uint16_t> ws;

    ws.reset(8, 20, false);
    ws.add(7, 2);
    ws.add(2, 1);
    ws.add(7, 2);
    ws.add(4, 2);
    const std::map<std::uint32_t, index_t> even_reference{{2, 1}, {4, 2}, {7, 4}};

    require_top_count_set(ws.argmax_n(2), even_reference, 2);
    require(ws.argmax_n(0).empty(), "zero CountWS sample count should be empty");
    require_top_count_set(ws.argmax_n(100), even_reference, 100);
    require(ws.argmax().bucket_id == 7 && ws.argmax().count == 4, "compact CountWS argmax mismatch");

    ws.reset(8, 20, true);
    ws.add(1, 1);
    ws.add(2, 2);
    ws.add(1, 2);
    const std::map<std::uint32_t, index_t> odd_reference{{1, 2}, {2, 1}};
    require_top_count_set(ws.argmax_n(2), odd_reference, 2);

    for (std::uint32_t generation = 0; generation < 300; ++generation) {
        ws.reset(8, 20, false);
        const std::uint32_t id = generation & 7U;
        ws.add(id, 2);
        const auto result = ws.argmax_n(8);
        require(result.size() == 1 && result[0].bucket_id == id && result[0].count == 2,
                "compact CountWS epoch wrap leaked an old bucket");
    }

    detail::PackedCountStorage<std::uint8_t, std::uint16_t> density_ws;
    density_ws.reset(64, 100, false);
    std::map<std::uint32_t, index_t> balanced_reference;
    for (std::uint32_t id = 0; id < 64; id += 2) {
        density_ws.add(id, 2);
        balanced_reference[id] = 2;
    }
    require_top_count_set(density_ws.argmax_n(64), balanced_reference, 64);

    density_ws.reset(64, 100, false);
    density_ws.add(0, 2);
    density_ws.add(63, 2);
    density_ws.add(63, 2);
    const std::map<std::uint32_t, index_t> sparse_reference{{0, 2}, {63, 4}};
    require_top_count_set(density_ws.argmax_n(64), sparse_reference, 64);

    CountWS narrow;
    narrow.reset(32, 9, false);
    require(narrow.count_bytes() == 1, "max bucket 9 should select byte counters");
    require(narrow.state_bytes() == 2, "byte counters should use 16-bit packed states");
    narrow.add(3, 2);
    narrow.add(19, 2);
    narrow.add(3, 2);
    const std::map<std::uint32_t, index_t> wrapper_reference{{3, 4}, {19, 2}};
    require_top_count_set(narrow.argmax_n(1), wrapper_reference, 1);

    CountWS medium;
    medium.reset(32, 128, false);
    require(medium.count_bytes() == 2, "max bucket 128 should select 16-bit counters");
    require(medium.state_bytes() == 4, "16-bit counters should use 32-bit packed states");
    medium.add(5, 256);
    require(medium.argmax().bucket_id == 5 && medium.argmax().count == 256,
            "16-bit packed counter lost its boundary value");

    CountWS large;
    large.reset(32, 32768, false);
    require(large.count_bytes() == 4, "max bucket 32768 should select 32-bit counters");
    require(large.state_bytes() == 8, "32-bit counters should use 64-bit packed states");
    large.add(6, 65536);
    require(large.argmax().bucket_id == 6 && large.argmax().count == 65536,
            "32-bit packed counter lost its boundary value");

    CountWS wide;
    constexpr index_t wide_bucket = std::numeric_limits<std::uint32_t>::max();
    wide.reset(32, wide_bucket, false);
    require(wide.count_bytes() == 8, "large buckets should select wide counters");
    require(wide.state_bytes() == 16, "wide counters should select the wide fallback");
    wide.add(7, wide_bucket * 2);
    require(wide.argmax().bucket_id == 7 && wide.argmax().count == wide_bucket * 2,
            "wide counter lost its boundary value");
}

void check_count_selection_tie_order() {
    detail::PackedCountStorage<std::uint8_t, std::uint16_t> ws;
    ws.reset(8, 8, false);
    ws.add(5, 2);
    ws.add(2, 2);
    ws.add(7, 2);

    const auto selected = ws.argmax_n(3);
    require(selected.size() == 3, "equal-count selection lost a bucket");
    require(selected[0].bucket_id == 2 && selected[1].bucket_id == 5 && selected[2].bucket_id == 7,
            "equal-count Z choices must use ascending bucket ID");
}

void require_scored_count_selection(index_t max_bucket) {
    CountWS ws;
    ws.reset(16, max_bucket, false);
    ws.add(5, 4);
    ws.add(2, 4);
    ws.add(7, 2);
    ws.add(1, 0);

    std::array<int, 16>                calls{};
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

void check_top_bucket_tie_order() {
    Matrix P(6, 3);
    P[2].set(0);
    P[3].set(0);
    P[4].set(1);
    P[5].set(1);

    ToddIndex index(P);
    const auto& top = index.top_sum_bucket_id_sizes_scratch(3);
    require(top.size() == 3, "top bucket selection lost a bucket");
    require(top[0].first == 1 && top[1].first == 2 && top[2].first == 0,
            "equal-size Todd buckets must use ascending ID");
}

void check_portable_random_and_seed_contract() {
    PyRNG rng(123);
    require(rng.rand_int(0, 100) == 47, "portable bounded RNG stream changed");
    require(rng.rand_u64() == 9824761154233434147ULL, "portable 64-bit RNG stream changed");

    if constexpr (std::numeric_limits<index_t>::digits > 32) {
        constexpr index_t high_seed = static_cast<index_t>(0x693e4e5ad9139746ULL);
        const auto normalized = detail::normalize_minstd_seed(high_seed);
        require(normalized == 730870781U, "high minstd seed normalization changed");

        PyRNG constructed(high_seed);
        PyRNG canonical_constructed(static_cast<index_t>(normalized));
        require(constructed.random_raw() == canonical_constructed.random_raw(),
                "PyRNG construction depends on uint_fast32_t width");

        PyRNG reseeded(1);
        PyRNG canonical_reseeded(1);
        reseeded.seed(high_seed);
        canonical_reseeded.seed(static_cast<index_t>(normalized));
        require(reseeded.random_raw() == canonical_reseeded.random_raw(),
                "PyRNG reseeding depends on uint_fast32_t width");
    }

    Matrix P(4, 3);
    P[0].set(0);
    P[1].set(0);
    P[2].set(1);
    P[3].set(2);
    require(matrix_seed(P, 7, 3) == 3727290120U, "portable matrix seed changed");
}

void check_candidate_tie_order() {
    Candidate tohpe;
    tohpe.source = CandidateSourceTohpe;
    Candidate todd;
    todd.source = CandidateSourceTodd;
    require(candidate_tie_preferred(tohpe, todd), "candidate ties must prefer source order");

    Candidate lower_bucket;
    lower_bucket.source = CandidateSourceTohpe;
    lower_bucket.bucket_id = 5;
    Candidate higher_bucket = lower_bucket;
    higher_bucket.bucket_id = 6;
    require(candidate_tie_preferred(lower_bucket, higher_bucket),
            "candidate ties must prefer lower bucket ID");

    Candidate lower_z = lower_bucket;
    lower_z.z = Row(3);
    lower_z.z.set(0);
    Candidate higher_z = lower_z;
    higher_z.z.reset(0);
    higher_z.z.set(1);
    require(candidate_tie_preferred(lower_z, higher_z), "candidate ties must prefer canonical Z order");
}

void check_exploration_score_feature_contract() {
    Candidate cand;
    cand.reduction   = 6;
    cand.basis_dim   = 3;
    cand.bucket_size = 4;
    cand.vec_weight  = 5;
    cand.z_weight    = 2;
    cand.z_size      = 8;

    ExplorationScore score({2.0f, -1.0f, 0.5f, 3.0f, -4.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                           ScoringFunction{});
    score.bn    = 10.0f;
    score.dn    = 12.0f;
    score.wvwn = 20.0f;

    const float direct = score.evaluate_features(6, 3, 4, 5, 2, 8);
    require(std::abs(direct - 0.3f) < 1e-7f, "exploration feature normalization changed");
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
    default_score.sc.type    = ScoringFunction::POLYNOM;
    require(!default_score.ranks_fixed_y_by_reduction(),
            "nonlinear scoring must use exact fixed-y ranking");
}

void check_score_aware_tohpe_z_ranking() {
    Matrix P    = matrix_from_words({1, 4, 11, 14, 16, 21, 26, 28, 31}, 5);
    auto   data = std::make_shared<MatrixWithData>(P, false);
    require(data->tohpe_basis().rows() == 1, "score-aware TOHPE fixture kernel changed");
    const Row      y(data->tohpe_basis()[0]);
    TohpeGenerator generator(data);

    std::vector<TohpeZInfo> reduction_ranked;
    generator.best_z_n_details_into(y, 8, reduction_ranked);
    require(!reduction_ranked.empty() && reduction_ranked[0].reduction == 2 &&
                reduction_ranked[0].z.count() == 4,
            "score-aware TOHPE fixture greedy candidate changed");

    ExplorationScore default_score(1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    default_score.bn    = static_cast<float>(data->index().max_bucket());
    default_score.dn    = static_cast<float>(data->tohpe_basis().rows() + 5);
    default_score.wvwn = static_cast<float>(P.rows());

    std::vector<TohpeZInfo> scored_like_default;
    generator.best_z_n_details_into(
        y, 8,
        [&](index_t reduction, index_t bucket_size, RowCView z) {
            return default_score.evaluate_features(static_cast<Int>(reduction), 1, static_cast<Int>(bucket_size),
                                                   static_cast<Int>(y.count()), static_cast<Int>(z.count()),
                                                   static_cast<Int>(z.size()));
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
    sparse_score.bn    = default_score.bn;
    sparse_score.dn    = default_score.dn;
    sparse_score.wvwn = default_score.wvwn;
    std::vector<TohpeZInfo> sparse_ranked;
    generator.best_z_n_details_into(
        y, 1,
        [&](index_t reduction, index_t bucket_size, RowCView z) {
            return sparse_score.evaluate_features(static_cast<Int>(reduction), 1, static_cast<Int>(bucket_size),
                                                  static_cast<Int>(y.count()), static_cast<Int>(z.count()),
                                                  static_cast<Int>(z.size()));
        },
        sparse_ranked);
    require(sparse_ranked.size() == 1 && sparse_ranked[0].reduction == 1 && sparse_ranked[0].z.count() == 1,
            "z-density score should select the lower-reduction sparse candidate");

    Candidate candidate(0, static_cast<Int>(sparse_ranked[0].reduction), k_single_sentinel<Int>(),
                        k_single_sentinel<Int>(), Row(y), Row(sparse_ranked[0].z), 1,
                        static_cast<Int>(sparse_ranked[0].bucket_size), sparse_ranked[0].bucket_id,
                        CandidateSourceTohpe);
    const float inner_score = sparse_score.evaluate_features(candidate.reduction, candidate.basis_dim,
                                                             candidate.bucket_size, candidate.vec_weight,
                                                             candidate.z_weight, candidate.z_size);
    const auto [outer_score, unused] = sparse_score(candidate);
    (void)unused;
    require(std::abs(inner_score - outer_score) < 1e-7f,
            "inner TOHPE z score differs from candidate pool score");

    const Matrix state =
        canonical_parity_matrix(generator.make(sparse_ranked[0].z, sparse_ranked[0].bucket_id).apply(y));
    require(P.rows() - state.rows() == sparse_ranked[0].reduction,
            "score-aware TOHPE reported reduction differs from applied reduction");
}

void check_policy_iteration_smoke() {
    Matrix P(4, 3);
    P[0].set(0);
    P[1].set(0);
    P[2].set(1);
    P[3].set(2);

    auto data = std::make_shared<MatrixWithData>(std::move(P), true);

    PolicyConfig cfg;
    cfg.selection.count          = 1;
    cfg.pool.final_size          = 4;
    cfg.tohpe = TohpeSearch{SamplingBudget{}, SourcePool{0, 0}, 1};
    cfg.tohpeprefix = TohpePrefixSearch{SamplingBudget{8, 8, 0, 2}, SourcePool{4, 0}, 2,
                                          ZBucketSearch{1, 8, 0.0f, 0.0f, 8}};
    cfg.todd = ToddSearch{SamplingBudget{8, 8, 0, 2}, SourcePool{4, 0}, 2,
                           ZBucketSearch{1, 8, 0.0f, 0.0f, 8}};

    auto result = policy_iteration_impl(data, cfg, 123, 0);
    require(result.chosen.size() == 1 && result.states.size() == 1, "policy iteration should choose one state");
    require(result.chosen[0].reduction > 0, "policy iteration should find a reducing candidate");
    require(result.states[0].rows() == 2 && result.states[0].cols() == 3, "policy iteration state shape mismatch");
    require(result.stats.accepted_tohpe == 0, "disabled TOHPE source should not emit candidates");
    require(result.stats.accepted_tohpeprefix > 0,
            "policy iteration should count accepted TOHPEprefix candidates");
    require(result.stats.accepted_todd > 0, "policy iteration should count accepted Todd candidates");
    require(result.stats.accepted == result.stats.accepted_tohpe + result.stats.accepted_tohpeprefix +
                                       result.stats.accepted_todd,
            "accepted total should match all three sources");
    require(result.stats.accepted == result.stats.nonzero, "accepted total should match nonzero candidate count");
    require(result.chosen[0].pool_tohpe_size + result.chosen[0].pool_tohpeprefix_size +
                result.chosen[0].pool_todd_size == result.chosen[0].pool_size,
            "final pool composition should contain all three sources");
}

void check_policy_iteration_repeatability() {
    Matrix P(4, 3);
    P[0].set(0);
    P[1].set(0);
    P[2].set(1);
    P[3].set(2);
    auto data = std::make_shared<MatrixWithData>(std::move(P), true);

    PolicyConfig cfg;
    cfg.selection.count = 2;
    cfg.selection.mode = "softmax";
    cfg.selection.temperature = 0.2f;
    cfg.pool.final_size = 4;
    cfg.tohpe = TohpeSearch{SamplingBudget{8, 8, 4, 2}, SourcePool{4, 0}, 2};
    cfg.tohpeprefix = TohpePrefixSearch{SamplingBudget{8, 8, 4, 2}, SourcePool{4, 0}, 2,
                                         ZBucketSearch{1, 8, 0.0f, 0.0f, 8}};
    cfg.todd = ToddSearch{SamplingBudget{8, 8, 4, 2}, SourcePool{4, 0}, 2,
                          ZBucketSearch{1, 8, 0.0f, 0.0f, 8}};

    const auto first  = policy_iteration_impl(data, cfg, 123, 0);
    const auto second = policy_iteration_impl(data, cfg, 123, 0);
    require(first.seed == second.seed && first.chosen.size() == second.chosen.size() &&
                first.states.size() == second.states.size(),
            "fixed-seed policy iteration result shape changed");
    for (std::size_t i = 0; i < first.chosen.size(); ++i) {
        const auto& a = first.chosen[i];
        const auto& b = second.chosen[i];
        require(a.final_score == b.final_score && a.pool_score == b.pool_score && a.reduction == b.reduction &&
                    a.k == b.k && a.l == b.l && a.basis_dim == b.basis_dim && a.tohpe_dim == b.tohpe_dim &&
                    a.bucket_size == b.bucket_size && a.source == b.source && first.states[i] == second.states[i],
                "fixed-seed policy iteration selected different candidates");
    }
}

void check_policy_iteration_merges_equivalent_parity_states() {
    Matrix P(4, 3);
    P[0].set(2); // 001
    P[1].set(0); // 100
    P[2].set(2); // 001
    P[3].set(0);
    P[3].set(2); // 101
    auto data = std::make_shared<MatrixWithData>(std::move(P), true);

    PolicyConfig cfg;
    cfg.selection = ActionSelection{16, "best", 0.0f};
    cfg.pool = ActionPool{16};
    cfg.tohpe = TohpeSearch{SamplingBudget{k_all_one_hot_samples, 32, 64, 3}, SourcePool{16, 0}, 8};
    cfg.tohpeprefix = TohpePrefixSearch{
        SamplingBudget{k_all_one_hot_samples, 32, 64, 3}, SourcePool{16, 0}, 4,
        ZBucketSearch{32, 0, 0.0f, 0.0f, 128}};
    cfg.todd = ToddSearch{SamplingBudget{k_all_one_hot_samples, 32, 64, 3}, SourcePool{16, 0}, 4,
                          ZBucketSearch{32, 0, 0.0f, 0.0f, 128}};

    const auto first = policy_iteration_impl(data, cfg, 123, 0);
    require(first.chosen.size() == 1 && first.states.size() == 1,
            "equivalent parity states should occupy one final action slot");

    Matrix expected(2, 3);
    expected[0].set(0); // 100
    expected[1].set(0);
    expected[1].set(2); // 101
    require(first.states[0] == expected, "selected parity state should use canonical row ordering");

    const auto& chosen = first.chosen[0];
    require(chosen.reduction == 4 && std::abs(chosen.final_score - (2.0f / 3.0f)) < 1e-6f,
            "equivalent parity states should retain the greatest finalization score");
    require(chosen.pool_size == 1 && chosen.pool_tohpe_size == 1 && chosen.pool_tohpeprefix_size == 0 &&
                chosen.pool_todd_size == 0,
            "final pool metadata should describe unique state representatives");

    const auto second = policy_iteration_impl(data, cfg, 123, 0);
    require(second.chosen.size() == 1 && second.states.size() == 1 && second.states[0] == first.states[0] &&
                second.chosen[0].final_score == chosen.final_score && second.chosen[0].source == chosen.source,
            "equivalent-state finalization should remain fixed-seed repeatable");
}

void check_tohpe_only_policy_continues_after_todd_stops() {
    Matrix P(4, 3);
    P[0].set(0);
    P[1].set(0);
    P[2].set(1);
    P[3].set(2);
    auto data = std::make_shared<MatrixWithData>(std::move(P), true);

    PolicyConfig cfg;
    cfg.selection.count = 1;
    cfg.pool.final_size = 4;
    cfg.tohpe = TohpeSearch{SamplingBudget{}, SourcePool{0, 0}, 1};
    cfg.tohpeprefix = TohpePrefixSearch{SamplingBudget{8, 8, 0, 2}, SourcePool{4, 0}, 2,
                                          ZBucketSearch{1, 8, 0.0f, 0.0f, 8}};
    cfg.todd = ToddSearch{SamplingBudget{8, 8, 0, 2}, SourcePool{0, 0}, 2,
                           ZBucketSearch{0, 0, 0.0f, 0.0f, 0}};

    auto result = policy_iteration_impl(data, cfg, 123, 0);
    require(result.stats.accepted_tohpe == 0, "disabled TOHPE source should not emit candidates");
    require(result.stats.accepted_tohpeprefix > 0, "TOHPEprefix traversal should retain candidates");
    require(result.stats.accepted_todd == 0, "disabled Todd source should not emit candidates");
}

void check_tohpe_policy() {
    Matrix P(4, 3);
    P[0].set(0);
    P[1].set(0);
    P[2].set(1);
    P[3].set(2);
    auto data = std::make_shared<MatrixWithData>(P, false);

    PolicyConfig cfg;
    cfg.selection.count = 1;
    cfg.pool.final_size = 4;
    cfg.tohpe = TohpeSearch{SamplingBudget{8, 8, 0, 2}, SourcePool{4, 0}, 2};
    cfg.tohpeprefix = TohpePrefixSearch{};
    cfg.todd = ToddSearch{SamplingBudget{8, 8, 0, 2}, SourcePool{0, 0}, 2,
                           ZBucketSearch{0, 0, 0.0f, 0.0f, 0}};

    auto result = policy_iteration_impl(data, cfg, 123, 0);
    require(result.stats.accepted_tohpe > 0, "TOHPE should emit candidates");
    require(result.stats.accepted_tohpeprefix == 0, "disabled TOHPEprefix source should not emit candidates");
    require(result.stats.accepted_todd == 0, "disabled Todd source should not emit candidates");
}

void check_tohpe_policy_scores_inner_z_choices() {
    const Matrix P    = matrix_from_words({1, 4, 11, 14, 16, 21, 26, 28, 31}, 5);
    auto         data = std::make_shared<MatrixWithData>(P, false);

    const auto run = [&](ExplorationScore exploration) {
        PolicyConfig cfg;
        cfg.scores.exploration = std::move(exploration);
        cfg.selection          = ActionSelection{1, "best", 0.0f};
        cfg.pool               = ActionPool{4};
        cfg.tohpe =
            TohpeSearch{SamplingBudget{k_all_one_hot_samples, 0, 0, 2}, SourcePool{4, 0}, 1};
        cfg.tohpeprefix = TohpePrefixSearch{SamplingBudget{}, SourcePool{0, 0}, 1,
                                             ZBucketSearch{0, 0, 0.0f, 0.0f, 0}};
        cfg.todd = ToddSearch{SamplingBudget{}, SourcePool{0, 0}, 1,
                              ZBucketSearch{0, 0, 0.0f, 0.0f, 0}};
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

void check_tohpe_and_tohpeprefix_stats_merge() {
    Matrix P(4, 3);
    P[0].set(0);
    P[1].set(0);
    P[2].set(1);
    P[3].set(2);
    auto data = std::make_shared<MatrixWithData>(P, true);

    const auto tohpe = TohpeSearch{SamplingBudget{8, 8, 0, 2}, SourcePool{4, 0}, 2};
    const auto prefix = TohpePrefixSearch{SamplingBudget{8, 8, 0, 2}, SourcePool{4, 0}, 2,
                                           ZBucketSearch{1, 8, 0.0f, 0.0f, 8}};
    const auto todd_off = ToddSearch{SamplingBudget{}, SourcePool{0, 0}, 2,
                                     ZBucketSearch{0, 0, 0.0f, 0.0f, 0}};

    PolicyConfig tohpe_cfg;
    tohpe_cfg.selection.count = 1;
    tohpe_cfg.pool.final_size = 8;
    tohpe_cfg.tohpe = tohpe;
    tohpe_cfg.todd = todd_off;

    PolicyConfig prefix_cfg = tohpe_cfg;
    prefix_cfg.tohpe = TohpeSearch{SamplingBudget{}, SourcePool{0, 0}, 1};
    prefix_cfg.tohpeprefix = prefix;

    PolicyConfig both_cfg = tohpe_cfg;
    both_cfg.tohpeprefix = prefix;

    const auto tohpe_result = policy_iteration_impl(data, tohpe_cfg, 123, 0);
    const auto prefix_result = policy_iteration_impl(data, prefix_cfg, 123, 0);
    const auto both_result = policy_iteration_impl(data, both_cfg, 123, 0);
    require(both_result.stats.accepted_tohpe == tohpe_result.stats.accepted_tohpe,
            "TOHPE statistics should retain their source identity");
    require(both_result.stats.accepted_tohpeprefix == prefix_result.stats.accepted_tohpeprefix,
            "TOHPEprefix statistics should retain their source identity");
}

void check_tohpe_continues_after_todd_pool_is_filled() {
    Matrix P(4, 3);
    P[0].set(0);
    P[1].set(0);
    P[2].set(1);
    P[3].set(2);
    auto data = std::make_shared<MatrixWithData>(std::move(P), true);

    PolicyConfig cfg;
    cfg.selection.count = 1;
    cfg.pool.final_size = 8;
    cfg.tohpe = TohpeSearch{SamplingBudget{}, SourcePool{0, 0}, 1};
    cfg.tohpeprefix = TohpePrefixSearch{SamplingBudget{0, 0, 0, 2}, SourcePool{4, 0}, 2,
                                          ZBucketSearch{3, 3, 0.0f, 0.0f, 3}};
    cfg.todd = ToddSearch{SamplingBudget{8, 8, 0, 2}, SourcePool{1, 0}, 2,
                           ZBucketSearch{1, 8, 0.0f, 0.0f, 8}};

    auto result = policy_iteration_impl(data, cfg, 123, 0);
    require(result.stats.total >= 3, "TOHPE minimum should continue shared bucket traversal");
    require(result.stats.accepted_todd > 0, "Todd source should fill its pool before TOHPE-only continuation");
}

void check_todd_tohpe_prefix() {
    Matrix P(4, 3);
    P[0].set(0);
    P[1].set(0);
    P[2].set(1);
    P[3].set(2);
    auto data = std::make_shared<MatrixWithData>(std::move(P), true);

    FullToddGenerator generator(data);
    auto              ns = generator.make(0);
    require(ns.tohpe_prefix_size() > 0, "TODD basis should retain a TOHPE prefix");
    require(ns.tohpe_prefix_size() <= ns.basis().rows(), "TOHPE prefix exceeds TODD basis");
}

void check_tohpe_only_basis_request() {
    Matrix P(4, 3);
    P[0].set(0);
    P[1].set(0);
    P[2].set(1);
    P[3].set(2);
    auto data = std::make_shared<MatrixWithData>(std::move(P), false);

    const auto bucket_id = data->index().single_id()[0];
    const Row  z         = data->index().key_of(bucket_id);
    std::vector<SumEntry> entries;
    require(data->index().materialize_bucket(bucket_id, entries) && !entries.empty(),
            "TOHPE-only request needs a materialized bucket");

    FullToddGenerator generator(data);
    auto generated = generator.solution_basis(z.cview(), entries.data(), static_cast<index_t>(entries.size()),
                                              SolutionBasisRequest::TohpeOnly);
    require(generated.basis.rows() <= data->tohpe_basis().rows(), "TOHPE-only basis exceeds TOHPE dimension");
    require(generated.tohpe_prefix_size == generated.basis.rows(),
            "TOHPE-only basis should consist entirely of its prefix");
}

void check_tohpe_only_request_keeps_full_todd_cache_lazy() {
    Matrix P(4, 3);
    P[0].set(0);
    P[1].set(0);
    P[2].set(1);
    P[3].set(2);
    auto data = std::make_shared<MatrixWithData>(std::move(P), true);
    require(!data->full_todd_ready(), "Full-Todd cache should start empty");

    const auto bucket_id = data->index().single_id()[0];
    const Row  z         = data->index().key_of(bucket_id);
    std::vector<SumEntry> entries;
    require(data->index().materialize_bucket(bucket_id, entries) && !entries.empty(),
            "lazy cache check needs a materialized bucket");

    FullToddGenerator generator(data);
    (void)generator.solution_basis(z.cview(), entries.data(), static_cast<index_t>(entries.size()),
                                   SolutionBasisRequest::TohpeOnly);
    require(!data->full_todd_ready(), "TOHPE-only request should not build Full-Todd data");
    auto both = generator.make(z.cview(), entries.data(), static_cast<index_t>(entries.size()),
                               SolutionBasisRequest::Both);
    require(both.tohpe_prefix_size() > 0 && both.tohpe_prefix_size() <= both.basis().rows(),
            "both request should expose the compact TOHPE prefix in the full basis");
    require(data->full_todd_ready(), "Todd request should build Full-Todd data");
}

void check_two_region_sampling() {
    PyRNG            rng(7);
    std::vector<Row> rows;
    rng.for_each_capped_bitvector_regions(
        3, 2, {2, 0, 3}, 2, {3, 0, 7}, 2, [&](RowCView coefs) { rows.emplace_back(coefs); });

    std::unordered_set<Row, RowHash, RowEq> unique(rows.begin(), rows.end());
    require(rows.size() == 7, "two-region sampler should emit the full non-zero universe once");
    require(unique.size() == rows.size(), "two-region sampler emitted a collision");
    require(!rows[0].test(2) && !rows[1].test(2), "prefix samples must be emitted first");
}

void check_two_region_sampling_routes_sources() {
    PyRNG            rng(17);
    std::vector<Row> prefix_rows;
    std::vector<Row> full_rows;
    rng.for_each_capped_bitvector_regions(
        3, 2, {2, 0, 0}, 2, {3, 0, 4}, 2, [&](RowCView coefs) { prefix_rows.emplace_back(coefs); },
        [&](RowCView coefs) { full_rows.emplace_back(coefs); });

    require(prefix_rows.size() == 2, "prefix callback should receive its complete budget");
    std::unordered_set<Row, RowHash, RowEq> seen(prefix_rows.begin(), prefix_rows.end());
    for (const Row& row : full_rows)
        require(seen.insert(row).second, "full callback collided with a prefix coefficient");
}

} // namespace

int main() {
    try {
        check_row_views();
        check_matrix_basics();
        check_canonical_parity_matrix();
        check_todd_index_basics();
        check_todd_index_duplicate_order();
        check_bucket_lengths();
        check_packed_count_storage();
        check_count_selection_tie_order();
        check_scored_count_selection();
        check_top_bucket_tie_order();
        check_portable_random_and_seed_contract();
        check_candidate_tie_order();
        check_exploration_score_feature_contract();
        check_score_aware_tohpe_z_ranking();
        check_policy_iteration_smoke();
        check_policy_iteration_repeatability();
        check_policy_iteration_merges_equivalent_parity_states();
        check_tohpe_only_policy_continues_after_todd_stops();
        check_tohpe_policy();
        check_tohpe_policy_scores_inner_z_choices();
        check_tohpe_and_tohpeprefix_stats_merge();
        check_tohpe_continues_after_todd_pool_is_filled();
        check_todd_tohpe_prefix();
        check_tohpe_only_basis_request();
        check_tohpe_only_request_keeps_full_todd_cache_lazy();
        check_two_region_sampling();
        check_two_region_sampling_routes_sources();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    std::cout << "vartodd_core_regression ok\n";
    return 0;
}
