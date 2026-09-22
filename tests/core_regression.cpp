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

// --- policy program builders -------------------------------------------------
//
// The scoring structs these checks used were replaced by compiled expression
// programs. These helpers assemble the equivalent programs so the checks keep
// asserting the same numbers against the same inputs.

// sum_i w[i] * feature_i, over the canonical feature order.
PolicyProgram linear_program(const std::vector<float>& weights,
                             PolicySite                site = PolicySite::ExplorationPool) {
    static constexpr Knob k_features[] = {Knob::red, Knob::ndim,  Knob::bucket,
                                          Knob::nyw,  Knob::nzw,   Knob::ntohpe};
    std::vector<Instr>    code;
    bool                  first = true;
    for (std::uint16_t i = 0; i < weights.size(); ++i) {
        if (weights[i] == 0.0f)
            continue; // a zero weight contributes no term, as in the old sum
        code.push_back(Instr{Op::LoadKnob, static_cast<std::uint16_t>(k_features[i])});
        code.push_back(Instr{Op::LoadConst, i});
        code.push_back(Instr{Op::Mul, 0});
        if (!first)
            code.push_back(Instr{Op::Add, 0});
        first = false;
    }
    if (code.empty()) { // all-zero weights: a constant zero score
        code.push_back(Instr{Op::LoadConst, 0});
    }
    return PolicyProgram(std::move(code), weights, 0, site);
}

// sum_i w[i] * |feature_i - c[i]| ** pow, with the first center scaled by
// Centers are expressed directly in the raw feature units.
PolicyProgram polynom_program(const std::vector<float>& weights, const std::vector<float>& centers, float pow,
                              PolicySite site = PolicySite::ExplorationPool) {
    static constexpr Knob k_features[] = {Knob::red, Knob::ndim,  Knob::bucket,
                                          Knob::nyw,  Knob::nzw,   Knob::ntohpe};
    std::vector<float>    consts       = weights;
    consts.insert(consts.end(), centers.begin(), centers.end());
    const auto center_index = static_cast<std::uint16_t>(weights.size());
    consts.push_back(2.0f);
    const auto two_index = static_cast<std::uint16_t>(consts.size() - 1);
    consts.push_back(pow);
    const auto pow_index = static_cast<std::uint16_t>(consts.size() - 1);

    std::vector<Instr> code;
    bool               first = true;
    const std::size_t  n     = std::min(weights.size(), centers.size());
    for (std::uint16_t i = 0; i < n; ++i) {
        if (weights[i] == 0.0f)
            continue;
        const auto center_slot = static_cast<std::uint16_t>(center_index + i);
        // |feature - center|
        code.push_back(Instr{Op::LoadKnob, static_cast<std::uint16_t>(k_features[i])});
        code.push_back(Instr{Op::LoadConst, center_slot});
        code.push_back(Instr{Op::Sub, 0});
        code.push_back(Instr{Op::Abs, 0});
        code.push_back(Instr{Op::LoadConst, pow_index});
        code.push_back(Instr{Op::Pow, 0});
        code.push_back(Instr{Op::LoadConst, i});
        code.push_back(Instr{Op::Mul, 0});
        if (!first)
            code.push_back(Instr{Op::Add, 0});
        first = false;
    }
    if (code.empty())
        code.push_back(Instr{Op::LoadConst, 0});
    return PolicyProgram(std::move(code), consts, 0, site);
}

// A PolicyScores holding the given exploration program and a reduction-greedy
// finalization, matching what the old default-constructed scores did.
PolicyScores scores_with_exploration(PolicyProgram exploration) {
    return PolicyScores{std::move(exploration), linear_program({1.0f}, PolicySite::Finalization), {}};
}

// Evaluates a program the way the engine does for one candidate.
float score_candidate(const PolicyProgram& program, const Candidate& cand, float bn, float dn, float wvwn) {
    KnobFrame frame{};
    (void)bn;
    frame.dn   = dn;
    frame.wvwn = wvwn;
    PolicyScorer scorer{&program, {}, &frame};
    return scorer.evaluate(cand);
}

float score_features(const PolicyProgram& program, Int reduction, Int basis_dim, Int bucket_size, Int vec_weight,
                     Int z_weight, Int z_size, float bn, float dn, float wvwn) {
    Candidate cand;
    cand.reduction   = reduction;
    cand.basis_dim   = basis_dim;
    cand.bucket_size = bucket_size;
    cand.vec_weight  = vec_weight;
    cand.z_weight    = z_weight;
    cand.z_size      = z_size;
    return score_candidate(program, cand, bn, dn, wvwn);
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


void check_scored_count_selection() {
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

    // The knob normalization these constants pin is the contract; only the
    // mechanism that evaluates it changed.
    const PolicyProgram score = linear_program({2.0f, -1.0f, 0.5f, 3.0f, -4.0f});

    const float direct = score_features(score, 6, 3, 4, 5, 2, 8, 10.0f, 12.0f, 20.0f);
    require(std::abs(direct - 13.5f) < 1e-7f, "exploration raw feature scoring changed");

    ExplorationScorer scorer;
    KnobFrame         frame{};
    frame.dn      = 12.0f;
    frame.wvwn    = 20.0f;
    scorer.program = &score;
    scorer.frame   = &frame;
    const auto [pooled, unused] = scorer(cand);
    (void)unused;
    require(std::abs(direct - pooled) < 1e-7f && std::abs(direct - cand.pool_score) < 1e-7f,
            "exploration feature scoring diverged from candidate scoring");

    // Centers are expressed in exact rank-reduction counts.
    const PolicyProgram raw_reduction_center =
        polynom_program({-4.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.8f, 0.0f, 0.0f, 0.0f, 0.0f}, 1.0f);
    const float raw_center_score = score_features(raw_reduction_center, 16, 0, 0, 0, 0, 1, 10.0f, 1.0f, 1.0f);
    require(std::abs(raw_center_score - (-60.8f)) < 1e-6f,
            "polynomial reduction center must be subtracted in raw units");

    Candidate final_candidate;
    final_candidate.reduction = 16;
    final_candidate.z_size    = 1;
    const PolicyProgram raw_final_center =
        polynom_program({-4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 1.0f,
                        PolicySite::Finalization);
    const float raw_final_score = score_candidate(raw_final_center, final_candidate, 10.0f, 1.0f, 1.0f);
    require(std::abs(raw_final_score - (-60.8f)) < 1e-6f,
            "final polynomial reduction center must be subtracted in raw units");
}

void check_tohpe_reduction_target_band() {
    // A wider fixture than the score-aware one: the band is only meaningful
    // when several distinct reductions are reachable from one y.
    const Matrix P    = Matrix::from_npy(data_path("init_npy/other/barenco_tof_4.qc.matrix.npy").string());
    auto         data = std::make_shared<MatrixWithData>(P, false);
    require(data->tohpe_basis().rows() > 0, "target band fixture has an empty TOHPE basis");
    const Row      y(data->tohpe_basis()[0]);
    TohpeGenerator generator(data);

    // Unbounded: the original reduction-descending order.
    std::vector<TohpeZInfo> unbounded;
    generator.best_z_n_details_into(y, 32, unbounded);
    require(!unbounded.empty(), "target band fixture produced no z candidates");
    for (std::size_t i = 1; i < unbounded.size(); ++i)
        require(unbounded[i - 1].reduction >= unbounded[i].reduction,
                "unbounded target must rank z by descending reduction");

    const index_t best = unbounded.front().reduction;
    index_t       smallest = best;
    for (const auto& info : unbounded)
        smallest = std::min(smallest, info.reduction);

    if (smallest < best) {
        // Targeting the smallest reduction present must pull it to the front,
        // ahead of the larger ones the default order would have preferred.
        std::vector<TohpeZInfo> banded;
        generator.best_z_n_details_into(y, 32, banded, TohpeRedTarget{smallest, smallest});
        require(!banded.empty(), "banded target produced no z candidates");
        require(banded.front().reduction == smallest,
                "a target band must rank its own reduction ahead of a larger one");
    }

    // Outside the band the nearest reduction wins, so a band strictly above
    // every candidate falls back to the largest available.
    std::vector<TohpeZInfo> above;
    generator.best_z_n_details_into(y, 32, above, TohpeRedTarget{best + 10, best + 10});
    require(!above.empty() && above.front().reduction == best,
            "an unreachable band must fall back to the closest reduction");

    // An unbounded band must reproduce the default ordering exactly, so the
    // feature costs nothing when unused.
    std::vector<TohpeZInfo> explicit_unbounded;
    generator.best_z_n_details_into(y, 32, explicit_unbounded, TohpeRedTarget{});
    require(explicit_unbounded.size() == unbounded.size(),
            "an unbounded band changed the candidate count");
    for (std::size_t i = 0; i < unbounded.size(); ++i)
        require(explicit_unbounded[i].bucket_id == unbounded[i].bucket_id &&
                    explicit_unbounded[i].reduction == unbounded[i].reduction,
                "an unbounded band changed the default z order");
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
    cfg.pool = ActionPool{256};
    cfg.tohpe = TohpeSearch{SamplingBudget{k_all_one_hot_samples, 32, 64, 3}, SourcePool{256, 0}, 8};
    cfg.tohpeprefix = TohpePrefixSearch{
        SamplingBudget{k_all_one_hot_samples, 32, 64, 3}, SourcePool{256, 0}, 256,
        ZBucketSearch{32, 0, 0.0f, 0.0f, 128}};
    cfg.todd = ToddSearch{SamplingBudget{k_all_one_hot_samples, 32, 64, 3}, SourcePool{256, 0}, 256,
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
    require(chosen.reduction == 2 && first.stats.mean_reduction == 2.0f,
            "noncanonical input must report the actual four-to-two row reduction for every source");
    require(first.stats.accepted > 1 && first.stats.accepted < 256,
            "collision fixture must admit all generated candidates without truncation");
    require(std::abs(chosen.final_score - first.stats.mean_reduction) < 1e-6f,
            "equivalent parity states should retain the winning representative score");
    require(chosen.pool_size == 1 && chosen.pool_tohpe_size == 1 && chosen.pool_tohpeprefix_size == 0 &&
                chosen.pool_todd_size == 0,
            "final pool metadata should describe unique state representatives");

    const auto second = policy_iteration_impl(data, cfg, 123, 0);
    require(second.chosen.size() == 1 && second.states.size() == 1 && second.states[0] == first.states[0] &&
                second.chosen[0].final_score == chosen.final_score && second.chosen[0].source == chosen.source,
            "equivalent-state finalization should remain fixed-seed repeatable");

    // A nonlinear source score plus pool metadata checks that the retained
    // exploration-score winner is scored using the settled unique pool.
    cfg.scores.final = PolicyProgram({
        {Op::LoadKnob, static_cast<std::uint16_t>(Knob::source)},
        {Op::LoadKnob, static_cast<std::uint16_t>(Knob::source)}, {Op::Mul, 0},
        {Op::LoadKnob, static_cast<std::uint16_t>(Knob::pool_size)},
        {Op::LoadConst, 0}, {Op::Mul, 0}, {Op::Add, 0}}, {10.0f}, 0, PolicySite::Finalization);
    cfg.scores.exploration = PolicyProgram(
        {{Op::LoadKnob, static_cast<std::uint16_t>(Knob::source)}}, {}, 0, PolicySite::ExplorationPool);
    const auto mixed = policy_iteration_impl(data, cfg, 123, 0);
    const auto& stats = mixed.stats;
    require(mixed.chosen.size() == 1 && stats.accepted_tohpe > 0 &&
                stats.accepted_tohpeprefix + stats.accepted_todd > 0,
            "collision fixture must exercise multiple sources");
    const auto& mixed_chosen = mixed.chosen[0];
    require(mixed_chosen.source == CandidateSourceTodd,
            "equivalent states must retain the candidate with the greatest exploration score");
    const double expected_score = 10.0 + static_cast<double>(mixed_chosen.source * mixed_chosen.source);
    require(std::abs(mixed_chosen.final_score - expected_score) < 1e-6,
            "collisions must score only the retained exploration-score winner");
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

    seen.clear();
    index_t prefix_count = 0;
    rng.for_each_capped_bitvector_regions(
        7, 3, {0, 7, 0}, 2, {0, 120, 7}, 2,
        [&](RowCView coefs) {
            ++prefix_count;
            require(coefs.size() == 7 && coefs.find_next(2) == Row::npos,
                    "prefix saturation escaped the prefix region");
            require(seen.emplace(coefs).second, "prefix saturation emitted a duplicate");
        },
        [&](RowCView coefs) {
            require(seen.emplace(coefs).second, "full saturation repeated a prefix vector");
        });
    require(prefix_count == 7 && seen.size() == 127, "two-region sparse saturation must cover both spaces");
}

void check_sampling_coverage_and_uniqueness() {
    for (index_t dim = 0; dim <= 8; ++dim) {
        const index_t universe = (index_t{1} << dim) - 1;
        const std::array<std::array<index_t, 3>, 5> budgets{{
            {0, universe, 0}, {0, universe / 2, universe - universe / 2},
            {dim, universe + 1, universe + 1}, {0, 0, universe}, {0, 0, 0}}};
        for (const auto& caps : budgets) {
            for (index_t seed = 1; seed <= 4; ++seed) {
                PyRNG rng(seed);
                std::unordered_set<Row, RowHash, RowEq> seen;
                rng.for_each_capped_bitvector(dim, caps, 2, [&](RowCView coefs, const char* src) {
                    require(!coefs.none(), "sampler emitted the zero vector");
                    require(seen.emplace(coefs).second, "sampling phases emitted a duplicate");
                    require(std::string(src) == "oh" || std::string(src) == "sparse" ||
                                std::string(src) == "dense", "sampler lost provenance");
                });
                const bool empty = caps == std::array<index_t, 3>{0, 0, 0};
                require(seen.size() == (empty ? 0 : universe),
                        "saturated sparse+dense budget must cover the full nonzero space");
            }
        }
    }

    // Below saturation, sparse samples stay in their requested weight range;
    // dense samples must fill their quota without duplicating either phase.
    PyRNG rng(41);
    std::unordered_set<Row, RowHash, RowEq> seen;
    std::array<index_t, 3> counts{};
    rng.for_each_capped_bitvector(8, {3, 20, 30}, 2, [&](RowCView coefs, const char* src) {
        require(seen.emplace(coefs).second, "mixed sampling emitted a duplicate");
        if (std::string(src) == "oh") {
            ++counts[0];
            require(coefs.count() == 1, "one-hot sample has the wrong weight");
        } else if (std::string(src) == "sparse") {
            ++counts[1];
            require(coefs.count() == 2, "sparse sample has the wrong weight");
        } else {
            ++counts[2];
        }
    });
    require(counts == std::array<index_t, 3>{3, 20, 30}, "mixed sampling underfilled its quotas");

    // The old fallback stopped at dimension 20. Unequal weight-class sizes
    // leave unseen weight-three vectors after the bounded random attempts.
    for (index_t dim : {index_t{21}, index_t{65}}) {
        const index_t pairs = dim * (dim - 1) / 2;
        const index_t triples = dim * (dim - 1) * (dim - 2) / 6;
        seen.clear();
        rng.for_each_capped_bitvector(dim, {0, pairs + triples, 0}, 3, [&](RowCView coefs) {
            require(coefs.count() == 2 || coefs.count() == 3, "sparse fallback escaped its weights");
            require(seen.emplace(coefs).second, "sparse fallback emitted a duplicate");
        });
        require(seen.size() == pairs + triples, "sparse sampling must exhaust spaces above dimension 20");
    }
}

void check_sampling_extreme_budgets() {
    const index_t max_count = std::numeric_limits<index_t>::max();
    const index_t api_max = static_cast<index_t>(std::numeric_limits<Int>::max());
    // Tiny/degenerate spaces with huge quotas must clamp before reserving or
    // adding budgets. The last case also tests overflow-safe internal addition.
    for (index_t dim : {0, 1, 2, 8, 12}) {
        const index_t universe = (index_t{1} << dim) - 1;
        const std::array<std::array<index_t, 3>, 5> budgets{{
            {0, api_max, 0}, {0, 0, api_max}, {max_count, api_max, api_max},
            {0, universe / 2, universe - universe / 2}, {0, max_count, max_count}}};
        for (const auto& caps : budgets) {
            PyRNG rng(91);
            std::unordered_set<Row, RowHash, RowEq> seen;
            rng.for_each_capped_bitvector(dim, caps, 2, [&](RowCView coefs) {
                require(coefs.size() == dim && !coefs.none(), "extreme budget emitted an invalid vector");
                require(seen.emplace(coefs).second, "extreme budget emitted a duplicate");
            });
            require(seen.size() == universe, "extreme budget missed part of the full space");
        }
    }

    // Exhaust a tractable sparse class even when the full space is enormous.
    // 243 and 729 match the observed GF32/GF64 TOHPE dimensions. Dense samples
    // must fill their quota without repeating any sparse or one-hot vector.
    for (index_t dim : {63, 64, 65, 129, 243, 729}) {
        const index_t pairs = dim * (dim - 1) / 2;
        PyRNG rng(102);
        std::unordered_set<Row, RowHash, RowEq> seen;
        std::array<index_t, 3> counts{};
        rng.for_each_capped_bitvector(dim, {max_count, pairs + 100, 257}, 2,
            [&](RowCView coefs, const char* src) {
                require(coefs.size() == dim && !coefs.none(), "wide sampler emitted an invalid vector");
                require(seen.emplace(coefs).second, "wide sampler emitted a duplicate");
                if (std::string(src) == "oh") {
                    require(coefs.count() == 1, "wide one-hot sample has wrong weight");
                    ++counts[0];
                } else if (std::string(src) == "sparse") {
                    require(coefs.count() == 2, "wide sparse sample has wrong weight");
                    ++counts[1];
                } else {
                    require(std::string(src) == "dense", "wide sampler lost provenance");
                    ++counts[2];
                }
            });
        require(counts == std::array<index_t, 3>{dim, pairs, 257},
                "wide sampler did not exhaust its sparse space or fill its dense quota");
    }
}

void check_sampling_primitives() {
    for (index_t seed = 1; seed <= 32; ++seed) {
        PyRNG rng(seed);
        for (std::uint32_t n = 0; n <= 16; ++n) {
            for (index_t k = 0; k <= n; ++k) {
                const auto zero_based = rng.floyd_sample_0n(n, k);
                const auto one_based = rng.floyd_sample_1n(n, k);
                const std::unordered_set<std::uint32_t> z(zero_based.begin(), zero_based.end());
                const std::unordered_set<std::uint32_t> o(one_based.begin(), one_based.end());
                require(z.size() == k && o.size() == k, "Floyd sampler repeated an index");
                for (auto value : z)
                    require(value < n, "zero-based sample escaped its range");
                for (auto value : o)
                    require(value >= 1 && value <= n, "one-based sample escaped its range");
            }
        }
        Matrix basis(4, 4);
        const auto special = rng.sample_special_bitvec(basis, 0, 0, 100);
        const std::unordered_set<Row, RowHash, RowEq> distinct(special.begin(), special.end());
        require(special.size() == 15 && distinct.size() == 15,
                "special sampler must cover the nonzero space without duplicates");
    }
    // Fixed seed, broad bounds: catch the former 2x bias toward all-ones when
    // a zero draw was replaced by 11 instead of being rejected and redrawn.
    PyRNG rng(777);
    std::array<index_t, 4> counts{};
    for (index_t i = 0; i < 6000; ++i) {
        const Row row = rng.sample_bitvector(2);
        ++counts[row.data()[0]];
    }
    require(counts[0] == 0, "dense sampler emitted zero");
    for (index_t i = 1; i < counts.size(); ++i)
        require(counts[i] >= 1700 && counts[i] <= 2300, "dense sampler strongly biases a nonzero vector");
}

void check_strict_population_ranks() {
    SeenValues empty;
    empty.finalize();
    require(empty.better_red(0) == 0 && empty.better_dim(0) == 0 && empty.better_score(0) == 0,
            "empty population has no better candidates");
    SeenValues values;
    values.observe(5, 3, 7);
    values.finalize();
    require(values.better_red(5) == 0 && values.better_dim(3) == 0 && values.better_score(7) == 0,
            "a single candidate must have zero rank");
    SeenValues other;
    other.observe(5, 3, 7); // tie
    other.observe(2, 1, -2);
    other.observe(9, 6, 11);
    values.merge_from(other);
    values.finalize();
    require(values.better_red(5) == 1 && values.better_dim(3) == 1 && values.better_score(7) == 1,
            "rank must count strictly better candidates, excluding ties");
    require(values.better_red(2) == 3 && values.better_dim(1) == 3 && values.better_score(-2) == 3,
            "worst candidate should have rank population_size - 1");
    require(values.better_red(-1) == 4 && values.better_dim(-1) == 4 && values.better_score(-3) == 4,
            "queries below all observations should count the whole population");
    require(values.better_red(std::numeric_limits<Int>::max()) == 0 &&
                values.better_dim(std::numeric_limits<Int>::max()) == 0 && values.better_score(11) == 0,
            "queries at or above the maximum must have zero rank");
}

void check_sampled_policy_states() {
    const Matrix input = Matrix::from_npy(data_path("init_npy/gf_mult_Vandaele_wo_ancilla/gf2^4_410.npy").string());
    const Tensor3D signature(input);
    for (int lazy_mode : {0, 20}) {
        for (bool enable_todd : {false, true}) {
            for (const char* selection : {"best", "softmax"}) {
                auto data = std::make_shared<MatrixWithData>(input, enable_todd, lazy_mode);
                PolicyConfig cfg;
                cfg.selection = ActionSelection{32, selection, 1.0f};
                cfg.pool = ActionPool{32};
                cfg.tohpe = TohpeSearch{SamplingBudget{4, 8, 8, 3}, SourcePool{32, 0}, 8};
                cfg.todd = ToddSearch{SamplingBudget{4, 8, 8, 3}, SourcePool{enable_todd ? 32 : 0, 0}, 8,
                                      ZBucketSearch{16, 16, 0.0f, 0.0f, 16}};
                const auto result = policy_iteration_impl(data, cfg, 37, 0);
                require(result.states.size() > 1 && result.chosen.size() == result.states.size(),
                        "GF4 sampling fixture should return multiple actions");
                for (std::size_t i = 0; i < result.states.size(); ++i) {
                    const auto& state = result.states[i];
                    require(Tensor3D(state) == signature, "sampled action changed the parity tensor");
                    require(state == canonical_parity_matrix(state), "final state is not canonical");
                    require(result.chosen[i].pool_size == result.states.size(),
                            "pool metadata must describe the unique pool before selection");
                    for (std::size_t j = 0; j < i; ++j)
                        require(state != result.states[j], "final pool contains duplicate states");
                }
            }
        }
    }
}

void check_policy_population_rank_normalization() {
    const Matrix input = Matrix::from_npy(data_path("init_npy/gf_mult_Vandaele_wo_ancilla/gf2^4_410.npy").string());
    const auto rank_sum = [](bool normalized) {
        const auto knobs = normalized
            ? std::array{Knob::nrank_red, Knob::nrank_dim, Knob::nrank_score}
            : std::array{Knob::rank_red, Knob::rank_dim, Knob::rank_score};
        return PolicyProgram({{Op::LoadKnob, static_cast<std::uint16_t>(knobs[0])},
                              {Op::LoadKnob, static_cast<std::uint16_t>(knobs[1])}, {Op::Add, 0},
                              {Op::LoadKnob, static_cast<std::uint16_t>(knobs[2])}, {Op::Add, 0}},
                             {}, 0, PolicySite::Finalization);
    };
    for (int mode : {0, 20}) {
        for (bool todd_enabled : {false, true}) {
            index_t population = 0;
            for (Int size : {1, 8, 32}) {
                auto data = std::make_shared<MatrixWithData>(input, todd_enabled, mode);
                PolicyConfig cfg;
                cfg.selection = ActionSelection{size, "best", 0};
                cfg.pool = ActionPool{size};
                cfg.tohpe = TohpeSearch{SamplingBudget{4, 8, 8, 3}, SourcePool{32, 0}, 8};
                cfg.todd = ToddSearch{SamplingBudget{4, 8, 8, 3}, SourcePool{todd_enabled ? 32 : 0, 0}, 8,
                                      ZBucketSearch{16, 16, 0, 0, 16}};
                cfg.scores.final = rank_sum(false);
                const auto raw = policy_iteration_impl(data, cfg, 37, 0);
                require(!raw.chosen.empty() && raw.stats.accepted > 32,
                        "rank fixture needs truncation of the generated population");
                if (population == 0) population = raw.stats.accepted;
                require(raw.stats.accepted == population, "final_size must not change rank population");

                cfg.scores.final = rank_sum(true);
                const auto norm = policy_iteration_impl(data, cfg, 37, 0);
                require(norm.states.size() == raw.states.size(), "rank normalization changed unique pool size");
                for (std::size_t i = 0; i < norm.states.size(); ++i) {
                    const auto it = std::find(raw.states.begin(), raw.states.end(), norm.states[i]);
                    require(it != raw.states.end(), "normalized pool lost a raw-ranked state");
                    const auto j = static_cast<std::size_t>(it - raw.states.begin());
                    const double expected = raw.chosen[j].final_score / double(population);
                    require(std::abs(norm.chosen[i].final_score - expected) < 1e-5,
                            "nrank denominator must include discarded and colliding candidates");
                    require(norm.chosen[i].final_score >= 0 && norm.chosen[i].final_score < 3,
                            "sum of three normalized ranks must lie in [0,3)");
                    if (!todd_enabled)
                        require(norm.chosen[i].num_better_dim == 0,
                                "constant TOHPE dimension must have zero strict dimension rank");
                }

                cfg.scores.final = PolicyProgram({{Op::LoadKnob, static_cast<std::uint16_t>(Knob::population_size)}},
                                                {}, 0, PolicySite::Finalization);
                const auto count = policy_iteration_impl(data, cfg, 37, 0);
                for (const auto& c : count.chosen)
                    require(c.final_score == population, "population_size knob must match all accepted candidates");
            }
        }
    }
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
        check_tohpe_reduction_target_band();
        check_policy_iteration_smoke();
        check_policy_iteration_repeatability();
        check_policy_iteration_merges_equivalent_parity_states();
        check_tohpe_only_policy_continues_after_todd_stops();
        check_tohpe_policy();
        check_tohpe_and_tohpeprefix_stats_merge();
        check_tohpe_continues_after_todd_pool_is_filled();
        check_todd_tohpe_prefix();
        check_tohpe_only_basis_request();
        check_tohpe_only_request_keeps_full_todd_cache_lazy();
        check_two_region_sampling();
        check_two_region_sampling_routes_sources();
        check_sampling_primitives();
        check_sampling_coverage_and_uniqueness();
        check_sampling_extreme_budgets();
        check_sampled_policy_states();
        check_strict_population_ranks();
        check_policy_population_rank_normalization();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    std::cout << "vartodd_core_regression ok\n";
    return 0;
}
