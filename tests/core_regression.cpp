#include "matrix.hpp"
#include "nullspace.hpp"
#include "todd_index.hpp"
#include "todd_generator.hpp"

#include <algorithm>
#include <array>
#include <compare>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>

namespace {

using namespace todd;

void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

std::filesystem::path data_path(const char* relative) {
    return std::filesystem::path(VARTODD_TEST_DATA_DIR) / relative;
}

std::filesystem::path tmp_path(const char* filename) {
    auto dir = std::filesystem::path(VARTODD_TEST_TMP_DIR);
    std::filesystem::create_directories(dir);
    return dir / filename;
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

void check_todd_index_basics() {
    Matrix    loaded = Matrix::from_npy(data_path("init_npy/other/adder_8.qc.matrix.npy").string());
    ToddIndex index(loaded);

    require(index.rows() == loaded.rows() && index.cols() == loaded.cols(), "ToddIndex shape mismatch");
    require(index.max_bucket() > 0 && index.buckets_num() > 0, "ToddIndex should build nonempty buckets");

    const SumEntry* ptr = nullptr;
    index_t         len = 0;
    require(index.sum_bucket(loaded[0], ptr, len) && len > 0, "single-row bucket lookup failed");

    Row pair = loaded[0] ^ loaded[1];
    const auto bucket_id = index.pair_bucket_id(0, 1);
    require(index.key_of(bucket_id) == pair, "pair bucket key mismatch");
    require(index.sum_bucket(bucket_id, ptr, len) && len > 0, "pair bucket id lookup failed");

    const SumEntry* ptr_by_key = nullptr;
    index_t         len_by_key = 0;
    require(index.sum_bucket(pair, ptr_by_key, len_by_key), "pair bucket key lookup failed");
    require(ptr == ptr_by_key && len == len_by_key, "bucket id/key lookup should resolve same range");
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
    cfg.tohpe.pool               = SourcePool{4, 0};
    cfg.tohpe.sampling           = SamplingBudget{8, 8, 0, 2};
    cfg.tohpe.z_choices          = 4;
    cfg.todd.pool                = SourcePool{4, 0};
    cfg.todd.sampling            = SamplingBudget{8, 8, 0, 2};
    cfg.todd.actions_per_bucket  = 2;
    cfg.todd.buckets             = ZBucketSearch{1, 8, 0.0f, 0.0f, 8};

    auto result = policy_iteration_impl(data, cfg, 123, 0);
    require(result.chosen.size() == 1 && result.states.size() == 1, "policy iteration should choose one state");
    require(result.chosen[0].reduction > 0, "policy iteration should find a reducing candidate");
    require(result.states[0].rows() == 2 && result.states[0].cols() == 3, "policy iteration state shape mismatch");
    require(result.stats.accepted_tohpe > 0, "policy iteration should count accepted TOHPE candidates");
    require(result.stats.accepted_todd > 0, "policy iteration should count accepted Todd candidates");
    require(result.stats.accepted == result.stats.accepted_tohpe + result.stats.accepted_todd,
            "accepted total should match accepted TOHPE plus accepted Todd");
    require(result.stats.accepted == result.stats.nonzero, "accepted total should match nonzero candidate count");
}

} // namespace

int main() {
    try {
        check_row_views();
        check_matrix_basics();
        check_todd_index_basics();
        check_packed_count_storage();
        check_policy_iteration_smoke();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    std::cout << "vartodd_core_regression ok\n";
    return 0;
}
