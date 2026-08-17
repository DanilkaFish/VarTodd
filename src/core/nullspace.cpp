#include "nullspace.hpp"

#include "algorithms.hpp"
#include "matrix.hpp"
#include "todd_index.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace todd {

namespace {

static void or_shifted(RowView dst, index_t offset, RowCView src) noexcept;

static std::uint32_t checked_u32_metadata(index_t value, const char* what) {
    if (value >= static_cast<index_t>(MatrixWithData::FullToddData::npos))
        throw std::overflow_error(what);
    return static_cast<std::uint32_t>(value);
}

static index_t checked_index_add(index_t a, index_t b, const char* what) {
    if (b > std::numeric_limits<index_t>::max() - a)
        throw std::overflow_error(what);
    return a + b;
}

static std::uint64_t checked_u64_add(std::uint64_t a, std::uint64_t b, const char* what) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error(what);
    return a + b;
}

static index_t checked_index_from_u64(std::uint64_t value, const char* what) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<index_t>::max()))
        throw std::overflow_error(what);
    return static_cast<index_t>(value);
}

static int checked_rank_delta(unsigned __int128 positive, unsigned __int128 negative, const char* what) {
    const auto int_max = static_cast<unsigned __int128>(std::numeric_limits<int>::max());
    if (positive >= negative) {
        const auto diff = positive - negative;
        if (diff > int_max)
            throw std::overflow_error(what);
        return static_cast<int>(diff);
    }

    const auto diff = negative - positive;
    if (diff > int_max)
        throw std::overflow_error(what);
    return -static_cast<int>(diff);
}

static std::vector<Row> transformed_rows_canonical(const Matrix& P, RowCView z, RowCView y) {
    struct RowEntry {
        Row         row;
        std::size_t order;
    };

    std::vector<RowEntry> rows;
    rows.reserve(checked_index_add(P.rows(), 1, "transformed row count overflow"));

    for (index_t i = 0; i < P.rows(); ++i) {
        Row row = y.test(i) ? (P[i] ^ z) : Row(P[i]);
        if (!row.none())
            rows.push_back({std::move(row), static_cast<std::size_t>(i)});
    }
    if ((y.count() & 1u) != 0 && !z.none())
        rows.push_back({Row(z), static_cast<std::size_t>(P.rows())});

    std::ranges::sort(rows, [](const RowEntry& lhs, const RowEntry& rhs) { return (lhs.row <=> rhs.row) < 0; });

    std::vector<RowEntry> kept;
    kept.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size();) {
        std::size_t j = i + 1;
        while (j < rows.size() && rows[i].row == rows[j].row)
            ++j;
        if (((j - i) & 1u) != 0) {
            auto first = rows.begin() + static_cast<std::ptrdiff_t>(i);
            auto last  = rows.begin() + static_cast<std::ptrdiff_t>(j);
            auto best  = std::min_element(first, last, [](const RowEntry& lhs, const RowEntry& rhs) {
                return lhs.order < rhs.order;
            });
            kept.push_back({std::move(best->row), best->order});
        }
        i = j;
    }

    std::ranges::sort(kept, [](const RowEntry& lhs, const RowEntry& rhs) { return lhs.order < rhs.order; });

    std::vector<Row> out;
    out.reserve(kept.size());
    for (auto& entry : kept)
        out.push_back(std::move(entry.row));
    return out;
}

static Matrix exact_apply(const Matrix& P, RowCView z, RowCView y) {
    auto rows = transformed_rows_canonical(P, z, y);
    Matrix out(0, P.cols());
    out.reserve_rows(static_cast<index_t>(rows.size()));
    for (const Row& row : rows)
        out.push_back(row);
    return out;
}

static int exact_rank_divergence(const Matrix& P, RowCView z, RowCView y) {
    const auto rows = transformed_rows_canonical(P, z, y);
    return checked_rank_delta(static_cast<unsigned __int128>(P.rows()),
                              static_cast<unsigned __int128>(rows.size()),
                              "rank divergence exceeds int range");
}

#ifndef NDEBUG
static bool rows_are_canonical(const Matrix& rows) {
    std::vector<Row> seen;
    seen.reserve(rows.rows());
    for (index_t i = 0; i < rows.rows(); ++i) {
        if (rows[i].none())
            return false;
        seen.emplace_back(rows[i]);
    }
    std::ranges::sort(seen);
    for (std::size_t i = 1; i < seen.size(); ++i) {
        if (seen[i - 1] == seen[i])
            return false;
    }
    return true;
}

static bool rows_are_linearly_independent(const Matrix& rows) {
    return basis_gauss_elimination(Matrix(rows)).rows() == rows.rows();
}
#endif

static auto build_tohpe_basis_precompute(const Matrix& P) -> Matrix {
    if (P.rows() == 0) {
        return Matrix(0, 0);
    }

    Matrix   L = L_expansion(P);
    Matrix   Y = Matrix::identity(P.rows());
    PivotMap pivots;
    pivots.reset(L.cols());
    gauss_elimination_inplace_rref(L, Y, pivots);

    Matrix tohpe_basis = extract_basis(Y, pivots);
#ifndef NDEBUG
    assert(tohpe_basis == get_tohpe_basis(P));
    assert(rows_are_linearly_independent(tohpe_basis));
#endif
    return tohpe_basis;
}

// Build both TOHPE and full-TODD data from one elimination pass over L(P).
static auto build_matrix_with_data_precompute(const Matrix& P)
    -> std::pair<Matrix, MatrixWithData::FullToddData> {
    if (P.rows() == 0) {
        return {Matrix(0, 0), MatrixWithData::FullToddData{}};
    }

    Matrix   L = L_expansion(P);
    Matrix   Y = Matrix::identity(P.rows());
    PivotMap pivots;
    pivots.reset(L.cols());
    gauss_elimination_inplace_rref(L, Y, pivots);

    Matrix tohpe_basis = extract_basis(Y, pivots);
#ifndef NDEBUG
    assert(tohpe_basis == get_tohpe_basis(P));
    assert(rows_are_linearly_independent(tohpe_basis));
#endif

    MatrixWithData::FullToddData out;
    const index_t                n         = P.cols();
    const index_t                full_cols = L.cols();

    out.pivot_row_of_col.assign(static_cast<std::size_t>(full_cols), MatrixWithData::FullToddData::npos);
    std::vector<index_t> pivot_source_rows;
    pivot_source_rows.reserve(pivots.size());
    for (index_t pivot_col = 0; pivot_col < full_cols; ++pivot_col) {
        auto row = pivots.get(pivot_col);
        if (row != PivotMap::npos) {
            out.pivot_row_of_col[static_cast<std::size_t>(pivot_col)] =
                checked_u32_metadata(static_cast<index_t>(pivot_source_rows.size()),
                                     "FullToddData pivot metadata overflow");
            pivot_source_rows.push_back(static_cast<index_t>(row));
        }
    }

    out.offset.assign(static_cast<std::size_t>(n), 0);
    if (n > 0) {
        out.offset[static_cast<std::size_t>(n - 1)] = 0;
        for (index_t b = n - 1; b > 0; --b) {
            out.offset[static_cast<std::size_t>(b - 1)] =
                checked_u64_add(out.offset[static_cast<std::size_t>(b)],
                                static_cast<std::uint64_t>(checked_index_add(b, 1, "FullToddData offset overflow")),
                                "FullToddData offset overflow");
        }
        const auto max_extra = static_cast<std::uint64_t>(n - 1);
        const auto max_index = static_cast<std::uint64_t>(std::numeric_limits<index_t>::max());
        for (const auto off : out.offset) {
            if (off > max_index - max_extra)
                throw std::overflow_error("FullToddData offset overflow");
        }
    }

    out.nonpivot_index.assign(static_cast<std::size_t>(full_cols), MatrixWithData::FullToddData::npos);
    index_t nonpiv_cols = 0;
    for (index_t col = 0; col < full_cols; ++col) {
        if (out.pivot_row_of_col[static_cast<std::size_t>(col)] == MatrixWithData::FullToddData::npos) {
            out.nonpivot_index[static_cast<std::size_t>(col)] =
                checked_u32_metadata(nonpiv_cols, "FullToddData nonpivot metadata overflow");
            ++nonpiv_cols;
        }
    }
    out.nonpiv_cols = nonpiv_cols;

    Matrix LY_nonpivot(static_cast<index_t>(pivot_source_rows.size()),
                       checked_index_add(nonpiv_cols, P.rows(), "FullToddData matrix width overflow"));
    for (index_t r = 0; r < LY_nonpivot.rows(); ++r) {
        auto       dst = LY_nonpivot[r];
        const auto lr  = L[pivot_source_rows[static_cast<std::size_t>(r)]];
        for (auto p = lr.find_first(); p != Row::npos; p = lr.find_next(p)) {
            const std::uint32_t idx = out.nonpivot_index[static_cast<std::size_t>(p)];
            if (idx != MatrixWithData::FullToddData::npos) {
                dst.set(static_cast<index_t>(idx));
            }
        }

        const auto yr = Y[pivot_source_rows[static_cast<std::size_t>(r)]];
        if (!yr.none()) {
            or_shifted(dst, nonpiv_cols, yr);
        }
    }
    out.LY_nonpivot = std::move(LY_nonpivot);
    return {std::move(tohpe_basis), std::move(out)};
}

static void or_shifted(RowView dst, index_t offset, RowCView src) noexcept {
    if (src.size() == 0)
        return;

    uint64_t*       d          = dst.data();
    const uint64_t* s          = src.data();
    const index_t   dst_blocks = dst.blocks();
    const index_t   src_blocks = src.blocks();
    const index_t   d0         = offset >> 6;
    const unsigned  sh         = static_cast<unsigned>(offset & 63);

    if (sh == 0) {
        for (index_t k = 0; k < src_blocks && d0 + k < dst_blocks; ++k)
            d[d0 + k] |= s[k];
    } else {
        for (index_t k = 0; k < src_blocks && d0 + k < dst_blocks; ++k) {
            const uint64_t w = s[k];
            d[d0 + k] |= (w << sh);
            if (d0 + k + 1 < dst_blocks)
                d[d0 + k + 1] |= (w >> (64u - sh));
        }
    }

    if (dst_blocks != 0)
        d[dst_blocks - 1] &= tail_mask_bits(dst.size());
}

} // namespace

MatrixWithData::MatrixWithData(Matrix P, bool build_full_todd)
    : P_{std::move(P)}, tohpe_basis_{build_tohpe_basis_precompute(P_)}, index_{P_},
      can_build_full_todd_{build_full_todd} {
#ifndef NDEBUG
    assert(rows_are_linearly_independent(tohpe_basis_));
#endif
}

const MatrixWithData::FullToddData& MatrixWithData::full_todd() const {
    if (!can_build_full_todd_)
        throw std::runtime_error("MatrixWithData was constructed without full TODD data");
    if (!full_todd_) {
        auto precomputed = build_matrix_with_data_precompute(P_);
#ifndef NDEBUG
        assert(precomputed.first == tohpe_basis_);
#endif
        full_todd_ = std::move(precomputed.second);
    }
    return *full_todd_;
}

Witness::Witness(std::shared_ptr<MatrixWithData> M, Row z) : M_{std::move(M)}, z_{std::move(z)} {
    const ToddIndex& idx = M_->index();

    std::vector<SumEntry> entries;
    if (!idx.materialize_bucket(z_, entries)) {
        return;
    }
    init_from_entries_(entries.data(), static_cast<index_t>(entries.size()));
}

Witness::Witness(std::shared_ptr<MatrixWithData> M, Row z, const SumEntry* ptr, index_t len)
    : M_{std::move(M)}, z_{std::move(z)} {
    init_from_entries_(ptr, len);
}

void Witness::init_from_entries_(const SumEntry* ptr, index_t len) {
    if (ptr == nullptr || len == 0)
        return;

    const Matrix& P = M_->P();
    const index_t m = P.rows();
    pairs_.reserve(std::min(static_cast<std::size_t>(len), static_cast<std::size_t>(m / 2)));
    pair_endpoints_ = Row(m);
    simple_entries_ = true;

    for (index_t t = 0; t < len; ++t) {
        if (ptr[t].is_pair())
            continue;
        const index_t a = ptr[t].a;
        if (special_ == k_single_sentinel<index_t>()) {
            special_ = a;
        } else {
            simple_entries_ = false;
        }
        if (pair_endpoints_.test(a))
            simple_entries_ = false;
        pair_endpoints_.set(a);
    }

    for (index_t t = 0; t < len; ++t) {
        if (!ptr[t].is_pair())
            continue;
        const index_t a = ptr[t].a;
        const index_t b = ptr[t].b;
        if (P[a].none() || P[b].none())
            simple_entries_ = false;
        if (pair_endpoints_.test(a) || pair_endpoints_.test(b))
            simple_entries_ = false;
        pair_endpoints_.set(a);
        pair_endpoints_.set(b);
        pairs_.emplace_back(a, b);
    }
}

TohpeWitness::TohpeWitness(std::shared_ptr<MatrixWithData> M, Row z) : Witness(std::move(M), std::move(z)) {}
TohpeWitness::TohpeWitness(std::shared_ptr<MatrixWithData> M, Row z, const SumEntry* ptr, index_t len)
    : Witness(std::move(M), std::move(z), ptr, len) {}

ToddWitness::ToddWitness(std::shared_ptr<MatrixWithData> M, Row z, Matrix&& Y)
    : Witness(std::move(M), std::move(z)), Y_{std::move(Y)} {}

ToddWitness::ToddWitness(std::shared_ptr<MatrixWithData> M, Row z, const SumEntry* ptr, index_t len, Matrix&& Y)
    : Witness(std::move(M), std::move(z), ptr, len), Y_{std::move(Y)} {}

int Witness::rank_divergence(RowCView y) const {
    if (!simple_entries_)
        return exact_rank_divergence(M_->P(), z_, y);

    const bool parity = (y.count() & 1u) != 0;
    unsigned __int128 ones_S = 0;
    const bool has_special = special_ != k_single_sentinel<index_t>();
    const unsigned __int128 S = has_special ? 1 : 0;
    if (has_special && y.test(special_))
        ++ones_S;
    unsigned __int128 diff_pairs = 0;
    for (const auto& pr : pairs_) {
        const index_t i = pr.first;
        const index_t j = pr.second;
        if (y.test(i) ^ y.test(j))
            ++diff_pairs;
    }
    if (!parity)
        return checked_rank_delta(ones_S + 2 * diff_pairs, 0, "rank divergence exceeds int range");
    return checked_rank_delta(2 * S + 2 * diff_pairs, ones_S + 1, "rank divergence exceeds int range");
}

Row NullSpace::linear_combination(RowCView coefs) const {
    if (coefs.size() > basis().rows()) {
        throw std::runtime_error("Number of coefs more than basis size");
    }
    Row out(basis().cols());
    for (auto i = coefs.find_first(); i != Row::npos; i = coefs.find_next(i)) {
        if (static_cast<index_t>(i) >= basis().rows())
            break;
        out ^= basis()[static_cast<index_t>(i)];
    }
    return out;
}

Matrix NullSpace::apply(RowCView y) const {
    std::vector<std::uint8_t> scratch_killed;
    return apply(y, scratch_killed);
}

Matrix NullSpace::apply(RowCView y, std::vector<std::uint8_t>& scratch_killed) const {
    const Matrix& P0     = M_->P();
    const index_t n_rows = P0.rows();
    if (y.size() != n_rows)
        throw std::invalid_argument("apply: y.size()!=P.rows()");

    const auto    z       = vector();
    const index_t special = w_->get_special();
    const auto&   pairs   = w_->get_pairs();

    if (!w_->has_simple_entries())
        return exact_apply(P0, z, y);

    if (scratch_killed.size() < static_cast<std::size_t>(n_rows))
        scratch_killed.resize(static_cast<std::size_t>(n_rows));
    std::fill_n(scratch_killed.begin(), static_cast<std::size_t>(n_rows), std::uint8_t{0});
    index_t killed = 0;
    for (const auto& pr : pairs) {
        const index_t a = pr.first;
        const index_t b = pr.second;
        if (y.test(a) ^ y.test(b)) {
            scratch_killed[static_cast<std::size_t>(a)] = 1;
            scratch_killed[static_cast<std::size_t>(b)] = 1;
            killed += 2;
        }
    }

    const bool parity      = (y.count() & 1) != 0;
    const bool add_z       = parity && (special == k_single_sentinel<index_t>() || y.test(special));
    const bool kill_special = special != k_single_sentinel<index_t>() && (y.test(special) || parity);
    if (kill_special) {
        scratch_killed[static_cast<std::size_t>(special)] = 1;
        ++killed;
    }

    if (add_z && n_rows == std::numeric_limits<index_t>::max())
        throw std::overflow_error("apply: row count overflow");
    const auto new_rows = n_rows - killed + (add_z ? 1 : 0);
    Matrix new_P(new_rows, P0.cols());

    index_t j = 0;
    for (index_t i = 0; i < n_rows; ++i) {
        if (scratch_killed[static_cast<std::size_t>(i)])
            continue;
        auto dst = new_P[j];
        assign(dst, P0[i]);
        if (y.test(i)) {
            dst ^= z;
        }
        ++j;
    }

    if (add_z) {
        assign(new_P[j], z);
    }
    return new_P;
}

TohpeGenerator::TohpeGenerator(std::shared_ptr<MatrixWithData> M) : M_{std::move(M)} {}

NullSpace TohpeGenerator::make(index_t row) const {
    return NullSpace(M_, std::make_unique<TohpeWitness>(M_, M_->P()[row]));
}

NullSpace TohpeGenerator::make(index_t row1, index_t row2) const {
    Row z(M_->P()[row1]);
    z ^= M_->P()[row2];
    return NullSpace(M_, std::make_unique<TohpeWitness>(M_, std::move(z)));
}

NullSpace TohpeGenerator::make(RowCView z) const {
    return NullSpace(M_, std::make_unique<TohpeWitness>(M_, z));
}

NullSpace TohpeGenerator::make(RowCView z, std::uint32_t bucket_id) const {
    std::vector<SumEntry> entries;
    M_->index().materialize_bucket(bucket_id, entries);
    return NullSpace(M_, std::make_unique<TohpeWitness>(
                             M_, z, entries.data(), static_cast<index_t>(entries.size())));
}

Row TohpeGenerator::best_z(RowCView y) const {
    const Matrix&    P   = M_->P();
    best_z_n_details_into(y, 1, scratch_z_infos_);
    if (scratch_z_infos_.empty())
        return Row(P.cols());
    return Row(scratch_z_infos_.front().z);
}

std::vector<std::pair<Row, index_t>> TohpeGenerator::best_z_n(RowCView y, index_t num_samples) const {
    std::vector<std::pair<Row, index_t>> out;
    best_z_n_into(y, num_samples, out);
    return out;
}

void TohpeGenerator::best_z_n_into(RowCView y, index_t num_samples,
                                   std::vector<std::pair<Row, index_t>>& scratch_out) const {
    best_z_n_details_into(y, num_samples, scratch_z_infos_);

    scratch_out.clear();
    scratch_out.reserve(scratch_z_infos_.size());
    for (auto& info : scratch_z_infos_)
        scratch_out.emplace_back(std::move(info.z), info.reduction);
}

void TohpeGenerator::count_z_reductions_(RowCView y) const {
    assert(y.count() != 0);
    const Matrix&    P       = M_->P();
    const ToddIndex& idx     = M_->index();
    const index_t    n       = P.rows();
    const index_t    y_count = y.count();

    if (scratch_ones_.size() < static_cast<std::size_t>(n))
        scratch_ones_.resize(static_cast<std::size_t>(n));
    if (scratch_zeros_.size() < static_cast<std::size_t>(n))
        scratch_zeros_.resize(static_cast<std::size_t>(n));

    std::size_t ones_size  = 0;
    std::size_t zeros_size = 0;
    for (index_t i = 0; i < n; ++i) {
        if (y.test(i))
            scratch_ones_[ones_size++] = i;
        else
            scratch_zeros_[zeros_size++] = i;
    }

    ws_.reset(idx.buckets_num(), idx.max_bucket(), (y_count & 1U) != 0);
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
    });
}

void TohpeGenerator::best_z_n_details_into(RowCView y, index_t num_samples,
                                           std::vector<TohpeZInfo>& scratch_out) const {
    count_z_reductions_(y);
    const ToddIndex& idx = M_->index();
    ws_.argmax_n_into(num_samples, scratch_candidates_);

    scratch_out.clear();
    scratch_out.reserve(scratch_candidates_.size());
    for (const auto& candidate : scratch_candidates_) {
        const auto bucket_id = candidate.bucket_id;
        scratch_out.push_back(TohpeZInfo{
            .z           = Row(idx.key_of(bucket_id)),
            .reduction   = candidate.count,
            .bucket_size = idx.bucket_size(bucket_id),
            .bucket_id   = bucket_id,
        });
    }
}

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
        const auto bucket_id = candidate.bucket_id;
        scratch_out.push_back(TohpeZInfo{
            .z           = idx.key_of(bucket_id),
            .reduction   = candidate.count,
            .bucket_size = idx.bucket_size(bucket_id),
            .bucket_id   = bucket_id,
        });
    }
}

FullToddGenerator::FullToddGenerator(std::shared_ptr<MatrixWithData> M) : M_{std::move(M)} {}

GeneratedSolutionBasis FullToddGenerator::tohpe_prefix_kernel(RowCView z, const SumEntry* ptr, index_t len) const {
    (void)z;
    static thread_local PivotMap pivY;
    Matrix basis = detail::build_transformed_tohpe_prefix(M_->tohpe_basis(), ptr, len, pivY);
    const index_t prefix_size = basis.rows();
    return {std::move(basis), prefix_size};
}

GeneratedSolutionBasis FullToddGenerator::solution_basis(RowCView z, const SumEntry* ptr, index_t len,
                                                          SolutionBasisRequest request) const {
    if (request == SolutionBasisRequest::TohpeOnly)
        return tohpe_prefix_kernel(z, ptr, len);
    return full_todd_kernel(z, ptr, len);
}

GeneratedSolutionBasis FullToddGenerator::full_todd_kernel(RowCView z, const SumEntry* ptr, index_t len) const {
    const auto&   ft         = M_->full_todd();
    const index_t n          = z.size();
    const index_t total_cols = checked_index_add(ft.nonpiv_cols, M_->P().rows(),
                                                 "full_todd_kernel column count overflow");

    std::vector<index_t> S;
    S.reserve(static_cast<std::size_t>(z.count()));
    for (auto i = z.find_first(); i != RowCView::npos; i = z.find_next(i)) {
        S.push_back(static_cast<index_t>(i));
    }
    auto add_col = [&](auto ra_row, index_t col) {
        const std::uint32_t prow = ft.pivot_row_of_col[static_cast<std::size_t>(col)];
        if (prow != MatrixWithData::FullToddData::npos) {
            ra_row ^= ft.LY_nonpivot[static_cast<index_t>(prow)];
        } else {
            ra_row.flip(static_cast<index_t>(ft.nonpivot_index[static_cast<std::size_t>(col)]));
        }
    };

    auto fill_row = [&](index_t row_idx, RowView row) {
        if (row_idx < n) {
            const index_t       gamma = row_idx;
            const std::uint64_t offg  = ft.offset[gamma];
            for (index_t s : S) {
                if (s == gamma)
                    continue;
                const index_t col =
                    static_cast<index_t>(s > gamma ? ft.offset[static_cast<std::size_t>(s)] + gamma : offg + s);
                add_col(row, col);
            }
        } else {
            for (std::size_t bi = 0; bi < S.size(); ++bi) {
                const index_t       b    = S[bi];
                const std::uint64_t offb = ft.offset[static_cast<std::size_t>(b)];
                for (std::size_t ai = 0; ai <= bi; ++ai) {
                    const index_t a = S[ai];
                    add_col(row, static_cast<index_t>(offb + a));
                }
            }
        }
    };

    return solve_and_build_solution_basis_generated(checked_index_add(n, 1, "full_todd_kernel row count overflow"),
                                                    total_cols, ft.nonpiv_cols, M_->tohpe_basis(), ptr, len, fill_row);
}

NullSpace FullToddGenerator::make(RowCView z) const {
    std::vector<SumEntry> entries;
    M_->index().materialize_bucket(z, entries);
    return make(z, entries.data(), static_cast<index_t>(entries.size()));
}

NullSpace FullToddGenerator::make(RowCView z, const SumEntry* ptr, index_t len) const {
    return make(z, ptr, len, SolutionBasisRequest::ToddOnly);
}

NullSpace FullToddGenerator::make(RowCView z, const SumEntry* ptr, index_t len, SolutionBasisRequest request) const {
    auto generated = solution_basis(z, ptr, len, request);
#ifndef NDEBUG
    assert(rows_are_linearly_independent(generated.basis));
#endif
    return NullSpace(M_, std::make_unique<ToddWitness>(M_, Row(z), ptr, len, std::move(generated.basis)),
                     generated.tohpe_prefix_size);
}

NullSpace FullToddGenerator::make_tohpe_prefix(RowCView z, const SumEntry* ptr, index_t len) const {
    return make(z, ptr, len, SolutionBasisRequest::TohpeOnly);
}

NullSpace FullToddGenerator::make(index_t row) const {
    Row                   z = M_->P()[row];
    std::vector<SumEntry> entries;
    M_->index().materialize_bucket(M_->index().single_id()[static_cast<std::size_t>(row)], entries);
    const auto len = static_cast<index_t>(entries.size());
    auto generated = full_todd_kernel(z, entries.data(), len);
#ifndef NDEBUG
    assert(rows_are_linearly_independent(generated.basis));
#endif
    return NullSpace(M_,
                     std::make_unique<ToddWitness>(M_, std::move(z), entries.data(), len,
                                                    std::move(generated.basis)),
                     generated.tohpe_prefix_size);
}

NullSpace FullToddGenerator::make(index_t row1, index_t row2) const {
    Row z = M_->P()[row1];
    z ^= M_->P()[row2];
    std::vector<SumEntry> entries;
    M_->index().materialize_bucket(M_->index().pair_bucket_id(row1, row2), entries);
    const auto len = static_cast<index_t>(entries.size());
    auto generated = full_todd_kernel(z, entries.data(), len);
#ifndef NDEBUG
    assert(rows_are_linearly_independent(generated.basis));
#endif
    return NullSpace(M_,
                     std::make_unique<ToddWitness>(M_, std::move(z), entries.data(), len,
                                                    std::move(generated.basis)),
                     generated.tohpe_prefix_size);
}

} // namespace todd
