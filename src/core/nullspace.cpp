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
#include <random>
#include <chrono>

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

MatrixWithData::MatrixWithData(Matrix P, bool build_full_todd, int lazy_mode)
    : P_{std::move(P)}, tohpe_basis_{lazy_mode >= 7 ? get_tohpe_basis(P_) : build_tohpe_basis_precompute(P_)}, lazy_mode_(lazy_mode),
      can_build_full_todd_{build_full_todd} {
    if (lazy_mode_) {
        row_lookup_.reserve(P_.rows());
        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
        hash_masks_.resize(P_.cols()); for (auto& h : hash_masks_) h = (lazy_mode == 13 || lazy_mode == 16 || lazy_mode == 19) ? 0 : rng(); // Mode 13 forces collisions in tests.
        row_hashes_.resize(P_.rows());
        for (index_t i = 0; i < P_.rows(); ++i)
            for (auto bit = P_[i].find_first(); bit != RowCView::npos; bit = P_[i].find_next(bit)) row_hashes_[i] ^= hash_masks_[bit];
        for (index_t i = 0; i < P_.rows(); ++i)
            if (P_[i].none() || !row_lookup_.emplace(Row(P_[i]), i).second) lazy_mode_ = 0;
    }
    if (lazy_mode_ >= 11) {
        row_hash_heads_.reserve(P_.rows());
        row_hash_next_.resize(P_.rows(), std::numeric_limits<index_t>::max());
        for (index_t i = 0; i < P_.rows(); ++i) {
            auto [it, inserted] = row_hash_heads_.try_emplace(row_hashes_[i], i);
            if (!inserted) { row_hash_next_[i] = it->second; it->second = i; }
        }
    }
    if (!lazy_mode_) index_.emplace(P_);
#ifndef NDEBUG
    assert(rows_are_linearly_independent(tohpe_basis_));
#endif
}

const ToddIndex& MatrixWithData::index() const {
    if (!index_) index_.emplace(P_);
    return *index_;
}

void MatrixWithData::row_bucket(RowCView z, std::vector<SumEntry>& entries) const {
    const auto started = std::chrono::steady_clock::now();
    ++row_bucket_calls;
    entries.clear();
    auto single = row_lookup_.find(z);
    if (single != row_lookup_.end()) entries.emplace_back(single->second);
    if (lazy_mode_ >= 11) {
        uint64_t zh = 0;
        for (auto bit = z.find_first(); bit != RowCView::npos; bit = z.find_next(bit)) zh ^= hash_masks_[bit];
        const auto none = std::numeric_limits<index_t>::max();
        for (index_t i = 0; i < P_.rows(); ++i) {
            auto found = row_hash_heads_.find(row_hashes_[i] ^ zh);
            if (found == row_hash_heads_.end()) continue;
            for (index_t j = found->second; j != none; j = row_hash_next_[j]) {
                if (j <= i) continue;
                bool same = true;
                for (index_t k = 0; k < z.blocks(); ++k)
                    if ((P_[i].data()[k] ^ z.data()[k]) != P_[j].data()[k]) { same = false; break; }
                if (same) entries.emplace_back(i, j);
            }
        }
    } else {
    Row partner(P_.cols());
    for (index_t i = 0; i < P_.rows(); ++i) {
        assign(partner, P_[i]); partner ^= z;
        auto found = row_lookup_.find(partner);
        if (found != row_lookup_.end() && i < found->second) entries.emplace_back(i, found->second);
    }
    }
    bucket_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
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
    if (z_.size() != M_->P().cols())
        throw std::invalid_argument("Witness: z.size()!=P.cols()");
    simple_entries_ = M_->has_canonical_rows() && !z_.none();
    const ToddIndex& idx = M_->index();

    std::vector<SumEntry> entries;
    if (!idx.materialize_bucket(z_, entries)) {
        return;
    }
    init_from_entries_(entries.data(), static_cast<index_t>(entries.size()));
}

Witness::Witness(std::shared_ptr<MatrixWithData> M, Row z, const SumEntry* ptr, index_t len)
    : M_{std::move(M)}, z_{std::move(z)} {
    if (z_.size() != M_->P().cols())
        throw std::invalid_argument("Witness: z.size()!=P.cols()");
    simple_entries_ = M_->has_canonical_rows() && !z_.none();
    init_from_entries_(ptr, len);
}

void Witness::init_from_entries_(const SumEntry* ptr, index_t len) {
    if (ptr == nullptr || len == 0)
        return;

    const Matrix& P = M_->P();
    const index_t m = P.rows();
    pairs_.reserve(std::min(static_cast<std::size_t>(len), static_cast<std::size_t>(m / 2)));
    pair_endpoints_ = Row(m);

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
    if (y.size() != M_->P().rows())
        throw std::invalid_argument("rank_divergence: y.size()!=P.rows()");
    if (!simple_entries_)
        return exact_rank_divergence(M_->P(), z_, y);

    // For distinct nonzero input rows and z!=0, translation by z partitions
    // rows into disjoint pairs {p,p^z}. A pair cancels iff its y bits differ,
    // contributing 2. Let p=parity(y), S=[z occurs], t=y at that row (0 if
    // absent). The special orbit {0,z} contributes t for p=0, or 2*S-t-1
    // for p=1: remove/retain z and account for the parity-dependent extra z.
    // Other orbits cannot collide. Noncanonical inputs and z=0 use exact count.
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
    return make(M_->P()[row]);
}

NullSpace TohpeGenerator::make(index_t row1, index_t row2) const {
    Row z(M_->P()[row1]);
    z ^= M_->P()[row2];
    return make(z.cview());
}

NullSpace TohpeGenerator::make(RowCView z) const {
    if (!M_->has_index()) {
        auto cached = topk_cache_.find(z);
        if (cached != topk_cache_.end())
            return NullSpace(M_, std::make_unique<TohpeWitness>(M_, z, cached->second.data(), cached->second.size()));
        std::vector<SumEntry> entries;
        M_->row_bucket(z, entries);
        return NullSpace(M_, std::make_unique<TohpeWitness>(M_, z, entries.data(), entries.size()));
    }
    return NullSpace(M_, std::make_unique<TohpeWitness>(M_, z));
}

NullSpace TohpeGenerator::make(RowCView z, std::uint32_t bucket_id) const {
    if (!M_->has_index() || bucket_id == std::numeric_limits<std::uint32_t>::max())
        return make(z);
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
                                           std::vector<TohpeZInfo>& scratch_out,
                                           const TohpeRedTarget& target) const {
    if (y.size() != M_->P().rows()) throw std::invalid_argument("TOHPE coefficient width mismatch");
    if (!M_->has_canonical_rows()) {
        // The per-pair accumulator assumes unique nonzero rows. Repeated rows
        // can cancel even in untouched buckets; evaluate every existing key.
        scratch_out.clear();
        if (num_samples == 0) return;
        const auto& idx = M_->index();
        for (index_t id = 0; id < idx.buckets_num(); ++id) {
            const auto bucket_id = static_cast<std::uint32_t>(id);
            Row z = idx.key_of(bucket_id);
            const int red = exact_rank_divergence(M_->P(), z, y);
            if (red > 0)
                scratch_out.push_back({std::move(z), static_cast<index_t>(red), idx.bucket_size(bucket_id), bucket_id});
        }
        const auto better = [&target](const TohpeZInfo& a, const TohpeZInfo& b) {
            const auto da = target.distance(a.reduction), db = target.distance(b.reduction);
            if (da != db) return da < db;
            if (a.reduction != b.reduction) return a.reduction > b.reduction;
            return a.bucket_id < b.bucket_id;
        };
        const auto count = std::min<std::size_t>(num_samples, scratch_out.size());
        if (count < scratch_out.size()) {
            std::nth_element(scratch_out.begin(), scratch_out.begin() + count, scratch_out.end(), better);
            scratch_out.resize(count);
        }
        std::sort(scratch_out.begin(), scratch_out.end(), better);
        return;
    }
    if (!M_->has_index()) {
        const auto started = std::chrono::steady_clock::now();
        if (M_->lazy_mode() >= 4) fingerprint_best_(y, num_samples, scratch_out, target);
        else lazy_best_z_(y, num_samples, scratch_out, target);
        M_->candidate_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return;
    }
    count_z_reductions_(y);
    const ToddIndex& idx = M_->index();
    ws_.argmax_n_into(num_samples, scratch_candidates_, target);

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


void TohpeGenerator::cache_topk(const std::vector<Row>& keys) const {
    if (M_->lazy_mode() >= 9) return; // Bounded cache already holds per-vector winners.
    topk_cache_.clear();
    if (M_->has_index() || (M_->lazy_mode() != 3 && M_->lazy_mode() != 5)) return;
    std::size_t bytes = 0;
    for (const auto& z : keys) {
        if (topk_cache_.contains(z)) continue;
        std::vector<SumEntry> entries; M_->row_bucket(z, entries);
        const auto added = entries.capacity() * sizeof(SumEntry) + z.blocks() * sizeof(uint64_t);
        if (bytes + added > 8 * 1024 * 1024) continue;
        bytes += added; topk_cache_.emplace(z, std::move(entries));
    }
    M_->topk_cache_bytes = bytes;
    M_->topk_cache_entries = topk_cache_.size();
}

index_t TohpeGenerator::cached_bucket_size_(RowCView z) const {
    if (M_->lazy_mode() >= 9) {
        if (auto it = topk_cache_.find(z); it != topk_cache_.end()) {
            ++M_->source_cache_hits;
            return it->second.size();
        }
    }
    std::vector<SumEntry> entries;
    M_->row_bucket(z, entries);
    const auto size = entries.size();
    const auto payload = entries.capacity() * sizeof(SumEntry) + z.blocks() * sizeof(uint64_t);
    // Payload cap plus entry cap bounds metadata as well as retained source arrays.
    if (M_->lazy_mode() >= 9 && cache_payload_ + payload <= 8 * 1024 * 1024 && topk_cache_.size() < 4096) {
        topk_cache_.emplace(Row(z), std::move(entries));
        cache_payload_ += payload;
        M_->topk_cache_bytes = cache_payload_;
        M_->topk_cache_entries = topk_cache_.size();
    }
    return size;
}

void TohpeGenerator::lazy_best_z_(RowCView y, index_t n, std::vector<TohpeZInfo>& out,
                                  const TohpeRedTarget& target) const {
    out.clear(); if (n == 0) return;
    const Matrix& P = M_->P();
    scratch_ones_.clear(); scratch_zeros_.clear();
    for (index_t i = 0; i < P.rows(); ++i)
        (y.test(i) ? scratch_ones_ : scratch_zeros_).push_back(i);
    const bool parity = y.count() & 1;
    auto evaluate = [&](auto& counts) {
        counts.clear();
        auto add = [&](RowCView key, index_t delta) {
            auto it = counts.find(key);
            if (it == counts.end()) counts.emplace(Row(key), delta - index_t(parity));
            else it->second += delta;
        };
        Row z(P.cols());
        for (auto i : scratch_ones_) {
            add(P[i], 1);
            for (auto j : scratch_zeros_) { assign(z, P[i]); z ^= P[j]; add(z.cview(), 2); }
        }
        if (parity) for (auto j : scratch_zeros_) add(P[j], 2);
        // Keep only n keys, not source lists for every bucket.
        using Pair = std::pair<Row, index_t>;
        std::vector<Pair> best;
        auto better = [&](const Pair& a, const Pair& b) {
            auto da = target.distance(a.second), db = target.distance(b.second);
            return da != db ? da < db : a.second != b.second ? a.second > b.second : a.first < b.first;
        };
        for (const auto& [key, count] : counts) {
            if (best.size() >= n) {
                auto da = target.distance(count), db = target.distance(best.front().second);
                if (da > db || (da == db && (count < best.front().second ||
                    (count == best.front().second && !(key < best.front().first))))) continue;
            }
            Pair item{key, count};
            if (best.size() < n) { best.push_back(std::move(item)); std::push_heap(best.begin(), best.end(), better); }
            else if (better(item, best.front())) { std::pop_heap(best.begin(), best.end(), better); best.back() = std::move(item); std::push_heap(best.begin(), best.end(), better); }
        }
        std::sort(best.begin(), best.end(), better);
        std::vector<SumEntry> entries;
        for (auto& item : best) {
            M_->row_bucket(item.first, entries);
            out.push_back({std::move(item.first), item.second, entries.size(), std::numeric_limits<uint32_t>::max()});
        }
    };
    if (M_->lazy_mode() == 1) {
        std::unordered_map<Row, index_t, RowHash, RowEq> counts; evaluate(counts);
    } else evaluate(dense_counts_);
}

void TohpeGenerator::fingerprint_best_(RowCView y, index_t n, std::vector<TohpeZInfo>& out,
                                         const TohpeRedTarget& target) const {
    out.clear(); if (!n) return;
    const Matrix& P = M_->P(); const auto& hashes = M_->row_hashes();
    const auto none = std::numeric_limits<index_t>::max();
    const auto no_link = std::numeric_limits<uint32_t>::max();
    const auto blocks = ceil_div64(P.cols());
    const bool parity = y.count() & 1;
    auto word = [&](index_t a, index_t b, index_t k) { return P[a].data()[k] ^ (b == none ? uint64_t(0) : P[b].data()[k]); };
    auto equal = [&](const LazyBucket& rep, index_t a, index_t b) {
        for (index_t k = 0; k < blocks; ++k) if (word(rep.a, rep.b, k) != word(a, b, k)) return false;
        return true;
    };
    scratch_ones_.clear(); scratch_zeros_.clear();
    for (index_t i = 0; i < P.rows(); ++i) (y.test(i) ? scratch_ones_ : scratch_zeros_).push_back(i);
    heads_.clear(); lazy_buckets_.clear();
    const bool flat = M_->lazy_mode() >= 17;
    if (flat) {
        const auto ones = scratch_ones_.size(), zeros = scratch_zeros_.size();
        const auto limit = std::numeric_limits<std::size_t>::max();
        if (zeros && ones > (limit - P.rows()) / zeros) throw std::overflow_error("flat table size overflow");
        const auto bound = ones * zeros + P.rows();
        if (bound > limit / 2) throw std::overflow_error("flat table capacity overflow");
        std::size_t capacity = 2;
        while (capacity < 2 * bound) {
            if (capacity > limit / 2) throw std::overflow_error("flat table capacity overflow");
            capacity *= 2;
        }
        flat_keys_.resize(capacity);
        flat_heads_.resize(capacity);
        std::fill(flat_heads_.begin(), flat_heads_.end(), no_link);
    }
    auto add = [&](index_t a, index_t b, index_t delta) {
        const auto h = hashes[a] ^ (b == none ? uint64_t(0) : hashes[b]);
        if (flat) {
            const auto mask = flat_heads_.size() - 1;
            auto slot = static_cast<std::size_t>(h) & mask;
            while (flat_heads_[slot] != no_link && flat_keys_[slot] != h) slot = (slot + 1) & mask;
            auto& head = flat_heads_[slot];
            for (auto id = head; id != no_link; id = lazy_buckets_[id].next)
                if (equal(lazy_buckets_[id], a, b)) { lazy_buckets_[id].count += delta; return; }
            if (lazy_buckets_.size() >= no_link) throw std::overflow_error("lazy bucket count overflow");
            auto id = static_cast<uint32_t>(lazy_buckets_.size());
            lazy_buckets_.push_back({a,b,delta-index_t(parity),head});
            flat_keys_[slot] = h; head = id;
            return;
        }
        const bool single_probe = M_->lazy_mode() >= 14;
        auto [found, existed] = [&] {
            if (single_probe) {
                auto [it, inserted] = heads_.try_emplace(h, no_link);
                return std::pair{it, !inserted};
            }
            auto it = heads_.find(h);
            return std::pair{it, it != heads_.end()};
        }();
        if (existed) {
            for (auto id = found->second; id != no_link; id = lazy_buckets_[id].next)
                if (equal(lazy_buckets_[id], a, b)) { lazy_buckets_[id].count += delta; return; }
        }
        if (lazy_buckets_.size() >= no_link) throw std::overflow_error("lazy bucket count overflow");
        auto id = static_cast<uint32_t>(lazy_buckets_.size());
        lazy_buckets_.push_back({a,b,delta-index_t(parity),existed ? found->second : no_link});
        if (!existed && !single_probe) heads_.emplace(h,id); else found->second = id;
    };
    for (auto i : scratch_ones_) {
        add(i,none,1);
        for (auto j : scratch_zeros_) add(i,j,2);
    }
    if (parity) for (auto j : scratch_zeros_) add(j,none,2);
    auto better = [&](uint32_t a, uint32_t b) {
        const auto& x=lazy_buckets_[a]; const auto& v=lazy_buckets_[b];
        auto dx=target.distance(x.count),dv=target.distance(v.count);
        if(dx!=dv)return dx<dv;
        if(x.count!=v.count)return x.count>v.count;
        for(index_t k=0;k<blocks;++k){auto wx=word(x.a,x.b,k),wv=word(v.a,v.b,k);if(wx!=wv)return wx<wv;}
        return false;
    };
    std::vector<uint32_t> best;
    for (uint32_t i=0;i<lazy_buckets_.size();++i) {
        if(best.size()<n){best.push_back(i);std::push_heap(best.begin(),best.end(),better);}
        else if(better(i,best.front())){std::pop_heap(best.begin(),best.end(),better);best.back()=i;std::push_heap(best.begin(),best.end(),better);}
    }
    std::sort(best.begin(),best.end(),better);
    std::vector<SumEntry> entries;
    for(auto id:best){const auto& b=lazy_buckets_[id];Row z(P[b.a]);if(b.b!=none)z^=P[b.b];
        auto size=cached_bucket_size_(z);out.push_back({std::move(z),b.count,size,no_link});}
}

FullToddGenerator::FullToddGenerator(std::shared_ptr<MatrixWithData> M) : M_{std::move(M)} {}

GeneratedSolutionBasis FullToddGenerator::tohpe_prefix_kernel(RowCView z, const SumEntry* ptr, index_t len) const {
    (void)z;
    static thread_local PivotMap pivY;
    Matrix basis = detail::build_transformed_tohpe_prefix(M_->tohpe_basis(), ptr, len, pivY, M_->P().rows());
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
