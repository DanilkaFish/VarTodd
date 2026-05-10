#include "nullspace.hpp"

#include "algorithms.hpp"
#include "matrix.hpp"
#include "todd_index.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ranges>
#include <stdexcept>

namespace todd {

namespace {

static void or_shifted(RowView dst, index_t offset, RowCView src) noexcept;

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
#endif

    MatrixWithData::FullToddData out;
    const index_t                n         = P.cols();
    const index_t                full_cols = L.cols();

    out.pivot_row_of_col.assign((std::size_t)full_cols, -1);
    std::vector<index_t> pivot_source_rows;
    pivot_source_rows.reserve(pivots.size());
    for (index_t pivot_col = 0; pivot_col < full_cols; ++pivot_col) {
        auto row = pivots.get(pivot_col);
        if (row != PivotMap::npos) {
            out.pivot_row_of_col[(std::size_t)pivot_col] = (std::int64_t)pivot_source_rows.size();
            pivot_source_rows.push_back((index_t)row);
        }
    }

    out.offset.assign((std::size_t)n, 0);
    if (n > 0) {
        out.offset[(std::size_t)(n - 1)] = 0;
        for (index_t b = n - 1; b > 0; --b) {
            out.offset[(std::size_t)(b - 1)] = out.offset[(std::size_t)b] + (std::uint64_t)(b + 1);
        }
    }

    out.nonpivot_index.assign((std::size_t)full_cols, -1);
    index_t nonpiv_cols = 0;
    for (index_t col = 0; col < full_cols; ++col) {
        if (out.pivot_row_of_col[(std::size_t)col] < 0) {
            out.nonpivot_index[(std::size_t)col] = (std::int64_t)nonpiv_cols;
            ++nonpiv_cols;
        }
    }
    out.nonpiv_cols = nonpiv_cols;

    Matrix               LY_nonpivot((index_t)pivot_source_rows.size(), nonpiv_cols + P.rows());
    std::vector<uint8_t> is_zero(LY_nonpivot.rows(), true);
    for (index_t r = 0; r < LY_nonpivot.rows(); ++r) {
        auto       dst = LY_nonpivot[r];
        const auto lr  = L[pivot_source_rows[(std::size_t)r]];
        for (auto p = lr.find_first(); p != Row::npos; p = lr.find_next(p)) {
            const std::int64_t idx = out.nonpivot_index[(std::size_t)p];
            if (idx >= 0) {
                dst.set((index_t)idx);
                is_zero[r] = false;
            }
        }

        const auto yr = Y[pivot_source_rows[(std::size_t)r]];
        if (!yr.none()) {
            or_shifted(dst, nonpiv_cols, yr);
            is_zero[r] = false;
        }
    }
    out.is_zero     = std::move(is_zero);
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
    const unsigned  sh         = (unsigned)(offset & 63);

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

void CountWS::reset(std::size_t K) {
    if (cnt.size() < K) {
        cnt.resize(K);
        tag.resize(K, 0);
    }
    ++epoch;
    if (epoch == 0) {
        std::fill(tag.begin(), tag.end(), 0);
        epoch = 1;
    }
    used.clear();
}

void CountWS::add(index_t id, index_t delta) {
    if (tag[id] != epoch) {
        tag[id] = epoch;
        cnt[id] = delta - parity;
        used.push_back(id);
    } else
        cnt[id] += delta;
}

index_t CountWS::argmax() const {
    assert(used.size() > 0);
    index_t best = used[0];
    for (auto id : used)
        if (cnt[id] > cnt[best] || (cnt[id] == cnt[best] && id < best))
            best = id;
    return best;
}

std::vector<index_t> CountWS::argmax_n(std::size_t n) const {
    std::vector<index_t> candidates;
    argmax_n_into(n, candidates);
    return candidates;
}

void CountWS::argmax_n_into(std::size_t n, std::vector<index_t>& scratch_out) const {
    scratch_out.clear();
    if (used.empty() || n == 0)
        return;

    auto better = [this](auto a, auto b) {
        if (cnt[a] != cnt[b])
            return cnt[a] > cnt[b];
        return a < b;
    };

    n = std::min(n, used.size());
    if (n <= 8) {
        scratch_out.reserve(n);
        for (auto id : used) {
            auto it = scratch_out.begin();
            while (it != scratch_out.end() && better(*it, id))
                ++it;
            if (it == scratch_out.end()) {
                if (scratch_out.size() < n)
                    scratch_out.push_back(id);
                continue;
            }
            if (scratch_out.size() < n) {
                scratch_out.insert(it, id);
                continue;
            }
            scratch_out.insert(it, id);
            scratch_out.pop_back();
        }
        return;
    }
    if (n == used.size()) {
        scratch_out.assign(used.begin(), used.end());
        std::ranges::sort(scratch_out, better);
        return;
    }

    scratch_out.reserve(n);
    for (auto id : used) {
        if (scratch_out.size() < n) {
            scratch_out.push_back(id);
            std::push_heap(scratch_out.begin(), scratch_out.end(), better);
            continue;
        }
        if (better(id, scratch_out.front())) {
            std::pop_heap(scratch_out.begin(), scratch_out.end(), better);
            scratch_out.back() = id;
            std::push_heap(scratch_out.begin(), scratch_out.end(), better);
        }
    }
    std::ranges::sort(scratch_out, better);
}

MatrixWithData::MatrixWithData(Matrix P) : P_{std::move(P)}, index_{P_} {
    auto precomputed = build_matrix_with_data_precompute(P_);
    tohpe_basis_     = std::move(precomputed.first);
    full_todd_       = std::move(precomputed.second);
}

const MatrixWithData::FullToddData& MatrixWithData::full_todd() const { return full_todd_; }

Witness::Witness(std::shared_ptr<MatrixWithData> M, Row z)
    : M_{M}, pair_endpoints_(M->P().rows()), z_{std::move(z)}, special_{-1} {
    const ToddIndex& idx = M_->index();

    const SumEntry* ptr = nullptr;
    index_t         len = 0;
    if (!idx.sum_bucket(z_, ptr, len)) {
        return;
    }
    init_from_entries_(ptr, len);
}

Witness::Witness(std::shared_ptr<MatrixWithData> M, Row z, const SumEntry* ptr, index_t len)
    : M_{M}, pair_endpoints_(M->P().rows()), z_{std::move(z)}, special_{-1} {
    init_from_entries_(ptr, len);
}

void Witness::init_from_entries_(const SumEntry* ptr, index_t len) {
    if (ptr == nullptr || len == 0)
        return;

    const Matrix& P = M_->P();
    const index_t m = P.rows();
    pairs_.reserve((std::size_t)(m / 2));

    for (index_t t = 0; t < len; ++t) {
        if (ptr[t].is_pair())
            continue;
        const index_t a = ptr[t].a;
        special_        = (int)a;
        break;
    }

    Row seen_endpoints(P.rows());

    for (index_t t = 0; t < len; ++t) {
        if (!ptr[t].is_pair())
            continue;
        const index_t a = ptr[t].a;
        const index_t b = ptr[t].b;
        assert(P[b].count() != 0);
        assert(special_ != a);
        assert(special_ != b);
        if (seen_endpoints.test(a) || seen_endpoints.test(b))
            pair_endpoints_disjoint_ = false;
        seen_endpoints.set(a);
        seen_endpoints.set(b);
        pair_endpoints_.set(a);
        pair_endpoints_.set(b);
        pairs_.emplace_back((int)a, (int)b);
    }
}

TohpeWitness::TohpeWitness(std::shared_ptr<MatrixWithData> M, Row z) : Witness(M, std::move(z)) {}
TohpeWitness::TohpeWitness(std::shared_ptr<MatrixWithData> M, Row z, const SumEntry* ptr, index_t len)
    : Witness(M, std::move(z), ptr, len) {}

ToddWitness::ToddWitness(std::shared_ptr<MatrixWithData> M, Row z, Matrix&& Y)
    : Witness(M, std::move(z)), Y_{std::move(Y)} {}

int Witness::rank_divergence(RowCView y) const {
    const int parity = (int)(y.count() & 1);
    int       ones_S = 0;
    int       S      = (special_ != -1) ? 1 : 0;
    if (S && y.test(special_))
        ++ones_S;
    int diff_pairs = 0;
    for (const auto& pr : pairs_) {
        const int i = pr.first;
        const int j = pr.second;
        if (y.test(i) ^ y.test(j))
            ++diff_pairs;
    }
    if (parity == 0)
        return ones_S + 2 * diff_pairs;
    return 2 * S - ones_S - 1 + 2 * diff_pairs;
}

Row NullSpace::linear_combination(RowCView coefs) const {
    if (coefs.size() > basis().rows()) {
        throw std::runtime_error("Number of coefs more than basis size");
    }
    Row out(basis().cols());
    for (auto i = coefs.find_first(); i != Row::npos; i = coefs.find_next(i)) {
        if ((index_t)i >= basis().rows())
            break;
        out ^= basis()[(index_t)i];
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

    const auto  z       = vector();
    const int   special = w_->get_special();
    const auto& pairs   = w_->get_pairs();

    scratch_killed.assign((std::size_t)n_rows, 0);
    index_t removed = 0;
    for (const auto& pr : pairs) {
        const index_t a = (index_t)pr.first;
        const index_t b = (index_t)pr.second;
        if (y.test(a) ^ y.test(b)) {
            scratch_killed[(std::size_t)a] = 1;
            scratch_killed[(std::size_t)b] = 1;
            removed += 2;
        }
    }

    const bool parity = (y.count() & 1) != 0;
    if (special != -1 && (y.test(special) || parity)) {
        scratch_killed[(std::size_t)special] = 1;
        ++removed;
    }
    if (parity && (special == -1 || y.test(special))) {
        --removed;
    }

    Matrix new_P(n_rows - removed, P0.cols());

    index_t j = 0;
    for (index_t i = 0; i < n_rows; ++i) {
        if (scratch_killed[(std::size_t)i])
            continue;
        if (y.test(i)) {
            assign(new_P[j], (P0[i] ^ z));
        } else {
            assign(new_P[j], P0[i]);
        }
        ++j;
    }

    if (parity && (special == -1 || y.test(special))) {
        assign(new_P[j], z);
    }
    return new_P;
}

TohpeGenerator::TohpeGenerator(std::shared_ptr<MatrixWithData> M) : M_{M} {
    const auto n = M_->P().rows();
    scratch_ones_.resize((std::size_t)n);
    scratch_zeros_.resize((std::size_t)n);
}

NullSpace TohpeGenerator::make(index_t row) const {
    return NullSpace(M_, std::make_unique<TohpeWitness>(TohpeWitness(M_, M_->P()[row])));
}

NullSpace TohpeGenerator::make(index_t row1, index_t row2) const {
    Row z(M_->P()[row1]);
    z ^= M_->P()[row2];
    return NullSpace(M_, std::make_unique<TohpeWitness>(TohpeWitness(M_, std::move(z))));
}

NullSpace TohpeGenerator::make(RowCView z) const {
    return NullSpace(M_, std::make_unique<TohpeWitness>(TohpeWitness(M_, z)));
}

NullSpace TohpeGenerator::make(RowCView z, std::uint32_t bucket_id) const {
    const SumEntry* ptr = nullptr;
    index_t         len = 0;
    M_->index().sum_bucket(bucket_id, ptr, len);
    return NullSpace(M_, std::make_unique<TohpeWitness>(TohpeWitness(M_, z, ptr, len)));
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

void TohpeGenerator::best_z_n_details_into(RowCView y, index_t num_samples,
                                           std::vector<TohpeZInfo>& scratch_out) const {
    assert(y.count() != 0);
    const Matrix&    P       = M_->P();
    const ToddIndex& idx     = M_->index();
    const index_t    n       = P.rows();
    const index_t    y_count = y.count();

    if (scratch_ones_.size() < (std::size_t)n)
        scratch_ones_.resize((std::size_t)n);
    if (scratch_zeros_.size() < (std::size_t)n)
        scratch_zeros_.resize((std::size_t)n);

    std::size_t ones_size  = 0;
    std::size_t zeros_size = 0;
    for (index_t i = 0; i < n; ++i) {
        if (y.test(i))
            scratch_ones_[ones_size++] = i;
        else
            scratch_zeros_[zeros_size++] = i;
    }

    ws_.reset(idx.buckets_num());
    ws_.parity = y_count % 2;
    for (std::size_t oi = 0; oi < ones_size; ++oi) {
        const index_t i = scratch_ones_[oi];
        ws_.add(idx.single_id()[(std::size_t)i], 1);

        for (std::size_t zi = 0; zi < zeros_size; ++zi) {
            const index_t j = scratch_zeros_[zi];
            const auto id = idx.pair_bucket_id(i, j);
            ws_.add(id, 2);
        }
    }
    if (ws_.parity) {
        for (std::size_t zi = 0; zi < zeros_size; ++zi) {
            const index_t i = scratch_zeros_[zi];
            ws_.add(idx.single_id()[(std::size_t)i], 2);
        }
    }

    scratch_out.clear();
    ws_.argmax_n_into(num_samples, scratch_candidates_);
    scratch_out.reserve(scratch_candidates_.size());
    for (auto id : scratch_candidates_) {
        const auto bucket_id = static_cast<std::uint32_t>(id);
        scratch_out.push_back(TohpeZInfo{
            .z           = Row(idx.key_of(bucket_id)),
            .reduction   = index_t(ws_.cnt[id]),
            .bucket_size = idx.bucket_size(bucket_id),
            .bucket_id   = bucket_id,
        });
    }
}

FullToddGenerator::FullToddGenerator(std::shared_ptr<MatrixWithData> M) : M_{M} { (void)M_->full_todd(); }

Matrix FullToddGenerator::full_todd_kernel(RowCView z, const SumEntry* ptr, int len) const {
    const auto&   ft         = M_->full_todd();
    const index_t n          = z.size();
    const index_t total_cols = ft.nonpiv_cols + M_->P().rows();

    std::vector<index_t> S;
    S.reserve((std::size_t)z.count());
    for (auto i = z.find_first(); i != RowView::npos; i = z.find_next(i)) {
        S.push_back((index_t)i);
    }
    auto add_col = [&](auto ra_row, index_t col) {
        const std::int64_t prow = ft.pivot_row_of_col[(std::size_t)col];
        if (prow >= 0) {
            ra_row ^= ft.LY_nonpivot[prow];
        } else {
            ra_row.flip(ft.nonpivot_index[(std::size_t)col]);
        }
    };

    auto fill_row = [&](index_t row_idx, RowView row) {
        if (row_idx < n) {
            const index_t       gamma = row_idx;
            const std::uint64_t offg  = ft.offset[(index_t)gamma];
            for (index_t s : S) {
                if (s == gamma)
                    continue;
                const index_t col = (s > gamma) ? (index_t)(ft.offset[(std::size_t)s] + gamma) : (index_t)(offg + s);
                add_col(row, col);
            }
        } else {
            for (std::size_t bi = 0; bi < S.size(); ++bi) {
                const index_t       b    = S[bi];
                const std::uint64_t offb = ft.offset[(std::size_t)b];
                for (std::size_t ai = 0; ai <= bi; ++ai) {
                    const index_t a = S[ai];
                    add_col(row, (index_t)(offb + a));
                }
            }
        }
    };

    return solve_and_build_solution_basis_generated(n + 1, total_cols, ft.nonpiv_cols, M_->tohpe_basis(), ptr, len,
                                                    fill_row);
}

NullSpace FullToddGenerator::make(RowCView z) const {
    const SumEntry* ptr = nullptr;
    index_t         len = 0;
    M_->index().sum_bucket(z, ptr, len);
    Matrix Y = full_todd_kernel(z, ptr, len);
    return NullSpace(M_, std::make_unique<ToddWitness>(ToddWitness(M_, z, std::move(Y))));
}

NullSpace FullToddGenerator::make(index_t row) const {
    Row             z   = M_->P()[row];
    const SumEntry* ptr = nullptr;
    index_t         len = 0;
    M_->index().sum_bucket(z, ptr, len);
    Matrix Y = full_todd_kernel(z, ptr, len);
    return NullSpace(M_, std::make_unique<ToddWitness>(ToddWitness(M_, std::move(z), std::move(Y))));
}

NullSpace FullToddGenerator::make(index_t row1, index_t row2) const {
    Row z = M_->P()[row1];
    z ^= M_->P()[row2];
    const SumEntry* ptr = nullptr;
    index_t         len = 0;
    M_->index().sum_bucket(z, ptr, len);
    Matrix Y = full_todd_kernel(z, ptr, len);
    return NullSpace(M_, std::make_unique<ToddWitness>(ToddWitness(M_, std::move(z), std::move(Y))));
}

} // namespace todd
