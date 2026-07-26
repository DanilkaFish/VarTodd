#ifndef GF2_TODD_INDEX_HPP
#define GF2_TODD_INDEX_HPP
#include "algorithms.hpp"
#include "matrix.hpp"

#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace todd {

using HashKey = std::uint64_t;
struct BucketRange {
    index_t off;
    index_t len;
};

// Compact row or row-pair reference stored inside a sum bucket.
class SumEntry {
  public:
    using storage_t = std::uint32_t;

    SumEntry(index_t a_, index_t b_) : a{static_cast<storage_t>(a_)}, b{static_cast<storage_t>(b_)} {
        assert(a_ < k_single_sentinel<storage_t>());
        assert(b_ < k_single_sentinel<storage_t>());
    }
    SumEntry(index_t a_) : a{static_cast<storage_t>(a_)}, b{k_single_sentinel<storage_t>()} {
        assert(a_ < k_single_sentinel<storage_t>());
    }
    bool is_pair() const { return b != k_single_sentinel<storage_t>(); }

    storage_t a;
    storage_t b;
};

Matrix solve_and_build_solution_basis(Matrix& A, index_t divider, const Matrix& tohpe, const SumEntry* ptr,
                                      index_t len);

namespace detail {
inline Row extract_right_for_solution(RowCView row, index_t divider, index_t nY) {
    Row y(nY);
    if (nY == 0)
        return y;

    uint64_t*       dst = y.data();
    const uint64_t* src = row.data();

    const index_t dst_blocks = y.blocks();
    const index_t src_blocks = row.blocks();

    const index_t  sb = divider >> 6;
    const unsigned sh = static_cast<unsigned>(divider & 63);

    if (sh == 0) {
        const index_t avail = (sb < src_blocks) ? (src_blocks - sb) : 0;
        const index_t take  = std::min(dst_blocks, avail);

        if (take)
            std::memcpy(dst, src + sb, static_cast<std::size_t>(take) * sizeof(uint64_t));
        if (take < dst_blocks)
            std::memset(dst + take, 0, static_cast<std::size_t>(dst_blocks - take) * sizeof(uint64_t));
    } else {
        for (index_t k = 0; k < dst_blocks; ++k) {
            const index_t  i0 = sb + k;
            const uint64_t lo = (i0 < src_blocks) ? src[i0] : 0ULL;
            const uint64_t hi = (i0 + 1 < src_blocks) ? src[i0 + 1] : 0ULL;
            dst[k]            = (lo >> sh) | (hi << (64u - sh));
        }
    }

    const index_t rem = nY & 63;
    if (rem != 0)
        dst[dst_blocks - 1] &= ((1ULL << rem) - 1ULL);

    return y;
}

inline void apply_solution_transform(Row& y, const SumEntry* ptr, index_t len) {
    bool odd = (y.count() & 1u) != 0;

    for (index_t t = 0; t < len; ++t) {
        const SumEntry& e = ptr[t];

        if (e.is_pair()) {
            if (y.test(e.b)) {
                y.flip(e.a);
                y.flip(e.b);
            }
        } else {
            if (odd) {
                y.flip(e.a);
                odd = false;
            }
        }
    }
}

inline bool insert_into_y_basis(Row& y, Matrix& basis, PivotMap& pivY) {
    auto yv = y.view();
    auto q  = yv.find_first();

    while (q != RowView::npos) {
        const auto brow = pivY.get(static_cast<std::size_t>(q));
        if (brow < 0) {
            pivY.set(static_cast<std::size_t>(q), basis.rows());
            basis.push_back(y.cview());
            return true;
        }

        yv ^= basis[static_cast<index_t>(brow)];
        q = yv.find_next(q);
    }

    return false;
}
} // namespace detail

template <class FillRow>
Matrix solve_and_build_solution_basis_generated(index_t rows, index_t cols, index_t divider, const Matrix& tohpe,
                                                const SumEntry* ptr, index_t len, FillRow&& fill_row) {
    if (divider > cols)
        throw std::invalid_argument("solve_and_build_solution_basis_generated: divider > cols");
    const index_t nY = cols - divider;
    if (nY == 0)
        return Matrix(0, 0);
    if (tohpe.rows() != 0 && tohpe.cols() != nY)
        throw std::invalid_argument("solve_and_build_solution_basis_generated: tohpe width mismatch");

    static thread_local PivotMap pivA;
    static thread_local PivotMap pivY;

    pivA.reset(static_cast<std::size_t>(divider));
    pivY.reset(static_cast<std::size_t>(nY));

    Matrix basis(0, nY);
    Matrix pivot_rows(0, cols);
    Row    row(cols);

    const index_t full_rank = nY;
    basis.reserve_rows(full_rank);
    pivot_rows.reserve_rows(std::min(rows, divider));

    for (index_t i = 0; i < rows && basis.rows() < full_rank; ++i) {
        row.reset();
        fill_row(i, row.view());

        auto rowv = row.view();
        auto p    = rowv.find_first();
        while (p != RowView::npos && p < divider) {
            const auto prow = pivA.get(static_cast<std::size_t>(p));
            if (prow < 0){
                pivA.set(static_cast<std::size_t>(p), pivot_rows.rows());
                pivot_rows.push_back(row.cview());
                break;
            }
            rowv ^= pivot_rows[static_cast<index_t>(prow)];
            p = rowv.find_next(p);
        }

        // if (p == RowView::npos)
            // continue;

        if (p < divider) {
            continue;
        }

        Row y = detail::extract_right_for_solution(row.cview(), divider, nY);
        detail::apply_solution_transform(y, ptr, len);
        detail::insert_into_y_basis(y, basis, pivY);
    }

    for (index_t i = 0; i < tohpe.rows() && basis.rows() < full_rank; ++i) {
        Row y = tohpe[i];

        detail::apply_solution_transform(y, ptr, len);
        detail::insert_into_y_basis(y, basis, pivY);
    }

    return basis;
}

struct HashKeyHash {
    static inline std::uint64_t mix(std::uint64_t x) noexcept {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }
    std::size_t operator()(HashKey k) const noexcept { return static_cast<std::size_t>(mix(k)); }
};

// Index row and row-pair sums by the resulting GF(2) vector.
class ToddIndex {
  public:
    using SumKeySize = std::pair<RowCView, index_t>;
    using BucketIdSize = std::pair<std::uint32_t, index_t>;

    explicit ToddIndex(const Matrix& P);
    const Matrix& matrix() const noexcept { return P_; }
    index_t       rows() const noexcept { return m_; }
    index_t       cols() const noexcept { return n_bits_; }
    HashKey       hash_vec(RowCView v) const noexcept;
    index_t       get_size_from_z(RowCView z) const;
    bool          sum_bucket(RowCView key, const SumEntry*& ptr, index_t& len) const noexcept;
    bool          sum_bucket(std::uint32_t id, const SumEntry*& ptr, index_t& len) const noexcept;
    bool          materialize_bucket(RowCView key, std::vector<SumEntry>& out) const;
    bool          materialize_bucket(std::uint32_t id, std::vector<SumEntry>& out) const;
    index_t       bucket_size(std::uint32_t id) const noexcept {
        return (id < bucket_len_.size()) ? bucket_len_[static_cast<std::size_t>(id)] : 0;
    }

    const std::vector<HashKey>&               row_hashes() const noexcept { return hP_; }
    std::vector<std::pair<RowCView, index_t>> sum_key_sizes() const;
    std::vector<SumKeySize>&                  sum_key_sizes_scratch() const;
    std::vector<BucketIdSize>&                sum_bucket_id_sizes_scratch() const;
    std::vector<BucketIdSize>&                top_sum_bucket_id_sizes_scratch(std::size_t max_count) const;
    const std::vector<std::uint32_t>&         single_id() const noexcept { return single_id_; }
    std::uint32_t pair_bucket_id(index_t i, index_t j) const noexcept {
        assert(i != j);
        return pair_id_[pair_index_fast_(i, j)];
    }
    RowCView      key_of(std::uint32_t id) const noexcept { return bucket_key_(id); }
    index_t       buckets_num() const noexcept { return bucket_len_.size(); }
    index_t       max_bucket() const noexcept { return max_bucket_; }

  private:
    const Matrix&                                                     P_;
    index_t                                                           m_{}, n_bits_{};
    std::vector<HashKey>                                              masks_;
    std::vector<HashKey>                                              hP_;
    std::vector<std::uint32_t>                                        single_id_;
    std::vector<std::uint32_t>                                        pair_id_;
    std::vector<index_t>                                              bucket_off_;
    std::vector<index_t>                                              bucket_len_;
    std::vector<std::uint32_t>                                        bucket_next_;
    std::vector<std::uint64_t>                                        bucket_key_blocks_;
    index_t                                                           bucket_key_blocks_per_row_{};
    index_t                                                           max_bucket_{};
    std::vector<SumEntry>                                             sum_entries_;
    mutable std::vector<SumKeySize>                                   scratch_sum_key_sizes_;
    mutable std::vector<BucketIdSize>                                 scratch_bucket_id_sizes_;
    ankerl::unordered_dense::map<HashKey, std::uint32_t, HashKeyHash> head_;

    void        build_masks_();
    void        build_row_hashes_();
    void        build_sum_buckets_();
    std::size_t pair_index_fast_(index_t i, index_t j) const noexcept {
        assert(i >= 0 && j >= 0);
        assert(i < m_ && j < m_);
        assert(i != j);
        if (i > j)
            std::swap(i, j);
        const auto ii = static_cast<std::size_t>(i);
        const auto jj = static_cast<std::size_t>(j);
        const auto mm = static_cast<std::size_t>(m_);
        return (ii * (2 * mm - ii - 1)) / 2 + (jj - ii - 1);
    }

    RowCView bucket_key_(std::uint32_t id) const noexcept {
        const auto* data = bucket_key_blocks_per_row_
                               ? bucket_key_blocks_.data() +
                                     static_cast<std::size_t>(id) * static_cast<std::size_t>(bucket_key_blocks_per_row_)
                               : nullptr;
        return RowCView(data, n_bits_, bucket_key_blocks_per_row_);
    }

    void push_bucket_key_(RowCView key);

    std::uint32_t get_bucket_id_(HashKey hk, RowCView sumv);
};
} // namespace todd
#endif
