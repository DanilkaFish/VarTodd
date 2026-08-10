#ifndef GF2_TODD_INDEX_HPP
#define GF2_TODD_INDEX_HPP
#include "algorithms.hpp"
#include "matrix.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
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

inline Matrix build_transformed_tohpe_prefix(const Matrix& tohpe, const SumEntry* ptr, index_t len, PivotMap& pivY) {
    Matrix basis(0, tohpe.cols());
    basis.reserve_rows(tohpe.rows());
    pivY.reset(static_cast<std::size_t>(tohpe.cols()));

    for (index_t i = 0; i < tohpe.rows(); ++i) {
        Row y = tohpe[i];
        apply_solution_transform(y, ptr, len);
        insert_into_y_basis(y, basis, pivY);
    }
    return basis;
}
} // namespace detail

struct GeneratedSolutionBasis {
    Matrix  basis;
    index_t tohpe_prefix_size = 0;
};

template <class FillRow>
GeneratedSolutionBasis solve_and_build_solution_basis_generated(index_t rows, index_t cols, index_t divider,
                                                                const Matrix& tohpe, const SumEntry* ptr, index_t len,
                                                                FillRow&& fill_row) {
    if (divider > cols)
        throw std::invalid_argument("solve_and_build_solution_basis_generated: divider > cols");
    const index_t nY = cols - divider;
    if (nY == 0)
        return {Matrix(0, 0), 0};
    if (tohpe.rows() != 0 && tohpe.cols() != nY)
        throw std::invalid_argument("solve_and_build_solution_basis_generated: tohpe width mismatch");

    static thread_local PivotMap pivA;
    static thread_local PivotMap pivY;

    pivA.reset(static_cast<std::size_t>(divider));
    Matrix basis = detail::build_transformed_tohpe_prefix(tohpe, ptr, len, pivY);
    Matrix pivot_rows(0, cols);
    Row    row(cols);

    const index_t full_rank = nY;
    pivot_rows.reserve_rows(std::min(rows, divider));
    const index_t tohpe_prefix_size = basis.rows();
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
    return {std::move(basis), tohpe_prefix_size};
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

namespace detail {

class BucketLengths {
  public:
    void assign(const std::vector<std::uint32_t>& counts, std::uint32_t max_count) {
        if (max_count <= std::numeric_limits<std::uint8_t>::max()) {
            storage_ = narrow_<std::uint8_t>(counts);
        } else if (max_count <= std::numeric_limits<std::uint16_t>::max()) {
            storage_ = narrow_<std::uint16_t>(counts);
        } else {
            storage_ = counts;
        }
    }

    index_t get(std::size_t id) const noexcept {
        return std::visit(
            [id](const auto& values) -> index_t {
                return id < values.size() ? static_cast<index_t>(values[id]) : 0;
            },
            storage_);
    }

    std::size_t size() const noexcept {
        return std::visit([](const auto& values) { return values.size(); }, storage_);
    }

    std::size_t count_bytes() const noexcept {
        return std::visit(
            [](const auto& values) {
                using value_type = typename std::decay_t<decltype(values)>::value_type;
                return sizeof(value_type);
            },
            storage_);
    }

    std::size_t storage_bytes() const noexcept {
        return std::visit(
            [](const auto& values) {
                using value_type = typename std::decay_t<decltype(values)>::value_type;
                return values.capacity() * sizeof(value_type);
            },
            storage_);
    }

    template <class F> decltype(auto) with_values(F&& fn) const {
        return std::visit(
            [&fn](const auto& values) -> decltype(auto) {
                return std::forward<F>(fn)(values);
            },
            storage_);
    }

  private:
    template <class Count>
    static std::vector<Count> narrow_(const std::vector<std::uint32_t>& counts) {
        std::vector<Count> out;
        out.reserve(counts.size());
        for (const std::uint32_t count : counts) {
            assert(count <= std::numeric_limits<Count>::max());
            out.push_back(static_cast<Count>(count));
        }
        return out;
    }

    std::variant<std::vector<std::uint8_t>, std::vector<std::uint16_t>, std::vector<std::uint32_t>> storage_;
};

} // namespace detail

// Index row and row-pair sums by the resulting GF(2) vector.
class ToddIndex {
  public:
    using BucketIdSize = std::pair<std::uint32_t, std::uint32_t>;

    explicit ToddIndex(const Matrix& P);
    const Matrix& matrix() const noexcept { return P_; }
    index_t       rows() const noexcept { return m_; }
    index_t       cols() const noexcept { return n_bits_; }
    HashKey       hash_vec(RowCView v) const noexcept;
    index_t       get_size_from_z(RowCView z) const;
    bool          materialize_bucket(RowCView key, std::vector<SumEntry>& out) const;
    bool          materialize_bucket(std::uint32_t id, std::vector<SumEntry>& out) const;
    index_t       bucket_size(std::uint32_t id) const noexcept {
        return bucket_len_.get(static_cast<std::size_t>(id));
    }
    std::size_t bucket_length_bytes() const noexcept { return bucket_len_.count_bytes(); }

    std::vector<BucketIdSize>&                sum_bucket_id_sizes_scratch() const;
    std::vector<BucketIdSize>&                top_sum_bucket_id_sizes_scratch(std::size_t max_count) const;
    const std::vector<std::uint32_t>&         single_id() const noexcept { return single_id_; }
    std::uint32_t pair_bucket_id(index_t i, index_t j) const noexcept {
        assert(i != j);
        return pair_id_[pair_index_fast_(i, j)];
    }
    Row           key_of(std::uint32_t id) const;
    index_t       buckets_num() const noexcept { return bucket_len_.size(); }
    index_t       max_bucket() const noexcept { return max_bucket_; }
    std::size_t   storage_bytes() const noexcept;

  private:
    const Matrix&                                                     P_;
    index_t                                                           m_{}, n_bits_{};
    std::vector<HashKey>                                              masks_;
    std::vector<HashKey>                                              hP_;
    std::vector<std::uint32_t>                                        single_id_;
    std::vector<std::uint32_t>                                        pair_id_;
    std::vector<std::uint32_t>                                        lookup_slots_;
    std::vector<std::uint32_t>                                        representative_pos_;
    std::vector<std::size_t>                                          pair_row_off_;
    detail::BucketLengths                                             bucket_len_;
    index_t                                                           max_bucket_{};
    mutable std::vector<BucketIdSize>                                 scratch_bucket_id_sizes_;
    std::size_t                                                       lookup_mask_{};
    std::uint32_t                                                     slot_id_mask_{};
    unsigned                                                          slot_id_bits_{};
    unsigned                                                          slot_fingerprint_bits_{};

    struct DuplicateBucket {
        std::uint32_t id;
        std::uint32_t begin;
        std::uint32_t end;
    };
    std::vector<DuplicateBucket> duplicate_buckets_;
    std::vector<std::uint32_t>   duplicate_positions_;

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

    SumEntry    decode_source_position_(std::uint32_t pos) const;
    Row         row_from_source_position_(std::uint32_t pos) const;
    bool        representative_equals_(std::uint32_t id, RowCView key) const noexcept;
    std::uint32_t find_bucket_(HashKey hk, RowCView key) const noexcept;
    std::pair<std::uint32_t, bool> get_bucket_id_(HashKey hk, RowCView sumv, std::uint32_t source_pos);
};
} // namespace todd
#endif
