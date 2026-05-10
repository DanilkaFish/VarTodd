#ifndef GF2_TODD_INDEX_HPP
#define GF2_TODD_INDEX_HPP
#include "matrix.hpp"

#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace todd {

using HashKey = std::uint32_t;
struct BucketRange {
    std::uint32_t off;
    std::uint32_t len;
};

// basic element
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

// Matrix solve_and_build_solution_basis(Matrix& A, index_t divider);
Matrix solve_and_build_solution_basis(Matrix& A, index_t divider, const Matrix& tohpe, const SumEntry* ptr, int len);
Matrix solve_and_build_solution_basis_generated(index_t rows, index_t cols, index_t divider, const Matrix& tohpe,
                                                const SumEntry* ptr, int len,
                                                const std::function<void(index_t, RowView)>& fill_row);

struct HashKeyHash {
    static inline std::uint64_t mix(std::uint64_t x) noexcept {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }
    size_t operator()(HashKey k) const noexcept { return (size_t)mix(k); }
};

// keeps buckets for columns of given matrix.
class ToddIndex {
  public:
    using SumKeySize = std::pair<RowCView, index_t>;

    explicit ToddIndex(const Matrix& P);
    const Matrix& matrix() const noexcept { return P_; }
    index_t       rows() const noexcept { return m_; }
    index_t       cols() const noexcept { return n_bits_; }
    HashKey       hash_vec(RowCView v) const noexcept;
    index_t       get_size_from_z(RowCView z) const;
    bool          sum_bucket(RowCView key, const SumEntry*& ptr, index_t& len) const noexcept;
    bool          sum_bucket(std::uint32_t id, const SumEntry*& ptr, index_t& len) const noexcept;
    index_t       bucket_size(std::uint32_t id) const noexcept {
        return (id < buckets_.size()) ? (index_t)buckets_[(std::size_t)id].len : 0;
    }

    const std::vector<HashKey>&               row_hashes() const noexcept { return hP_; }
    std::vector<std::pair<RowCView, index_t>> sum_key_sizes() const;
    std::vector<SumKeySize>&                  sum_key_sizes_scratch() const;
    const std::vector<std::uint32_t>&         single_id() const noexcept { return single_id_; }
    const std::vector<std::uint32_t>&         pair_id() const noexcept { return pair_id_; }
    std::uint32_t pair_bucket_id(index_t i, index_t j) const noexcept {
        assert(i != j);
        return pair_id_[pair_index_fast_(i, j)];
    }
    RowCView      key_of(std::uint32_t id) const noexcept { return bucket_keys_[(index_t)id]; }
    index_t       buckets_num() const noexcept { return buckets_.size(); }
    index_t       max_bucket() const noexcept;

  private:
    struct BucketInfo {
        std::uint32_t off = 0, len = 0, cur = 0, next = std::numeric_limits<std::uint32_t>::max();
    };
    const Matrix&                                                     P_;
    index_t                                                           m_{}, n_bits_{};
    std::vector<HashKey>                                              masks_;
    std::vector<HashKey>                                              hP_;
    std::vector<std::uint32_t>                                        single_id_;
    std::vector<std::uint32_t>                                        pair_id_;
    std::vector<BucketInfo>                                           buckets_;
    Matrix                                                            bucket_keys_;
    std::vector<SumEntry>                                             sum_entries_;
    mutable std::vector<SumKeySize>                                   scratch_sum_key_sizes_;
    ankerl::unordered_dense::map<HashKey, std::uint32_t, HashKeyHash> head_;

    void        build_masks_();
    void        build_row_hashes_();
    void        build_sum_buckets_();
    std::size_t pair_index_fast_(index_t i, index_t j) const noexcept {
        assert(i >= 0 && j >= 0);
        assert(i < m_ && j < m_);
        return static_cast<std::size_t>(i) * static_cast<std::size_t>(m_) + static_cast<std::size_t>(j);
    }

    std::uint32_t get_bucket_id_(HashKey hk, RowCView sumv);
};
} // namespace todd
#endif
