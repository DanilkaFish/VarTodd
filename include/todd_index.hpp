#ifndef GF2_TODD_INDEX_HPP
#define GF2_TODD_INDEX_HPP
#include "matrix.hpp"

#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace todd {

using HashKey = std::uint64_t;
struct BucketRange {
    std::uint32_t off;
    std::uint32_t len;
};

// basic element
class SumEntry {
  public:
    SumEntry(index_t a, index_t b) : a{a}, b{b} {}
    SumEntry(index_t a) : a{a}, b{k_single_sentinel<index_t>()} {}
    bool is_pair() { return b != k_single_sentinel<index_t>(); }

//   private:
    index_t a;
    index_t b;
};

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

static inline std::size_t pair_index(index_t i, index_t j, index_t n) noexcept {
    if (i > j)
        std::swap(i, j);
    const std::size_t ii = (std::size_t)i, nn = (std::size_t)n;
    const std::size_t prefix = ii * (nn - 1) - (ii * (ii - 1)) / 2;
    return prefix + (std::size_t)(j - i - 1);
}

// keeps buckets for columns of given matrix.
class ToddIndex {
  public:
    explicit ToddIndex(const Matrix& P);

    const Matrix& matrix() const noexcept { return P_; }
    index_t       rows() const noexcept { return m_; }
    index_t       cols() const noexcept { return n_bits_; }
    HashKey       hash_vec(RowCView v) const noexcept;
    index_t       get_size_from_z(RowCView z) const;
    bool          sum_bucket(RowCView key, const SumEntry*& ptr, index_t& len) const noexcept;

    const std::vector<HashKey>&               row_hashes() const noexcept { return hP_; }
    std::vector<std::pair<RowCView, index_t>> sum_key_sizes() const;
    const std::vector<std::uint32_t>&         single_id() const noexcept { return single_id_; }
    const std::vector<std::uint32_t>&         pair_id() const noexcept { return pair_id_; }
    RowCView key_of(std::uint32_t id) const noexcept { return buckets_[(std::size_t)id].key.cview(); }
    index_t  buckets_num() const noexcept { return buckets_.size(); }
    index_t  max_bucket() const noexcept;

  private:
    struct BucketInfo {
        std::uint32_t off = 0, len = 0, cur = 0, next = std::numeric_limits<std::uint32_t>::max();
        Row           key;
    };
    const Matrix&                                                     P_;
    index_t                                                           m_{}, n_bits_{};
    std::vector<HashKey>                                              masks_;
    std::vector<HashKey>                                              hP_;
    std::vector<std::uint32_t>                                        single_id_;
    std::vector<std::uint32_t>                                        pair_id_;
    std::vector<BucketInfo>                                           buckets_;
    std::vector<SumEntry>                                             sum_entries_;
    ankerl::unordered_dense::map<HashKey, std::uint32_t, HashKeyHash> head_;

    void build_masks_();
    void build_row_hashes_();
    void build_sum_buckets_();

    std::uint32_t get_bucket_id_(HashKey hk, RowCView sumv);
};
} // namespace todd
#endif
