#pragma once
#include "typedef.hpp"
#include <algorithm>
#include <bit>
#include <bitset>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <type_traits>
#include <vector>

namespace todd {
static inline index_t ceil_div64(index_t bits) noexcept { return (bits / 64) + ((bits & 63) != 0); }

static inline uint64_t tail_mask_bits(index_t nbits) {
    const index_t rem = nbits & 63;
    return rem ? ((1ULL << rem) - 1ULL) : ~0ULL;
}
static inline unsigned popcount64(uint64_t x) noexcept { return static_cast<unsigned>(__builtin_popcountll(x)); }

// -------------------- Views --------------------
// Non-owning bit-row view. `size_` is the logical bit width; `blocks_` is the stored word count.
template <class Ptr> class BasicRowView {
    Ptr     p_      = nullptr;
    index_t size_   = 0;
    index_t blocks_ = 0;

  public:
    static constexpr index_t npos = k_single_sentinel<index_t>();

    BasicRowView() = default;
    BasicRowView(Ptr p, index_t size) noexcept : p_(p), size_(size), blocks_(ceil_div64(size_)) {}
    BasicRowView(Ptr p, index_t size, index_t blocks_) noexcept : p_(p), size_(size), blocks_(blocks_) {}
    BasicRowView& operator=(const auto&) = delete;
    index_t       size() const noexcept { return size_; }
    index_t       blocks() const noexcept { return blocks_; }

    const uint64_t* data() const noexcept { return p_; }

    // only for mutable view
    uint64_t* data() noexcept
        requires(!std::is_const_v<std::remove_pointer_t<Ptr>>)
    {
        return p_;
    }

    bool test(index_t i) const noexcept { return (data()[i >> 6] >> (i & 63)) & 1ULL; }
    bool none() const noexcept {
        const index_t logical_blocks = std::min(blocks_, ceil_div64(size_));
        if (logical_blocks == 0)
            return true;

        const auto* __restrict p = data();
        const index_t          last = logical_blocks - 1;

        for (index_t k = 0; k < last; ++k)
            if (p[k] != 0)
                return false;
        return (p[last] & tail_mask_bits(size_)) == 0;
    }
    index_t count() const noexcept {
        const index_t logical_blocks = std::min(blocks_, ceil_div64(size_));
        if (logical_blocks == 0)
            return 0;
        index_t total = 0;
        for (index_t k = 0; k + 1 < logical_blocks; ++k) {
            total += popcount64(p_[k]);
        }
        uint64_t last = p_[logical_blocks - 1] & tail_mask_bits(size_);
        total += popcount64(last);
        return total;
    }

    // mutators only exist for mutable view
    void set(index_t i) noexcept
        requires(!std::is_const_v<std::remove_pointer_t<Ptr>>)
    {
        data()[i >> 6] |= (1ULL << (i & 63));
    }
    void reset(index_t i) noexcept
        requires(!std::is_const_v<std::remove_pointer_t<Ptr>>)
    {
        data()[i >> 6] &= ~(1ULL << (i & 63));
    }
    void flip(index_t i) noexcept
        requires(!std::is_const_v<std::remove_pointer_t<Ptr>>)
    {
        data()[i >> 6] ^= (1ULL << (i & 63));
    }
    void clear() noexcept
        requires(!std::is_const_v<std::remove_pointer_t<Ptr>>)
    {
        std::fill_n(data(), blocks_, 0ULL);
    }
    index_t find_first() const noexcept {
        const index_t logical_blocks = std::min(blocks_, ceil_div64(size_));
        if (logical_blocks == 0)
            return npos;

        const index_t last = logical_blocks - 1;
        for (index_t k = 0; k < last; ++k) {
            const uint64_t w = p_[k];
            if (w != 0)
                return (k << 6) + std::countr_zero(w);
        }

        const uint64_t w = p_[last] & tail_mask_bits(size_);
        if (w == 0)
            return npos;
        return (last << 6) + std::countr_zero(w);
    }

    index_t find_next(index_t pos) const noexcept {
        if (size_ == 0 || pos >= size_ - 1)
            return npos;
        ++pos;

        const index_t logical_blocks = std::min(blocks_, ceil_div64(size_));
        index_t       k              = pos >> 6;
        if (k >= logical_blocks)
            return npos;

        uint64_t w = p_[k] & (~0ULL << (pos & 63));
        const index_t last = logical_blocks - 1;
        if (k == last)
            w &= tail_mask_bits(size_);

        if (w) {
            return (k << 6) + std::countr_zero(w);
        }

        for (++k; k < last; ++k) {
            w = p_[k];
            if (w != 0)
                return (k << 6) + std::countr_zero(w);
        }

        if (k == last) {
            w = p_[last] & tail_mask_bits(size_);
            if (w != 0)
                return (last << 6) + std::countr_zero(w);
        }
        return npos;
    }
    // implicit downgrade: RowView -> RowCView
    operator BasicRowView<const uint64_t*>() const noexcept { return {data(), size_, blocks_}; }
};

using RowView  = BasicRowView<uint64_t*>;
using RowCView = BasicRowView<const uint64_t*>;

// -------------------- Concepts --------------------

template <class R>
concept ReadableRow = requires(const R& r) {
    { r.size() } -> std::convertible_to<index_t>;
    { r.blocks() } -> std::convertible_to<index_t>;
    { r.data() } -> std::same_as<const uint64_t*>;
};

template <class R>
concept WritableRow = ReadableRow<R> && requires(R& r) {
    { r.data() } -> std::same_as<uint64_t*>;
};

// ------------------- OPS --------------------------

inline static void xor_rows(uint64_t* __restrict dst, const uint64_t* __restrict src, index_t n) noexcept {
    for (index_t i = 0; i < n; ++i)
        dst[i] ^= src[i];
}

template <ReadableRow R> inline uint64_t read_word_or_zero(const R& row, index_t block) noexcept {
    return block < row.blocks() ? row.data()[block] : 0ULL;
}

template <WritableRow R> inline void mask_tail_inplace(R& row) noexcept {
    const index_t rem = row.size() & 63;
    if (rem != 0 && row.blocks() != 0)
        row.data()[row.blocks() - 1] &= ((1ULL << rem) - 1ULL);
}

template <WritableRow L, ReadableRow R> inline L& operator^=(L& lhs, const R& rhs) noexcept {
    assert(lhs.size() == rhs.size());
    const index_t xor_blocks = std::min(lhs.blocks(), rhs.blocks());
    xor_rows(lhs.data(), rhs.data(), xor_blocks);
    mask_tail_inplace(lhs);
    return lhs;
}

template <WritableRow L, ReadableRow R> inline L&& operator^=(L&& lhs, const R& rhs) noexcept {
    assert(lhs.size() == rhs.size());
    const index_t xor_blocks = std::min(lhs.blocks(), rhs.blocks());
    xor_rows(lhs.data(), rhs.data(), xor_blocks);
    mask_tail_inplace(lhs);
    return static_cast<L&&>(lhs);
}

template <ReadableRow L, ReadableRow R> inline bool operator==(const L& lhs, const R& rhs) noexcept {
    if (lhs.size() != rhs.size())
        return false;

    const index_t nb = ceil_div64(lhs.size());
    if (nb == 0)
        return true;

    const index_t last = nb - 1;
    for (index_t k = 0; k < last; ++k)
        if (read_word_or_zero(lhs, k) != read_word_or_zero(rhs, k))
            return false;

    const uint64_t mask = tail_mask_bits(lhs.size());
    return (read_word_or_zero(lhs, last) & mask) == (read_word_or_zero(rhs, last) & mask);
}

template <ReadableRow L, ReadableRow R> inline std::strong_ordering operator<=>(const L& lhs, const R& rhs) noexcept {
    if (lhs.size() < rhs.size())
        return std::strong_ordering::less;
    if (lhs.size() > rhs.size())
        return std::strong_ordering::greater;

    const index_t nb = ceil_div64(lhs.size());
    if (nb == 0)
        return std::strong_ordering::equal;

    const index_t last = nb - 1;

    for (index_t k = 0; k < last; ++k) {
        const uint64_t a = read_word_or_zero(lhs, k);
        const uint64_t b = read_word_or_zero(rhs, k);
        if (a < b)
            return std::strong_ordering::less;
        if (a > b)
            return std::strong_ordering::greater;
    }

    const uint64_t mask = tail_mask_bits(lhs.size());
    const uint64_t al   = read_word_or_zero(lhs, last) & mask;
    const uint64_t bl   = read_word_or_zero(rhs, last) & mask;
    if (al < bl)
        return std::strong_ordering::less;
    if (al > bl)
        return std::strong_ordering::greater;
    return std::strong_ordering::equal;
}

template <ReadableRow L, ReadableRow R> inline void assign(L& lhs, const R& rhs) noexcept {
    assert(lhs.size() == rhs.size());

    auto* __restrict dst       = lhs.data();
    const auto* __restrict src = rhs.data();
    const index_t            copy_blocks = std::min(lhs.blocks(), rhs.blocks());

    if (copy_blocks != 0 && dst != src)
        std::memcpy(dst, src, static_cast<std::size_t>(copy_blocks) * sizeof(uint64_t));
    if (copy_blocks < lhs.blocks())
        std::fill_n(dst + copy_blocks, lhs.blocks() - copy_blocks, 0ULL);
    mask_tail_inplace(lhs);
}

// Allows assignment into temporary row views such as matrix[i].
template <ReadableRow L, ReadableRow R> inline void assign(L&& lhs, const R& rhs) noexcept {
    assert(lhs.size() == rhs.size());

    auto* __restrict dst       = lhs.data();
    const auto* __restrict src = rhs.data();
    const index_t            copy_blocks = std::min(lhs.blocks(), rhs.blocks());

    if (copy_blocks != 0 && dst != src)
        std::memcpy(dst, src, static_cast<std::size_t>(copy_blocks) * sizeof(uint64_t));
    if (copy_blocks < lhs.blocks())
        std::fill_n(dst + copy_blocks, lhs.blocks() - copy_blocks, 0ULL);
    mask_tail_inplace(lhs);
}

// -------------------- Owning Row --------------------

class Row {
    index_t               size_ = 0;
    std::vector<uint64_t> blk_;

  public:
    static constexpr index_t npos = k_single_sentinel<index_t>();

    Row() = default;
    template <ReadableRow T>
    Row(const T& readable_row) : size_(readable_row.size()) {
        const index_t dst_blocks = ceil_div64(size_);
        blk_.assign(static_cast<std::size_t>(dst_blocks), 0ULL);
        const index_t take = std::min(dst_blocks, readable_row.blocks());
        if (take != 0)
            std::memcpy(blk_.data(), readable_row.data(), static_cast<std::size_t>(take) * sizeof(uint64_t));
        mask_tail();
    }
    explicit Row(index_t size) : size_(size), blk_(ceil_div64(size), 0ULL) {}

    index_t         size() const noexcept { return size_; }
    index_t         blocks() const noexcept { return blk_.size(); }
    uint64_t*       data() noexcept { return blk_.data(); }
    const uint64_t* data() const noexcept { return blk_.data(); }

    bool    test(index_t i) const noexcept { return (data()[i >> 6] >> (i & 63)) & 1ULL; }
    void    reset() noexcept { std::fill_n(blk_.data(), blk_.size(), 0); }
    void    reset(index_t i) noexcept { data()[i >> 6] &= ~(1ULL << (i & 63)); }
    index_t count() const noexcept { return this->cview().count(); }
    bool    none() const noexcept { return this->cview().none(); }
    void    set(index_t i) noexcept { data()[i >> 6] |= (1ULL << (i & 63)); }
    void    flip(index_t i) noexcept { data()[i >> 6] ^= (1ULL << (i & 63)); }

    RowView  view() noexcept { return RowView{blk_.data(), size_}; }
    RowCView cview() const noexcept { return RowCView{blk_.data(), size_}; }

    operator RowView() noexcept { return view(); }
    operator RowCView() const noexcept { return cview(); }

    void mask_tail() noexcept {
        const index_t rem = size_ & 63;
        if (rem != 0 && !blk_.empty())
            blk_.back() &= ((1ULL << rem) - 1ULL);
    }
};

template <ReadableRow L, ReadableRow R> inline Row operator^(const L& a, const R& b) {
    assert(a.size() == b.size());

    Row out(a.size());
    const index_t copy_blocks = std::min(out.blocks(), a.blocks());
    if (copy_blocks != 0)
        std::memcpy(out.data(), a.data(), static_cast<std::size_t>(copy_blocks) * sizeof(uint64_t));
    const index_t xor_blocks = std::min(out.blocks(), b.blocks());
    for (index_t k = 0; k < xor_blocks; ++k)
        out.data()[k] ^= b.data()[k];
    out.mask_tail();
    return out;
}

// ------------------- MATRIX --------------------------
struct RowHash {
    // enables find(RowCView) without constructing Row
    using is_transparent = void;

    std::size_t operator()(const Row& r) const noexcept;
    std::size_t operator()(RowCView v) const noexcept;

  private:
    static std::size_t hash_view(RowCView v) noexcept;
};

struct RowEq {
    using is_transparent = void;
    bool operator()(RowCView a, RowCView b) const noexcept;
};

class Matrix {
  public:
    Matrix() = default;
    explicit Matrix(index_t rows, index_t cols);
    Matrix(const Matrix&) = default;
    static Matrix zeros(index_t rows, index_t cols);
    static Matrix identity(index_t n);
    static Matrix from_rows(const std::vector<Row>& rows);
    static Matrix from_npy(const std::string& npy_path);
    void          save_npy(const std::string& npy_path) const;

    Matrix& operator=(const Matrix&) = default;

    void reset();
    void reserve_rows(index_t rows);
    auto rows() const noexcept -> index_t { return rows_; }
    auto cols() const noexcept -> index_t { return cols_; }

    RowView  operator[](index_t i) noexcept;
    RowCView operator[](index_t i) const noexcept;

    auto transpose() const -> Matrix;
    auto append_right_inplace(const Matrix& rhs) -> Matrix&;
    auto append_down_inplace(const Matrix& rhs) -> Matrix&;
    auto operator+=(const Matrix& rhs) -> Matrix&;
    auto operator+(const Matrix& rhs) const -> Matrix;
    auto operator*(const Matrix& rhs) const -> Matrix;
    auto operator*(RowCView v) const -> Row;
    bool operator==(const Matrix& rhs) const noexcept;
    bool operator!=(const Matrix& rhs) const noexcept { return !(*this == rhs); }
    void push_back(RowCView bv);

  private:
    index_t rows_           = 0;
    index_t cols_           = 0;
    index_t blocks_per_row_ = 0;
    // row-major contiguous storage: row i starts at data_[i * blocks_per_row_]
    std::vector<uint64_t> data_;
};

std::uint32_t matrix_seed(const Matrix& mat, std::uint32_t base_seed, std::uint32_t step) noexcept;

class Tensor3D {
  public:
    explicit Tensor3D(index_t n);
    explicit Tensor3D(const Matrix& M);

    bool get(index_t i, index_t j, index_t k) const;
    bool operator==(const Tensor3D& other) const noexcept;
    bool operator!=(const Tensor3D& other) const noexcept { return !(*this == other); }

    index_t    n() const noexcept { return n_; }
    const Row& data() const noexcept { return t_; }

  private:
    index_t lin(index_t i, index_t j, index_t k) const noexcept;
    index_t n_{0};
    Row     t_;
};

template <ReadableRow T> std::ostream& operator<<(std::ostream& os, const T& row) {
    for (index_t k = 0; k < row.blocks(); ++k)
        os << std::bitset<64>(row.data()[k]);
    return os;
}
std::ostream& operator<<(std::ostream& os, const Matrix& mat);
} // namespace todd
