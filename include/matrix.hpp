#pragma once
#include "typedef.hpp"
#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace todd {
static inline index_t ceil_div64(index_t bits) { return (bits + 63) / 64; }

static inline uint64_t tail_mask_bits(index_t nbits) {
    const index_t rem = nbits & 63;
    return rem ? ((1ULL << rem) - 1ULL) : ~0ULL;
}
static inline unsigned popcount64(uint64_t x) noexcept { return static_cast<unsigned>(__builtin_popcountll(x)); }

// -------------------- Views --------------------
template <class Ptr> class BasicRowView {
    Ptr     p_      = nullptr;
    index_t size_   = 0;
    index_t blocks_ = 0;

  public:
    static constexpr index_t npos = (index_t)-1;

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
        auto* __restrict dst = this->data();

        for (index_t k = 0; k < this->blocks(); ++k)
            if (dst[k] != 0)
                return false;
        return true;
    }
    index_t count() const noexcept {
        if (ceil_div64(size_) == 0)
            return 0;
        index_t total = 0;
        for (index_t k = 0; k + 1 < ceil_div64(size_); ++k) {
            total += popcount64(p_[k]);
        }
        const index_t rem  = size_ & 63;
        uint64_t      last = p_[ceil_div64(size_) - 1];
        if (rem != 0) {
            const uint64_t mask = (rem == 64) ? ~0ULL : ((1ULL << rem) - 1ULL);
            last &= mask;
        }
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
        data()[i >> 6] &= !(1ULL << (i & 63));
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
        const auto* __restrict p   = p_;
        const auto* __restrict end = p_ + blocks_;
        while (p != end && *p == 0)
            ++p;
        if (p == end)
            return npos;

        size_t k   = size_t(p - p_);
        size_t bit = std::countr_zero(*p);
        return (k << 6) + bit;
    }

    index_t find_next(index_t pos) const noexcept {
        if (size_ == 0 || pos >= size_ - 1)
            return npos;
        ++pos;

        size_t   k = pos >> 6;
        uint64_t w = p_[k] & (~0ULL << (pos & 63));

        if (w) {
            return (k << 6) + std::countr_zero(w);
        }

        const auto* __restrict p   = p_ + k + 1;
        const auto* __restrict end = p_ + blocks_;
        while (p != end && *p == 0)
            ++p;
        if (p == end)
            return npos;

        return ((p - p_) << 6) + std::countr_zero(*p);
    }
    // implicit downgrade: RowView -> RowCView
    operator BasicRowView<const uint64_t*>() const noexcept { return {data(), size_}; }
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

template <WritableRow L, ReadableRow R> inline L& operator^=(L& lhs, const R& rhs) noexcept {
    assert(lhs.size() == rhs.size());
    assert(lhs.blocks() == rhs.blocks());
    assert(lhs.data() != rhs.data());
    xor_rows(lhs.data(), rhs.data(), lhs.blocks());
    return lhs;
}

// TODO: not good lvalue usage
template <WritableRow L, ReadableRow R> inline L&& operator^=(L&& lhs, const R& rhs) noexcept {
    assert(lhs.size() == rhs.size());
    assert(lhs.blocks() == rhs.blocks());
    assert(lhs.data() != rhs.data());
    xor_rows(lhs.data(), rhs.data(), lhs.blocks());
    return lhs;
}

template <ReadableRow L, ReadableRow R> inline bool operator==(const L& lhs, const R& rhs) noexcept {
    assert(lhs.blocks() == rhs.blocks());

    const uint64_t* a = lhs.data();
    const uint64_t* b = rhs.data();

    const index_t nb = lhs.blocks();
    if (nb == 0)
        return true;

    const index_t last = nb - 1;
    for (index_t k = 0; k < last; ++k)
        if (a[k] != b[k])
            return false;

    return ((a[last]) == (b[last]));
}

template <ReadableRow L, ReadableRow R> inline std::strong_ordering operator<=>(const L& lhs, const R& rhs) noexcept {
    assert(lhs.blocks() == rhs.blocks());

    const uint64_t* a  = lhs.data();
    const uint64_t* b  = rhs.data();
    const index_t   nb = lhs.blocks();
    if (nb == 0)
        return std::strong_ordering::equal;

    const index_t last = nb;

    for (index_t k = 0; k < last; ++k) {
        if (a[k] < b[k])
            return std::strong_ordering::less;
        if (a[k] > b[k])
            return std::strong_ordering::greater;
    }
    return std::strong_ordering::equal;
}

template <ReadableRow L, ReadableRow R> inline void assign(L& lhs, const R& rhs) noexcept {
    assert(lhs.size() == rhs.size());
    assert(lhs.blocks() == rhs.blocks());

    auto* __restrict dst       = lhs.data();
    const auto* __restrict src = rhs.data();

    for (index_t k = 0; k < lhs.blocks(); ++k)
        dst[k] = src[k];
}

// used for view object which keep memories
template <ReadableRow L, ReadableRow R> inline void assign(L&& lhs, const R& rhs) noexcept {
    assert(lhs.size() == rhs.size());
    assert(lhs.blocks() == rhs.blocks());

    auto* __restrict dst       = lhs.data();
    const auto* __restrict src = rhs.data();

    for (index_t k = 0; k < lhs.blocks(); ++k)
        dst[k] = src[k];
}

// -------------------- Owning Row --------------------

class Row {
    index_t               size_ = 0;
    std::vector<uint64_t> blk_;

  public:
    static constexpr index_t npos = (index_t)-1;

    Row() = default;
    template <ReadableRow T>
    Row(const T& readable_row)
        : size_(readable_row.size()), blk_(readable_row.data(), readable_row.data() + readable_row.blocks()) {}
    explicit Row(index_t size) : size_(size), blk_(ceil_div64(size), 0ULL) {}

    index_t         size() const noexcept { return size_; }
    index_t         blocks() const noexcept { return blk_.size(); }
    uint64_t*       data() noexcept { return blk_.data(); }
    const uint64_t* data() const noexcept { return blk_.data(); }

    bool    test(index_t i) const noexcept { return (data()[i >> 6] >> (i & 63)) & 1ULL; }
    void    reset(index_t i) noexcept { data()[i >> 6] &= !(1ULL << (i & 63)); }
    index_t count() const noexcept { return this->cview().count(); }
    bool    none() const noexcept { return this->cview().none(); }
    void    set(index_t i) noexcept { data()[i >> 6] |= (1ULL << (i & 63)); }
    void    flip(index_t i) noexcept { data()[i >> 6] ^= (1ULL << (i & 63)); }

    RowView  view() noexcept { return RowView{blk_.data(), size_}; }
    RowCView cview() const noexcept { return RowCView{blk_.data(), size_}; }

    operator RowView() noexcept { return view(); }
    operator RowCView() const noexcept { return cview(); }

    void mask_tail() noexcept {
        if (!blk_.empty()) {
            blk_.back() &= tail_mask_bits(size_);
        }
    }
};

template <ReadableRow L, ReadableRow R> inline Row operator^(const L& a, const R& b) {
    assert(a.size() == b.size());
    assert(a.blocks() == b.blocks());

    Row out(a.size());
    std::copy_n(a.data(), a.blocks(), out.data());
    out ^= b; // uses operator^= above (Row is WritableRow)
    out.mask_tail();
    return out;
}

// ------------------- MATRIX --------------------------
struct RowHash {
    // enables find(RowCView) without constructing Row
    using is_transparent = void;

    size_t operator()(const Row& r) const noexcept;
    size_t operator()(RowCView v) const noexcept;

  private:
    static size_t hash_view(RowCView v) noexcept;
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
	void save_npy(const std::string& npy_path) const;

    Matrix& operator=(const Matrix&) = default;

    void reset();
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
    for (auto bl = row.data(); bl - row.data() < row.blocks(); bl++) {
        os << std::bitset<64>(*bl);
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const Matrix& mat);
} // namespace todd