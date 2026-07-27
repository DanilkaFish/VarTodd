#include "matrix.hpp"

#include <algorithm>

#include <cnpy++.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace todd {

static std::size_t checked_mul_size(std::size_t a, std::size_t b, const char* what) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error(what);
    return a * b;
}

static std::size_t checked_add_size(std::size_t a, std::size_t b, const char* what) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error(what);
    return a + b;
}

static index_t checked_tensor_bits(index_t n) {
    const auto N  = static_cast<std::size_t>(n);
    const auto N2 = checked_mul_size(N, N, "Tensor3D allocation overflow");
    const auto N3 = checked_mul_size(N2, N, "Tensor3D allocation overflow");
    if (N3 > std::numeric_limits<index_t>::max())
        throw std::overflow_error("Tensor3D allocation overflow");
    return static_cast<index_t>(N3);
}

static bool points_into_storage(const std::vector<uint64_t>& storage, const uint64_t* p) noexcept {
    if (p == nullptr || storage.empty())
        return false;
    const auto* begin = storage.data();
    const auto* end   = begin + storage.size();
    auto        less  = std::less<const uint64_t*>{};
    return !less(p, begin) && less(p, end);
}

static Matrix canonicalize_rows(std::vector<Row>&& rows, index_t cols) {
    struct RowEntry {
        Row         row;
        std::size_t order;
    };

    std::vector<RowEntry> entries;
    entries.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (!rows[i].none())
            entries.push_back({std::move(rows[i]), i});
    }
    if (entries.empty())
        return Matrix(0, 0);

    std::ranges::sort(entries, [](const RowEntry& lhs, const RowEntry& rhs) { return (lhs.row <=> rhs.row) < 0; });

    std::vector<RowEntry> kept;
    kept.reserve(entries.size());
    index_t active_cols = 0;
    for (std::size_t i = 0; i < entries.size();) {
        std::size_t j = i + 1;
        while (j < entries.size() && entries[i].row == entries[j].row)
            ++j;
        if (((j - i) & 1u) != 0) {
            auto first = entries.begin() + static_cast<std::ptrdiff_t>(i);
            auto last  = entries.begin() + static_cast<std::ptrdiff_t>(j);
            auto best  = std::min_element(first, last, [](const RowEntry& lhs, const RowEntry& rhs) {
                return lhs.order < rhs.order;
            });
            RowCView row = best->row.cview();
            for (auto p = row.find_first(); p != RowCView::npos; p = row.find_next(p))
                active_cols = std::max(active_cols, static_cast<index_t>(p + 1));
            kept.push_back({std::move(best->row), best->order});
        }
        i = j;
    }

    if (kept.empty())
        return Matrix(0, 0);

    std::ranges::sort(kept, [](const RowEntry& lhs, const RowEntry& rhs) { return lhs.order < rhs.order; });

    active_cols = std::min(active_cols, cols);
    Matrix out(0, active_cols);
    out.reserve_rows(static_cast<index_t>(kept.size()));
    for (const RowEntry& entry : kept) {
        if (active_cols == entry.row.size()) {
            out.push_back(entry.row.cview());
            continue;
        }
        Row cropped(active_cols);
        RowCView row = entry.row.cview();
        for (auto p = row.find_first(); p != RowCView::npos && static_cast<index_t>(p) < active_cols;
             p = row.find_next(p)) {
            cropped.set(static_cast<index_t>(p));
        }
        out.push_back(cropped);
    }
    return out;
}

static inline uint64_t mix64(uint64_t x) noexcept {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static inline bool equal_row_blocks(RowCView a, RowCView b) noexcept {
    const index_t nblocks = ceil_div64(a.size());
    if (nblocks == 0)
        return true;
    const index_t last = nblocks - 1;
    for (index_t i = 0; i < last; ++i) {
        if (read_word_or_zero(a, i) != read_word_or_zero(b, i))
            return false;
    }
    const uint64_t mask = tail_mask_bits(a.size());
    return (read_word_or_zero(a, last) & mask) == (read_word_or_zero(b, last) & mask);
}

std::size_t RowHash::operator()(const Row& r) const noexcept { return hash_view(r.cview()); }
std::size_t RowHash::operator()(RowCView v) const noexcept { return hash_view(v); }

std::size_t RowHash::hash_view(RowCView v) noexcept {
    uint64_t      h  = mix64(static_cast<uint64_t>(v.size()));
    const index_t nb = ceil_div64(v.size());
    if (nb) {
        const index_t last = nb - 1;
        for (index_t i = 0; i < last; ++i) {
            h ^= mix64(read_word_or_zero(v, i) + 0x9e3779b97f4a7c15ULL + h);
        }
        h ^= mix64((read_word_or_zero(v, last) & tail_mask_bits(v.size())) + 0x9e3779b97f4a7c15ULL + h);
    }
    return static_cast<std::size_t>(h);
}

bool RowEq::operator()(RowCView a, RowCView b) const noexcept {
    if (a.size() != b.size())
        return false;
    return equal_row_blocks(a, b);
}

std::uint32_t matrix_seed(const Matrix& mat, std::uint32_t base_seed, std::uint32_t step) noexcept {
    constexpr std::uint64_t fnv_offset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t fnv_prime  = 0x100000001b3ULL;
    std::uint64_t           hash       = fnv_offset;
    const auto add = [&](std::uint64_t value) {
        hash ^= value;
        hash *= fnv_prime;
    };

    add(mat.rows());
    add(mat.cols());
    add(base_seed);
    add(step);
    for (index_t i = 0; i < mat.rows(); ++i) {
        const RowCView row = mat[i];
        for (index_t block = 0; block < row.blocks(); ++block) {
            std::uint64_t value = row.data()[block];
            if (block + 1 == row.blocks())
                value &= tail_mask_bits(row.size());
            add(value);
        }
    }
    return static_cast<std::uint32_t>(hash ^ (hash >> 32));
}
// ---------------- Matrix ----------------

Matrix::Matrix(index_t rows, index_t cols)
    : rows_(rows), cols_(cols), blocks_per_row_(ceil_div64(cols)),
      data_(checked_mul_size(static_cast<std::size_t>(rows), static_cast<std::size_t>(blocks_per_row_),
                             "Matrix allocation overflow"),
            0ULL) {}

Matrix Matrix::zeros(index_t rows, index_t cols) { return Matrix(rows, cols); }

Matrix Matrix::identity(index_t n) {
    Matrix I(n, n);
    for (index_t i = 0; i < n; ++i)
        I[i].set(i);
    return I;
}

Matrix Matrix::from_rows(const std::vector<Row>& rows) {
    if (rows.empty())
        return Matrix();
    const index_t cols = rows[0].size();
    Matrix        M(rows.size(), cols);

    for (index_t r = 0; r < M.rows_; ++r) {
        if (static_cast<index_t>(rows[r].size()) != cols)
            throw std::runtime_error("from_rows: inconsistent row width");
        auto rr = M[r];
        assign(rr, rows[r].cview());
        if (rr.blocks() != 0)
            rr.data()[rr.blocks() - 1] &= tail_mask_bits(cols);
    }
    return M;
}

Matrix Matrix::from_npy(const std::string& npy_path) {
    // Load, canonicalize, and compact a uint8 GF(2) matrix from NumPy storage.
    cnpypp::NpyArray arr = cnpypp::npy_load(npy_path);

    if (arr.word_sizes.back() != sizeof(std::uint8_t)) {
        throw std::runtime_error("build_context: expected uint8 .npy data");
    }

    if (arr.shape.size() != 2) {
        throw std::runtime_error("build_context: expected 2D array in .npy");
    }

    const std::size_t n_rows = arr.shape[0];
    const std::size_t n_cols = arr.shape[1];

    const std::uint8_t* data = arr.data<std::uint8_t>();

    std::vector<Row> rows;
    rows.reserve(n_rows);

    const bool fortran_order = arr.memory_order == cnpypp::MemoryOrder::Fortran;
    for (std::size_t i = 0; i < n_rows; ++i) {
        Row row(static_cast<index_t>(n_cols));
        for (std::size_t j = 0; j < n_cols; ++j) {
            const std::size_t offset =
                fortran_order
                    ? checked_add_size(checked_mul_size(j, n_rows, "Matrix::from_npy offset overflow"), i,
                                       "Matrix::from_npy offset overflow")
                    : checked_add_size(checked_mul_size(i, n_cols, "Matrix::from_npy offset overflow"), j,
                                       "Matrix::from_npy offset overflow");
            std::uint8_t v = data[offset];
            if (v != 0) {
                row.set(static_cast<index_t>(j));
            }
        }
        rows.push_back(std::move(row));
    }

    return canonicalize_rows(std::move(rows), static_cast<index_t>(n_cols));
}

void Matrix::save_npy(const std::string& npy_path) const {
    const index_t n_rows = this->rows();
    const index_t n_cols = this->cols();
    const auto    total =
        checked_mul_size(static_cast<std::size_t>(n_rows), static_cast<std::size_t>(n_cols),
                         "Matrix::save_npy allocation overflow");

    std::vector<std::uint8_t> data(total, 0);

    for (index_t i = 0; i < n_rows; ++i) {
        RowCView   row        = (*this)[i];
        const auto row_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n_cols);
        for (auto p = row.find_first(); p != RowCView::npos && p < n_cols; p = row.find_next(p))
            data[row_offset + static_cast<std::size_t>(p)] = 1;
    }

    std::vector<std::size_t> shape = {static_cast<std::size_t>(n_rows), static_cast<std::size_t>(n_cols)};

    cnpypp::npy_save(npy_path, data.data(), shape, "w");
}

void Matrix::reset() { std::ranges::fill(data_, 0); }

void Matrix::reserve_rows(index_t rows) {
    if (rows <= rows_ || cols_ == 0)
        return;
    data_.reserve(checked_mul_size(static_cast<std::size_t>(rows), static_cast<std::size_t>(blocks_per_row_),
                                   "Matrix::reserve_rows overflow"));
}

RowCView Matrix::operator[](index_t i) const noexcept {
    const auto* ptr = blocks_per_row_ == 0 ? nullptr : data_.data() + i * blocks_per_row_;
    return RowCView(ptr, cols_, blocks_per_row_);
}

RowView Matrix::operator[](index_t i) noexcept {
    auto* ptr = blocks_per_row_ == 0 ? nullptr : data_.data() + i * blocks_per_row_;
    return RowView(ptr, cols_, blocks_per_row_);
}

void Matrix::push_back(RowCView bv) {
    if (rows_ == 0 && cols_ == 0) {
        cols_           = static_cast<index_t>(bv.size());
        blocks_per_row_ = ceil_div64(cols_);
    }
    if (static_cast<index_t>(bv.size()) != cols_)
        throw std::runtime_error("push_back: wrong row width");
    if (rows_ == std::numeric_limits<index_t>::max())
        throw std::overflow_error("Matrix::push_back row count overflow");

    const bool source_in_storage = points_into_storage(data_, bv.data());
    Row        source_copy;
    if (source_in_storage)
        source_copy = Row(bv);

    data_.resize(checked_mul_size(static_cast<std::size_t>(rows_ + 1), static_cast<std::size_t>(blocks_per_row_),
                                  "Matrix::push_back overflow"),
                 0ULL);
    assign((*this)[rows_], source_in_storage ? source_copy.cview() : bv);
    if (blocks_per_row_ != 0)
        data_[static_cast<std::size_t>(rows_) * static_cast<std::size_t>(blocks_per_row_) +
              static_cast<std::size_t>(blocks_per_row_ - 1)] &= tail_mask_bits(cols_);
    ++rows_;
}

Matrix& Matrix::append_down_inplace(const Matrix& rhs) {
    if (rhs.rows_ == 0)
        return *this;
    if (this == &rhs) {
        Matrix copy(rhs);
        return append_down_inplace(copy);
    }
    if (rows_ == 0) {
        *this = rhs;
        return *this;
    }
    if (cols_ != rhs.cols_)
        throw std::runtime_error("append_down_inplace: col mismatch");
    if (rhs.rows_ > std::numeric_limits<index_t>::max() - rows_)
        throw std::overflow_error("Matrix::append_down_inplace row count overflow");

    const index_t old_rows = rows_;
    rows_ += rhs.rows_;
    data_.resize(checked_mul_size(static_cast<std::size_t>(rows_), static_cast<std::size_t>(blocks_per_row_),
                                  "Matrix::append_down_inplace overflow"));
    const auto dst_word = checked_mul_size(static_cast<std::size_t>(old_rows), static_cast<std::size_t>(blocks_per_row_),
                                           "Matrix::append_down_inplace offset overflow");
    const auto copy_words =
        checked_mul_size(static_cast<std::size_t>(rhs.rows_), static_cast<std::size_t>(blocks_per_row_),
                         "Matrix::append_down_inplace copy overflow");
    if (copy_words != 0)
        std::memcpy(data_.data() + dst_word, rhs.data_.data(),
                    checked_mul_size(copy_words, sizeof(uint64_t), "Matrix::append_down_inplace byte overflow"));
    return *this;
}

Matrix& Matrix::append_right_inplace(const Matrix& rhs) {
    if (rhs.rows_ == 0)
        return *this;
    if (rows_ == 0) {
        *this = rhs;
        return *this;
    }
    if (rows_ != rhs.rows_)
        throw std::runtime_error("append_right_inplace: row mismatch");
    if (rhs.cols_ == 0)
        return *this;
    if (rhs.cols_ > std::numeric_limits<index_t>::max() - cols_)
        throw std::overflow_error("Matrix::append_right_inplace col count overflow");

    const index_t old_cols = cols_;
    const index_t new_cols = cols_ + rhs.cols_;
    const index_t old_bpr  = blocks_per_row_;
    const index_t rhs_bpr  = rhs.blocks_per_row_;
    const index_t new_bpr  = ceil_div64(new_cols);

    std::vector<uint64_t> new_data(
        checked_mul_size(static_cast<std::size_t>(rows_), static_cast<std::size_t>(new_bpr),
                         "Matrix::append_right_inplace allocation overflow"),
        0ULL);

    const index_t dst_block0 = old_cols >> 6;
    const index_t shift      = old_cols & 63;

    for (index_t r = 0; r < rows_; ++r) {
        const uint64_t* a = old_bpr == 0 ? nullptr : data_.data() + r * old_bpr;
        const uint64_t* b = rhs_bpr == 0 ? nullptr : rhs.data_.data() + r * rhs_bpr;
        uint64_t*       d = &new_data[r * new_bpr];

        if (old_bpr != 0)
            std::memcpy(d, a, old_bpr * sizeof(uint64_t));

        if (shift == 0) {
            std::memcpy(d + dst_block0, b, rhs_bpr * sizeof(uint64_t));
        } else {
            for (index_t k = 0; k < rhs_bpr; ++k) {
                const uint64_t w = b[k];
                d[dst_block0 + k] |= (w << shift);
                if (dst_block0 + k + 1 < new_bpr) {
                    d[dst_block0 + k + 1] |= (w >> (64 - shift));
                }
            }
        }

        const index_t rem = new_cols & 63;
        if (rem != 0)
            d[new_bpr - 1] &= ((1ULL << rem) - 1ULL);
    }

    cols_           = new_cols;
    blocks_per_row_ = new_bpr;
    data_.swap(new_data);
    return *this;
}

Matrix Matrix::transpose() const {
    Matrix T(cols_, rows_);
    for (index_t r = 0; r < rows_; ++r) {
        RowCView rr = (*this)[r];
        for (auto c = rr.find_first(); c != RowCView::npos && c < cols_; c = rr.find_next(c))
            T[c].set(r);
    }
    return T;
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
    if (rows_ != rhs.rows_ || cols_ != rhs.cols_)
        throw std::runtime_error("operator+=: dim mismatch");

    const auto n = checked_mul_size(static_cast<std::size_t>(rows_), static_cast<std::size_t>(blocks_per_row_),
                                    "Matrix::operator+= overflow");
    for (std::size_t i = 0; i < n; ++i)
        data_[i] ^= rhs.data_[i];
    return *this;
}

Matrix Matrix::operator+(const Matrix& rhs) const {
    Matrix out = *this;
    out += rhs;
    return out;
}

Row Matrix::operator*(RowCView v) const {
    if (static_cast<index_t>(v.size()) != cols_)
        throw std::runtime_error("A*v: dim mismatch");
    if (blocks_per_row_ == 0)
        return Row(rows_);

    std::vector<uint64_t> vb_storage;
    const uint64_t*       vb = v.data();
    if (v.blocks() != blocks_per_row_) {
        vb_storage.assign(static_cast<std::size_t>(blocks_per_row_), 0ULL);
        for (auto p = v.find_first(); p != RowCView::npos; p = v.find_next(p)) {
            const index_t ip = static_cast<index_t>(p);
            if (ip < cols_)
                vb_storage[static_cast<std::size_t>(ip >> 6)] |= (1ULL << (ip & 63));
        }
        vb = vb_storage.data();
    }

    Row y(rows_);
    for (index_t r = 0; r < rows_; ++r) {
        const uint64_t* a      = &data_[r * blocks_per_row_];
        unsigned        parity = 0;
        for (index_t k = 0; k < blocks_per_row_; ++k) {
            parity ^= (popcount64(a[k] & vb[k]) & 1U);
        }
        if (parity)
            y.set(r);
    }
    return y;
}

Matrix Matrix::operator*(const Matrix& rhs) const {
    if (cols_ != rhs.rows_)
        throw std::runtime_error("A*B: dim mismatch");

    Matrix C(rows_, rhs.cols_);
    if (blocks_per_row_ == 0)
        return C;

    Matrix Bt = rhs.transpose();

    for (index_t i = 0; i < rows_; ++i) {
        const uint64_t* a = &data_[i * blocks_per_row_];
        for (index_t j = 0; j < Bt.rows_; ++j) {
            const uint64_t* b = &Bt.data_[j * Bt.blocks_per_row_];

            unsigned parity = 0;
            for (index_t k = 0; k < blocks_per_row_; ++k) {
                parity ^= (popcount64(a[k] & b[k]) & 1U);
            }
            if (parity)
                C[i].set(j);
        }
    }
    return C;
}

bool Matrix::operator==(const Matrix& rhs) const noexcept {
    return rows_ == rhs.rows_ && cols_ == rhs.cols_ && data_ == rhs.data_;
}

Tensor3D::Tensor3D(index_t n)
    : n_{n}, t_(checked_tensor_bits(n)) {}

Tensor3D::Tensor3D(const Matrix& M) : Tensor3D(static_cast<index_t>(M.cols())) {
    const index_t rN = static_cast<index_t>(M.rows());
    std::vector<index_t> idx;
    for (index_t r = 0; r < rN; ++r) {
        auto z = M[r];
        idx.clear();
        idx.reserve(z.count());
        for (auto p = z.find_first(); p != RowCView::npos; p = z.find_next(p))
            idx.push_back(static_cast<index_t>(p));
        for (index_t i : idx)
            for (index_t j : idx)
                for (index_t k : idx)
                    t_.flip(static_cast<std::size_t>(lin(i, j, k)));
    }
}

index_t Tensor3D::lin(index_t i, index_t j, index_t k) const noexcept {
    const std::size_t N = static_cast<std::size_t>(n_);
    return static_cast<index_t>(static_cast<std::size_t>(i) + N * static_cast<std::size_t>(j) +
                                N * N * static_cast<std::size_t>(k));
}

bool Tensor3D::get(index_t i, index_t j, index_t k) const {
    if (i >= n_ || j >= n_ || k >= n_)
        throw std::out_of_range("Tensor3D::get");
    return t_.test(static_cast<std::size_t>(lin(i, j, k)));
}

bool Tensor3D::operator==(const Tensor3D& other) const noexcept { return n_ == other.n_ && t_ == other.t_; }
std::ostream&  operator<<(std::ostream& os, const Matrix& mat) {
    os << "[\n";
    for (index_t i = 0; i < mat.rows(); i++) {
        os << mat[i];
    }
    return os << "]";
}
} // namespace todd
