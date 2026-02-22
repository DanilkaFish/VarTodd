#include "matrix.hpp"

#include <algorithm>
#include <bit>
#include <boost/dynamic_bitset.hpp>
#include <boost/functional/hash.hpp>

#include <cnpy++.hpp>
#include <cstdint>
#include <iterator>
#include <stdexcept>

namespace todd {
constexpr auto __bl_size = 4;

static inline uint64_t mix64(uint64_t x) noexcept {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static inline bool equal_row_blocks(const uint64_t* a, const uint64_t* b, index_t nblocks, index_t cols) noexcept {
    if (nblocks == 0)
        return true;
    const index_t last = nblocks - 1;
    for (index_t i = 0; i < last; ++i) {
        if (a[i] != b[i])
            return false;
    }
    const uint64_t mask = tail_mask_bits(cols);
    return ((a[last] & mask) == (b[last] & mask));
}

size_t RowHash::operator()(const Row& r) const noexcept { return hash_view(r.cview()); }
size_t RowHash::operator()(RowCView v) const noexcept { return hash_view(v); }

size_t RowHash::hash_view(RowCView v) noexcept {
    uint64_t        h  = mix64((uint64_t)v.size());
    const uint64_t* p  = v.data();
    const index_t   nb = v.blocks();
    if (nb) {
        const index_t last = nb - 1;
        for (index_t i = 0; i < last; ++i) {
            h ^= mix64(p[i] + 0x9e3779b97f4a7c15ULL + h);
        }
        h ^= mix64((p[last] & tail_mask_bits(v.size())) + 0x9e3779b97f4a7c15ULL + h);
    }
    return (size_t)h;
}

bool RowEq::operator()(RowCView a, RowCView b) const noexcept {
    return a.size() == b.size() && a.blocks() == b.blocks() &&
           equal_row_blocks(a.data(), b.data(), a.blocks(), a.size());
}

std::uint32_t matrix_seed(const Matrix& mat, std::uint32_t base_seed, std::uint32_t step) noexcept {
    std::size_t h = 0xcbf29ce484222325ULL; // some non-zero starting value

    boost::hash_combine(h, static_cast<std::size_t>(base_seed));
    boost::hash_combine(h, static_cast<std::size_t>(step));
    RowHash rh{};

    for (index_t i = 0; i < mat.rows(); i++) {
        boost::hash_combine(h, rh(mat[i]));
    }

    std::uint64_t h64  = static_cast<std::uint64_t>(h);
    std::uint32_t seed = static_cast<std::uint32_t>(h64 ^ (h64 >> 32));
    return seed;
}
// ---------------- Matrix ----------------

Matrix::Matrix(index_t rows, index_t cols)
    : rows_(rows), cols_(cols), blocks_per_row_(ceil_div64(cols)), data_(rows * blocks_per_row_, 0ULL) {}

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
        auto rr    = M[r];
        auto newrr = rows[r].cview();
        for (auto p = newrr.find_first(); p != RowCView::npos; p = newrr.find_next(p)) {
            rr.set(static_cast<index_t>(p));
        }
    }
    return M;
}

// TODO : elimination of double columns
Matrix Matrix::from_npy(const std::string& npy_path) {
    // Load the .npy file
    cnpypp::NpyArray arr = cnpypp::npy_load(npy_path);

    if (arr.word_sizes.back() != sizeof(std::uint8_t)) {
        throw std::runtime_error("build_context: expected uint8 .npy data");
    }

    if (arr.shape.size() != 2) {
        throw std::runtime_error("build_context: expected 2D array in .npy");
    }

    const std::size_t n_rows = arr.shape[1];
    const std::size_t n_cols = arr.shape[0];

    const std::uint8_t* data = arr.data<std::uint8_t>();

    // Construct GF(2) Matrix of size n_rows x n_cols
    Matrix mat(static_cast<index_t>(n_rows), static_cast<index_t>(n_cols));

    // Assuming row-major layout in the .npy file:
    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            std::uint8_t v = data[j * n_rows + i];
            if (v & 1u) {
                mat[static_cast<index_t>(i)].set(static_cast<index_t>(j));
            }
        }
    }

    return mat;
}

void Matrix::save_npy(const std::string& npy_path) const {
    // Get matrix dimensions
    const index_t n_rows = this->rows();
    const index_t n_cols = this->cols();
    
    // Create a buffer for the uint8 data (row-major layout)
    std::vector<std::uint8_t> data(static_cast<std::size_t>(n_rows * n_cols), 0);
    
    // Fill the buffer with matrix data (assuming row-major layout)
    for (index_t i = 0; i < n_rows; ++i) {
        const auto& row = (*this)[i];
        for (index_t j = 0; j < n_cols; ++j) {
            if (row.test(j)) {
                // Convert to column-major indexing if that's what from_npy expects
                // data[j * n_rows + i] = 1;
                
                // Or use row-major if that's your preferred format
                data[static_cast<std::size_t>(i * n_cols + j)] = 1;
            }
        }
    }
    
    // Save as .npy file with shape [n_cols, n_rows] to match from_npy's expected layout
    std::vector<std::size_t> shape = {static_cast<std::size_t>(n_cols), 
                                      static_cast<std::size_t>(n_rows)};
    
    // If you want row-major storage in the file, use shape [n_rows, n_cols] instead
    // std::vector<std::size_t> shape = {static_cast<std::size_t>(n_rows), 
    //                                   static_cast<std::size_t>(n_cols)};
    
    cnpypp::npy_save(npy_path, data.data(), shape, "w");
}

void Matrix::reset() { std::ranges::fill(data_, 0); }

RowCView Matrix::operator[](index_t i) const noexcept {
    return RowCView(&data_[i * blocks_per_row_], cols_, blocks_per_row_);
}

RowView Matrix::operator[](index_t i) noexcept { return RowView(&data_[i * blocks_per_row_], cols_, blocks_per_row_); }

void Matrix::push_back(RowCView bv) {
    if (rows_ == 0 && cols_ == 0) {
        cols_           = static_cast<index_t>(bv.size());
        blocks_per_row_ = ceil_div64(cols_);
    }
    if (static_cast<index_t>(bv.size()) != cols_)
        throw std::runtime_error("push_back: wrong row width");

    data_.resize((rows_ + 1) * blocks_per_row_, 0ULL);
    assign((*this)[rows_], bv);
    ++rows_;
}

Matrix& Matrix::append_down_inplace(const Matrix& rhs) {
    if (rhs.rows_ == 0)
        return *this;
    if (rows_ == 0) {
        *this = rhs;
        return *this;
    }
    if (cols_ != rhs.cols_)
        throw std::runtime_error("append_down_inplace: col mismatch");

    const index_t old_rows = rows_;
    rows_ += rhs.rows_;
    data_.resize(rows_ * blocks_per_row_);
    std::memcpy(&data_[old_rows * blocks_per_row_], rhs.data_.data(), rhs.rows_ * blocks_per_row_ * sizeof(uint64_t));
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

    const index_t old_cols = cols_;
    const index_t new_cols = cols_ + rhs.cols_;
    const index_t old_bpr  = blocks_per_row_;
    const index_t rhs_bpr  = rhs.blocks_per_row_;
    const index_t new_bpr  = ceil_div64(new_cols);

    std::vector<uint64_t> new_data(rows_ * new_bpr, 0ULL);

    const index_t dst_block0 = old_cols >> 6;
    const index_t shift      = old_cols & 63;

    for (index_t r = 0; r < rows_; ++r) {
        const uint64_t* a = &data_[r * old_bpr];
        const uint64_t* b = &rhs.data_[r * rhs_bpr];
        uint64_t*       d = &new_data[r * new_bpr];

        // copy left part
        std::memcpy(d, a, old_bpr * sizeof(uint64_t));

        // append right part at bit offset old_cols
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

        // mask unused high bits in final block
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
        for (index_t c = 0; c < cols_; ++c) {
            if (rr.test(c))
                T[c].set(r);
        }
    }
    return T;
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
    if (rows_ != rhs.rows_ || cols_ != rhs.cols_)
        throw std::runtime_error("operator+=: dim mismatch");

    const index_t n = rows_ * blocks_per_row_;
    for (index_t i = 0; i < n; ++i)
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

    // convert v to blocks once
    std::vector<uint64_t> vb(blocks_per_row_, 0ULL);
    for (auto p = v.find_first(); p != RowCView::npos; p = v.find_next(p)) {
        index_t ip = static_cast<index_t>(p);
        vb[ip >> 6] |= (1ULL << (ip & 63));
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

    Matrix Bt = rhs.transpose();
    Matrix C(rows_, rhs.cols_);

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
    : n_{n}, t_(static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(n)) {}

Tensor3D::Tensor3D(const Matrix& M) : Tensor3D(static_cast<index_t>(M.cols())) {
    const index_t rN = static_cast<index_t>(M.rows());
    for (index_t r = 0; r < rN; ++r) {
        auto                 z = M[r];
        std::vector<index_t> idx;
        idx.reserve(z.count());
        for (auto p = z.find_first(); p != RowView::npos; p = z.find_next(p))
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
