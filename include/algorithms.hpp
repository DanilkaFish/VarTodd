#pragma once
#include "matrix.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace todd {
struct PivotMap {
    std::vector<std::int32_t> row;
    // Row                       has_pivot; // packed bitvector: 1 bit/column instead of a byte/word per column
    std::size_t               len  = 0;
    static constexpr auto     npos = std::int32_t{-1};
    std::size_t               size() const { return len; }
    void                      reset(std::size_t n) {
        len = 0;
        if (row.size() < n) {
            row.resize(n);
            // has_pivot = Row(n);
        }
        std::fill(row.begin(), row.end(), npos);
        // has_pivot.reset(); // O(n) but paid once per call, vs. O(n^2) get()s below
    }

    // Branchless: build the row[col] vs npos select from the raw bit word,
    // avoiding a data-dependent (matrix-structure-dependent, hence
    // unpredictable) jump, which used to cost ~10% of total runtime here as
    // a mispredicted branch.
    inline std::int32_t get(std::size_t col) const {
        return row[col];
        // return has_pivot.test(col) ? row[col] : npos;
        // const std::uint64_t bit  = (has_pivot.data()[col >> 6] >> (col & 63)) & 1ULL;
        // const std::int32_t  mask = -static_cast<std::int32_t>(bit); // 0 -> 0x0, 1 -> 0xFFFFFFFF
        // return (row[col] & mask) | (npos & ~mask);
    }
    inline void set(std::size_t col, index_t r) {
        if (r > static_cast<index_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::overflow_error("PivotMap row index exceeds int32 storage");
        row[col] = static_cast<std::int32_t>(r);
        // has_pivot.set(col);
        len++;
    }
};

struct RrefResult {
    Matrix               rref;
    Matrix               transform;
    std::vector<index_t> pivot_cols;
    index_t              rank{};
};
void          gauss_elimination_inplace_rref(Matrix& A, Matrix& aug, PivotMap& pivots);
Matrix        get_tohpe_basis(const Matrix& P);
Matrix        row_dependency_basis(Matrix&& A);
Matrix        basis_gauss_elimination(Matrix&& A);
Matrix        extract_basis(const Matrix& kernel, const PivotMap& pivots);
Matrix        L_expansion(const Matrix& mat);
std::ostream& operator<<(std::ostream& os, const Matrix& mat);

} // namespace todd
