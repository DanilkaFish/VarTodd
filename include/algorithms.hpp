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
    std::vector<std::int32_t>  row;
    std::vector<std::uint32_t> tag;
    std::size_t                len   = 0;
    static constexpr auto      npos  = std::int32_t{-1};
    std::uint32_t              epoch = 1;
    std::size_t                size() const { return len; }
    void                       reset(std::size_t n) {
        len = 0;
        if (row.size() < n) {
            row.resize(n);
            tag.resize(n, 0);
        }
        ++epoch;
        if (epoch == 0) {
            std::fill(tag.begin(), tag.end(), 0);
            epoch = 1;
        }
    }

    inline std::int32_t get(std::size_t col) const { return (tag[col] == epoch) ? row[col] : npos; }
    inline void         set(std::size_t col, index_t r) {
        if (r > static_cast<index_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::overflow_error("PivotMap row index exceeds int32 storage");
        row[col] = static_cast<std::int32_t>(r);
        tag[col] = epoch;
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
