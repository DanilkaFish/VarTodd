#include "algorithms.hpp"

#include "matrix.hpp"
#include "random.hpp"
#include "todd_index.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <sys/types.h>
#include <unordered_map>

namespace todd {
constexpr int MAX_ROWS_IN_BASIS = 100;

static inline Row extract_right(RowCView row, index_t divider, index_t nY) {
    Row y(nY);
    if (nY == 0)
        return y;

    uint64_t*       dst = y.data();
    const uint64_t* src = row.data();

    const index_t dst_blocks = y.blocks();
    const index_t src_blocks = row.blocks();

    const index_t  sb = divider >> 6;
    const unsigned sh = (unsigned)(divider & 63);

    if (sh == 0) {
        const index_t avail = (sb < src_blocks) ? (src_blocks - sb) : 0;
        const index_t take  = std::min(dst_blocks, avail);

        if (take)
            std::memcpy(dst, src + sb, (std::size_t)take * sizeof(uint64_t));
        if (take < dst_blocks)
            std::memset(dst + take, 0, (std::size_t)(dst_blocks - take) * sizeof(uint64_t));
    } else {
        for (index_t k = 0; k < dst_blocks; ++k) {
            const index_t  i0 = sb + k;
            const uint64_t lo = (i0 < src_blocks) ? src[i0] : 0ULL;
            const uint64_t hi = (i0 + 1 < src_blocks) ? src[i0 + 1] : 0ULL;
            dst[k]            = (lo >> sh) | (hi << (64u - sh));
        }
    }

    // mask tail bits
    const index_t rem = nY & 63;
    if (rem != 0)
        dst[dst_blocks - 1] &= ((1ULL << rem) - 1ULL);

    return y;
}

static inline void apply_solution_transform(Row& y, const SumEntry* ptr, int len) {
    bool odd = (y.count() & 1u) != 0;

    for (int t = 0; t < len; ++t) {
        const SumEntry& e = ptr[t];

        if (e.is_pair()) {
            // Original semantics:
            // if (y.test(b)) { y.flip(a); y.flip(b); }
            //
            // Parity is unchanged here:
            // - either 0 flips
            // - or 2 flips
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

static inline bool insert_into_y_basis(Row& y, Matrix& basis, PivotMap& pivY) {
    auto yv = y.view();
    auto q  = yv.find_first();

    while (q != RowView::npos) {
        const int32_t brow = pivY.get((std::size_t)q);
        if (brow < 0) {
            pivY.set((std::size_t)q, (int32_t)basis.rows());
            basis.push_back(y.cview());   // copy reduced row into basis
            return true;
        }

        yv ^= basis[(index_t)brow];
        q = yv.find_first();
    }

    return false; // dependent
}

Matrix solve_and_build_solution_basis(Matrix& A,
                                      index_t divider,
                                      const Matrix& tohpe,
                                      const SumEntry* ptr,
                                      int len) {
    // If not, fall back to divider + tohpe.cols().
    const index_t nTotal = (A.cols() != 0) ? A.cols() : (divider + tohpe.cols());

    if (divider > nTotal)
        throw std::invalid_argument("solve_and_build_solution_basis: divider > cols");
    const index_t nY = nTotal - divider;

    if (nY == 0) return Matrix(0, 0);

    if (tohpe.rows() != 0 && tohpe.cols() != nY)
        throw std::invalid_argument("solve_and_build_solution_basis: tohpe width mismatch");

    static thread_local PivotMap pivA;
    static thread_local PivotMap pivY;

    pivA.reset((std::size_t)divider);
    pivY.reset((std::size_t)nY);

    const index_t full_rank = nY;
    Matrix        basis(0, nY);
    basis.reserve_rows(full_rank);

    for (index_t i = 0; i < A.rows() && basis.rows() < full_rank; ++i) {
        auto row = A[i];

        auto p = row.find_first();
        while (p != RowView::npos && p < divider) {
            const int32_t prow = pivA.get((std::size_t)p);
            if (prow < 0)
                break;

            row ^= A[(index_t)prow];
            p = row.find_next(p);
        }

        if (p == RowView::npos)
            continue;

        if (p < divider) {
            pivA.set((std::size_t)p, (int32_t)i);
            continue;
        }

        Row y = extract_right((RowCView)row, divider, nY);
        apply_solution_transform(y, ptr, len);
        insert_into_y_basis(y, basis, pivY);
    }

    for (index_t i = 0; i < tohpe.rows() && basis.rows() < full_rank; ++i) {
        Row y = tohpe[i];

        apply_solution_transform(y, ptr, len);
        insert_into_y_basis(y, basis, pivY);
    }

    return basis;
}

Matrix solve_and_build_solution_basis_generated(index_t rows,
                                                index_t cols,
                                                index_t divider,
                                                const Matrix& tohpe,
                                                const SumEntry* ptr,
                                                int len,
                                                const std::function<void(index_t, RowView)>& fill_row) {
    if (divider > cols)
        throw std::invalid_argument("solve_and_build_solution_basis_generated: divider > cols");
    const index_t nY = cols - divider;
    if (nY == 0)
        return Matrix(0, 0);
    if (tohpe.rows() != 0 && tohpe.cols() != nY)
        throw std::invalid_argument("solve_and_build_solution_basis_generated: tohpe width mismatch");

    static thread_local PivotMap pivA;
    static thread_local PivotMap pivY;

    pivA.reset((std::size_t)divider);
    pivY.reset((std::size_t)nY);

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
            const int32_t prow = pivA.get((std::size_t)p);
            if (prow < 0)
                break;

            rowv ^= pivot_rows[(index_t)prow];
            p = rowv.find_next(p);
        }

        if (p == RowView::npos)
            continue;

        if (p < divider) {
            pivA.set((std::size_t)p, (int32_t)pivot_rows.rows());
            pivot_rows.push_back(row.cview());
            continue;
        }

        Row y = extract_right(row.cview(), divider, nY);
        apply_solution_transform(y, ptr, len);
        insert_into_y_basis(y, basis, pivY);
    }

    for (index_t i = 0; i < tohpe.rows() && basis.rows() < full_rank; ++i) {
        Row y = tohpe[i];

        apply_solution_transform(y, ptr, len);
        insert_into_y_basis(y, basis, pivY);
    }

    return basis;
}

Matrix basis_gauss_elimination(Matrix&& A) {
    Matrix linear_indep;
    linear_indep.reserve_rows(A.rows());
    std::vector<index_t> pivot_cols;
    pivot_cols.reserve(A.rows());

    for (index_t r = 0; r < A.rows(); ++r) {
        auto row = A[r];

        for (size_t k = 0; k < pivot_cols.size(); ++k) {
            const index_t p = pivot_cols[k];
            if (row.test(p)) {
                row ^= linear_indep[k];
            }
        }

        const index_t p = row.find_first();
        if (p != RowView::npos) {
            linear_indep.push_back(row);
            pivot_cols.push_back(p);
            assign(A[r], row);
        }
    }
    return linear_indep;
}

Matrix row_dependency_basis(Matrix&& A) {
    if (A.rows() == 0)
        return Matrix(0, 0);

    Matrix   row_basis;
    Matrix   transform_basis;
    Matrix   dependencies;
    row_basis.reserve_rows(A.rows());
    transform_basis.reserve_rows(A.rows());
    dependencies.reserve_rows(A.rows());
    PivotMap pivots;
    pivots.reset(A.cols());

    for (index_t i = 0; i < A.rows(); ++i) {
        auto row = A[i];
        Row  y(A.rows());
        y.set(i);

        while (true) {
            const auto p = row.find_first();
            if (p == RowView::npos) {
                dependencies.push_back(y);
                break;
            }

            const int32_t brow = pivots.get((std::size_t)p);
            if (brow < 0) {
                pivots.set((std::size_t)p, (int32_t)row_basis.rows());
                row_basis.push_back(row);
                transform_basis.push_back(y);
                break;
            }

            row ^= row_basis[(index_t)brow];
            y ^= transform_basis[(index_t)brow];
        }
    }

    return dependencies;
}

void gauss_elimination_inplace_rref(Matrix& A, Matrix& aug, PivotMap& pivots) {
    if (A.rows() != aug.rows())
        throw std::invalid_argument("gauss_jordan_rref_inplace: row mismatch");
    const index_t m = A.rows();

    for (index_t i = 0; i < m; ++i) {
        auto row  = A[i];
        auto arow = aug[i];

        while (true) {
            // TODO DOES find_first really neccesary
            auto p = row.find_first();
            if (p == RowView::npos) break;
            const int32_t prow = pivots.get((size_t)p); 
            if (prow < 0) break;              
            if ((index_t)prow == i) break;
            row  ^= A[(index_t)prow];
            arow ^= aug[(index_t)prow];
        }

        auto p = row.find_first();
        if (p == RowView::npos) continue;
        if (pivots.get((size_t)p) >= 0) {
            continue;
        }
        pivots.set((size_t)p, (int32_t)i);

        for (index_t r = 0; r < m; ++r) {
            if (r == i) continue;
            if (A[r].test(p)) {
                A[r]   ^= row;
                aug[r] ^= arow;
            }
        }
    }
}

Matrix extract_basis(const Matrix& kernel, const PivotMap& pivots) {
    Matrix _kernel(0, 0);
    // TODO memore allocation overhead
    std::vector<std::size_t> not_used_rows{};
    not_used_rows.resize(kernel.rows(), 1);
    for (auto row: pivots.row) {
        if (row != PivotMap::npos) {
            not_used_rows[row] = 0;
        }
    }
    for (index_t i = 0; i < kernel.rows(); i++) {
        if (not_used_rows[i]) {
            _kernel.push_back(kernel[i]);
        }
    }
    return _kernel;
}

static inline void or_copy_prefix_shifted(uint64_t* dst, std::size_t dst_nblocks, std::size_t dst_bit_off,
                                          const uint64_t* src, std::size_t prefix_bits) {
    if (!prefix_bits)
        return;

    const std::size_t d0   = dst_bit_off >> 6;
    const unsigned    sh   = (unsigned)(dst_bit_off & 63);
    const std::size_t full = prefix_bits >> 6;
    const unsigned    rem  = (unsigned)(prefix_bits & 63);

    auto or_word = [&](std::size_t idx, uint64_t w) {
        dst[d0 + idx] |= (sh ? (w << sh) : w);
        if (sh && d0 + idx + 1 < dst_nblocks)
            dst[d0 + idx + 1] |= (w >> (64u - sh));
    };

    for (std::size_t k = 0; k < full; ++k)
        or_word(k, src[k]);
    if (rem)
        or_word(full, src[full] & ((1ULL << rem) - 1ULL));
}

Matrix L_expansion(const Matrix& mat) {
    const index_t n = mat.rows(), c = mat.cols();
    if (!n || !c)
        return Matrix(n, 0);

    Matrix out(n, (c * (c + 1)) / 2);

    for (index_t i = 0; i < n; ++i) {
        auto            srow = mat[i]; // RowCRef
        auto            drow = out[i]; // RowRef
        uint64_t*       d    = drow.data();
        const uint64_t* s    = srow.data();

        std::size_t pos = 0;
        for (index_t L = c; L > 0; --L) {
            if (srow.test(L - 1))
                or_copy_prefix_shifted(d, (std::size_t)drow.blocks(), pos, s, (std::size_t)L);
            pos += (std::size_t)L;
        }
    }
    return out;
}

index_t PyRNG::rand_int(index_t low, index_t high) {
    std::uniform_int_distribution<index_t> dist(low, high);
    return dist(rng);
}
uint64_t PyRNG::rand_u64() {
    std::uniform_int_distribution<uint64_t> dist((uint64_t)0, std::numeric_limits<uint64_t>::max());
    return dist(rng);
}
double PyRNG::rand_double(double low, double high) {
    std::uniform_real_distribution<double> dist(low, high);
    return dist(rng);
}

index_t PyRNG::random_raw() { return rng(); }

void PyRNG::seed(index_t seed_val) { rng.seed(seed_val); }

std::minstd_rand& PyRNG::get_engine() { return rng; }

static inline Row mask_to_bv(index_t dim, uint64_t mask) {
    Row v(dim);
    v.data()[0] = mask;
    return v;
}

std::vector<Row> PyRNG::sample_special_bitvec(const Matrix& basis, index_t i, index_t j, index_t num_samples) {
    auto                                   dim = basis.rows();
    const index_t                          N   = 1ULL << dim;
    std::vector<Row>                       out;
    std::unordered_map<uint64_t, uint64_t> remap;
    remap.reserve((size_t)num_samples * 2);
    for (index_t j = std::max(N - num_samples, (index_t)1); j < N; ++j) {
        index_t t   = rand_int(1, j); // <- your RNG: inclusive range
        auto    itT = remap.find(t);
        index_t x   = (itT == remap.end()) ? t : itT->second;
        auto    itJ = remap.find(j);
        index_t y   = (itJ == remap.end()) ? j : itJ->second;
        remap[t]    = y;
        out.push_back(mask_to_bv(dim, x));
    }

    return out;
}

std::vector<uint32_t> PyRNG::floyd_sample_0n(uint32_t n, index_t k)
{
    std::unordered_map<uint32_t, uint32_t> map;
    map.reserve(k * 2);

    std::vector<uint32_t> out;
    out.reserve(k);

    for (uint32_t j = (uint32_t)(n - k); j < (uint32_t)n; ++j) {
        uint32_t t = (uint32_t)rand_int(0, j); // inclusive
        auto itT = map.find(t);
        uint32_t x = (itT == map.end()) ? t : itT->second;

        auto itJ = map.find(j);
        uint32_t y = (itJ == map.end()) ? j : itJ->second;

        map[t] = y;
        out.push_back(x);
    }
    return out;
}

// Floyd sampling without replacement from [1..n], returns k unique ints, no retries.
std::vector<uint32_t> PyRNG::floyd_sample_1n(uint32_t n, index_t k)
{
    std::unordered_map<uint32_t, uint32_t> map;
    map.reserve(k * 2);

    std::vector<uint32_t> out;
    out.reserve(k);

    // sample k numbers from [1..n]
    for (uint32_t j = (uint32_t)(n - k + 1); j <= n; ++j) {
        uint32_t t = (uint32_t)rand_int(1, j);
        auto itT = map.find(t);
        uint32_t x = (itT == map.end()) ? t : itT->second;

        auto itJ = map.find(j);
        uint32_t y = (itJ == map.end()) ? j : itJ->second;

        map[t] = y;
        out.push_back(x);
    }
    return out;
}

Row PyRNG::sample_bitvector(index_t dim) {
    Row r(dim);
    sample_bitvector_inplace(r);
    return r;
}

void PyRNG::sample_bitvector_inplace(Row& r) {
    const index_t dim = r.size();
    if (dim == 0)
        return;

    uint64_t*     dst = r.data();
    const index_t nb  = r.blocks();

    for (index_t i = 0; i < nb; ++i) {
        dst[i] = rand_u64();
    }

    // Clear unused high bits in the last block
    const index_t rem = dim & 63;
    if (rem != 0) {
        dst[nb - 1] &= ((1ULL << rem) - 1ULL);
    }
    if (r.count() == 0) {
        for (index_t i = 0; i < nb; ++i) {
            dst[i] = -1;
        }

        const index_t rem = dim & 63;
        if (rem != 0) {
            dst[nb - 1] &= ((1ULL << rem) - 1ULL);
        }
    }
}
Matrix get_tohpe_basis(const Matrix& P) {
    return row_dependency_basis(L_expansion(P));
}

} // namespace todd
