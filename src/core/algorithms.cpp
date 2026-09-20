#include "algorithms.hpp"

#include "matrix.hpp"
#include "random.hpp"
#include "todd_index.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace todd {

static std::size_t checked_mul_size(std::size_t a, std::size_t b, const char* what) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error(what);
    return a * b;
}

static index_t checked_add_index(index_t a, index_t b, const char* what) {
    if (b > std::numeric_limits<index_t>::max() - a)
        throw std::overflow_error(what);
    return a + b;
}

static std::size_t triangular_count_size(index_t n, const char* what) {
    const auto N = static_cast<std::size_t>(n);
    if (N == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error(what);
    return (N & 1U) == 0 ? checked_mul_size(N / 2, N + 1, what) : checked_mul_size(N, (N + 1) / 2, what);
}

static index_t triangular_col_count(index_t n) {
    const std::size_t cols = triangular_count_size(n, "L_expansion column overflow");
    if (cols > std::numeric_limits<index_t>::max())
        throw std::overflow_error("L_expansion column overflow");
    return static_cast<index_t>(cols);
}

// Solve the left block while collecting independent transformed right-block rows.
Matrix solve_and_build_solution_basis(Matrix& A,
                                      index_t divider,
                                      const Matrix& tohpe,
                                      const SumEntry* ptr,
                                      index_t len) {
    // Empty A carries no width; infer it from the split point and TOHPE basis.
    const index_t nTotal = (A.cols() != 0) ? A.cols()
                                           : checked_add_index(divider, tohpe.cols(),
                                                               "solve_and_build_solution_basis width overflow");

    if (divider > nTotal)
        throw std::invalid_argument("solve_and_build_solution_basis: divider > cols");
    const index_t nY = nTotal - divider;

    if (nY == 0)
        return Matrix(0, 0);

    if (tohpe.rows() != 0 && tohpe.cols() != nY)
        throw std::invalid_argument("solve_and_build_solution_basis: tohpe width mismatch");

    static thread_local PivotMap pivA;
    static thread_local PivotMap pivY;

    pivA.reset(static_cast<std::size_t>(divider));
    pivY.reset(static_cast<std::size_t>(nY));

    const index_t full_rank = nY;
    Matrix        basis(0, nY);
    basis.reserve_rows(full_rank);

    for (index_t i = 0; i < A.rows() && basis.rows() < full_rank; ++i) {
        auto row = A[i];

        auto p = row.find_first();
        while (p != RowView::npos && p < divider) {
            const auto prow = pivA.get(static_cast<std::size_t>(p));
            if (prow < 0)
                break;

            row ^= A[static_cast<index_t>(prow)];
            p = row.find_next(p);
        }

        if (p == RowView::npos)
            continue;

        if (p < divider) {
            pivA.set(static_cast<std::size_t>(p), i);
            continue;
        }

        Row y = detail::extract_right_for_solution(static_cast<RowCView>(row), divider, nY);
        detail::apply_solution_transform(y, ptr, len);
        detail::insert_into_y_basis(y, basis, pivY);
    }

    for (index_t i = 0; i < tohpe.rows() && basis.rows() < full_rank; ++i) {
        Row y = tohpe[i];

        detail::apply_solution_transform(y, ptr, len);
        detail::insert_into_y_basis(y, basis, pivY);
    }

    return basis;
}

Matrix basis_gauss_elimination(Matrix&& A) {
    Matrix linear_indep(0, A.cols());
    linear_indep.reserve_rows(A.rows());
    std::vector<index_t> pivot_cols;
    pivot_cols.reserve(A.rows());

    for (index_t r = 0; r < A.rows(); ++r) {
        auto row = A[r];

        for (std::size_t k = 0; k < pivot_cols.size(); ++k) {
            const index_t p = pivot_cols[k];
            if (row.test(p)) {
                row ^= linear_indep[k];
            }
        }

        const index_t p = row.find_first();
        if (p != RowView::npos) {
            linear_indep.push_back(row);
            pivot_cols.push_back(p);
        }
    }
    return linear_indep;
}

Matrix row_dependency_basis(Matrix&& A) {
    if (A.rows() == 0)
        return Matrix(0, 0);

    Matrix   row_basis;
    Matrix   transform_basis;
    Matrix   dependencies(0, A.rows());
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

            const auto brow = pivots.get(static_cast<std::size_t>(p));
            if (brow < 0) {
                pivots.set(static_cast<std::size_t>(p), row_basis.rows());
                row_basis.push_back(row);
                transform_basis.push_back(y);
                break;
            }

            row ^= row_basis[static_cast<index_t>(brow)];
            y ^= transform_basis[static_cast<index_t>(brow)];
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
            auto p = row.find_first();
            if (p == RowView::npos)
                break;
            const auto prow = pivots.get(static_cast<std::size_t>(p));
            if (prow < 0)
                break;
            if (static_cast<index_t>(prow) == i)
                break;
            row  ^= A[static_cast<index_t>(prow)];
            arow ^= aug[static_cast<index_t>(prow)];
        }

        auto p = row.find_first();
        if (p == RowView::npos)
            continue;
        if (pivots.get(static_cast<std::size_t>(p)) >= 0) {
            continue;
        }
        pivots.set(static_cast<std::size_t>(p), i);

        for (index_t r = 0; r < m; ++r) {
            if (r == i)
                continue;
            if (A[r].test(p)) {
                A[r]   ^= row;
                aug[r] ^= arow;
            }
        }
    }
}

Matrix extract_basis(const Matrix& kernel, const PivotMap& pivots) {
    Matrix basis(0, kernel.cols());
    basis.reserve_rows(kernel.rows());
    std::vector<std::uint8_t> is_pivot_row(kernel.rows(), 0);
    for (std::size_t col = 0; col < pivots.row.size(); ++col) {
        const auto row = pivots.get(col);
        if (row != PivotMap::npos) {
            assert(row >= 0);
            assert(static_cast<index_t>(row) < kernel.rows());
            is_pivot_row[static_cast<std::size_t>(row)] = 1;
        }
    }
    for (index_t i = 0; i < kernel.rows(); i++) {
        if (!is_pivot_row[i]) {
            basis.push_back(kernel[i]);
        }
    }
    return basis;
}

static inline void or_copy_prefix_shifted(std::uint64_t* dst, std::size_t dst_nblocks, std::size_t dst_bit_off,
                                          const std::uint64_t* src, std::size_t prefix_bits) {
    if (!prefix_bits)
        return;

    const std::size_t d0   = dst_bit_off >> 6;
    const unsigned    sh   = static_cast<unsigned>(dst_bit_off & 63);
    const std::size_t full = prefix_bits >> 6;
    const unsigned    rem  = static_cast<unsigned>(prefix_bits & 63);

    auto or_word = [&](std::size_t idx, std::uint64_t w) {
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

    Matrix out(n, triangular_col_count(c));

    for (index_t i = 0; i < n; ++i) {
        auto            srow = mat[i];
        auto            drow = out[i];
        std::uint64_t*       d    = drow.data();
        const std::uint64_t* s    = srow.data();

        const std::size_t total_cols = static_cast<std::size_t>(out.cols());
        for (auto p = srow.find_first(); p != RowCView::npos && p < c; p = srow.find_next(p)) {
            const index_t     L   = p + 1;
            const std::size_t pos = total_cols - triangular_count_size(L, "L_expansion column overflow");
            or_copy_prefix_shifted(d, static_cast<std::size_t>(drow.blocks()), pos, s, static_cast<std::size_t>(L));
        }
    }
    return out;
}

unsigned __int128 PyRNG::bounded_raw_(unsigned __int128 bound) {
    if (bound == 0)
        throw std::invalid_argument("PyRNG bounded draw requires a non-zero bound");

    constexpr std::uint64_t raw_base = static_cast<std::uint64_t>(std::minstd_rand::max()) -
                                       static_cast<std::uint64_t>(std::minstd_rand::min()) + 1ULL;
    constexpr unsigned __int128 raw_space = static_cast<unsigned __int128>(raw_base) * raw_base * raw_base;
    if (bound > raw_space)
        throw std::overflow_error("PyRNG bounded draw exceeds its raw sampling space");

    for (;;) {
        unsigned __int128 space = raw_base;
        unsigned __int128 value = static_cast<std::uint64_t>(rng() - std::minstd_rand::min());
        while (space < bound) {
            value = value * raw_base + static_cast<std::uint64_t>(rng() - std::minstd_rand::min());
            space *= raw_base;
        }
        const unsigned __int128 limit = space - space % bound;
        if (value < limit)
            return value % bound;
    }
}

index_t PyRNG::rand_int(index_t low, index_t high) {
    if (low > high)
        throw std::invalid_argument("PyRNG rand_int requires low <= high");
    const auto span = static_cast<unsigned __int128>(high) - low + 1;
    return low + static_cast<index_t>(bounded_raw_(span));
}

uint64_t PyRNG::rand_u64() {
    return static_cast<uint64_t>(bounded_raw_(static_cast<unsigned __int128>(1) << 64));
}

double PyRNG::rand_double(double low, double high) {
    if (!(low <= high))
        throw std::invalid_argument("PyRNG rand_double requires low <= high");
    const double unit = std::ldexp(static_cast<double>(rand_u64() >> 11), -53);
    return low + (high - low) * unit;
}

index_t PyRNG::random_raw() { return rng(); }

void PyRNG::seed(index_t seed_val) { rng.seed(detail::normalize_minstd_seed(seed_val)); }

std::minstd_rand& PyRNG::get_engine() { return rng; }

static inline Row mask_to_bv(index_t dim, uint64_t mask) {
    Row v(dim);
    if (v.blocks() != 0)
        v.data()[0] = mask;
    return v;
}

std::vector<Row> PyRNG::sample_special_bitvec(const Matrix& basis, index_t i, index_t j, index_t num_samples) {
    (void)i;
    (void)j;
    const auto dim = basis.rows();
    if (dim == 0 || num_samples == 0)
        return {};
    if (dim >= std::numeric_limits<index_t>::digits)
        throw std::overflow_error("sample_special_bitvec: dimension is too large");

    const index_t                          N      = index_t{1} << dim;
    const index_t                          target = std::min(num_samples, N - 1);
    std::vector<Row>                       out;
    std::unordered_set<index_t> selected;
    out.reserve(static_cast<std::size_t>(target));
    selected.reserve(checked_mul_size(static_cast<std::size_t>(target), 2,
                                   "sample_special_bitvec reserve overflow"));
    for (index_t upper = N - target; upper < N; ++upper) {
        const index_t t = rand_int(1, upper); // inclusive
        const index_t x = selected.insert(t).second ? t : upper;
        selected.insert(x);
        out.push_back(mask_to_bv(dim, x));
    }

    return out;
}

std::vector<std::uint32_t> PyRNG::floyd_sample_0n(std::uint32_t n, index_t k) {
    if (k > n)
        throw std::invalid_argument("floyd_sample_0n: k > n");
    std::unordered_set<std::uint32_t> selected;
    selected.reserve(checked_mul_size(static_cast<std::size_t>(k), 2, "floyd_sample_0n reserve overflow"));

    std::vector<std::uint32_t> out;
    out.reserve(k);

    for (index_t j = static_cast<index_t>(n) - k; j < static_cast<index_t>(n); ++j) {
        const auto t = static_cast<std::uint32_t>(rand_int(0, j)); // inclusive
        // Floyd's algorithm grows the sampled set. When t was already picked,
        // j is guaranteed fresh because earlier iterations only reached j-1.
        const auto x = selected.insert(t).second ? t : static_cast<std::uint32_t>(j);
        selected.insert(x);
        out.push_back(x);
    }
    return out;
}

// Floyd sampling without replacement from [1..n], returns k unique ints, no retries.
std::vector<std::uint32_t> PyRNG::floyd_sample_1n(std::uint32_t n, index_t k) {
    if (k > n)
        throw std::invalid_argument("floyd_sample_1n: k > n");
    auto out = floyd_sample_0n(n, k);
    for (auto& value : out)
        ++value;
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

    const index_t rem = dim & 63;
    // Reject zero rather than mapping it to all-ones, which gave that vector
    // twice the probability of any other nonzero vector in small spaces.
    do {
        for (index_t i = 0; i < nb; ++i) {
            dst[i] = rand_u64();
        }
        if (rem != 0) {
            dst[nb - 1] &= ((1ULL << rem) - 1ULL);
        }
    } while (r.none());
}
Matrix get_tohpe_basis(const Matrix& P) {
    return row_dependency_basis(L_expansion(P));
}

} // namespace todd
