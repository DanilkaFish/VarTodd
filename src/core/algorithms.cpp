#include "algorithms.hpp"

#include "random.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace todd {
constexpr int MAX_ROWS_IN_BASIS = 100;

Matrix solution_gaussian_elimination(Matrix& A, Matrix& aug) {
    if (A.rows() != aug.rows()) {
        throw std::invalid_argument("A and aug must have the same number of rows");
    }
    if (A.rows() == 0) {
        return Matrix();
    }

    const index_t m = A.rows();
    const index_t n = A.cols();

    std::vector<std::int64_t> pivot_row_of_col(n, -1);

    Matrix solution;

    for (index_t i = 0; i < m; ++i) {
        auto a_row   = A[i];
        auto aug_row = aug[i];

        auto p = a_row.find_first();
        while (p != decltype(a_row)::npos) {
            const std::int64_t prow = pivot_row_of_col[static_cast<index_t>(p)];
            if (prow < 0)
                break;

            a_row ^= A[static_cast<index_t>(prow)];
            aug_row ^= aug[static_cast<index_t>(prow)];

            p = a_row.find_next(p);
        }

        p = a_row.find_first();
        if (p != decltype(a_row)::npos) {
            pivot_row_of_col[static_cast<index_t>(p)] = static_cast<std::int64_t>(i);
        } else {

            if (!aug_row.none()) {
                solution.push_back(aug_row);
                if (solution.rows() >= MAX_ROWS_IN_BASIS) {
                    return solution;
                }
            }
        }
    }

    return solution;
}

struct PivotMap {
    std::vector<int32_t>  row;
    std::vector<uint32_t> tag;
    uint32_t              epoch = 1;

    void reset(std::size_t n) {
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

    inline int32_t get(std::size_t col) const { return (tag[col] == epoch) ? row[col] : -1; }
    inline void    set(std::size_t col, int32_t r) {
        row[col] = r;
        tag[col] = epoch;
    }
};

Matrix drop_left_block(const Matrix& A, index_t divider) {
    const index_t m = A.rows();
    const index_t n = A.cols();

    if (divider > n)
        throw std::runtime_error("drop_left_block: divider > cols()");
    const index_t new_cols = n - divider;

    if (new_cols == 0)
        return Matrix(m, 0);

    Matrix out(m, new_cols);

    const index_t  sb = divider >> 6;
    const unsigned sh = (unsigned)(divider & 63);

    for (index_t r = 0; r < m; ++r) {
        auto srcRow = A[r];
        auto dstRow = out[r];

        const uint64_t* src        = srcRow.data();
        const index_t   src_blocks = srcRow.blocks();

        uint64_t*     dst        = dstRow.data();
        const index_t dst_blocks = dstRow.blocks();

        for (index_t k = 0; k < dst_blocks; ++k) {
            const index_t  i0 = sb + k;
            const uint64_t lo = (i0 < src_blocks) ? src[i0] : 0ULL;

            if (sh == 0) {
                dst[k] = lo;
            } else {
                const uint64_t hi = (i0 + 1 < src_blocks) ? src[i0 + 1] : 0ULL;
                dst[k]            = (lo >> sh) | (hi << (64u - sh));
            }
        }

        const unsigned rem = (unsigned)(new_cols & 63);
        if (rem)
            dst[dst_blocks - 1] &= ((1ULL << rem) - 1ULL);
    }

    return out;
}

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
Matrix solve_and_build_solution_basis(Matrix& A, index_t divider) {
    if (A.rows() == 0)
        return Matrix();

    const index_t m      = A.rows();
    const index_t nTotal = A.cols();
    if (divider > nTotal)
        return Matrix();

    const index_t nY = nTotal - divider;
    if (nY == 0)
        return Matrix();

    static thread_local PivotMap pivA;
    static thread_local PivotMap pivY;

    pivA.reset((std::size_t)divider);
    pivY.reset((std::size_t)nY);

    Matrix basis(0, nY);

    for (index_t i = 0; i < m; ++i) {
        auto row = A[i];

        auto p = row.find_first();
        while (p != RowView::npos && p < divider) {
            const int32_t prow = pivA.get((std::size_t)p);
            if (prow < 0)
                break;
            row ^= A[(index_t)prow];
            p = row.find_next(p);
        }

        p = row.find_first();
        if (p == RowView::npos)
            continue;

        if (p < divider) {
            pivA.set((std::size_t)p, (int32_t)i);
            continue;
        }

        Row  y  = extract_right((RowCView)row, divider, nY);
        auto yv = y.view();

        auto q = yv.find_first();
        while (q != RowView::npos) {
            const int32_t brow = pivY.get((std::size_t)q);
            if (brow < 0)
                break;
            yv ^= basis[(index_t)brow];
            q = yv.find_next(q);
        }

        q = yv.find_first();
        if (q == RowView::npos)
            continue;

        pivY.set((std::size_t)q, (int32_t)basis.rows());
        basis.push_back(y.cview());
        if (basis.rows() >= MAX_ROWS_IN_BASIS)
            break;
    }

    return basis;
}

Matrix basis_gauss_elimination(Matrix&& A) {
    Matrix               linear_indep;
    std::vector<index_t> pivot_columns;
    for (index_t i = 0; i < A.rows(); i++) {
        auto current_row = A[i];
        for (size_t i = 0; i < pivot_columns.size(); ++i) {
            size_t pivot_col = pivot_columns[i];
            if (current_row.test(pivot_col)) {
                current_row ^= linear_indep[i];
            }
        }

        size_t pivot_index = current_row.find_first();
        if (pivot_index != RowView::npos) {
            linear_indep.push_back(current_row);
            pivot_columns.push_back(pivot_index);
            assign(A[i], current_row);
        }
    }
    return linear_indep;
}
void gauss_elimination_inplace(Matrix& A, Matrix& aug, pivot_map& pivots) {
    if (A.rows() != aug.rows())
        throw std::invalid_argument("gauss_elimination_inplace: row mismatch");
    for (index_t i = 0; i < A.rows(); i++) {
        auto row = A[i];
        if (pivots.count(i)) {
            continue;
        }
        for (auto [k, v] : pivots) {
            if (row.test(v)) {
                A[i] ^= A[k];
                aug[i] ^= aug[k];
            }
        }

        index_t index = row.find_first();
        if (index != RowView::npos) {
            for (auto [k, v] : pivots) {
                if (A[k].test(index)) {
                    A[k] ^= row;
                    aug[k] ^= aug[i];
                }
            }
            pivots.insert_or_assign(i, index);
        }
    }
}

Matrix extract_basis(const Matrix& kernel, const pivot_map& pivots) {
    Matrix _kernel(0, 0);

    for (index_t i = 0; i < kernel.rows(); i++) {
        if (pivots.find(i) == pivots.end()) {
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

std::ostream& operator<<(std::ostream& os, const pivot_map& m) {
    os << '{';
    bool first = true;
    for (const auto& [k, v] : m) {
        if (!first)
            os << ", ";
        first = false;
        os << k << ':' << v;
    }
    return os << '}';
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

auto& PyRNG::get_engine() { return rng; }

static inline Row mask_to_bv(index_t dim, uint64_t mask) {
    Row v(dim);
    for (index_t i = 0; i < dim; ++i) {
        if (mask & (1ULL << i))
            v.set(i);
    }
    return v;
}
std::vector<Row> PyRNG::sample_special_bitvec(const Matrix& basis, index_t i, index_t j, index_t num_samples) {
    auto                                   dim = basis.rows();
    const index_t                          N   = 1ULL << dim;
    std::vector<Row>                       out;
    std::unordered_map<uint64_t, uint64_t> remap;
    remap.reserve((size_t)num_samples * 2);
    for (index_t j = N - num_samples; j < N; ++j) {
        index_t t   = rand_int(0, j); // <- your RNG: inclusive range
        auto    itT = remap.find(t);
        index_t x   = (itT == remap.end()) ? t : itT->second;
        auto    itJ = remap.find(j);
        index_t y   = (itJ == remap.end()) ? j : itJ->second;
        remap[t]    = y;
        out.push_back(mask_to_bv(dim, x));
    }

    return out;
    // if (j != k_single_sentinel<decltype(j)>()) {
    // 	std::vector<index_t> i11;
    // 	std::vector<index_t> i00;
    // 	std::vector<index_t> i10;
    // 	std::vector<index_t> i01;
    // 	for (index_t k=0; k<basis.rows(); k++ ){
    // 		switch (basis[k].test(i) << 1 + basis[k].test(j))
    // 		{
    // 		case 0:
    // 			i00.push_back(k);
    // 			break;
    // 		case 1:
    // 			i01.push_back(k);
    // 			break;
    // 		case 2:
    // 			i10.push_back(k);
    // 			break;
    // 		case 3:
    // 			i11.push_back(k);
    // 			break;
    // 		}
    // 	}
    // 	std::vector<Row> out;
    // 	for (auto k: i01) {
    // 		Row y(dim);
    // 		y.set(k);
    // 		out.push_back(y);
    // 		for (auto l: i00) {
    // 			Row y(dim);
    // 			y.set(k);
    // 			y.set(l);
    // 			out.push_back(y);
    // 		}
    // 		for (auto l: i11) {
    // 			Row y(dim);
    // 			y.set(k);
    // 			y.set(l);
    // 			out.push_back(y);
    // 		}
    // 	}
    // 	for (auto k: i10) {
    // 		Row y(dim);
    // 		y.set(k);
    // 		out.push_back(y);
    // 		for (auto l: i00) {
    // 			Row y(dim);
    // 			y.set(k);
    // 			y.set(l);
    // 			out.push_back(y);
    // 		}
    // 		for (auto l: i11) {
    // 			Row y(dim);
    // 			y.set(k);
    // 			y.set(l);
    // 			out.push_back(y);
    // 		}
    // 	}
    // 	return out;
    // } else {
    // 	std::vector<index_t> i1;
    // 	std::vector<index_t> i0;
    // 	for (index_t k=0; k<basis.rows(); k++ ){
    // 		switch (basis[k].test(i))
    // 		{
    // 		case 0:
    // 			i0.push_back(k);
    // 			break;
    // 		case 1:
    // 			i1.push_back(k);
    // 			break;
    // 		}
    // 	}
    // 	std::vector<Row> out;
    // 	for (auto k: i1) {
    // 		Row y(dim);
    // 		y.set(k);
    // 		out.push_back(y);
    // 		for (auto l: i0) {
    // 			Row y(dim);
    // 			y.set(k);
    // 			y.set(l);
    // 			out.push_back(y);
    // 		}
    // 	}
    // 	return out;
    // }
}

std::vector<Row> PyRNG::sample_small_unique_bitvectors(index_t dim, index_t num_samples, float generator_part) {
    if (dim == 0 || num_samples == 0)
        return {};

    std::vector<Row> out;

    if (dim > 15) {
        // index_t init = rand_int(dim)
        for (index_t j = dim - int(dim * generator_part); j < dim; ++j) {
            std::unordered_map<uint64_t, uint64_t> remap;
            index_t t   = rand_int(0, j); // <- your RNG: inclusive range
            auto    itT = remap.find(t);
            index_t x   = (itT == remap.end()) ? t : itT->second;
            auto    itJ = remap.find(j);
            index_t y   = (itJ == remap.end()) ? j : itJ->second;

            remap[t]    = y;
            Row v(dim);
            v.set(x);
            out.push_back(v);
        }   
        // for (index_t i = 0; i < dim; i++) {
            // Row v(dim);
            // bvs.push_back(v);
        // }
        for (index_t i = 0; i < num_samples; i++) {
            out.push_back(sample_bitvector(dim));
        }
        return out;
    }

    const index_t N = 1ULL << dim;
    // num_samples = std::min(num_samples, N);
    if (N <= num_samples) {
        out.reserve(N);
        for (index_t i = 1; i < N - 1; i++) {
            out.push_back(mask_to_bv(dim, i));
        }
        return out;
    }

    std::unordered_map<uint64_t, uint64_t> remap;
    remap.reserve((size_t)num_samples * 2);
    for (index_t j = N - num_samples; j < N; ++j) {
        index_t t   = rand_int(0, j); // <- your RNG: inclusive range
        auto    itT = remap.find(t);
        index_t x   = (itT == remap.end()) ? t : itT->second;
        auto    itJ = remap.find(j);
        index_t y   = (itJ == remap.end()) ? j : itJ->second;
        remap[t]    = y;
        out.push_back(mask_to_bv(dim, x));
    }
    for (index_t i = 0; i < dim; i++) {
    	auto itT = remap.find(1 << i);
    	if (itT == remap.end()) {
    		Row v(dim);
    		v.set(i);
    		out.push_back(v);
    	}
    }
    return out;
}

Row PyRNG::sample_bitvector(index_t dim) {
    Row r(dim);
    if (dim == 0)
        return r;

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
    return r;
}
Matrix get_tohpe_basis(const Matrix& P) {
    Matrix    L = L_expansion(P);
    Matrix    Y = Matrix::identity(P.rows());
    pivot_map pivots;
    gauss_elimination_inplace(L, Y, pivots);
    Matrix M = extract_basis(Y, pivots);
    return M;
}

} // namespace todd
