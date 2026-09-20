#include "algorithms.hpp"
#include "nullspace.hpp"
#include "todd_generator.hpp"
#include "todd_index.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

using namespace todd;
namespace {
using U = std::uint64_t;
using Bits = std::vector<unsigned char>;
using Binary = std::vector<Bits>;
std::size_t matrices = 0, directions = 0, rewrites = 0, split_cases = 0;
std::size_t embedded_directions = 0;
std::size_t rank_checks = 0;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Independent byte-array RREF. No production elimination or packed bit helpers.
std::vector<std::size_t> reduce(Binary& a, std::size_t width) {
    std::vector<std::size_t> pivots;
    for (std::size_t c = 0; c < width && pivots.size() < a.size(); ++c) {
        const auto r = pivots.size();
        auto p = r;
        while (p < a.size() && !a[p][c]) ++p;
        if (p == a.size()) continue;
        std::swap(a[r], a[p]);
        for (std::size_t j = 0; j < a.size(); ++j)
            if (j != r && a[j][c])
                for (std::size_t k = c; k < width; ++k) a[j][k] ^= a[r][k];
        pivots.push_back(c);
    }
    return pivots;
}
Binary kernel(Binary constraints, std::size_t width) {
    const auto pivots = reduce(constraints, width);
    Binary basis;
    for (std::size_t c = 0; c < width; ++c) {
        if (std::find(pivots.begin(), pivots.end(), c) != pivots.end()) continue;
        Bits y(width, 0); y[c] = 1;
        for (std::size_t i = 0; i < pivots.size(); ++i) y[pivots[i]] = constraints[i][c];
        basis.push_back(std::move(y));
    }
    return basis;
}
Binary unpack(const Matrix& m) {
    Binary out(m.rows(), Bits(m.cols()));
    for (index_t i = 0; i < m.rows(); ++i)
        for (index_t j = 0; j < m.cols(); ++j) out[i][j] = m[i].test(j);
    return out;
}
bool same_span(Binary a, Binary b, std::size_t width) {
    const auto ra = reduce(a, width).size();
    const auto rb = reduce(b, width).size();
    if (ra != rb) return false;
    a.insert(a.end(), b.begin(), b.end());
    return reduce(a, width).size() == ra;
}
Matrix words(const std::vector<U>& rows, std::size_t n) {
    Matrix p(rows.size(), n);
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < n; ++j) if ((rows[i] >> j) & 1) p[i].set(j);
    return p;
}
U word(RowCView row) {
    U w = 0;
    for (std::size_t i = 0; i < row.size(); ++i) if (row.test(i)) w |= U{1} << i;
    return w;
}
// All distinct symmetric cubic entries, including repeated indices. For n<=6
// these fit in 56 bits; this does not use Tensor3D or L_expansion.
U cubic(U x, std::size_t n) {
    U out = 0; std::size_t b = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j)
            for (std::size_t k = j; k < n; ++k, ++b)
                if ((x >> i & 1) && (x >> j & 1) && (x >> k & 1)) out |= U{1} << b;
    return out;
}
void toggle(std::set<U>& rows, U x) {
    if (x && !rows.insert(x).second) rows.erase(x);
}
std::set<U> rewrite(const std::vector<U>& p, U z, U y) {
    std::set<U> out; unsigned parity = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const bool bit = y >> i & 1; parity ^= bit;
        toggle(out, p[i] ^ (bit ? z : 0));
    }
    if (parity) toggle(out, z);
    return out;
}
std::set<U> rows_of(const Matrix& p) {
    std::set<U> out;
    for (index_t i = 0; i < p.rows(); ++i) toggle(out, word(p[i]));
    return out;
}

// Reduction is a row-count identity for every y, not only null-space members.
void check_rank_counts(const std::vector<U>& p, std::size_t n, int mode) {
    const auto m = p.size();
    auto data = std::make_shared<MatrixWithData>(words(p, n), false, mode);
    TohpeGenerator generator(data);
    std::set<U> buckets(p.begin(), p.end());
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = i + 1; j < m; ++j) buckets.insert(p[i] ^ p[j]);
    for (U y = 0; y < (U{1} << m); ++y) {
        const Row yy = words({y}, m)[0];
        std::set<U> expected_positive;
        for (U z = 0; z < (U{1} << n); ++z) {
            const Row zz = words({z}, n)[0];
            auto ns = generator.make(zz.cview());
            const auto reference = rewrite(p, z, y);
            const int red = int(m) - int(reference.size());
            if (ns.rank_divergence(yy.cview()) != red) {
                std::cerr << "Rank mismatch: mode=" << mode << " z=" << z << " y=" << y << " P=";
                for (auto x : p) std::cerr << x << ',';
                std::cerr << " predicted=" << ns.rank_divergence(yy.cview()) << " expected=" << red << '\n';
                throw std::runtime_error("rank_divergence differs from independent row count");
            }
            const auto applied = ns.apply(yy.cview());
            require(applied.rows() == reference.size() && rows_of(applied) == reference,
                    "apply did not produce the canonical row multiset");
            if (buckets.contains(z) && red > 0) expected_positive.insert(z);
            ++rank_checks;
        }
        // y=0 is not a policy sample; keep the public z-search audit within its
        // existing nonzero-y contract, while rank/apply above also check zero.
        if (!y) continue;
        std::vector<TohpeZInfo> infos;
        generator.best_z_n_details_into(yy.cview(), buckets.size() + 1, infos);
        std::set<U> actual_positive;
        for (const auto& info : infos) {
            const auto z = word(info.z);
            const int red = int(m) - int(rewrite(p, z, y).size());
            require(red >= 0 && info.reduction == static_cast<index_t>(red),
                    "TOHPE z-search reduction differs from independent row count");
            if (red > 0) require(actual_positive.insert(z).second, "z-search returned duplicate keys");
        }
        require(actual_positive == expected_positive, "z-search missed a positive bucket");
    }
}

void small_matrix(const std::vector<U>& p, std::size_t n, int mode) {
    ++matrices;
    const auto m = p.size();
    auto data = std::make_shared<MatrixWithData>(words(p, n), true, mode);
    Binary quadratic;
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = a; b < n; ++b) {
            Bits row(m);
            for (std::size_t i = 0; i < m; ++i) row[i] = (p[i] >> a & 1) & (p[i] >> b & 1);
            quadratic.push_back(std::move(row));
        }
    const auto tohpe = unpack(data->tohpe_basis());
    require(data->tohpe_basis().cols() == m, "empty TOHPE basis lost its y width");
    require(same_span(tohpe, kernel(quadratic, m), m), "TOHPE null space differs from direct quadratic constraints");
    auto independent = tohpe;
    require(reduce(independent, m).size() == tohpe.size(), "TOHPE basis is dependent");
    FullToddGenerator generator(data);
    for (U z = 1; z < (U{1} << n); ++z) {
        ++directions;
        const auto zrow = words({z}, n);
        auto ns = generator.make(zrow[0]);
        require(ns.basis().cols() == m, "TODD basis lost its y width");
        auto basis = unpack(ns.basis());
        auto reduced = basis;
        require(reduce(reduced, m).size() == basis.size(), "TODD basis is dependent");
        require(ns.tohpe_prefix_size() <= ns.basis().rows(), "invalid compact prefix dimension");
        std::vector<U> delta(m);
        for (std::size_t i = 0; i < m; ++i) delta[i] = cubic(p[i] ^ z, n) ^ cubic(p[i], n) ^ cubic(z, n);
        std::set<std::set<U>> expected, actual;
        for (U y = 0; y < (U{1} << m); ++y) {
            U change = 0;
            for (std::size_t i = 0; i < m; ++i) if (y >> i & 1) change ^= delta[i];
            if (change == 0) expected.insert(rewrite(p, z, y));
        }
        for (U coeffs = 0; coeffs < (U{1} << ns.basis().rows()); ++coeffs) {
            U y = 0;
            for (index_t i = 0; i < ns.basis().rows(); ++i) if (coeffs >> i & 1) y ^= word(ns.basis()[i]);
            U change = 0;
            for (std::size_t i = 0; i < m; ++i) if (y >> i & 1) change ^= delta[i];
            require(change == 0, "TODD basis contains a tensor-changing direction");
            const auto reference = rewrite(p, z, y);
            const Row yy = words({y}, m)[0];
            const auto result = ns.apply(yy.cview());
            require(rows_of(result) == reference, "apply differs from direct row rewrite/parity cancellation");
            require(ns.rank_divergence(yy.cview()) == int(m) - int(reference.size()),
                    "predicted reduction disagrees with direct row count");
            actual.insert(reference); ++rewrites;
        }
        if (actual != expected) {
            std::cerr << "Incomplete space: n=" << n << " z=" << z << " P=";
            for (auto x : p) std::cerr << x << ',';
            std::cerr << " expected=" << expected.size() << " actual=" << actual.size() << '\n';
            throw std::runtime_error("TODD misses tensor-preserving resulting states");
        }
    }
}

void split_solver() {
    std::mt19937 rng(7103);
    for (index_t divider : {0, 1, 63, 64, 65, 127, 128, 129})
        for (index_t width : {1, 63, 64, 65, 129}) {
            ++split_cases;
            constexpr index_t r = 9;
            Matrix a(r, divider + width);
            for (index_t i = 0; i < r; ++i)
                for (index_t j = 0; j < a.cols(); ++j) if (rng() & 1) a[i].set(j);
            // Force a left-block dependency with a right block that remains random.
            for (index_t j = 0; j < divider; ++j) {
                if (a[0].test(j)) a[r-1].set(j);
                else a[r-1].reset(j);
            }
            Binary constraints(divider, Bits(r));
            for (index_t j = 0; j < divider; ++j)
                for (index_t i = 0; i < r; ++i) constraints[j][i] = a[i].test(j);
            Binary expected;
            for (auto& coefs : kernel(constraints, r)) {
                Bits y(width);
                for (index_t i = 0; i < r; ++i) if (coefs[i])
                    for (index_t j = 0; j < width; ++j) y[j] ^= a[i].test(divider+j);
                expected.push_back(std::move(y));
            }
            Matrix empty; // Legacy 0x0 prefix: the solver must infer width from cols-divider.
            auto generated = solve_and_build_solution_basis_generated(r, a.cols(), divider, empty, nullptr, 0,
                [&](index_t i, RowView row) { assign(row, a[i]); });
            Matrix copy(a);
            auto materialized = solve_and_build_solution_basis(copy, divider, empty, nullptr, 0);
            require(generated.basis.cols() == width && materialized.cols() == width, "split solver lost output width");
            require(same_span(unpack(generated.basis), expected, width), "generated split solver misses a dependency");
            require(same_span(unpack(materialized), expected, width), "materialized split solver misses a dependency");
        }
}

void wide_tohpe() {
    std::mt19937 rng(7199);
    for (index_t n : {0, 1, 7, 13, 63, 64, 65})
        for (index_t m : {0, 1, 7, 63, 64, 65, 129}) {
            Matrix p(m, n);
            for (index_t i = 0; i < m; ++i)
                for (index_t j = 0; j < n; ++j) if (rng() & 1) p[i].set(j);
            Binary constraints;
            for (index_t a = 0; a < n; ++a)
                for (index_t b = a; b < n; ++b) {
                    Bits row(m);
                    for (index_t i = 0; i < m; ++i) row[i] = p[i].test(a) && p[i].test(b);
                    constraints.push_back(std::move(row));
                }
            const auto expected = kernel(constraints, m);
            for (int mode : {0, 20}) {
                MatrixWithData data(p, false, mode);
                const auto& basis = data.tohpe_basis();
                require(basis.cols() == m, "wide/empty TOHPE basis lost coefficient width");
                require(same_span(unpack(basis), expected, m), "wide TOHPE basis differs from byte-array oracle");
            }
        }
}

void wide_todd() {
    // Injectively embed a four-coordinate problem into many columns. Keeping
    // all four original coordinates makes its cubic constraints equivalent to
    // the small oracle, while extra columns exercise cross-word indexing.
    std::mt19937 rng(1997);
    for (std::size_t n : {63, 64, 65, 127, 128, 129, 192}) {
        std::vector<U> columns(n);
        for (auto& c : columns) c = 1 + rng()%15;
        columns[0] = 1; columns[n/2] = 2; columns[n-2] = 4; columns[n-1] = 8;
        auto embed = [&](U x) {
            Bits row(n);
            for (std::size_t j = 0; j < n; ++j) {
                U bits = x & columns[j];
                while (bits) { row[j] ^= 1; bits &= bits-1; }
            }
            return row;
        };
        for (bool has_tohpe : {false, true}) {
            std::vector<U> p{1,2,3,4,5,6,8,9};
            if (has_tohpe) p.push_back(7);
            Matrix input(p.size(), n);
            for (std::size_t i = 0; i < p.size(); ++i) {
                const auto row = embed(p[i]);
                for (std::size_t j = 0; j < n; ++j) if (row[j]) input[i].set(j);
            }
            for (int mode : {0, 20}) {
                auto data = std::make_shared<MatrixWithData>(input, true, mode);
                FullToddGenerator generator(data);
                for (U z = 1; z < 16; ++z) {
                    ++embedded_directions;
                    Row zrow(n);
                    const auto bits = embed(z);
                    for (std::size_t j = 0; j < n; ++j) if (bits[j]) zrow.set(j);
                    auto ns = generator.make(zrow.cview());
                    std::vector<U> delta;
                    for (auto x : p) delta.push_back(cubic(x^z,4)^cubic(x,4)^cubic(z,4));
                    std::set<std::set<Bits>> expected, actual;
                    auto encoded = [&](U y) {
                        std::set<Bits> state;
                        for (auto x : rewrite(p,z,y)) state.insert(embed(x));
                        return state;
                    };
                    for (U y = 0; y < (U{1} << p.size()); ++y) {
                        U change = 0;
                        for (std::size_t i = 0; i < p.size(); ++i) if (y >> i & 1) change ^= delta[i];
                        if (!change) expected.insert(encoded(y));
                    }
                    auto basis = unpack(ns.basis());
                    require(reduce(basis,p.size()).size() == ns.basis().rows(), "wide TODD basis is dependent");
                    for (U c = 0; c < (U{1} << ns.basis().rows()); ++c) {
                        U y = 0;
                        for (index_t i = 0; i < ns.basis().rows(); ++i) if (c >> i & 1) y ^= word(ns.basis()[i]);
                        const Row yy = words({y},p.size())[0];
                        const auto result = unpack(ns.apply(yy.cview()));
                        const std::set<Bits> state(result.begin(),result.end());
                        require(state.size() == result.size(), "wide apply emitted duplicate rows");
                        require(state == encoded(y), "wide apply differs from independent embedded rewrite");
                        require(ns.rank_divergence(yy.cview()) == int(p.size()) - int(state.size()),
                                "wide reduction differs from independent row count");
                        actual.insert(state);
                    }
                    require(actual == expected, "wide TODD basis changes or misses tensor-preserving states");
                }
            }
        }
    }
}
} // namespace

int main() {
    try {
        // This must run first: thread-local pivot capacity can mask the bug.
        small_matrix({1,2,3,4,5,6,8,9}, 4, 0);
        split_solver();
        wide_tohpe();
        wide_todd();
        check_rank_counts({1}, 1, 0); // z=0, odd y: actual reduction is zero.
        check_rank_counts({1,1,2}, 3, 0); // duplicates outside an empty z bucket.
        for (U mask = 0; mask < 128; ++mask) {
            std::vector<U> p;
            for (U x = 1; x < 8; ++x) if (mask >> (x-1) & 1) p.push_back(x);
            for (int mode : {0,20}) check_rank_counts(p,3,mode);
        }
        for (std::size_t m = 0; m <= 4; ++m)
            for (U code = 0; code < (U{1} << (2*m)); ++code) {
                std::vector<U> p(m);
                for (std::size_t i = 0; i < m; ++i) p[i] = (code >> (2*i)) & 3;
                for (int mode : {0,20}) check_rank_counts(p,2,mode);
            }
        for (std::size_t n = 1; n <= 3; ++n)
            for (U mask = 0; mask < (U{1} << ((U{1} << n)-1)); ++mask) {
                std::vector<U> p;
                for (U x = 1; x < (U{1} << n); ++x) if (mask >> (x-1) & 1) p.push_back(x);
                for (int mode : {0,20}) small_matrix(p,n,mode);
            }
        std::mt19937 rng(9917);
        for (std::size_t n = 4; n <= 6; ++n)
            for (int sample = 0; sample < 16; ++sample) {
                std::vector<U> p((U{1} << n)-1);
                std::iota(p.begin(),p.end(),1);
                std::shuffle(p.begin(),p.end(),rng);
                p.resize(1 + rng()%10);
                for (int mode : {0,20}) small_matrix(p,n,mode);
            }
        for (int sample = 0; sample < 16; ++sample) {
            std::vector<U> p(1 + rng()%8);
            for (auto& x : p) x = rng()%8;
            small_matrix(p,3,0);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
    std::cout << "algebra ok: matrices=" << matrices << " z_directions=" << directions
              << " rewrites=" << rewrites << " split_cases=" << split_cases
              << " wide_shapes=49 embedded_directions=" << embedded_directions
              << " rank_checks=" << rank_checks << '\n';
}
