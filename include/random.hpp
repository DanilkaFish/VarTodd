#pragma once
#include "matrix.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace todd {
namespace detail {
inline index_t binom_bounded(index_t n, index_t k, index_t limit) {
    if (limit == 0 || k > n)
        return 0;
    k = std::min(k, n - k);
    if (k == 0)
        return 1;

    unsigned __int128 c = 1;
    for (index_t i = 1; i <= k; ++i) {
        c = (c * static_cast<unsigned __int128>(n - k + i)) / i;
        if (c >= limit)
            return limit;
    }
    return static_cast<index_t>(c);
}

inline index_t sparse_space_bounded(index_t dim, index_t lo, index_t hi, index_t limit) {
    index_t total = 0;
    for (index_t w = lo; w <= hi && total < limit; ++w) {
        total += binom_bounded(dim, w, limit - total);
    }
    return total;
}

inline index_t checked_add_count(index_t a, index_t b, const char* what) {
    if (b > std::numeric_limits<index_t>::max() - a)
        throw std::overflow_error(what);
    return a + b;
}

inline std::size_t reserve_twice(index_t n, const char* what) {
    const auto N = static_cast<std::size_t>(n);
    if (N > std::numeric_limits<std::size_t>::max() / 2)
        throw std::overflow_error(what);
    return N * 2;
}

inline index_t capped_add_count(index_t a, index_t b, index_t cap) noexcept {
    if (a >= cap || b > cap - a)
        return cap;
    return a + b;
}

inline std::minstd_rand::result_type normalize_minstd_seed(index_t seed) noexcept {
    return static_cast<std::minstd_rand::result_type>(
        seed % static_cast<index_t>(std::minstd_rand::modulus));
}
} // namespace detail

class PyRNG {
  public:
    PyRNG(index_t seed = std::random_device{}()) : rng(detail::normalize_minstd_seed(seed)) {}
    std::vector<Row>      sample_special_bitvec(const Matrix& basis, index_t i, index_t j, index_t num_samples);
    Row                   sample_bitvector(index_t dim);
    void                  sample_bitvector_inplace(Row& r);
    std::vector<std::uint32_t> floyd_sample_0n(std::uint32_t n, index_t k);
    std::vector<std::uint32_t> floyd_sample_1n(std::uint32_t n, index_t k);
    template <class Fn>
    void for_each_capped_bitvector(index_t dim, std::array<index_t, 3> sample_caps, index_t sparse_max_weight,
                                   Fn&& fn);
    template <class Fn>
    void for_each_capped_bitvector_regions(index_t full_dim, index_t prefix_dim,
                                           std::array<index_t, 3> prefix_caps,
                                           index_t prefix_sparse_max_weight,
                                           std::array<index_t, 3> full_caps,
                                           index_t full_sparse_max_weight, Fn&& fn);
    template <class PrefixFn, class FullFn>
    void for_each_capped_bitvector_regions(index_t full_dim, index_t prefix_dim,
                                           std::array<index_t, 3> prefix_caps,
                                           index_t prefix_sparse_max_weight,
                                           std::array<index_t, 3> full_caps,
                                           index_t full_sparse_max_weight, PrefixFn&& prefix_fn, FullFn&& full_fn);

    index_t  rand_int(index_t low, index_t high);
    uint64_t rand_u64();
    double   rand_double(double low, double high);
    index_t  random_raw();
    void     seed(index_t seed_val);
    template <class T> void shuffle(std::vector<T>& values) {
        for (std::size_t i = values.size(); i > 1; --i)
            std::swap(values[i - 1], values[static_cast<std::size_t>(rand_int(0, i - 1))]);
    }
    std::minstd_rand& get_engine();

  private:
    unsigned __int128 bounded_raw_(unsigned __int128 bound);
    template <class Fn>
    void for_each_capped_bitvector_in_region(index_t full_dim, index_t region_dim,
                                              std::array<index_t, 3> sample_caps,
                                              index_t sparse_max_weight,
                                              std::unordered_set<Row, RowHash, RowEq>& seen, Fn&& fn);

    std::minstd_rand rng;
};

template <class Fn>
void PyRNG::for_each_capped_bitvector(index_t dim, std::array<index_t, 3> sample_caps, index_t sparse_max_weight,
                                      Fn&& fn) {
    std::unordered_set<Row, RowHash, RowEq> seen;
    for_each_capped_bitvector_in_region(dim, dim, sample_caps, sparse_max_weight, seen, fn);
}

template <class Fn>
void PyRNG::for_each_capped_bitvector_regions(index_t full_dim, index_t prefix_dim,
                                               std::array<index_t, 3> prefix_caps,
                                               index_t prefix_sparse_max_weight,
                                               std::array<index_t, 3> full_caps,
                                               index_t full_sparse_max_weight, Fn&& fn) {
    for_each_capped_bitvector_regions(full_dim, prefix_dim, prefix_caps, prefix_sparse_max_weight, full_caps,
                                      full_sparse_max_weight, fn, fn);
}

template <class PrefixFn, class FullFn>
void PyRNG::for_each_capped_bitvector_regions(index_t full_dim, index_t prefix_dim,
                                               std::array<index_t, 3> prefix_caps,
                                               index_t prefix_sparse_max_weight,
                                               std::array<index_t, 3> full_caps,
                                               index_t full_sparse_max_weight, PrefixFn&& prefix_fn,
                                               FullFn&& full_fn) {
    std::unordered_set<Row, RowHash, RowEq> seen;
    for_each_capped_bitvector_in_region(full_dim, std::min(prefix_dim, full_dim), prefix_caps,
                                        prefix_sparse_max_weight, seen, std::forward<PrefixFn>(prefix_fn));
    for_each_capped_bitvector_in_region(full_dim, full_dim, full_caps, full_sparse_max_weight, seen,
                                        std::forward<FullFn>(full_fn));
}

template <class Fn>
void PyRNG::for_each_capped_bitvector_in_region(index_t full_dim, index_t region_dim,
                                                 std::array<index_t, 3> sample_caps,
                                                 index_t sparse_max_weight,
                                                 std::unordered_set<Row, RowHash, RowEq>& seen, Fn&& fn) {
    // Emit unique non-zero full-width coefficient vectors supported on [0, region_dim).
    if (full_dim == 0 || region_dim == 0)
        return;
    if (full_dim > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("for_each_capped_bitvector: dimension exceeds 32-bit sampler range");

    const index_t one_hot_count = std::min(sample_caps[0], region_dim);
    const index_t sparse_high   = std::min(region_dim, std::max<index_t>(2, sparse_max_weight));
    const index_t sparse_count =
        (region_dim >= 2) ? detail::sparse_space_bounded(region_dim, 2, sparse_high, sample_caps[1]) : 0;
    const index_t dense_count = sample_caps[2];
    const bool    finite_universe = region_dim < std::numeric_limits<index_t>::digits;
    const index_t universe        = finite_universe ? (index_t{1} << region_dim) - 1 : 0;
    const index_t reserve_count =
        finite_universe
            ? detail::capped_add_count(detail::capped_add_count(one_hot_count, sparse_count, universe), dense_count,
                                       universe)
            : detail::checked_add_count(detail::checked_add_count(one_hot_count, sparse_count,
                                                                  "for_each_capped_bitvector sample count overflow"),
                                        dense_count, "for_each_capped_bitvector sample count overflow");

    Row scratch_bitvec(full_dim);
    Row region_bitvec(region_dim);
    seen.reserve(detail::reserve_twice(std::max<index_t>(16, reserve_count),
                                       "for_each_capped_bitvector reserve overflow"));

    auto emit_unique = [&](RowCView coefs) -> bool {
        if (coefs.none())
            return false;
        auto [it, inserted] = seen.emplace(coefs);
        if (!inserted)
            return false;
        fn(it->cview());
        return true;
    };

    auto fill_weight = [&](index_t weight) {
        scratch_bitvec.reset();
        auto bits = floyd_sample_0n(static_cast<std::uint32_t>(region_dim), weight);
        for (auto b : bits)
            scratch_bitvec.set(b);
    };

    auto fill_dense = [&]() {
        if (region_dim == full_dim) {
            sample_bitvector_inplace(scratch_bitvec);
            return;
        }
        sample_bitvector_inplace(region_bitvec);
        scratch_bitvec.reset();
        RowCView region_view = region_bitvec.cview();
        for (auto bit = region_view.find_first(); bit != Row::npos; bit = region_view.find_next(bit))
            scratch_bitvec.set(bit);
    };

    auto random_attempts = [](index_t target) {
        const index_t max_value = std::numeric_limits<index_t>::max();
        const index_t by_trials = (target > max_value / 8) ? max_value : target * 8;
        const index_t by_slack  = (target > max_value - 16) ? max_value : target + 16;
        return std::max(by_trials, by_slack);
    };

    auto exhaustive_small_masks = [&](index_t target, bool sparse_only, index_t sparse_limit) {
        constexpr index_t exhaustive_dim_limit = 20;
        if (region_dim > exhaustive_dim_limit)
            return index_t{0};

        index_t emitted = 0;
        const index_t end = index_t{1} << region_dim;
        for (index_t mask = 1; mask < end && emitted < target; ++mask) {
            const index_t weight = popcount64(static_cast<std::uint64_t>(mask));
            if (sparse_only && (weight < 2 || weight > sparse_limit))
                continue;

            scratch_bitvec.reset();
            scratch_bitvec.data()[0] = static_cast<std::uint64_t>(mask);
            emitted += emit_unique(scratch_bitvec.cview()) ? 1 : 0;
        }
        return emitted;
    };

    if (one_hot_count > 0) {
        index_t emitted = 0;
        auto bits = floyd_sample_0n(static_cast<std::uint32_t>(region_dim), one_hot_count);
        for (auto b : bits) {
            scratch_bitvec.reset();
            scratch_bitvec.set(b);
            emitted += emit_unique(scratch_bitvec.cview()) ? 1 : 0;
        }
        for (index_t b = 0; emitted < one_hot_count && b < region_dim; ++b) {
            scratch_bitvec.reset();
            scratch_bitvec.set(b);
            emitted += emit_unique(scratch_bitvec.cview()) ? 1 : 0;
        }
    }

    if (sparse_count > 0) {
        index_t emitted = 0;
        for (index_t i = 0; emitted < sparse_count && i < random_attempts(sparse_count); ++i) {
            fill_weight(rand_int(2, sparse_high));
            emitted += emit_unique(scratch_bitvec.cview()) ? 1 : 0;
        }
        if (emitted < sparse_count)
            exhaustive_small_masks(sparse_count - emitted, true, sparse_high);
    }

    index_t dense_target = dense_count;
    if (finite_universe) {
        index_t seen_in_region = 0;
        for (const Row& value : seen) {
            bool in_region = true;
            RowCView value_view = value.cview();
            for (auto bit = value_view.find_first(); bit != Row::npos; bit = value_view.find_next(bit)) {
                if (bit >= region_dim) {
                    in_region = false;
                    break;
                }
            }
            seen_in_region += in_region ? 1 : 0;
        }
        dense_target = std::min<index_t>(dense_target, universe > seen_in_region ? universe - seen_in_region : 0);
    }
    index_t emitted = 0;
    for (index_t i = 0; emitted < dense_target && i < random_attempts(dense_target); ++i) {
        fill_dense();
        emitted += emit_unique(scratch_bitvec.cview()) ? 1 : 0;
    }
    if (emitted < dense_target)
        exhaustive_small_masks(dense_target - emitted, false, sparse_high);
}

} // namespace todd
