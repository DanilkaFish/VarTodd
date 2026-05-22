#pragma once
#include "matrix.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
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
} // namespace detail

class PyRNG {
  public:
    PyRNG(index_t seed = std::random_device{}()) : rng(seed) {}
    std::vector<Row>      sample_special_bitvec(const Matrix& basis, index_t i, index_t j, index_t num_samples);
    Row                   sample_bitvector(index_t dim);
    void                  sample_bitvector_inplace(Row& r);
    std::vector<uint32_t> floyd_sample_0n(uint32_t n, index_t k);
    std::vector<uint32_t> floyd_sample_1n(uint32_t n, index_t k);
    template <class Fn>
    void for_each_capped_bitvector(index_t dim, std::array<index_t, 3> sample_caps, index_t sparse_max_weight,
                                   Fn&& fn);

    index_t  rand_int(index_t low, index_t high);
    uint64_t rand_u64();
    double   rand_double(double low, double high);
    index_t  random_raw();
    void     seed(index_t seed_val);
    std::minstd_rand& get_engine();

  private:
    std::minstd_rand rng;
};

template <class Fn>
void PyRNG::for_each_capped_bitvector(index_t dim, std::array<index_t, 3> sample_caps, index_t sparse_max_weight,
                                      Fn&& fn) {
    if (dim == 0)
        return;

    const index_t one_hot_count = std::min(sample_caps[0], dim);
    const index_t sparse_high   = std::min(dim, std::max<index_t>(2, sparse_max_weight));
    const index_t sparse_count =
        (dim >= 2) ? detail::sparse_space_bounded(dim, 2, sparse_high, sample_caps[1]) : 0;
    const index_t dense_count   = sample_caps[2];
    const index_t total_count   = one_hot_count + sparse_count + dense_count;

    Row scratch_bitvec(dim);
    std::unordered_set<Row, RowHash, RowEq> seen;
    seen.reserve(static_cast<std::size_t>(std::max<index_t>(16, total_count * 2)));

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
        auto bits = floyd_sample_0n(static_cast<uint32_t>(dim), weight);
        for (auto b : bits)
            scratch_bitvec.set(b);
    };

    auto random_attempts = [](index_t target) {
        return std::max<index_t>(target * 8, target + 16);
    };

    auto exhaustive_small_masks = [&](index_t target, bool sparse_only, index_t sparse_limit) {
        constexpr index_t exhaustive_dim_limit = 20;
        if (dim > exhaustive_dim_limit)
            return index_t{0};

        index_t emitted = 0;
        const index_t end = index_t{1} << dim;
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
        auto bits = floyd_sample_0n(static_cast<uint32_t>(dim), one_hot_count);
        for (auto b : bits) {
            scratch_bitvec.reset();
            scratch_bitvec.set(b);
            emit_unique(scratch_bitvec.cview());
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
    if (dim < sizeof(index_t) * 8) {
        const index_t universe = (index_t{1} << dim) - 1;
        dense_target = std::min<index_t>(dense_target, universe > seen.size() ? universe - seen.size() : 0);
    }
    index_t emitted = 0;
    for (index_t i = 0; emitted < dense_target && i < random_attempts(dense_target); ++i) {
        sample_bitvector_inplace(scratch_bitvec);
        emitted += emit_unique(scratch_bitvec.cview()) ? 1 : 0;
    }
    if (emitted < dense_target)
        exhaustive_small_masks(dense_target - emitted, false, sparse_high);
}

}; // namespace todd
