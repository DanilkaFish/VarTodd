#pragma once
#include "matrix.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <random>

namespace todd {
class PyRNG {
  public:
    PyRNG(index_t seed = std::random_device{}()) : rng(seed) {}
    std::vector<Row>      sample_special_bitvec(const Matrix& basis, index_t i, index_t j, index_t num_samples);
    std::vector<Row>      sample_unique_bitvectors(index_t dim, index_t num_samples, float generator_part);
    Row                   sample_bitvector(index_t dim);
    void                  sample_bitvector_inplace(Row& r);
    std::vector<uint32_t> floyd_sample_0n(uint32_t n, index_t k);
    std::vector<uint32_t> floyd_sample_1n(uint32_t n, index_t k);
    template <class Fn> void for_each_unique_bitvector(index_t dim, index_t num_samples, float generator_part, Fn&& fn);

    index_t  rand_int(index_t low, index_t high);
    uint64_t rand_u64();
    double   rand_double(double low, double high);
    index_t  random_raw();
    void     seed(index_t seed_val);
    auto&    get_engine();

  private:
    std::minstd_rand rng;
};

template <class Fn>
void PyRNG::for_each_unique_bitvector(index_t dim, index_t num_samples, float generator_part, Fn&& fn) {
    if (dim == 0)
        return;

    generator_part = std::clamp(generator_part, 0.0f, 1.0f);
    const auto g = static_cast<uint32_t>(dim * generator_part);
    Row        scratch_bitvec(dim);

    auto emit_mask = [&](uint64_t mask) {
        scratch_bitvec.reset();
        scratch_bitvec.data()[0] = mask;
        fn(scratch_bitvec.cview());
    };

    if (dim > 30) {
        auto bits = floyd_sample_0n(dim, g);
        for (auto b : bits) {
            scratch_bitvec.reset();
            scratch_bitvec.set(b);
            fn(scratch_bitvec.cview());
        }
        for (index_t i = 0; i < num_samples; ++i) {
            sample_bitvector_inplace(scratch_bitvec);
            fn(scratch_bitvec.cview());
        }
        return;
    }

    const index_t Nminus1 = (1ULL << dim) - 1;
    if (Nminus1 <= g + num_samples) {
        for (index_t i = 1; i < Nminus1 + 1; ++i)
            emit_mask(i);
        return;
    }

    std::vector<uint32_t> gen_masks;
    gen_masks.reserve(g);
    auto bits = floyd_sample_0n(dim, g);
    for (auto b : bits) {
        const uint32_t mask = 1u << b;
        gen_masks.push_back(mask);
        emit_mask(mask);
    }
    std::sort(gen_masks.begin(), gen_masks.end());

    auto compact = floyd_sample_1n(Nminus1 - g, num_samples);
    auto compact_to_mask = [&](uint32_t c) -> uint32_t {
        uint32_t x = c;
        for (uint32_t e : gen_masks) {
            if (e <= x)
                ++x;
            else
                break;
        }
        return x;
    };

    for (uint32_t c : compact)
        emit_mask(compact_to_mask(c));
}

}; // namespace todd
