#pragma once
#include "matrix.hpp"
#include "typedef.hpp"

#include <random>

namespace todd {
std::vector<index_t> fisher_yates_shuffle(index_t max_unique, index_t n, std::mt19937& rng);
class PyRNG {
  public:
    PyRNG(index_t seed = std::random_device{}()) : rng(seed) {}
    std::vector<Row>     sample_special_bitvec(const Matrix& basis, index_t i, index_t j, index_t num_samples);
    std::vector<Row>     sample_unique_bitvectors(index_t dim, index_t num_samples);
    std::vector<index_t> fisher_yates(index_t max_unique, index_t n);
    std::vector<Row>     sample_small_unique_bitvectors(index_t dim, index_t num_samples);
    Row                  sample_bitvector(index_t dim);

    index_t  rand_int(index_t low, index_t high);
    uint64_t rand_u64();
    double   rand_double(double low, double high);
    index_t  random_raw();
    void     seed(index_t seed_val);
    auto&    get_engine();

  private:
    std::minstd_rand rng;
};

}; // namespace todd