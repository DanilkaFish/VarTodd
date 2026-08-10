#include "matrix.hpp"
#include "todd_index.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace todd;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: bench_todd_index <npy_path> <repetitions>\n";
        return 1;
    }

    const std::string path        = argv[1];
    const int         repetitions = std::stoi(argv[2]);
    if (repetitions <= 0)
        throw std::invalid_argument("repetitions must be positive");

    Matrix P = Matrix::from_npy(path);

    std::size_t storage_bytes = 0;
    index_t     buckets       = 0;
    index_t     max_bucket    = 0;
    std::size_t checksum      = 0;

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        ToddIndex index(P);
        storage_bytes = index.storage_bytes();
        buckets       = index.buckets_num();
        max_bucket    = index.max_bucket();
        checksum += static_cast<std::size_t>(buckets);
    }
    const auto end = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "rows=" << P.rows() << " cols=" << P.cols() << " buckets=" << buckets
              << " max_bucket=" << max_bucket << '\n';
    std::cout << "storage_bytes=" << storage_bytes
              << " storage_mib=" << static_cast<double>(storage_bytes) / (1024.0 * 1024.0) << '\n';
    std::cout << "total_ms=" << total_ms << " ms_per_build=" << total_ms / repetitions
              << " checksum=" << checksum << '\n';
}
