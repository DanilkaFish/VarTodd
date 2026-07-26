#include <chrono>
#include <iostream>
#include <string>
#include "matrix.hpp"
#include "nullspace.hpp"
#include "todd_generator.hpp"
#include <stdexcept>

using namespace todd;
int main(int argc, char* argv[]){
    if (argc < 6) {
        std::cerr << "usage: bench <npy_path> <iterations> <num_dense_samples> <tohpe_z_choices> <min_buckets>\n";
        return 1;
    }
    std::string path = argv[1];
    int iterations = std::stoi(argv[2]);
    int num_dense_samples = std::stoi(argv[3]);
    int seed = 1;
    int z_choices = std::stoi(argv[4]);
    int min_buckets = std::stoi(argv[5]);

    std::cout << "sizeof(Matrix)="   << sizeof(Matrix)
              << " sizeof(Row)="     << sizeof(Row)
              << " sizeof(Candidate)=" << sizeof(Candidate)
              << " sizeof(CountWS)="  << sizeof(CountWS) << "\n";

    Matrix mat = Matrix::from_npy(path);
    // Defaults mirror demos/vartodd.cpp so only the CLI-named knobs deviate
    // from the maintained reference configuration.
    PolicyConfig pcfg{};
    pcfg.tohpe.sampling.dense = num_dense_samples;
    pcfg.todd.sampling.dense  = num_dense_samples;
    pcfg.tohpe.z_choices      = z_choices;
    pcfg.todd.buckets.min_buckets = min_buckets;
    // leave max_buckets = 0 (unlimited) and limit_bucket = -1 (unlimited),
    // matching demo defaults, so we don't over-constrain the search radius

    auto start = std::chrono::steady_clock::now();

    auto result = policy_iteration_impl(std::make_shared<MatrixWithData>(std::move(mat)), pcfg, seed, -1);
    int i = 0;
    for (; i < iterations && result.states.size() >= 1; ++i) {
        result = policy_iteration_impl(
            std::make_shared<MatrixWithData>(std::move(result.states.back())), pcfg, seed, i);
    }

    auto end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "iterations_run=" << i
              << " total_ms=" << total_ms
              << " ms_per_iter=" << (i > 0 ? total_ms / i : 0.0) << "\n";
}