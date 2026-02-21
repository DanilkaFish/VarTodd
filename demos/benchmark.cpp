#include <cstdint>
#include <iostream>

// Adjust these includes to your actual library layout:
#include "algorithms.hpp"
#include "matrix.hpp"
#include "nullspace.hpp"
#include "random.hpp"
// #include "cpp_kernel/include/.hpp"
// #include "cpp_kernel/include/Tensor3D.hpp"
using namespace gf2;

Matrix build_context() {
	// Default location – change this to wherever your .npy lives
	// e.g. "../data/context.npy" or something passed via CMake
	// const std::string default_path = "../qc_utils/data/ham_med15.matrix.npy";
	const std::string default_path = "../qc_utils/data/gf_2pow6_mult_comp1.matrix.npy";
	// const std::string default_path = "../qc_utils/data/mod.matrix.npy";
	return build_context(default_path);
}

// Light research: always uses TohpeGenerator, like Python `light_research`
static Matrix light_research(const Matrix& mat) {
	std::shared_ptr<MatrixWithData> md{new MatrixWithData(Matrix(mat))};
	TohpeGenerator gen(md);
	int best_reduction = 0;
	bitvec_t best_vector;
	bool have_best_vector = false;
	NullSpace best_ns = gen.make(0);

	// ns = make1(gen, 0)
	NullSpace ns_temp = gen.make(0);
	const Matrix& Y = ns_temp.basis();

	PyRNG rng(42);
	const int num_samples = 8;

	// rng.sample_unique_bitvecs(...) should return some iterable/container of bitvec_t
	auto samples = rng.sample_small_unique_bitvectors(Y.rows(), num_samples);

	for (const bitvec_t& coeffs : samples) {
		// y = ns.linear_combination(vec)
		bitvec_t y = ns_temp.linear_combination(coeffs);

		// best_z = gen.best_z(y)
		bitvec_t best_z = gen.best_z(y);

		// ns = gen.make(best_z)
		NullSpace ns = gen.make(best_z);

		int reduction = ns.rank_divergence(y);
		if (reduction > best_reduction) {
			best_reduction = reduction;
			best_vector = y;
			best_ns = std::move(ns);
			have_best_vector = true;
		}
	}

	if (have_best_vector) {
		std::cout << "best_reduction (light) = " << best_reduction << "\n";
		Matrix new_mat = best_ns.apply(best_vector);
		std::cout << "rank drop (light): " << (mat.rows() - new_mat.rows()) << "\n";
		return new_mat;
	}

	return mat;
}

// Heavy research: generic over generator, mirrors Python `research`
template <typename Gen> static Matrix research(const Matrix& mat) {
	std::shared_ptr<MatrixWithData> md{new MatrixWithData(Matrix(mat))};
	Gen gen(md);

	int best_reduction = 0;
	bitvec_t best_vector;
	bool have_best_vector = false;
	NullSpace best_ns = gen.make(0);

	const int num_samples = 32;
	PyRNG rng(43);

	const index_t n_rows = mat.rows();

	for (index_t k = 0; k < n_rows; ++k) {
		// Single index transformations: ns = make1(gen, k)
		{
			NullSpace ns = gen.make(k);
			const Matrix& Y = ns.basis();
			auto samples = rng.sample_small_unique_bitvectors(Y.rows(), num_samples);

			for (const bitvec_t& coeffs : samples) {
				bitvec_t y = ns.linear_combination(coeffs);
				int reduction = ns.rank_divergence(y);
				if (reduction > best_reduction) {
					best_reduction = reduction;
					best_vector = y;
					best_ns = gen.make(k);
					// best_ns = std::move(ns);
					have_best_vector = true;
				}
			}
		}

		// Pair transformations: ns = make2(gen, k, l)
		for (index_t l = k + 1; l < n_rows; ++l) {
			NullSpace ns = gen.make(k, l);
			const Matrix& Y = ns.basis();
			auto samples = rng.sample_small_unique_bitvectors(Y.rows(), num_samples);

			for (const bitvec_t& coeffs : samples) {
				bitvec_t y = ns.linear_combination(coeffs);
				int reduction = ns.rank_divergence(y);
				if (reduction > best_reduction) {
					best_reduction = reduction;
					best_vector = y;
					best_ns = gen.make(k, l);
					// best_ns = std::move(ns);
					have_best_vector = true;
				}
			}
		}
	}

	if (have_best_vector) {
		std::cout << "best_reduction (heavy) = " << best_reduction << "\n";
		Matrix new_mat = best_ns.apply(best_vector);
		std::cout << "rank drop (heavy): " << (mat.rows() - new_mat.rows()) << "\n";
		return new_mat;
	}

	return mat;
}

// Full optimize loop, closely matching Python's optimize()
static Matrix optimize(Matrix mat) {
	index_t old_rank = mat.rows() + 1;
	index_t rank = mat.rows();

	while (rank < old_rank) {
		// inner loop with light_research using TohpeGenerator
		while (rank < old_rank) {
			Matrix new_mat = light_research(mat);
			old_rank = rank;
			rank = new_mat.rows();
			mat = std::move(new_mat);
		}

		// heavy research using ToddGenerator
		{
			std::cerr << "begining todd";
			Matrix new_mat = research<FullToddGenerator>(mat);
			old_rank = rank;
			rank = new_mat.rows();
			mat = std::move(new_mat);
		}
	}

	return mat;
}

int main() {
	try {
		// build_context() should load your matrix from .npy (as you said)
		Matrix mat = build_context();
		Matrix new_mat = mat;

		// direct call to Todd-like optimization, as in Python main
		new_mat = optimize(new_mat);

		std::cout << "Initial rows: " << mat.rows() << "  Final rows: " << new_mat.rows() << "\n";

		Tensor3D t0(mat);
		Tensor3D t1(new_mat);
		std::cout << "Tensor3D equality: " << (t0 == t1 ? "true" : "false") << "\n";

		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "Exception: " << ex.what() << "\n";
		return 1;
	}
}
