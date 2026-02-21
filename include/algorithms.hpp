#pragma once
#include "matrix.hpp"
#include "typedef.hpp"

#include <ostream>
#include <vector>

namespace gf2 {

	struct RrefResult {
		Matrix rref;
		Matrix transform;
		std::vector<index_t> pivot_cols;
		index_t rank{};
	};
	Matrix solve_and_build_solution_basis(Matrix& A, Matrix& aug);
	Matrix solve_and_build_solution_basis(Matrix& A, index_t divider);
	Matrix get_tohpe_basis(const Matrix& P);
	void gauss_elimination_inplace(Matrix& A, Matrix& aug, pivot_map& pivots);
	Matrix solution_gaussian_elimination(Matrix& A, Matrix& aug);
	Matrix basis_gauss_elimination(Matrix&& A);
	Matrix extract_basis(const Matrix& basis_kernel, const pivot_map& pivots);
	Matrix L_expansion(const Matrix& mat);
	std::ostream& operator<<(std::ostream& os, const Matrix& mat);
	std::ostream& operator<<(std::ostream& os, const pivot_map& m);

} // namespace gf2
