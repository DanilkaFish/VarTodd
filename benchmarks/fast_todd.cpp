#include "matrix.hpp"
#include "nullspace.hpp"
#include "todd_generator.hpp"

#include <iostream>
#include <memory>

using namespace gf2;

int main() {
	auto name = "../npy_corrected/gf2^16_mult.qc.matrix.npy";
	// auto name = "/home/danilkaf/projects/GF2LIB/npy_corrected/mod_adder_1024.qc.matrix.npy";
	auto init_matrix = Matrix::from_npy(name);
	auto finit_matrix = init_matrix;
	std::cerr << init_matrix.rows() << " " << init_matrix.cols() << std::endl;
	auto md_ptr = std::make_shared<MatrixWithData>(MatrixWithData(std::move(init_matrix)));
	auto policy_cfg = PolicyConfig{};
	policy_cfg.num_samples = 40;
	policy_cfg.max_z_to_research = 10000;
	policy_cfg.max_z_to_research_fraction = 1.0;
	// policy_cfg.escore.wred = 1
	policy_cfg.escore.wred = 1;
	policy_cfg.fscore.wred = -1;
	policy_cfg.min_reduction = 1;
	// policy_cfg.fscore.wred = -1;
	// policy_cfg.wdim = -1;
	// policy_cfg.max_z_to_research = 10000;
	// policy_cfg.wred = 1;
	policy_cfg.threads = 4;
	// policy_cfg.wpossible_red = 1;
	auto result = policy_iteration_impl(md_ptr, policy_cfg, 11, 1);
	auto rank = result.states.back().rows();
	while (result.states.size() >= 1) {
		std::cerr << "total reduction : " << result.chosen.back().reduction << "    from source -- "
				  << ((result.chosen.back().k == k_single_sentinel<Int>()) ? "tohpe" : "todd") << std::endl
				  << result.states.back().rows() << " " << result.states.back().cols() << std::endl;
		rank = result.states.back().rows();
		finit_matrix = result.states.front();
		result = policy_iteration_impl(std::make_shared<MatrixWithData>(MatrixWithData(std::move(result.states.back()))), policy_cfg);
		// if ((result.chosen.back().k != k_single_sentinel<Int>())) break;
	}
	// finit_matrix[0].flip(0);
	if (Tensor3D(init_matrix) != Tensor3D(finit_matrix)) {
		throw std::runtime_error("wtf : wrong final tensor");
	}
	// std::cerr << "total reduction : " << result.best_reduction
	//    << "    from source -- " << ((result.chosen.back().k == k_single_sentinel<Int>()) ? "tohpe" : "todd") << std::endl
	// std::cerr << "final rank : " << rank << std::endl;
}