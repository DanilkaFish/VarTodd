#include "matrix.hpp"
#include "nullspace.hpp"
#include "todd_generator.hpp"
#include <boost/program_options.hpp>

#include <iostream>
#include <memory>

namespace po = boost::program_options;
using namespace todd;

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("file,f", po::value<std::string>()->required(), "input matrix file")
        ("output_file,o", po::value<std::string>()->default_value(""), "output matrix file")
        ("num-samples,ns", po::value<int>()->default_value(40), "number of samples")
        ("max-z,mz", po::value<int>()->default_value(10000), "max Z to research")
        ("max-z-fraction", po::value<double>()->default_value(1.0), "max Z fraction")
        ("escore-wred,er", po::value<int>()->default_value(1), "exploration score wred")
        ("fscore-wred,fr", po::value<int>()->default_value(-1), "finalization score wred")
        ("min-reduction", po::value<int>()->default_value(1), "minimum reduction")
        ("threads,r", po::value<int>()->default_value(4), "number of threads")
        ("seed,s", po::value<int>()->default_value(4), "number of threads")
    ;
    
    po::positional_options_description p;
    p.add("file", 1);
    
    po::variables_map vm;
    
    try {
        po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .positional(p)
                  .run(), vm);
        
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }
    
    auto filename = vm["file"].as<std::string>();
    
    PolicyConfig policy_cfg;
    policy_cfg.num_samples = vm["num-samples"].as<int>();
    policy_cfg.max_z_to_research = vm["max-z"].as<int>();
    policy_cfg.max_z_to_research_fraction = vm["max-z-fraction"].as<double>();
    policy_cfg.escore.wred = vm["escore-wred"].as<int>();
    policy_cfg.fscore.wred = vm["fscore-wred"].as<int>();
    policy_cfg.min_reduction = vm["min-reduction"].as<int>();
    policy_cfg.threads = vm["threads"].as<int>();
	
	auto init_matrix = Matrix::from_npy(filename);
	auto finit_matrix = init_matrix;
	std::cerr << init_matrix.rows() << " " << init_matrix.cols() << std::endl;
	auto md_ptr = std::make_shared<MatrixWithData>(MatrixWithData(std::move(init_matrix)));
	auto result = policy_iteration_impl(md_ptr, policy_cfg, 11, 1);
	auto rank = result.states.back().rows();
	while (result.states.size() >= 1) {
		std::cerr << "total reduction : " << result.chosen.back().reduction << "    from source -- "
				  << ((result.chosen.back().k == k_single_sentinel<Int>()) ? "tohpe" : "todd") << std::endl
				  << result.states.back().rows() << " " << result.states.back().cols() << std::endl;
		rank = result.states.back().rows();
		finit_matrix = result.states.front();
		result = policy_iteration_impl(std::make_shared<MatrixWithData>(MatrixWithData(std::move(result.states.back()))), policy_cfg, vm["seed"].as<int>());
	}
    std::string output = vm["output_file"].as<std::string>();
    if (!output.empty()) {}
        finit_matrix.save_npy(output);
	if (Tensor3D(init_matrix) != Tensor3D(finit_matrix)) {
		throw std::runtime_error("CORE LINALG ERROR");
	}
}