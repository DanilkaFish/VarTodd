#include "matrix.hpp"
#include "nullspace.hpp"
#include "todd_generator.hpp"
#include <boost/program_options.hpp>

#include <array>
#include <iostream>
#include <memory>
#include <filesystem>
#include <vector>

namespace po = boost::program_options;
using namespace todd;

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("file,f", po::value<std::string>()->required(), "input matrix file")
        ("output_file,o", po::value<std::string>()->default_value(""), "output matrix file")
        ("tohpe-vector-samples", po::value<std::vector<int>>()->multitoken()->default_value(std::vector<int>{16, 32, 16}, "16 32 16"), "TOHPE sample caps: one-hot sparse dense")
        ("todd-vector-samples", po::value<std::vector<int>>()->multitoken()->default_value(std::vector<int>{16, 32, 16}, "16 32 16"), "full Todd sample caps: one-hot sparse dense")
        ("min-z,mz", po::value<int>()->default_value(10000), "minimum Z buckets to research")
        ("max-z", po::value<int>()->default_value(10000), "maximum Z buckets to research")
        ("sparse-max-weight", po::value<int>()->default_value(8), "maximum sparse vector weight")
        ("tohpe-pool-size", po::value<int>()->default_value(1), "max TOHPE candidates kept; 0 disables TOHPE")
        ("todd-pool-size", po::value<int>()->default_value(1), "max full Todd candidates kept; 0 disables full Todd")
        ("min-tohpe-actions", po::value<int>()->default_value(0), "minimum TOHPE actions before final merge")
        ("min-todd-actions", po::value<int>()->default_value(0), "minimum full Todd actions before final merge")
        ("tohpe-sample", po::value<int>()->default_value(1), "number of TOHPE z candidates per vector")
        ("bucket-temperature", po::value<double>()->default_value(0.0), "Gumbel temperature for bucket choice")
        ("bucket-random-fraction", po::value<double>()->default_value(0.0), "fraction of researched buckets chosen randomly")
        ("max-per-signature", po::value<int>()->default_value(2), "candidate cap per diversity signature; 0 disables")
        ("escore-wred,er", po::value<int>()->default_value(1), "exploration score wred")
        ("fscore-wred,fr", po::value<int>()->default_value(-1), "finalization score wred")
        ("min-reduction", po::value<int>()->default_value(1), "minimum reduction")
        ("seed,s", po::value<int>()->default_value(4), "seed")
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
    if (!std::filesystem::exists(filename)) {
        std::cerr << "Error: input file does not exist: " << filename << std::endl;
        return 1;
    }
    
    PolicyConfig policy_cfg;
    auto sample_caps = [](const std::vector<int>& values) {
        std::array<Int, 3> caps = {0, 0, 0};
        for (std::size_t i = 0; i < std::min<std::size_t>(caps.size(), values.size()); ++i)
            caps[i] = values[i];
        return caps;
    };
    policy_cfg.tohpe_vector_samples = sample_caps(vm["tohpe-vector-samples"].as<std::vector<int>>());
    policy_cfg.todd_vector_samples = sample_caps(vm["todd-vector-samples"].as<std::vector<int>>());
    policy_cfg.min_z_to_research = vm["min-z"].as<int>();
    policy_cfg.max_z_to_research = vm["max-z"].as<int>();
    policy_cfg.sparse_max_weight = vm["sparse-max-weight"].as<int>();
    policy_cfg.tohpe_pool_size = vm["tohpe-pool-size"].as<int>();
    policy_cfg.todd_pool_size = vm["todd-pool-size"].as<int>();
    policy_cfg.min_tohpe_actions = vm["min-tohpe-actions"].as<int>();
    policy_cfg.min_todd_actions = vm["min-todd-actions"].as<int>();
    policy_cfg.tohpe_sample = vm["tohpe-sample"].as<int>();
    policy_cfg.bucket_temperature = static_cast<float>(vm["bucket-temperature"].as<double>());
    policy_cfg.bucket_random_fraction = static_cast<float>(vm["bucket-random-fraction"].as<double>());
    policy_cfg.max_per_signature = vm["max-per-signature"].as<int>();
    policy_cfg.escore.weights[0] = vm["escore-wred"].as<int>();
    policy_cfg.fscore.weights[0] = vm["fscore-wred"].as<int>();
    policy_cfg.min_reduction = vm["min-reduction"].as<int>();
	
	auto init_matrix = Matrix::from_npy(filename);
	auto finit_matrix = init_matrix;
	std::cerr << init_matrix.rows() << " " << init_matrix.cols() << std::endl;
	auto md_ptr = std::make_shared<MatrixWithData>(MatrixWithData(std::move(init_matrix)));
	auto result = policy_iteration_impl(md_ptr, policy_cfg, vm["seed"].as<int>(), 1);
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
    if (!output.empty()) {
        finit_matrix.save_npy(output);
        std::cerr << "Saved output matrix to: " << output << std::endl;
    } else {
        std::cerr << "No output file provided; skipping save." << std::endl;
    }
	if (Tensor3D(init_matrix) != Tensor3D(finit_matrix)) {
		throw std::runtime_error("CORE LINALG ERROR");
	}
}
