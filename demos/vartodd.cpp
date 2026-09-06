#include "matrix.hpp"
#include "nullspace.hpp"
#include "todd_generator.hpp"
#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace po = boost::program_options;
using namespace todd;

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("file,f", po::value<std::string>()->required(), "input matrix file")
        ("output_file,o", po::value<std::string>()->default_value(""), "output matrix file")
        ("action-count", po::value<int>()->default_value(1), "number of final actions to apply")
        ("selection-mode", po::value<std::string>()->default_value("best"), "final action selection mode: best or softmax")
        ("selection-temperature", po::value<double>()->default_value(0.0), "softmax temperature for final action selection")
        ("action-pool-final-size", po::value<int>()->default_value(16), "final merged action pool size")
        ("tohpe-sampling", po::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{"all", "0", "32", "2"}, "all 0 32 2"), "TOHPE sampling: one_hot sparse dense sparse_max_weight")
        ("todd-sampling", po::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{"all", "0", "32", "2"}, "all 0 32 2"), "full Todd sampling: one_hot sparse dense sparse_max_weight")
        ("tohpe-pool-keep", po::value<int>()->default_value(2), "TOHPE candidates kept; 0 disables TOHPE")
        ("tohpe-pool-reserve", po::value<int>()->default_value(0), "minimum TOHPE actions reserved before final merge")
        ("tohpe-z-choices", po::value<int>()->default_value(8), "number of TOHPE z candidates per vector")
        ("todd-pool-keep", po::value<int>()->default_value(16), "full Todd candidates kept; 0 disables full Todd")
        ("todd-pool-reserve", po::value<int>()->default_value(0), "minimum full Todd actions reserved before final merge")
        ("todd-actions-per-bucket", po::value<int>()->default_value(4), "full Todd actions kept from each z bucket")
        ("z-min-buckets,mz", po::value<int>()->default_value(32), "minimum Z buckets to research")
        ("z-max-buckets", po::value<int>()->default_value(0), "maximum Z buckets to research; 0 means unlimited")
        ("z-temperature", po::value<double>()->default_value(0.0), "Gumbel temperature for bucket choice")
        ("z-random-fraction", po::value<double>()->default_value(0.0), "fraction of researched buckets chosen randomly")
        ("escore-wred,er", po::value<int>()->default_value(1), "exploration score wred")
        ("fscore-wred,fr", po::value<int>()->default_value(1), "finalization score wred")
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
    auto parse_one_hot = [](std::string value) -> Int {
        std::ranges::transform(value, value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (value == "all")
            return k_all_one_hot_samples;
        return static_cast<Int>(std::max(0, std::stoi(value)));
    };
    auto parse_sampling = [&](const std::vector<std::string>& values) {
        if (values.size() != 4)
            throw std::runtime_error("sampling budget expects four values: one_hot sparse dense sparse_max_weight");
        return SamplingBudget{parse_one_hot(values[0]), static_cast<Int>(std::max(0, std::stoi(values[1]))),
                              static_cast<Int>(std::max(0, std::stoi(values[2]))),
                              static_cast<Int>(std::max(0, std::stoi(values[3])))};
    };
    policy_cfg.selection = ActionSelection{static_cast<Int>(vm["action-count"].as<int>()),
                                           vm["selection-mode"].as<std::string>(),
                                           static_cast<float>(vm["selection-temperature"].as<double>())};
    policy_cfg.pool      = ActionPool{static_cast<Int>(vm["action-pool-final-size"].as<int>())};
    policy_cfg.tohpe     = TohpeSearch{
        parse_sampling(vm["tohpe-sampling"].as<std::vector<std::string>>()),
        SourcePool{static_cast<Int>(vm["tohpe-pool-keep"].as<int>()),
                   static_cast<Int>(vm["tohpe-pool-reserve"].as<int>())},
        static_cast<Int>(vm["tohpe-z-choices"].as<int>())};
    policy_cfg.todd = ToddSearch{
        parse_sampling(vm["todd-sampling"].as<std::vector<std::string>>()),
        SourcePool{static_cast<Int>(vm["todd-pool-keep"].as<int>()),
                   static_cast<Int>(vm["todd-pool-reserve"].as<int>())},
        static_cast<Int>(vm["todd-actions-per-bucket"].as<int>()),
        ZBucketSearch{static_cast<Int>(vm["z-min-buckets"].as<int>()),
                      static_cast<Int>(vm["z-max-buckets"].as<int>()),
                      static_cast<float>(vm["z-temperature"].as<double>()),
                      static_cast<float>(vm["z-random-fraction"].as<double>())}};
    auto wred_program = [](int wred, PolicySite site) {
        return PolicyProgram(
            {Instr{Op::LoadKnob, static_cast<std::uint16_t>(Knob::nred)}, Instr{Op::LoadConst, 0},
             Instr{Op::Mul, 0}},
            {static_cast<float>(wred)}, 0, site);
    };
    policy_cfg.scores.exploration = wred_program(vm["escore-wred"].as<int>(), PolicySite::ExplorationZ);
    policy_cfg.scores.final       = wred_program(vm["fscore-wred"].as<int>(), PolicySite::Finalization);
	
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
