#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <vector>

// Adjust these includes to your actual paths:
#include "algorithms.hpp" // build_context()
#include "matrix.hpp"
#include "nullspace.hpp"
#include "random.hpp"

namespace gf2 {

	// Simple Node equivalent to the Python Node for this search
	struct Node {
		Matrix mat;
		std::size_t rank = 0;
		int depth = 0;
		std::size_t order = 0; // tie-breaker for the queue

		explicit Node(const Matrix& m, int depth_ = 0) : mat(m), rank(static_cast<std::size_t>(m.rows())), depth(depth_) {}

		PyRNG build_rng(std::uint32_t base_seed = 42, std::uint32_t add_seed = 0) const { return PyRNG(matrix_seed(mat, base_seed, add_seed)); }
	};

	using NodePtr = std::shared_ptr<Node>;

	struct NodeCmp {
		bool operator()(const NodePtr& a, const NodePtr& b) const {
			if (a->rank != b->rank) {
				// smaller rank = higher priority → so "bigger" in heap terms
				return a->rank > b->rank;
			}
			// tie-break on insertion order (FIFO-ish)
			return a->order > b->order;
		}
	};

	using Frontier = std::priority_queue<NodePtr, std::vector<NodePtr>, NodeCmp>;
	// m_of(node): rank
	inline std::size_t m_of(const Node* node) { return node->rank; }

	// successors: generate child nodes by applying Todd nullspaces
	// Equivalent to the Python `successors(node)` generator (but returning a vector).
	static std::vector<Matrix> successors(const Node& node) {
		std::vector<Matrix> result;

		const int num_samples = 4;
		PyRNG rng = node.build_rng();
		const Matrix& mat = node.mat;
		std::shared_ptr<MatrixWithData> md{new MatrixWithData(Matrix(mat))};
		FullToddGenerator gen_todd(md);
		const std::size_t n_rows = static_cast<std::size_t>(mat.rows());

		for (std::size_t k = 0; k < n_rows; ++k) {
			// 1-sparse
			{
				NullSpace ns = gen_todd.make(static_cast<index_t>(k));
				const Matrix& Y = ns.basis();
				auto samples = rng.sample_small_unique_bitvectors(Y.rows(), num_samples);

				for (const bitvec_t& coeffs : samples) {
					bitvec_t y = ns.linear_combination(coeffs);
					int red = ns.rank_divergence(y);
					if (red > 0) {
						Matrix child_mat = ns.apply(y);
						result.push_back(std::move(child_mat));
					}
				}
			}
			// 2-sparse
			for (std::size_t l = k + 1; l < n_rows; ++l) {
				NullSpace ns = gen_todd.make(static_cast<index_t>(k), static_cast<index_t>(l));
				const Matrix& Y = ns.basis();
				auto samples = rng.sample_small_unique_bitvectors(Y.rows(), num_samples);

				for (const bitvec_t& coeffs : samples) {
					bitvec_t y = ns.linear_combination(coeffs);
					int red = ns.rank_divergence(y);
					if (red > 0) {
						Matrix child_mat = ns.apply(y);
						result.push_back(std::move(child_mat));
					}
				}
			}
		}

		return result;
	}

	static std::pair<Matrix, std::size_t> search_min_m(const Matrix& root_mat) {
		Frontier frontier;

		std::size_t counter = 0;

		// Push root
		{
			auto root = std::make_unique<Node>(root_mat, /*depth=*/0);
			root->order = counter++;
			frontier.push(std::move(root)); // ✅ move into the queue
		}

		Matrix best_mat = root_mat;
		std::size_t best_m = best_mat.rows();

		// const std::size_t MAX_EXPANSIONS = 1'000'000;
		std::size_t print_counter = 0;

		while (!frontier.empty()) {
			// while (!frontier.empty() && counter < MAX_EXPANSIONS) {
			// TAKE OWNERSHIP of the top node
			NodePtr current = frontier.top(); // ✅ move out of the queue
			frontier.pop();
			Node& s = *current;

			std::size_t m = s.rank;

			if (print_counter < counter / 1000) {
				++print_counter;
				std::cout << "counter=" << counter << " best_m=" << best_m << " frontier_size=" << frontier.size() << "\n";
			}

			bool has_child = false;
			auto child_mats = successors(s); // returns std::vector<Matrix>

			for (auto& child_mat : child_mats) {
				has_child = true;
				auto child = std::make_unique<Node>(child_mat, s.depth + 1);
				child->order = counter++;
				frontier.push(std::move(child)); // ✅ move into queue
			}

			if (!has_child) {
				if (m < best_m) {
					best_m = m;
					best_mat = s.mat; // copy the matrix you care about
				}
			}
			// if (counter > 1000) {
			//     return {best_mat, best_m};
			// }
			// current goes out of scope here → Node is destroyed automatically.
		}

		return {best_mat, best_m};
	}

} // namespace gf2

// ------------------ main ------------------

int main() {
	try {
		using namespace gf2;

		// Build matrix from context (.npy) as we did earlier
		Matrix mat = build_context();
		Tensor3D tensor(mat);

		std::cout << "mat.rows = " << mat.rows() << "\n";
		std::cout << "mat.cols = " << mat.cols() << "\n";

		// Root node
		// Node root(mat, 0);

		auto [best_mat, best_m] = search_min_m(mat);

		// Matrix best_mat = best_node->mat;

		std::cout << "best_m = " << best_m << "\n";
		std::cout << "best_mat.rows = " << best_mat.rows() << "\n";
		std::cerr << "best_mat.cols = " << best_mat.cols() << "\n";

		Tensor3D tensor_best(best_mat);

		std::cout << "Tensor3D equal: " << ((tensor_best == tensor) ? "true" : "false") << "\n";

		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "Exception: " << ex.what() << "\n";
		return 1;
	}
}
