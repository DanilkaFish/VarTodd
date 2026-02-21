#pragma once

#include "matrix.hpp"
#include "todd_index.hpp"
#include "typedef.hpp"

#include <ankerl/unordered_dense.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace gf2 {

	struct CountWS {
		std::vector<index_t> cnt;
		std::vector<std::uint32_t> tag;
		std::vector<std::uint32_t> used;
		std::uint32_t epoch = 1;
		bool parity;
		void reset(std::size_t K) {
			if (cnt.size() < K) {
				cnt.resize(K);
				tag.resize(K, 0);
			}
			++epoch;
			if (epoch == 0) {
				std::fill(tag.begin(), tag.end(), 0);
				epoch = 1;
			}
			used.clear();
		}

		void add(std::uint32_t id, index_t delta) {
			if (tag[id] != epoch) {
				tag[id] = epoch;
				cnt[id] = delta - parity;
				used.push_back(id);
			} else
				cnt[id] += delta;
		}

		std::uint32_t argmax() const {
			std::uint32_t best = used[0];
			for (auto id : used)
				if (cnt[id] > cnt[best])
					best = id;
			return best;
		}

		std::vector<std::uint32_t> argmax_n(std::size_t n) const {
			if (used.empty()) return {};
			
			std::vector<std::uint32_t> candidates(used.begin(), used.end());
			n = std::min(n, candidates.size());
			
			std::nth_element(candidates.begin(), candidates.begin() + n, candidates.end(),
				[this](auto a, auto b) { return cnt[a] > cnt[b]; });
			
			candidates.resize(n);
			return candidates;
		}
	};

	class MatrixWithData {
	  public:
		// explicit MatrixWithData(const Matrix& P);
		MatrixWithData(Matrix&& P);

		const Matrix& P() const noexcept { return P_; }
		const ToddIndex& index() const noexcept { return index_; }

		const Matrix& tohpe_basis() const noexcept { return tohpe_basis_; }

		struct FullToddData {
			Matrix LY_nonpivot;
			std::vector<std::uint64_t> offset;
			std::vector<std::int64_t> pivot_row_of_col;
			std::vector<std::int64_t> nonpivot_index;
			std::vector<bool> is_zero;
			index_t nonpiv_cols{};
		};

		const FullToddData& full_todd() const noexcept { return full_todd_; }

	  private:
		Matrix P_;
		ToddIndex index_;

		Matrix tohpe_basis_;
		FullToddData full_todd_;
	};

	class Witness {
	  public:
		virtual ~Witness() = default;
		const auto vector() const { return z_; }
		int rank_divergence(RowCView y) const;
		int get_special() const { return special_; }
		const std::vector<std::pair<int, int>>& get_pairs() const { return pairs_; }
		virtual const Matrix& get_Y() const = 0;

	  protected:
		explicit Witness(std::shared_ptr<MatrixWithData> M, Row z);
		std::shared_ptr<MatrixWithData> M_;
		Row z_;
		int special_ = -1;
		std::vector<std::pair<int, int>> pairs_;
	};

	class TohpeWitness final : public Witness {
	  public:
		TohpeWitness(std::shared_ptr<MatrixWithData> M, Row z);
		const Matrix& get_Y() const override { return M_->tohpe_basis(); }
	};

	class ToddWitness final : public Witness {
	  public:
		ToddWitness(std::shared_ptr<MatrixWithData> M, Row z, Matrix&& Y);
		const Matrix& get_Y() const override { return Y_; }

	  private:
		Matrix Y_;
	};

	class NullSpace {
	  public:
		NullSpace(std::shared_ptr<MatrixWithData> M, std::unique_ptr<Witness>&& w) : M_{M}, w_{std::move(w)} {}

		[[nodiscard]] Matrix apply(RowCView y) const;
		[[nodiscard]] int rank_divergence(RowCView y) const { return w_->rank_divergence(y); }
		[[nodiscard]] Row linear_combination(RowCView coefs) const;
		[[nodiscard]] const Matrix& basis() const noexcept { return w_->get_Y(); }
		[[nodiscard]] const Matrix& P() const noexcept { return M_->P(); }
		[[nodiscard]] const auto vector() const noexcept { return w_->vector(); }

		NullSpace(NullSpace&&) noexcept = default;
		NullSpace& operator=(NullSpace&& other) noexcept {
			if (this != &other) {
				w_ = std::move(other.w_);
			}
			return *this;
		}

		NullSpace(const NullSpace&) = delete;
		NullSpace& operator=(const NullSpace&) = delete;

	  private:
		std::shared_ptr<MatrixWithData> M_;
		std::unique_ptr<Witness> w_;
	};

	class TohpeGenerator {
	  public:
		TohpeGenerator(std::shared_ptr<MatrixWithData> M);
		std::vector<std::pair<Row, index_t>> best_z_n(RowCView y, index_t num_samples) const;
		[[nodiscard]] NullSpace make(RowCView z) const;
		[[nodiscard]] NullSpace make(index_t row) const;
		[[nodiscard]] NullSpace make(index_t row1, index_t row2) const;
		[[nodiscard]] const Matrix& P() const noexcept { return M_->P(); }
		[[nodiscard]] Row best_z(RowCView y) const;

	  private:
		std::shared_ptr<MatrixWithData> M_;
		mutable CountWS ws_;
	};

	class FullToddGenerator {
	  public:
		FullToddGenerator(std::shared_ptr<MatrixWithData> M);
		[[nodiscard]] NullSpace make(RowCView z) const;
		[[nodiscard]] NullSpace make(index_t row) const;
		[[nodiscard]] NullSpace make(index_t row1, index_t row2) const;
		[[nodiscard]] const Matrix& P() const noexcept { return M_->P(); }
		Matrix full_todd_kernel(RowCView z) const;

	  private:
		std::shared_ptr<MatrixWithData> M_;
		mutable Matrix R_and_AUG_nonpivot_scratch_;
	};

} // namespace gf2
