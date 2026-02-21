#include "nullspace.hpp"

#include "algorithms.hpp"
#include "matrix.hpp"

#include <algorithm>
#include <random>
#include <ranges>
#include <stdexcept>

namespace gf2 {

	MatrixWithData::MatrixWithData(Matrix&& P) : P_{std::move(P)}, index_{P} {
		if (P_.rows() == 0) {
			tohpe_basis_ = Matrix(0, 0);
			full_todd_ = FullToddData{};
			return;
		}

		Matrix L = L_expansion(P_);
		Matrix Y = Matrix::identity(P_.rows());
		pivot_map pivots;
		gauss_elimination_inplace(L, Y, pivots);

		tohpe_basis_ = extract_basis(Y, pivots);

		const index_t n = P_.cols();
		const index_t full_cols = L.cols();

		Matrix L_reduced(pivots.size(), full_cols);
		Matrix Y_reduced(pivots.size(), Y.cols());
		std::vector<bool> is_zero(L_reduced.rows(), true);

		std::vector<std::pair<index_t, index_t>> pivot_mapping;
		pivot_mapping.reserve(pivots.size());

		index_t counter = 0;
		for (const auto& kv : pivots) {
			const index_t row = kv.first;
			const index_t pivot_col = kv.second;
			assign(L_reduced[counter], L[row]);
			if (!Y[row].none()) {
				assign(Y_reduced[counter], Y[row]);
				is_zero[counter] = false;
			}
			pivot_mapping.emplace_back(pivot_col, counter);
			++counter;
		}

		std::sort(pivot_mapping.begin(), pivot_mapping.end());

		full_todd_.offset.assign((std::size_t)n, 0);
		if (n > 0) {
			full_todd_.offset[(std::size_t)(n - 1)] = 0;
			for (index_t b = n - 1; b > 0; --b) {
				full_todd_.offset[(std::size_t)(b - 1)] = full_todd_.offset[(std::size_t)b] + (std::uint64_t)(b + 1);
			}
		}

		full_todd_.pivot_row_of_col.assign((std::size_t)full_cols, -1);
		for (const auto& pr : pivot_mapping) {
			full_todd_.pivot_row_of_col[(std::size_t)pr.first] = (std::int64_t)pr.second;
		}

		full_todd_.nonpivot_index.assign((std::size_t)full_cols, -1);
		index_t nonpiv_cols = 0;
		for (index_t col = 0; col < full_cols; ++col) {
			if (full_todd_.pivot_row_of_col[(std::size_t)col] < 0) {
				full_todd_.nonpivot_index[(std::size_t)col] = (std::int64_t)nonpiv_cols;
				++nonpiv_cols;
			}
		}
		full_todd_.nonpiv_cols = nonpiv_cols;

		Matrix L_nonpivot(L_reduced.rows(), nonpiv_cols);
		for (index_t r = 0; r < L_reduced.rows(); ++r) {
			Row proj(nonpiv_cols);
			const auto lr = L_reduced[r];
			for (auto p = lr.find_first(); p != Row::npos; p = lr.find_next(p)) {
				const std::int64_t idx = full_todd_.nonpivot_index[(std::size_t)p];
				if (idx >= 0) {
					proj.set((index_t)idx);
					is_zero[r] = false;
				}
			}
			if (!is_zero[r])
				assign(L_nonpivot[r], proj);
		}
		full_todd_.is_zero = std::move(is_zero);
		full_todd_.LY_nonpivot = std::move(L_nonpivot.append_right_inplace(Y_reduced));
	}

	Witness::Witness(std::shared_ptr<MatrixWithData> M, Row z) : M_{M}, z_{std::move(z)}, special_{-1} {
		const Matrix& P = M_->P();
		const ToddIndex& idx = M_->index();
		const index_t m = P.rows();
		pairs_.reserve((std::size_t)(m / 2));

		std::vector<std::uint32_t> used_tag(m, 0);

		auto is_used = [&](index_t i) -> bool { return used_tag[(std::size_t)i] == 1; };
		auto mark_used = [&](index_t i) { used_tag[(std::size_t)i] = 1; };

		const SumEntry* ptr = nullptr;
		index_t len = 0;
		if (!idx.sum_bucket(z_, ptr, len)) {
			return;
		}

		for (index_t t = 0; t < len; ++t) {
			const index_t a = ptr[t].a;
			const index_t b = ptr[t].b;
			if (b != k_single_sentinel<decltype(b)>())
				continue;
			if (P[a] == z_) {
				special_ = (int)a;
				mark_used(a);
				break;
			}
		}

		Row buf(P.cols());

		for (index_t t = 0; t < len; ++t) {
			const index_t a = ptr[t].a;
			const index_t b = ptr[t].b;
			if (b == k_single_sentinel<decltype(b)>())
				continue;
			if (is_used(a) || is_used(b))
				continue;

			assign(buf, P[a]);
			buf ^= P[b];
			if (buf == z_) {
				pairs_.emplace_back((int)a, (int)b);
				mark_used(a);
				mark_used(b);
			}
		}
	}

	TohpeWitness::TohpeWitness(std::shared_ptr<MatrixWithData> M, Row z) : Witness(M, std::move(z)) {}

	ToddWitness::ToddWitness(std::shared_ptr<MatrixWithData> M, Row z, Matrix&& Y) : Witness(M, std::move(z)), Y_{std::move(Y)} {}

	int Witness::rank_divergence(RowCView y) const {
		const int parity = (int)(y.count() & 1);
		int ones_S = 0;
		int S = (special_ != -1) ? 1 : 0;
		if (S && y.test(special_))
			++ones_S;
		int diff_pairs = 0;
		for (const auto& pr : pairs_) {
			const int i = pr.first;
			const int j = pr.second;
			if (y.test(i) ^ y.test(j))
				++diff_pairs;
		}
		if (parity == 0)
			return ones_S + 2 * diff_pairs;
		return 2 * S - ones_S - 1 + 2 * diff_pairs;
	}

	Row NullSpace::linear_combination(RowCView coefs) const {
		if (coefs.size() > basis().rows()) {
			throw std::runtime_error("Number of coefs more than basis size");
		}
		Row out(basis().cols());
		for (auto i = coefs.find_first(); i != Row::npos; i = coefs.find_next(i)) {
			if ((index_t)i >= basis().rows())
				break;
			out ^= basis()[(index_t)i];
		}
		return out;
	}

	Matrix NullSpace::apply(RowCView y) const {
		const Matrix& P0 = M_->P();
		const index_t n_rows = P0.rows();
		if (y.size() != n_rows)
			throw std::invalid_argument("apply: y.size()!=P.rows()");

		const auto z = vector();
		const int special = w_->get_special();
		const auto& pairs = w_->get_pairs();

		std::vector<std::uint8_t> killed((std::size_t)n_rows, 0);
		index_t removed = 0;
		for (const auto& pr : pairs) {
			const index_t a = (index_t)pr.first;
			const index_t b = (index_t)pr.second;
			if (y.test(a) ^ y.test(b)) {
				killed[(std::size_t)a] = 1;
				killed[(std::size_t)b] = 1;
				removed += 2;
			}
		}

		const bool parity = (y.count() & 1) != 0;
		if (special != -1 && (y.test(special) || parity)) {
			killed[(std::size_t)special] = 1;
			++removed;
		}
		if (parity && (special == -1 || y.test(special))) {
			--removed;
		}

		Matrix new_P(n_rows - removed, P0.cols());

		index_t j = 0;
		for (index_t i = 0; i < n_rows; ++i) {
			if (killed[(std::size_t)i])
				continue;
			if (y.test(i)) {
				assign(new_P[j], (P0[i] ^ z));
			} else {
				assign(new_P[j], P0[i]);
			}
			++j;
		}

		if (parity && (special == -1 || y.test(special))) {
			assign(new_P[j], z);
		}
		return new_P;
	}

	TohpeGenerator::TohpeGenerator(std::shared_ptr<MatrixWithData> M) : M_{M} {}

	NullSpace TohpeGenerator::make(index_t row) const { return NullSpace(M_, std::make_unique<TohpeWitness>(TohpeWitness(M_, M_->P()[row]))); }

	NullSpace TohpeGenerator::make(index_t row1, index_t row2) const {
		Row z(M_->P()[row1]);
		z ^= M_->P()[row2];
		return NullSpace(M_, std::make_unique<TohpeWitness>(TohpeWitness(M_, std::move(z))));
	}

	NullSpace TohpeGenerator::make(RowCView z) const { return NullSpace(M_, std::make_unique<TohpeWitness>(TohpeWitness(M_, z))); }

	Row TohpeGenerator::best_z(RowCView y) const {
		const Matrix& P = M_->P();
		const ToddIndex& idx = M_->index();
		const index_t n = P.rows();

		std::vector<index_t> ones;
		std::vector<index_t> zeros;
		ones.reserve((std::size_t)y.count());
		zeros.reserve((std::size_t)(n - y.count()));

		for (index_t i = 0; i < n; ++i) {
			(y.test(i) ? ones : zeros).push_back(i);
		}

		ws_.reset(idx.buckets_num());
		ws_.parity = y.count() % 2;
		for (index_t i : ones) {
			ws_.add(idx.single_id()[(std::size_t)i], 1);

			for (index_t j : zeros) {
				const auto id = idx.pair_id()[pair_index(i, j, n)];
				ws_.add(id, 2);
			}
		}
		if (ws_.parity){
			for (index_t i : zeros) {
				ws_.add(idx.single_id()[(std::size_t)i], 1 );
			}
		}
		if (ws_.used.empty())
			return Row(P.cols()); 

		const std::uint32_t best_id = ws_.argmax();
		return idx.key_of((std::size_t)best_id); // copy out
	}

	std::vector<std::pair<Row, index_t>> TohpeGenerator::best_z_n(RowCView y, index_t num_samples) const {
		const Matrix& P = M_->P();
		const ToddIndex& idx = M_->index();
		const index_t n = P.rows();

		std::vector<index_t> ones;
		std::vector<index_t> zeros;
		ones.reserve((std::size_t)y.count());
		zeros.reserve((std::size_t)(n - y.count()));

		for (index_t i = 0; i < n; ++i) {
			(y.test(i) ? ones : zeros).push_back(i);
		}

		ws_.reset(idx.buckets_num());
		ws_.parity = y.count() % 2;
		for (index_t i : ones) {
			ws_.add(idx.single_id()[(std::size_t)i], 1);

			for (index_t j : zeros) {
				const auto id = idx.pair_id()[pair_index(i, j, n)];
				ws_.add(id, 2);
			}
		}
		if (ws_.parity){
			for (index_t i : zeros) {
				ws_.add(idx.single_id()[(std::size_t)i], 1 );
			}
		}
		std::vector<Row> out{};

		return ws_.argmax_n(num_samples) 
			| std::views::all 
			| std::views::transform([&](auto id) { return std::pair{Row(idx.key_of(id)), index_t(ws_.cnt[id])}; })
			| std::ranges::to<std::vector>();
	}
	

	FullToddGenerator::FullToddGenerator(std::shared_ptr<MatrixWithData> M) : M_{M} {
		const index_t n = M_->P().cols();
		const index_t rows = n + 1;
		const index_t nonp = M_->full_todd().nonpiv_cols;
		R_and_AUG_nonpivot_scratch_ = Matrix(rows, nonp + M_->P().rows());
	}

	Matrix FullToddGenerator::full_todd_kernel(RowCView z) const {
		const auto& ft = M_->full_todd();
		const index_t n = z.size();
		Matrix& RA = R_and_AUG_nonpivot_scratch_;
		RA.reset();

		std::vector<index_t> S;
		S.reserve((std::size_t)z.count());
		for (auto i = z.find_first(); i != RowView::npos; i = z.find_next(i)) {
			S.push_back((index_t)i);
		}
		auto add_col = [&](auto ra_row, index_t col) {
			const std::int64_t prow = ft.pivot_row_of_col[(std::size_t)col];
			if (prow >= 0) {
				// if (!ft.is_zero[prow])
					ra_row ^= ft.LY_nonpivot[prow];
			} else {
				ra_row.flip(ft.nonpivot_index[(std::size_t)col]);
			}
		};

		for (index_t gamma = 0; gamma < n; ++gamma) {
			auto row = RA[gamma];
			const std::uint64_t offg = ft.offset[(index_t)gamma];
			for (index_t s : S) {
				if (s == gamma)
					continue;
				const index_t col = (s > gamma) ? (index_t)(ft.offset[(std::size_t)s] + gamma) : (index_t)(offg + s);
				add_col(row, col);
			}
		}

		{
			auto brow_ra = RA[RA.rows() - 1];
			for (std::size_t bi = 0; bi < S.size(); ++bi) {
				const index_t b = S[bi];
				const std::uint64_t offb = ft.offset[(std::size_t)b];
				for (std::size_t ai = 0; ai <= bi; ++ai) {
					const index_t a = S[ai];
					add_col(brow_ra, (index_t)(offb + a));
				}
			}
		}
		Matrix solution = solve_and_build_solution_basis(RA, M_->full_todd().nonpiv_cols);
		solution.append_down_inplace(M_->tohpe_basis());
		return solution;
	}

	NullSpace FullToddGenerator::make(RowCView z) const {
		Matrix Y = full_todd_kernel(z);
		return NullSpace(M_, std::make_unique<ToddWitness>(ToddWitness(M_, z, std::move(Y))));
	}

	NullSpace FullToddGenerator::make(index_t row) const {

		Row z = M_->P()[row];
		Matrix Y = full_todd_kernel(z);
		return NullSpace(M_, std::make_unique<ToddWitness>(ToddWitness(M_, std::move(z), std::move(Y))));
	}

	NullSpace FullToddGenerator::make(index_t row1, index_t row2) const {
		Row z = M_->P()[row1];
		z ^= M_->P()[row2];
		Matrix Y = full_todd_kernel(z);
		return NullSpace(M_, std::make_unique<ToddWitness>(ToddWitness(M_, std::move(z), std::move(Y))));
	}

} // namespace gf2
