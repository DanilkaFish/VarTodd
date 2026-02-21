#include "todd_index.hpp"

#include <cstring>
namespace gf2 {

	static inline bool equal_masked(RowCView a, const Row& b) noexcept {
		return a==b;
	}
	ToddIndex::ToddIndex(const Matrix& P) : P_(P), m_(P.rows()), n_bits_(P.cols()) {
		build_masks_();
		build_row_hashes_();
		build_sum_buckets_();
	}
	void ToddIndex::build_masks_() {
		masks_.assign((std::size_t)n_bits_, 0);
		std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
		for (index_t i = 0; i < n_bits_; ++i)
			masks_[(std::size_t)i] = rng();
	}
	HashKey ToddIndex::hash_vec(RowCView v) const noexcept {
		HashKey h = 0;
		for (auto p = v.find_first(); p != RowCView::npos; p = v.find_next(p))
			h ^= masks_[(std::size_t)p];
		return h;
	}
	void ToddIndex::build_row_hashes_() {
		hP_.assign((std::size_t)m_, 0);
		for (index_t i = 0; i < m_; ++i)
			hP_[(std::size_t)i] = hash_vec(P_[i]);
	}
	std::uint32_t ToddIndex::get_bucket_id_(HashKey hk, RowCView sumv) {
		auto [it, ins] = head_.try_emplace(hk, std::numeric_limits<std::uint32_t>::max());
		std::uint32_t& head_id = it->second;
		for (std::uint32_t id = head_id; id != std::numeric_limits<std::uint32_t>::max(); id = buckets_[(std::size_t)id].next) {
			if (equal_masked(sumv, buckets_[(std::size_t)id].key))
				return id;
		}
		const std::uint32_t new_id = (std::uint32_t)buckets_.size();
		BucketInfo b;
		b.key = Row(sumv);
		b.next = head_id;
		head_id = new_id;
		buckets_.push_back(std::move(b));
		return new_id;
	}
	void ToddIndex::build_sum_buckets_() {
		const index_t n = m_;
		single_id_.assign((std::size_t)n, 0);
		pair_id_.assign((std::size_t)n * (n - 1) / 2, 0);
		head_.clear();
		head_.reserve((std::size_t)std::min<index_t>(n * 4, 2000000));
		head_.max_load_factor(0.7f);
		buckets_.clear();
		buckets_.reserve((std::size_t)std::min<index_t>(n * 4, 2000000));
		const index_t nb = (n_bits_ + 63) / 64;
		std::vector<std::uint64_t> tmp((std::size_t)nb);
		for (index_t i = 0; i < n; ++i) {
			const std::uint32_t id = get_bucket_id_(hP_[(std::size_t)i], P_[i]);
			++buckets_[(std::size_t)id].len;
			single_id_[(std::size_t)i] = id;
		}
		for (index_t i = 0; i < n; ++i) {
			auto ri = P_[i];
			const HashKey hi = hP_[(std::size_t)i];
			for (index_t j = i + 1; j < n; ++j) {
				auto rj = P_[j];
				const HashKey hk = hi ^ hP_[(std::size_t)j];
				for (index_t k = 0; k < nb; ++k)
					tmp[(std::size_t)k] = ri.data()[k] ^ rj.data()[k];
				if (nb)
					tmp.back() &= tail_mask_bits(n_bits_);
				RowCView sumv(tmp.data(), n_bits_, nb);
				const std::uint32_t id = get_bucket_id_(hk, sumv);
				++buckets_[(std::size_t)id].len;
				pair_id_[pair_index(i, j, n)] = id;
			}
		}
		std::uint32_t off = 0;
		for (auto& b : buckets_) {
			b.off = off;
			off += b.len;
			b.cur = 0;
		}
		sum_entries_.assign((std::size_t)off, SumEntry{0, 0});
		for (index_t i = 0; i < n; ++i) {
			const std::uint32_t id = single_id_[(std::size_t)i];
			auto& b = buckets_[(std::size_t)id];
			sum_entries_[(std::size_t)(b.off + b.cur++)] = SumEntry{i, k_single_sentinel<index_t>()};
		}
		for (index_t i = 0; i < n; ++i) {
			for (index_t j = i + 1; j < n; ++j) {
				const std::uint32_t id = pair_id_[pair_index(i, j, n)];
				auto& b = buckets_[(std::size_t)id];
				sum_entries_[(std::size_t)(b.off + b.cur++)] = SumEntry{i, j};
			}
		}
	}
	bool ToddIndex::sum_bucket(RowCView key, const SumEntry*& ptr, index_t& len) const noexcept {
		const HashKey hk = hash_vec(key);
		auto it = head_.find(hk);
		if (it == head_.end()) {
			ptr = nullptr;
			len = 0;
			return false;
		}
		for (std::uint32_t id = it->second; id != std::numeric_limits<std::uint32_t>::max(); id = buckets_[(std::size_t)id].next) {
			const auto& b = buckets_[(std::size_t)id];
			if (equal_masked(key, b.key)) {
				ptr = sum_entries_.data() + (std::size_t)b.off;
				len = (index_t)b.len;
				return true;
			}
		}
		ptr = nullptr;
		len = 0;
		return false;
	}
	index_t ToddIndex::get_size_from_z(RowCView z) const {
		const SumEntry* p = nullptr;
		index_t len = 0;
		sum_bucket(z, p, len);
		return len;
	}
	std::vector<std::pair<RowCView, index_t>> ToddIndex::sum_key_sizes() const {
		std::vector<std::pair<RowCView, index_t>> out;
		out.reserve(buckets_.size());
		for (const auto& b : buckets_)
			out.emplace_back(b.key.cview(), (index_t)b.len);
		return out;
	}
	index_t ToddIndex::max_bucket() const noexcept {
		index_t max = 0;
		for (const auto& b : buckets_)
			max = std::max(max,(index_t)b.len);
		return max;
	}
} // namespace gf2
