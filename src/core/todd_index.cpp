#include "todd_index.hpp"

#include <cstring>
#include <random>
namespace todd {

static inline bool equal_masked(RowCView a, RowCView b) noexcept { return a == b; }

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
    auto [it, ins]         = head_.try_emplace(hk, std::numeric_limits<std::uint32_t>::max());
    std::uint32_t& head_id = it->second;
    for (std::uint32_t id = head_id; id != std::numeric_limits<std::uint32_t>::max();
         id               = bucket_next_[(std::size_t)id]) {
        if (equal_masked(sumv, bucket_key_(id)))
            return id;
    }
    const std::uint32_t new_id = (std::uint32_t)bucket_len_.size();
    push_bucket_key_(sumv);
    bucket_off_.push_back(0);
    bucket_len_.push_back(0);
    bucket_next_.push_back(head_id);
    head_id = new_id;
    return new_id;
}

void ToddIndex::push_bucket_key_(RowCView key) {
    assert(key.blocks() == bucket_key_blocks_per_row_);
    const auto* src = key.data();
    const std::size_t blocks   = static_cast<std::size_t>(bucket_key_blocks_per_row_);
    switch (blocks) {
    case 0:
        return;
    case 1:
        bucket_key_blocks_.push_back(src[0]);
        return;
    case 2:
        bucket_key_blocks_.push_back(src[0]);
        bucket_key_blocks_.push_back(src[1]);
        return;
    case 3:
        bucket_key_blocks_.push_back(src[0]);
        bucket_key_blocks_.push_back(src[1]);
        bucket_key_blocks_.push_back(src[2]);
        return;
    case 4:
        bucket_key_blocks_.push_back(src[0]);
        bucket_key_blocks_.push_back(src[1]);
        bucket_key_blocks_.push_back(src[2]);
        bucket_key_blocks_.push_back(src[3]);
        return;
    }
    const std::size_t old_size = bucket_key_blocks_.size();
    bucket_key_blocks_.resize(old_size + blocks);
    std::memcpy(bucket_key_blocks_.data() + old_size, src, blocks * sizeof(std::uint64_t));
}

void ToddIndex::build_sum_buckets_() {
    const index_t n = m_;
    single_id_.assign((std::size_t)n, 0);

    std::size_t pair_count = 0;
    for (index_t i = 0; i < n; ++i)
        pair_count += static_cast<std::size_t>(n - i - 1);
    pair_id_.assign(pair_count, 0);
    const std::size_t sum_count    = static_cast<std::size_t>(n) + pair_count;
    const std::size_t reserve_hint = std::min<std::size_t>(sum_count, 2000000);

    head_.clear();
    head_.reserve(reserve_hint);
    head_.max_load_factor(0.7f);
    bucket_off_.clear();
    bucket_len_.clear();
    bucket_next_.clear();
    max_bucket_ = 0;
    bucket_off_.reserve(reserve_hint);
    bucket_len_.reserve(reserve_hint);
    bucket_next_.reserve(reserve_hint);
    const index_t              nb = (n_bits_ + 63) / 64;
    bucket_key_blocks_per_row_ = nb;
    bucket_key_blocks_.clear();
    bucket_key_blocks_.reserve(reserve_hint * static_cast<std::size_t>(nb));
    const std::uint64_t        tail_mask = tail_mask_bits(n_bits_);
    std::vector<std::uint64_t> tmp((std::size_t)nb);
    std::vector<const std::uint64_t*> row_data((std::size_t)n, nullptr);
    for (index_t i = 0; i < n; ++i)
        row_data[(std::size_t)i] = P_[i].data();

    for (index_t i = 0; i < n; ++i) {
        RowCView            row(row_data[(std::size_t)i], n_bits_, nb);
        const std::uint32_t id = get_bucket_id_(hP_[(std::size_t)i], row);
        ++bucket_len_[(std::size_t)id];
        single_id_[(std::size_t)i] = id;
    }
    for (index_t i = 0; i < n; ++i) {
        const auto*   ri = row_data[(std::size_t)i];
        const HashKey hi = hP_[(std::size_t)i];
        for (index_t j = i + 1; j < n; ++j) {
            const auto*   rj = row_data[(std::size_t)j];
            const HashKey hk = hi ^ hP_[(std::size_t)j];
            for (index_t k = 0; k < nb; ++k)
                tmp[(std::size_t)k] = ri[k] ^ rj[k];
            if (nb)
                tmp.back() &= tail_mask;
            RowCView sumv(tmp.data(), n_bits_, nb);

            const std::uint32_t id = get_bucket_id_(hk, sumv);
            ++bucket_len_[(std::size_t)id];
            pair_id_[pair_index_fast_(i, j)] = id;
        }
    }
    std::uint32_t off = 0;
    for (std::uint32_t id = 0; id < bucket_len_.size(); ++id) {
        bucket_off_[(std::size_t)id] = off;
        off += bucket_len_[(std::size_t)id];
        max_bucket_ = std::max(max_bucket_, (index_t)bucket_len_[(std::size_t)id]);
    }
    std::vector<std::uint32_t> bucket_cur(bucket_len_.size(), 0);
    sum_entries_.assign((std::size_t)off, SumEntry{0, 0});
    for (index_t i = 0; i < n; ++i) {
        const std::uint32_t id  = single_id_[(std::size_t)i];
        const std::uint32_t cur = bucket_cur[(std::size_t)id]++;
        sum_entries_[(std::size_t)(bucket_off_[(std::size_t)id] + cur)] = SumEntry{i};
    }
    for (index_t i = 0; i < n; ++i) {
        for (index_t j = i + 1; j < n; ++j) {
            const std::uint32_t id  = pair_id_[pair_index_fast_(i, j)];
            const std::uint32_t cur = bucket_cur[(std::size_t)id]++;
            sum_entries_[(std::size_t)(bucket_off_[(std::size_t)id] + cur)] = SumEntry{i, j};
        }
    }
}

bool ToddIndex::sum_bucket(RowCView key, const SumEntry*& ptr, index_t& len) const noexcept {
    const HashKey hk = hash_vec(key);
    auto          it = head_.find(hk);
    if (it == head_.end()) {
        ptr = nullptr;
        len = 0;
        return false;
    }
    for (std::uint32_t id = it->second; id != std::numeric_limits<std::uint32_t>::max();
         id               = bucket_next_[(std::size_t)id]) {
        if (equal_masked(key, bucket_key_(id))) {
            ptr = sum_entries_.data() + (std::size_t)bucket_off_[(std::size_t)id];
            len = (index_t)bucket_len_[(std::size_t)id];
            return true;
        }
    }
    ptr = nullptr;
    len = 0;
    return false;
}

bool ToddIndex::sum_bucket(std::uint32_t id, const SumEntry*& ptr, index_t& len) const noexcept {
    if (id >= bucket_len_.size()) {
        ptr = nullptr;
        len = 0;
        return false;
    }
    ptr = sum_entries_.data() + (std::size_t)bucket_off_[(std::size_t)id];
    len = (index_t)bucket_len_[(std::size_t)id];
    return len != 0;
}

index_t ToddIndex::get_size_from_z(RowCView z) const {
    const SumEntry* p   = nullptr;
    index_t         len = 0;
    sum_bucket(z, p, len);
    return len;
}

std::vector<std::pair<RowCView, index_t>> ToddIndex::sum_key_sizes() const {
    std::vector<std::pair<RowCView, index_t>> out;
    out.reserve(bucket_len_.size());
    for (std::uint32_t id = 0; id < bucket_len_.size(); ++id)
        out.emplace_back(bucket_key_(id), (index_t)bucket_len_[(std::size_t)id]);
    return out;
}

std::vector<ToddIndex::SumKeySize>& ToddIndex::sum_key_sizes_scratch() const {
    scratch_sum_key_sizes_.clear();
    scratch_sum_key_sizes_.reserve(bucket_len_.size());
    for (std::uint32_t id = 0; id < bucket_len_.size(); ++id)
        scratch_sum_key_sizes_.emplace_back(bucket_key_(id), (index_t)bucket_len_[(std::size_t)id]);
    return scratch_sum_key_sizes_;
}

std::vector<ToddIndex::BucketIdSize>& ToddIndex::sum_bucket_id_sizes_scratch() const {
    scratch_bucket_id_sizes_.clear();
    scratch_bucket_id_sizes_.reserve(bucket_len_.size());
    for (std::uint32_t id = 0; id < bucket_len_.size(); ++id)
        scratch_bucket_id_sizes_.emplace_back(id, (index_t)bucket_len_[(std::size_t)id]);
    return scratch_bucket_id_sizes_;
}

std::vector<ToddIndex::BucketIdSize>& ToddIndex::top_sum_bucket_id_sizes_scratch(std::size_t max_count) const {
    scratch_bucket_id_sizes_.clear();
    const std::size_t N = bucket_len_.size();
    if (N == 0 || max_count == 0)
        return scratch_bucket_id_sizes_;

    const std::size_t cap = std::min(N, max_count);
    std::vector<std::size_t> idx(N);
    for (std::size_t i = 0; i < N; ++i)
        idx[i] = i;

    auto better = [&](std::size_t a, std::size_t b) {
        return bucket_len_[a] > bucket_len_[b];
    };

    if (cap < N) {
        std::nth_element(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(cap), idx.end(), better);
        idx.resize(cap);
    }
    std::sort(idx.begin(), idx.end(), better);

    scratch_bucket_id_sizes_.reserve(idx.size());
    for (auto id : idx)
        scratch_bucket_id_sizes_.emplace_back(static_cast<std::uint32_t>(id), (index_t)bucket_len_[id]);
    return scratch_bucket_id_sizes_;
}
} // namespace todd
