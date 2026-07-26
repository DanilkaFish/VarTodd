#include "todd_index.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <random>
#include <stdexcept>
namespace todd {

static inline bool equal_masked(RowCView a, RowCView b) noexcept { return a == b; }

static std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error(what);
    return a * b;
}

static std::size_t triangular_pair_count(std::size_t n) {
    if (n < 2)
        return 0;
    if ((n & 1U) == 0)
        return checked_mul(n / 2, n - 1, "ToddIndex pair count overflow");
    return checked_mul(n, (n - 1) / 2, "ToddIndex pair count overflow");
}

static void require_u32_capacity(std::size_t n, const char* what) {
    if (n > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error(what);
}

static std::size_t flat_table_capacity(std::size_t max_buckets) {
    if (max_buckets == 0)
        return 0;
    const std::size_t scaled = checked_mul(max_buckets, 10, "ToddIndex lookup table overflow");
    if (scaled > std::numeric_limits<std::size_t>::max() - 6)
        throw std::overflow_error("ToddIndex lookup table overflow");
    const std::size_t needed = (scaled + 6) / 7;
    const std::size_t cap    = std::bit_ceil(needed);
    if (cap == 0)
        throw std::overflow_error("ToddIndex lookup table overflow");
    return cap;
}

ToddIndex::ToddIndex(const Matrix& P) : P_(P), m_(P.rows()), n_bits_(P.cols()) {
    build_masks_();
    build_row_hashes_();
    build_sum_buckets_();
}

void ToddIndex::build_masks_() {
    masks_.assign(static_cast<std::size_t>(n_bits_), 0);
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
    for (index_t i = 0; i < n_bits_; ++i)
        masks_[static_cast<std::size_t>(i)] = rng();
}

HashKey ToddIndex::hash_vec(RowCView v) const noexcept {
    HashKey h = 0;
    for (auto p = v.find_first(); p != RowCView::npos; p = v.find_next(p))
        h ^= masks_[static_cast<std::size_t>(p)];
    return h;
}

void ToddIndex::build_row_hashes_() {
    hP_.assign(static_cast<std::size_t>(m_), 0);
    for (index_t i = 0; i < m_; ++i)
        hP_[static_cast<std::size_t>(i)] = hash_vec(P_[i]);
}

SumEntry ToddIndex::decode_source_position_(std::uint32_t pos) const {
    const auto row_count = static_cast<std::size_t>(m_);
    if (pos < row_count)
        return SumEntry{static_cast<index_t>(pos)};

    const std::size_t pair_pos = static_cast<std::size_t>(pos) - row_count;
    const auto upper = std::upper_bound(pair_row_off_.begin(), pair_row_off_.end(), pair_pos);
    assert(upper != pair_row_off_.begin());
    const std::size_t i = static_cast<std::size_t>(upper - pair_row_off_.begin() - 1);
    assert(i + 1 < row_count);
    const std::size_t j = i + 1 + (pair_pos - pair_row_off_[i]);
    assert(j < row_count);
    return SumEntry{static_cast<index_t>(i), static_cast<index_t>(j)};
}

Row ToddIndex::row_from_source_position_(std::uint32_t pos) const {
    const SumEntry entry = decode_source_position_(pos);
    Row            row(P_[static_cast<index_t>(entry.a)]);
    if (entry.is_pair())
        row ^= P_[static_cast<index_t>(entry.b)];
    return row;
}

bool ToddIndex::representative_equals_(std::uint32_t id, RowCView key) const noexcept {
    if (key.size() != n_bits_ || id >= representative_pos_.size())
        return false;
    const SumEntry entry = decode_source_position_(representative_pos_[id]);
    if (!entry.is_pair())
        return equal_masked(key, P_[static_cast<index_t>(entry.a)]);

    const RowCView a = P_[static_cast<index_t>(entry.a)];
    const RowCView b = P_[static_cast<index_t>(entry.b)];
    for (index_t block = 0; block < key.blocks(); ++block) {
        if (key.data()[block] != (a.data()[block] ^ b.data()[block]))
            return false;
    }
    return true;
}

std::uint32_t ToddIndex::find_bucket_(HashKey hk, RowCView key) const noexcept {
    constexpr std::uint32_t npos = std::numeric_limits<std::uint32_t>::max();
    if (lookup_slots_.empty() || key.size() != n_bits_)
        return npos;

    const std::uint64_t mixed = HashKeyHash::mix(hk);
    const std::uint32_t fingerprint =
        slot_fingerprint_bits_ == 0
            ? 0
            : static_cast<std::uint32_t>(mixed >> (64U - slot_fingerprint_bits_));
    std::size_t slot_pos = static_cast<std::size_t>(mixed) & lookup_mask_;
    for (std::size_t probe = 0; probe < lookup_slots_.size(); ++probe) {
        const std::uint32_t slot = lookup_slots_[slot_pos];
        if (slot == 0)
            return npos;
        const std::uint32_t encoded_id = slot & slot_id_mask_;
        const std::uint32_t stored_fingerprint =
            slot_fingerprint_bits_ == 0 ? 0 : static_cast<std::uint32_t>(slot >> slot_id_bits_);
        if (stored_fingerprint == fingerprint) {
            assert(encoded_id != 0);
            const std::uint32_t id = encoded_id - 1;
            if (representative_equals_(id, key))
                return id;
        }
        slot_pos = (slot_pos + 1) & lookup_mask_;
    }
    return npos;
}

std::pair<std::uint32_t, bool> ToddIndex::get_bucket_id_(HashKey hk, RowCView sumv,
                                                        std::uint32_t source_pos) {
    const std::uint64_t mixed = HashKeyHash::mix(hk);
    const std::uint32_t fingerprint =
        slot_fingerprint_bits_ == 0
            ? 0
            : static_cast<std::uint32_t>(mixed >> (64U - slot_fingerprint_bits_));
    std::size_t slot_pos = static_cast<std::size_t>(mixed) & lookup_mask_;
    for (std::size_t probe = 0; probe < lookup_slots_.size(); ++probe) {
        std::uint32_t& slot = lookup_slots_[slot_pos];
        if (slot == 0) {
            if (representative_pos_.size() >= std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("ToddIndex bucket id overflow");
            const std::uint32_t id = static_cast<std::uint32_t>(representative_pos_.size());
            representative_pos_.push_back(source_pos);
            const std::uint32_t encoded_id = id + 1;
            slot = slot_fingerprint_bits_ == 0
                       ? encoded_id
                       : static_cast<std::uint32_t>((fingerprint << slot_id_bits_) | encoded_id);
            return {id, true};
        }

        const std::uint32_t encoded_id = slot & slot_id_mask_;
        const std::uint32_t stored_fingerprint =
            slot_fingerprint_bits_ == 0 ? 0 : static_cast<std::uint32_t>(slot >> slot_id_bits_);
        if (stored_fingerprint == fingerprint) {
            assert(encoded_id != 0);
            const std::uint32_t id = encoded_id - 1;
            if (representative_equals_(id, sumv))
                return {id, false};
        }
        slot_pos = (slot_pos + 1) & lookup_mask_;
    }
    throw std::runtime_error("ToddIndex lookup table is full");
}

void ToddIndex::build_sum_buckets_() {
    // Build the full bucket index for every row and row-pair sum.
    const index_t    n          = m_;
    const auto        row_count  = static_cast<std::size_t>(n);
    const std::size_t pair_count = triangular_pair_count(row_count);
    if (pair_count > std::numeric_limits<std::size_t>::max() - row_count)
        throw std::overflow_error("ToddIndex sum count overflow");
    const std::size_t sum_count = row_count + pair_count;
    require_u32_capacity(sum_count, "ToddIndex sum count exceeds 32-bit source storage");
    require_u32_capacity(row_count, "ToddIndex row count exceeds 32-bit entry storage");

    single_id_.assign(row_count, 0);
    pair_id_.assign(pair_count, 0);

    const std::size_t table_capacity = flat_table_capacity(sum_count);
    lookup_slots_.assign(table_capacity, 0);
    lookup_mask_ = table_capacity == 0 ? 0 : table_capacity - 1;
    slot_id_bits_ = sum_count == 0 ? 0 : std::bit_width(sum_count);
    slot_fingerprint_bits_ = 32U - slot_id_bits_;
    slot_id_mask_ =
        slot_id_bits_ == 32U ? std::numeric_limits<std::uint32_t>::max()
                             : static_cast<std::uint32_t>((std::uint32_t{1} << slot_id_bits_) - 1U);

    representative_pos_.clear();
    max_bucket_ = 0;
    representative_pos_.reserve(sum_count);
    duplicate_buckets_.clear();
    duplicate_positions_.clear();
    std::vector<std::uint32_t> build_lengths;
    build_lengths.reserve(sum_count);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> duplicate_build;

    pair_row_off_.assign(row_count + 1, 0);
    std::size_t pair_off = 0;
    for (std::size_t i = 0; i < row_count; ++i) {
        pair_row_off_[i] = pair_off;
        pair_off += row_count - i - 1;
    }
    pair_row_off_[row_count] = pair_off;
    assert(pair_off == pair_count);

    const index_t              nb        = (n_bits_ + 63) / 64;
    const std::uint64_t        tail_mask = tail_mask_bits(n_bits_);
    std::vector<std::uint64_t> tmp(static_cast<std::size_t>(nb));
    std::vector<const std::uint64_t*> row_data(static_cast<std::size_t>(n), nullptr);
    for (index_t i = 0; i < n; ++i)
        row_data[static_cast<std::size_t>(i)] = P_[i].data();

    for (index_t i = 0; i < n; ++i) {
        RowCView            row(row_data[static_cast<std::size_t>(i)], n_bits_, nb);
        const auto source_pos = static_cast<std::uint32_t>(i);
        const auto [id, inserted] = get_bucket_id_(hP_[static_cast<std::size_t>(i)], row, source_pos);
        if (inserted)
            build_lengths.push_back(0);
        else
            duplicate_build.emplace_back(id, source_pos);
        ++build_lengths[static_cast<std::size_t>(id)];
        single_id_[static_cast<std::size_t>(i)] = id;
    }
    std::size_t pair_pos = 0;
    for (index_t i = 0; i < n; ++i) {
        const auto*   ri = row_data[static_cast<std::size_t>(i)];
        const HashKey hi = hP_[static_cast<std::size_t>(i)];
        for (index_t j = i + 1; j < n; ++j) {
            const auto*   rj = row_data[static_cast<std::size_t>(j)];
            const HashKey hk = hi ^ hP_[static_cast<std::size_t>(j)];
            for (index_t k = 0; k < nb; ++k)
                tmp[static_cast<std::size_t>(k)] = ri[k] ^ rj[k];
            if (nb)
                tmp.back() &= tail_mask;
            RowCView sumv(tmp.data(), n_bits_, nb);

            const auto source_pos = static_cast<std::uint32_t>(row_count + pair_pos);
            const auto [id, inserted] = get_bucket_id_(hk, sumv, source_pos);
            if (inserted)
                build_lengths.push_back(0);
            else
                duplicate_build.emplace_back(id, source_pos);
            ++build_lengths[static_cast<std::size_t>(id)];
            pair_id_[pair_pos++] = id;
        }
    }

    std::uint32_t max_bucket = 0;
    for (const std::uint32_t len : build_lengths)
        max_bucket = std::max(max_bucket, len);
    max_bucket_ = static_cast<index_t>(max_bucket);
    bucket_len_.assign(build_lengths, max_bucket);

    std::ranges::sort(duplicate_build);
    duplicate_positions_.reserve(duplicate_build.size());
    duplicate_buckets_.reserve(duplicate_build.size());
    for (std::size_t i = 0; i < duplicate_build.size();) {
        const std::uint32_t id    = duplicate_build[i].first;
        const auto begin = static_cast<std::uint32_t>(duplicate_positions_.size());
        std::size_t j = i;
        while (j < duplicate_build.size() && duplicate_build[j].first == id) {
            duplicate_positions_.push_back(duplicate_build[j].second);
            ++j;
        }
        const auto end = static_cast<std::uint32_t>(duplicate_positions_.size());
        duplicate_buckets_.push_back(DuplicateBucket{.id = id, .begin = begin, .end = end});
        assert(bucket_size(id) == static_cast<index_t>(end - begin + 1));
        i = j;
    }

    hP_.clear();
    hP_.shrink_to_fit();
}

Row ToddIndex::key_of(std::uint32_t id) const {
    if (id >= representative_pos_.size())
        return Row(n_bits_);
    return row_from_source_position_(representative_pos_[id]);
}

bool ToddIndex::materialize_bucket(RowCView key, std::vector<SumEntry>& out) const {
    out.clear();
    if (key.size() != n_bits_)
        return false;
    const auto id = find_bucket_(hash_vec(key), key);
    return id != std::numeric_limits<std::uint32_t>::max() && materialize_bucket(id, out);
}

bool ToddIndex::materialize_bucket(std::uint32_t id, std::vector<SumEntry>& out) const {
    out.clear();
    if (id >= representative_pos_.size())
        return false;
    out.reserve(static_cast<std::size_t>(bucket_size(id)));
    out.push_back(decode_source_position_(representative_pos_[id]));
    if (bucket_size(id) == 1)
        return true;

    const auto it = std::lower_bound(
        duplicate_buckets_.begin(), duplicate_buckets_.end(), id,
        [](const DuplicateBucket& bucket, std::uint32_t value) { return bucket.id < value; });
    assert(it != duplicate_buckets_.end() && it->id == id);
    for (std::uint32_t pos = it->begin; pos < it->end; ++pos)
        out.push_back(decode_source_position_(duplicate_positions_[pos]));
    assert(out.size() == static_cast<std::size_t>(bucket_size(id)));
    return true;
}

index_t ToddIndex::get_size_from_z(RowCView z) const {
    if (z.size() != n_bits_)
        return 0;
    const auto id = find_bucket_(hash_vec(z), z);
    return id == std::numeric_limits<std::uint32_t>::max() ? 0 : bucket_size(id);
}

std::vector<ToddIndex::BucketIdSize>& ToddIndex::sum_bucket_id_sizes_scratch() const {
    scratch_bucket_id_sizes_.clear();
    scratch_bucket_id_sizes_.reserve(bucket_len_.size());
    bucket_len_.with_values([&](const auto& lengths) {
        for (std::uint32_t id = 0; id < lengths.size(); ++id)
            scratch_bucket_id_sizes_.emplace_back(id, static_cast<std::uint32_t>(lengths[id]));
    });
    return scratch_bucket_id_sizes_;
}

std::vector<ToddIndex::BucketIdSize>& ToddIndex::top_sum_bucket_id_sizes_scratch(std::size_t max_count) const {
    scratch_bucket_id_sizes_.clear();
    const std::size_t N = bucket_len_.size();
    if (N == 0 || max_count == 0)
        return scratch_bucket_id_sizes_;

    const std::size_t cap = std::min(N, max_count);
    bucket_len_.with_values([&](const auto& lengths) {
        auto better = [&](std::size_t a, std::size_t b) { return lengths[a] > lengths[b]; };

        std::vector<std::size_t> idx;
        if (cap == N) {
            idx.resize(N);
            for (std::size_t i = 0; i < N; ++i)
                idx[i] = i;
            std::sort(idx.begin(), idx.end(), better);
        } else {
            idx.reserve(cap);
            for (std::size_t id = 0; id < N; ++id) {
                if (idx.size() < cap) {
                    idx.push_back(id);
                    std::push_heap(idx.begin(), idx.end(), better);
                } else if (better(id, idx.front())) {
                    std::pop_heap(idx.begin(), idx.end(), better);
                    idx.back() = id;
                    std::push_heap(idx.begin(), idx.end(), better);
                }
            }
            std::sort(idx.begin(), idx.end(), better);
        }

        scratch_bucket_id_sizes_.reserve(idx.size());
        for (auto id : idx)
            scratch_bucket_id_sizes_.emplace_back(static_cast<std::uint32_t>(id),
                                                  static_cast<std::uint32_t>(lengths[id]));
    });
    return scratch_bucket_id_sizes_;
}
} // namespace todd
