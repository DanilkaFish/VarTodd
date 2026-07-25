#pragma once

#include "matrix.hpp"
#include "todd_index.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace todd {

struct TohpeZInfo {
    Row           z;
    index_t       reduction   = 0;
    index_t       bucket_size = 0;
    std::uint32_t bucket_id   = 0;
};

struct CountWSScore {
    std::uint32_t bucket_id = 0;
    index_t       count     = 0;
};

namespace detail {

template <class Count, class StateWord>
class PackedCountStorage {
    static_assert(std::is_integral_v<Count> && std::is_unsigned_v<Count>);
    static_assert(std::is_integral_v<StateWord> && std::is_unsigned_v<StateWord>);
    static_assert(std::numeric_limits<StateWord>::digits >= 2 * std::numeric_limits<Count>::digits);

  public:
    void reset(std::size_t buckets, index_t max_count, bool parity) {
        if (buckets > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("PackedCountStorage supports at most 2^32-1 buckets");
        if (max_count > static_cast<index_t>(std::numeric_limits<Count>::max()))
            throw std::overflow_error("PackedCountStorage count type is too narrow");

        const bool layout_changed = buckets_ != buckets;
        if (buckets > capacity_) {
            state_by_id_ = std::make_unique<StateWord[]>(buckets);
            touched_ids_ = std::make_unique_for_overwrite<std::uint32_t[]>(buckets);
            capacity_ = buckets;
        } else if (layout_changed && capacity_ != 0) {
            std::fill_n(state_by_id_.get(), capacity_, StateWord{0});
        }

        buckets_ = buckets;
        touched_size_ = 0;
        max_count_ = max_count;
        parity_ = parity;

        if (layout_changed)
            epoch_ = 0;
        if (epoch_ == std::numeric_limits<Count>::max()) {
            if (capacity_ != 0)
                std::fill_n(state_by_id_.get(), capacity_, StateWord{0});
            epoch_ = 1;
        } else {
            ++epoch_;
        }
    }

    void add(index_t id, index_t delta) noexcept {
        assert(id < buckets_);
        assert(id <= std::numeric_limits<std::uint32_t>::max());
        assert(delta >= static_cast<index_t>(parity_));

        constexpr unsigned count_bits = std::numeric_limits<Count>::digits;
        constexpr StateWord count_mask = static_cast<StateWord>(std::numeric_limits<Count>::max());
        const auto idx = static_cast<std::size_t>(id);
        const StateWord state = state_by_id_[idx];
        const StateWord epoch_prefix = static_cast<StateWord>(epoch_) << count_bits;
        if ((state >> count_bits) == static_cast<StateWord>(epoch_)) {
            const index_t next = static_cast<index_t>(state & count_mask) + delta;
            assert(next <= max_count_);
            assert(next <= static_cast<index_t>(std::numeric_limits<Count>::max()));
            state_by_id_[idx] = static_cast<StateWord>(epoch_prefix | static_cast<StateWord>(next));
            return;
        }

        const index_t initial = delta - static_cast<index_t>(parity_);
        assert(touched_size_ < buckets_);
        assert(initial <= max_count_);
        assert(initial <= static_cast<index_t>(std::numeric_limits<Count>::max()));
        state_by_id_[idx] = static_cast<StateWord>(epoch_prefix | static_cast<StateWord>(initial));
        touched_ids_[touched_size_++] = static_cast<std::uint32_t>(id);
    }

    std::vector<CountWSScore> argmax_n(std::size_t n) {
        std::vector<CountWSScore> out;
        argmax_n_into(n, out);
        return out;
    }

    void argmax_n_into(std::size_t n, std::vector<CountWSScore>& out) {
        out.clear();
        if (n == 0 || touched_size_ == 0)
            return;

        n = std::min(n, touched_size_);
        auto* const first = touched_ids_.get();
        auto* const last = first + touched_size_;
        if (n < touched_size_) {
            std::ranges::nth_element(first, first + n, last, [this](std::uint32_t a, std::uint32_t b) {
                return count_of_(a) > count_of_(b);
            });
        }

        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t id = first[i];
            out.push_back(CountWSScore{
                .bucket_id = id,
                .count = count_of_(id),
            });
        }
    }

    CountWSScore argmax() const {
        assert(touched_size_ != 0);
        auto* const first = touched_ids_.get();
        auto* const last = first + touched_size_;
        const auto best = std::ranges::max_element(first, last, [this](std::uint32_t a, std::uint32_t b) {
            return count_of_(a) < count_of_(b);
        });
        return CountWSScore{
            .bucket_id = *best,
            .count = count_of_(*best),
        };
    }

    std::size_t touched_size() const noexcept { return touched_size_; }

  private:
    index_t count_of_(std::uint32_t id) const noexcept {
        constexpr StateWord count_mask = static_cast<StateWord>(std::numeric_limits<Count>::max());
        return static_cast<index_t>(state_by_id_[id] & count_mask);
    }

    std::unique_ptr<StateWord[]>     state_by_id_;
    std::unique_ptr<std::uint32_t[]> touched_ids_;
    std::size_t                      capacity_ = 0;
    std::size_t                      buckets_ = 0;
    std::size_t                      touched_size_ = 0;
    Count                            epoch_ = 0;
    index_t                          max_count_ = 0;
    bool                             parity_ = false;
};

template <class Count, class MapWord>
class CompactCountStorage {
    static_assert(std::is_integral_v<Count> && std::is_unsigned_v<Count>);
    static_assert(std::is_integral_v<MapWord> && std::is_unsigned_v<MapWord>);

  public:
    void reset(std::size_t buckets, index_t max_count, bool parity) {
        if (buckets > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("CompactCountStorage supports at most 2^32-1 buckets");
        if (max_count > static_cast<index_t>(std::numeric_limits<Count>::max()))
            throw std::overflow_error("CompactCountStorage count type is too narrow");

        const unsigned map_bits = std::numeric_limits<MapWord>::digits;
        const unsigned position_bits =
            (buckets <= 1) ? 0U : std::bit_width(static_cast<std::uint64_t>(buckets - 1));
        if (position_bits >= map_bits)
            throw std::overflow_error("CompactCountStorage mapping type is too narrow");

        const bool layout_changed = buckets_ != buckets || position_bits_ != position_bits;
        if (buckets > capacity_) {
            id_to_pos_ = std::make_unique<MapWord[]>(buckets);
            counts_ = std::make_unique_for_overwrite<Count[]>(buckets);
            pos_to_id_ = std::make_unique_for_overwrite<std::uint32_t[]>(buckets);
            selection_positions_ = std::make_unique_for_overwrite<std::uint32_t[]>(buckets);
            capacity_ = buckets;
        } else if (layout_changed && capacity_ != 0) {
            std::fill_n(id_to_pos_.get(), capacity_, MapWord{0});
        }

        buckets_ = buckets;
        position_bits_ = position_bits;
        position_mask_ = (position_bits_ == 0) ? MapWord{0}
                                               : static_cast<MapWord>((MapWord{1} << position_bits_) - 1);
        max_epoch_ = static_cast<MapWord>(std::numeric_limits<MapWord>::max() >> position_bits_);
        max_count_ = max_count;
        parity_ = parity;
        touched_size_ = 0;

        if (layout_changed)
            epoch_ = 0;
        if (epoch_ == max_epoch_) {
            if (capacity_ != 0)
                std::fill_n(id_to_pos_.get(), capacity_, MapWord{0});
            epoch_ = 1;
        } else {
            ++epoch_;
        }
    }

    void add(index_t id, index_t delta) noexcept {
        assert(id < buckets_);
        assert(id <= std::numeric_limits<std::uint32_t>::max());
        assert(delta >= static_cast<index_t>(parity_));

        const auto idx = static_cast<std::size_t>(id);
        const MapWord state = id_to_pos_[idx];
        if ((state >> position_bits_) == epoch_) {
            const auto pos = static_cast<std::size_t>(state & position_mask_);
            const index_t next = static_cast<index_t>(counts_[pos]) + delta;
            assert(next <= max_count_);
            assert(next <= static_cast<index_t>(std::numeric_limits<Count>::max()));
            counts_[pos] = static_cast<Count>(next);
            return;
        }

        const auto pos = touched_size_++;
        const index_t initial = delta - static_cast<index_t>(parity_);
        assert(pos < buckets_);
        assert(initial <= max_count_);
        assert(initial <= static_cast<index_t>(std::numeric_limits<Count>::max()));
        counts_[pos] = static_cast<Count>(initial);
        pos_to_id_[pos] = static_cast<std::uint32_t>(id);
        id_to_pos_[idx] =
            static_cast<MapWord>((epoch_ << position_bits_) | static_cast<MapWord>(pos));
    }

    std::vector<CountWSScore> argmax_n(std::size_t n) {
        std::vector<CountWSScore> out;
        argmax_n_into(n, out);
        return out;
    }

    void argmax_n_into(std::size_t n, std::vector<CountWSScore>& out) {
        out.clear();
        if (n == 0 || touched_size_ == 0)
            return;

        n = std::min(n, touched_size_);
        auto* const first = selection_positions_.get();
        auto* const last = first + touched_size_;
        std::iota(first, last, std::uint32_t{0});
        if (n < touched_size_) {
            std::ranges::nth_element(first, first + n, last, [this](std::uint32_t a, std::uint32_t b) {
                return counts_[a] > counts_[b];
            });
        }

        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto pos = static_cast<std::size_t>(first[i]);
            out.push_back(CountWSScore{
                .bucket_id = pos_to_id_[pos],
                .count = static_cast<index_t>(counts_[pos]),
            });
        }
    }

    CountWSScore argmax() const {
        assert(touched_size_ != 0);
        const auto values = std::ranges::subrange(counts_.get(), counts_.get() + touched_size_);
        const auto best = std::ranges::max_element(values);
        const auto pos = static_cast<std::size_t>(best - counts_.get());
        return CountWSScore{
            .bucket_id = pos_to_id_[pos],
            .count = static_cast<index_t>(*best),
        };
    }

    std::size_t touched_size() const noexcept { return touched_size_; }

  private:
    std::unique_ptr<MapWord[]>       id_to_pos_;
    std::unique_ptr<Count[]>         counts_;
    std::unique_ptr<std::uint32_t[]> pos_to_id_;
    std::unique_ptr<std::uint32_t[]> selection_positions_;
    std::size_t                      capacity_ = 0;
    std::size_t                      buckets_ = 0;
    std::size_t                      touched_size_ = 0;
    unsigned                         position_bits_ = 0;
    MapWord                          position_mask_ = 0;
    MapWord                          epoch_ = 0;
    MapWord                          max_epoch_ = 0;
    index_t                          max_count_ = 0;
    bool                             parity_ = false;
};

} // namespace detail

class CountWS {
    using Packed8 = detail::PackedCountStorage<std::uint8_t, std::uint16_t>;
    using Packed16 = detail::PackedCountStorage<std::uint16_t, std::uint32_t>;
    using Packed32 = detail::PackedCountStorage<std::uint32_t, std::uint64_t>;
    using Wide = detail::CompactCountStorage<std::uint64_t, std::uint64_t>;
    using Storage = std::variant<Packed8, Packed16, Packed32, Wide>;

  public:
    void reset(std::size_t buckets, index_t max_bucket, bool parity) {
        if (buckets > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("CountWS supports at most 2^32-1 buckets");
        if (max_bucket > std::numeric_limits<index_t>::max() / 2)
            throw std::overflow_error("CountWS maximum count overflow");

        const index_t max_count = max_bucket * 2;
        auto select = [this, buckets, max_count, parity]<class Selected>(std::size_t count_bytes,
                                                                        std::size_t state_bytes) {
            if (!std::holds_alternative<Selected>(storage_))
                storage_.template emplace<Selected>();
            std::get<Selected>(storage_).reset(buckets, max_count, parity);
            count_bytes_ = count_bytes;
            state_bytes_ = state_bytes;
        };

        if (max_count <= std::numeric_limits<std::uint8_t>::max())
            select.template operator()<Packed8>(1, 2);
        else if (max_count <= std::numeric_limits<std::uint16_t>::max())
            select.template operator()<Packed16>(2, 4);
        else if (max_count <= std::numeric_limits<std::uint32_t>::max())
            select.template operator()<Packed32>(4, 8);
        else
            select.template operator()<Wide>(8, 16);
    }

    template <class F>
    decltype(auto) with_storage(F&& function) {
        return std::visit(
            [&function](auto& storage) -> decltype(auto) {
                return std::forward<F>(function)(storage);
            },
            storage_);
    }

    template <class F>
    decltype(auto) with_storage(F&& function) const {
        return std::visit(
            [&function](const auto& storage) -> decltype(auto) {
                return std::forward<F>(function)(storage);
            },
            storage_);
    }

    void add(index_t id, index_t delta) {
        with_storage([id, delta](auto& storage) { storage.add(id, delta); });
    }

    std::vector<CountWSScore> argmax_n(std::size_t n) {
        return with_storage([n](auto& storage) { return storage.argmax_n(n); });
    }

    void argmax_n_into(std::size_t n, std::vector<CountWSScore>& out) {
        with_storage([n, &out](auto& storage) { storage.argmax_n_into(n, out); });
    }

    CountWSScore argmax() const {
        return with_storage([](const auto& storage) { return storage.argmax(); });
    }

    std::size_t count_bytes() const noexcept { return count_bytes_; }
    std::size_t state_bytes() const noexcept { return state_bytes_; }

  private:
    Storage     storage_;
    std::size_t count_bytes_ = 1;
    std::size_t state_bytes_ = 2;
};

// Owns a matrix plus the indexes/bases reused during one policy iteration.
class MatrixWithData {
  public:
    explicit MatrixWithData(Matrix P, bool build_full_todd = true);

    const ToddIndex& index() const noexcept { return index_; }
    const Matrix&    P() const noexcept { return P_; }
    const Matrix&    tohpe_basis() const noexcept { return tohpe_basis_; }

    // RREF-derived data used to generate full-TODD nullspace bases.
    struct FullToddData {
        static constexpr std::uint32_t npos = std::numeric_limits<std::uint32_t>::max();

        Matrix                     LY_nonpivot;
        std::vector<std::uint64_t> offset;
        std::vector<std::uint32_t> pivot_row_of_col;
        std::vector<std::uint32_t> nonpivot_index;
        index_t                    nonpiv_cols{};
    };

    const FullToddData& full_todd() const;

  private:
    Matrix                              P_;
    Matrix                              tohpe_basis_;
    const ToddIndex                     index_;
    std::optional<FullToddData>         full_todd_;
};

// Base witness for predicting/applying a candidate rank reduction.
class Witness {
  public:
    virtual ~Witness()                          = default;
    virtual auto get_Y() const -> const Matrix& = 0;

    auto vector() const -> const auto& { return z_; }
    auto rank_divergence(RowCView y) const -> int;
    auto get_special() const -> index_t { return special_; }
    auto get_pairs() const -> const std::vector<std::pair<index_t, index_t>>& { return pairs_; }
    bool has_simple_entries() const noexcept { return simple_entries_; }

  protected:
    explicit Witness(std::shared_ptr<MatrixWithData> M, Row z);
    Witness(std::shared_ptr<MatrixWithData> M, Row z, const SumEntry* ptr, index_t len);
    void init_from_entries_(const SumEntry* ptr, index_t len);

    std::shared_ptr<MatrixWithData>  M_;
    std::vector<std::pair<index_t, index_t>> pairs_;
    Row                                      pair_endpoints_;
    bool                                     simple_entries_ = true;

    Row z_;
    index_t special_ = k_single_sentinel<index_t>();
};

class TohpeWitness final : public Witness {
  public:
    TohpeWitness(std::shared_ptr<MatrixWithData> M, Row z);
    TohpeWitness(std::shared_ptr<MatrixWithData> M, Row z, const SumEntry* ptr, index_t len);
    const Matrix& get_Y() const override { return M_->tohpe_basis(); }
};

class ToddWitness final : public Witness {
  public:
    ToddWitness(std::shared_ptr<MatrixWithData> M, Row z, Matrix&& Y);
    ToddWitness(std::shared_ptr<MatrixWithData> M, Row z, const SumEntry* ptr, index_t len, Matrix&& Y);
    const Matrix& get_Y() const override { return Y_; }

  private:
    Matrix Y_;
};

class NullSpace {
  public:
    NullSpace(std::shared_ptr<MatrixWithData> M, std::unique_ptr<Witness>&& w) : M_{std::move(M)}, w_{std::move(w)} {}

    Matrix        apply(RowCView y) const;
    Matrix        apply(RowCView y, std::vector<std::uint8_t>& scratch_killed) const;
    int           rank_divergence(RowCView y) const { return w_->rank_divergence(y); }
    Row           linear_combination(RowCView coefs) const;
    const Matrix& basis() const noexcept { return w_->get_Y(); }
    const Matrix& P() const noexcept { return M_->P(); }
    const auto    vector() const noexcept { return w_->vector(); }

    NullSpace(NullSpace&&) noexcept = default;
    NullSpace& operator=(NullSpace&& other) noexcept {
        if (this != &other) {
            M_ = std::move(other.M_);
            w_ = std::move(other.w_);
        }
        return *this;
    }

    NullSpace(const NullSpace&)            = delete;
    NullSpace& operator=(const NullSpace&) = delete;

  private:
    std::shared_ptr<MatrixWithData> M_;
    std::unique_ptr<Witness>        w_;
};

class TohpeGenerator {
  public:
    TohpeGenerator(std::shared_ptr<MatrixWithData> M);
    auto best_z_n(RowCView y, index_t num_samples) const -> std::vector<std::pair<Row, index_t>>;
    void best_z_n_into(RowCView y, index_t num_samples, std::vector<std::pair<Row, index_t>>& scratch_out) const;
    void best_z_n_details_into(RowCView y, index_t num_samples, std::vector<TohpeZInfo>& scratch_out) const;

    auto make(RowCView z) const -> NullSpace;
    auto make(RowCView z, std::uint32_t bucket_id) const -> NullSpace;
    auto make(index_t row) const -> NullSpace;
    auto make(index_t row1, index_t row2) const -> NullSpace;
    Row  best_z(RowCView y) const;

    const Matrix& P() const noexcept { return M_->P(); }

  private:
    std::shared_ptr<MatrixWithData> M_;

    mutable CountWS                 ws_;
    mutable std::vector<index_t>    scratch_ones_;
    mutable std::vector<index_t>    scratch_zeros_;
    mutable std::vector<CountWSScore> scratch_candidates_;
    mutable std::vector<TohpeZInfo> scratch_z_infos_;
};

class FullToddGenerator {
  public:
    explicit FullToddGenerator(std::shared_ptr<MatrixWithData> M);
    auto make(RowCView z) const -> NullSpace;
    auto make(RowCView z, const SumEntry* ptr, index_t len) const -> NullSpace;
    auto make(index_t row) const -> NullSpace;
    auto make(index_t row1, index_t row2) const -> NullSpace;
    auto full_todd_kernel(RowCView z) const -> Matrix;
    auto full_todd_kernel(RowCView z, const SumEntry* ptr, index_t len) const -> Matrix;

    const Matrix& P() const noexcept { return M_->P(); }

  private:
    std::shared_ptr<MatrixWithData> M_;
};

} // namespace todd
