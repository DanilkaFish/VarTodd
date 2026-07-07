#pragma once

#include "matrix.hpp"
#include "todd_index.hpp"
#include "typedef.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace todd {

struct TohpeZInfo {
    Row           z;
    index_t       reduction   = 0;
    index_t       bucket_size = 0;
    std::uint32_t bucket_id   = 0;
};

struct CountWS {
    std::vector<index_t>       cnt;
    std::vector<std::uint16_t> tag;
    std::vector<std::uint32_t> used;
    std::uint16_t              epoch = 1;
    bool                       parity = false;

    auto argmax_n(std::size_t n) const -> std::vector<index_t>;
    void argmax_n_into(std::size_t n, std::vector<index_t>& scratch_out) const;
    auto argmax() const -> index_t;
    void reset(std::size_t K);
    void add(index_t id, index_t delta);
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
    mutable std::vector<index_t>    scratch_candidates_;
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
