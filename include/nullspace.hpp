#pragma once

#include "matrix.hpp"
#include "todd_index.hpp"
#include "typedef.hpp"

// #include <ankerl/unordered_dense.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace todd {

struct CountWS {
    std::vector<index_t> cnt;
    std::vector<index_t> tag;
    std::vector<index_t> used;
    index_t              epoch = 1;
    bool                 parity;

    auto argmax_n(std::size_t n) const -> std::vector<index_t>;
    auto argmax() const -> index_t;
    void reset(std::size_t K);
    void add(index_t id, index_t delta);
};

// object with precacluated data used to build nullspaces and new ranks
class MatrixWithData {
  public:
    MatrixWithData(Matrix&& P);

    const ToddIndex& index() const noexcept { return index_; }
    const Matrix&    P() const noexcept { return P_; }
    const Matrix&    tohpe_basis() const noexcept { return tohpe_basis_; }

    // keeps linear independent columns of (L|Y) system
    struct FullToddData {
        Matrix                     LY_nonpivot;
        std::vector<std::uint64_t> offset;
        std::vector<std::int64_t>  pivot_row_of_col;
        std::vector<std::int64_t>  nonpivot_index;
        std::vector<bool>          is_zero;
        index_t                    nonpiv_cols{};
    };

    const FullToddData& full_todd() const noexcept { return full_todd_; }

  private:
    Matrix       P_;
    Matrix       tohpe_basis_;
    ToddIndex    index_;
    FullToddData full_todd_;
};

// base object for prediction of rank divergence
class Witness {
  public:
    virtual ~Witness()                          = default;
    virtual auto get_Y() const -> const Matrix& = 0;

    auto vector() const -> const auto& { return z_; }
    auto rank_divergence(RowCView y) const -> int;
    auto get_special() const -> int { return special_; }
    auto get_pairs() const -> const std::vector<std::pair<int, int>>& { return pairs_; }

  protected:
    explicit Witness(std::shared_ptr<MatrixWithData> M, Row z);
    std::shared_ptr<MatrixWithData>  M_;
    std::vector<std::pair<int, int>> pairs_;

    Row z_;
    int special_ = -1;
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

    Matrix        apply(RowCView y) const;
    int           rank_divergence(RowCView y) const { return w_->rank_divergence(y); }
    Row           linear_combination(RowCView coefs) const;
    const Matrix& basis() const noexcept { return w_->get_Y(); }
    const Matrix& P() const noexcept { return M_->P(); }
    const auto    vector() const noexcept { return w_->vector(); }

    NullSpace(NullSpace&&) noexcept = default;
    NullSpace& operator=(NullSpace&& other) noexcept {
        if (this != &other) {
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

    auto make(RowCView z) const -> NullSpace;
    auto make(index_t row) const -> NullSpace;
    auto make(index_t row1, index_t row2) const -> NullSpace;
    Row  best_z(RowCView y) const;

    const Matrix& P() const noexcept { return M_->P(); }

  private:
    std::shared_ptr<MatrixWithData> M_;

    mutable CountWS ws_;
};

class FullToddGenerator {
  public:
    FullToddGenerator(std::shared_ptr<MatrixWithData> M);
    auto make(RowCView z) const -> NullSpace;
    auto make(index_t row) const -> NullSpace;
    auto make(index_t row1, index_t row2) const -> NullSpace;
    auto full_todd_kernel(RowCView z) const -> Matrix;

    const Matrix& P() const noexcept { return M_->P(); }

  private:
    std::shared_ptr<MatrixWithData> M_;

    mutable Matrix R_and_AUG_nonpivot_scratch_;
};

} // namespace todd
