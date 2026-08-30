#pragma once

// Stack-machine evaluator for Python-authored policy scoring expressions.
//
// A policy expression is compiled in Python to a flat postfix program over the
// opcodes below, then evaluated here once per scored candidate. The design
// constraint is that evaluation must stay in the same cost range as the
// weighted sum it replaces: no allocation, no virtual dispatch, no GIL.

#include "typedef.hpp"
#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace todd {

// Every quantity a policy expression may read. Raw knobs carry the value as the
// engine computes it; normalized knobs divide by the per-iteration normalizers
// (bn, dn, wvwn) so a policy can be written scale-free.
//
// The order here is the wire format shared with Python: appending is safe,
// reordering is not.
enum class Knob : std::uint16_t {
    // reduction
    red,
    nred,
    // basis dimension
    dim,
    ndim,
    // bucket size
    bucket,
    nbucket,
    // y (coefficient vector) weight
    yw,
    nyw,
    // z weight and size
    zw,
    nzw,
    zsize,
    // theoretical reduction ceiling for the bucket
    max_red,
    nmax_red,
    // TOHPE dimension of the resulting state (finalization only)
    tohpe,
    ntohpe,
    // population ranks from SeenValues (finalization only)
    rank_red,
    nrank_red,
    rank_dim,
    nrank_dim,
    rank_score,
    nrank_score,
    // pool composition (finalization only)
    pool_size,
    pool_tohpe,
    pool_prefix,
    pool_todd,
    f_tohpe,
    f_prefix,
    f_todd,
    // provenance
    source,
    bucket_id,
    k_idx,
    l_idx,
    // normalizers themselves
    bn,
    dn,
    wvwn,

    count_
};

inline constexpr std::size_t k_knob_count = static_cast<std::size_t>(Knob::count_);

using KnobMask = std::bitset<k_knob_count>;

// Where a program is evaluated. Determines which knobs are meaningful: the
// inner-z and pool sites run before finalization, so tohpe_dim, the SeenValues
// ranks and the pool composition are not yet known there.
enum class PolicySite : std::uint8_t {
    ExplorationZ,   // D1: ranking z buckets for a fixed y
    ExplorationPool,// D2: ranking candidates into a source pool
    Finalization,   // D5: ranking the merged pool
};

const char* policy_site_name(PolicySite site) noexcept;

// Knobs readable at each site. Single source of truth; mirrored to Python.
KnobMask site_available_knobs(PolicySite site) noexcept;

const char* knob_name(Knob knob) noexcept;

enum class Op : std::uint8_t {
    LoadKnob,  // arg = Knob
    LoadConst, // arg = index into consts_
    LoadParam, // arg = index into the bound params span

    // binary, pop 2 push 1
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Pow,

    // unary, pop 1 push 1
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Tanh,
    Sigmoid,
    Floor,
    Sign,

    // comparisons and logic, results are 0.0f / 1.0f
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    And,
    Or,
    Not,

    // ternary, pop 3 push 1: (cond, a, b) -> cond != 0 ? a : b
    Select,
};

struct Instr {
    Op            op;
    std::uint16_t arg = 0;
};

// Raw knob values for one candidate plus the per-iteration normalizers.
// Normalized knobs are derived on load rather than precomputed, so an
// expression that never reads them pays nothing.
struct KnobFrame {
    float red        = 0.0f;
    float dim        = 0.0f;
    float bucket     = 0.0f;
    float yw         = 0.0f;
    float zw         = 0.0f;
    float zsize      = 1.0f;
    float max_red    = 0.0f;
    float tohpe      = 0.0f;
    float rank_red   = 0.0f;
    float rank_dim   = 0.0f;
    float rank_score = 0.0f;
    float pool_size  = 0.0f;
    float pool_tohpe = 0.0f;
    float pool_prefix = 0.0f;
    float pool_todd  = 0.0f;
    float source     = 0.0f;
    float bucket_id  = 0.0f;
    float k_idx      = 0.0f;
    float l_idx      = 0.0f;

    // normalizers, injected once per policy iteration
    float bn   = 1.0f;
    float dn   = 1.0f;
    float wvwn = 1.0f;
};

// Guards matching the arithmetic the replaced scoring functions used, so that
// swapping a legacy score for its expression form is bit-comparable.
inline constexpr float k_div_epsilon = 0.001f; // polynom_scoring's inverse-distance floor
inline constexpr float k_log_floor   = 1e-6f;  // logarithmic_scoring's clamp

inline constexpr std::size_t k_max_stack_depth = 32;

class PolicyProgramError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class PolicyProgram {
  public:
    PolicyProgram() = default;

    // Validates stack discipline, depth, argument ranges and site availability,
    // then precomputes used_knobs() and the fast-path predicate. Throws
    // PolicyProgramError on any violation, so a constructed program is always
    // safe to evaluate.
    PolicyProgram(std::vector<Instr> code, std::vector<float> consts, std::size_t n_params, PolicySite site);

    float eval(const KnobFrame& f, std::span<const float> params) const noexcept;

    bool        empty() const noexcept { return code_.empty(); }
    std::size_t n_params() const noexcept { return n_params_; }
    PolicySite  site() const noexcept { return site_; }
    KnobMask    used_knobs() const noexcept { return used_; }
    bool        uses(Knob k) const noexcept { return used_.test(static_cast<std::size_t>(k)); }

    std::span<const Instr> code() const noexcept { return code_; }
    std::span<const float> consts() const noexcept { return consts_; }

    // True when, for a fixed y, the program orders candidates identically to
    // ranking by reduction descending. Lets the inner-z search keep its
    // reduction-only path instead of scoring every touched bucket.
    bool ranks_fixed_y_by_reduction() const noexcept { return ranks_by_reduction_; }

  private:
    std::vector<Instr> code_;
    std::vector<float> consts_;
    std::size_t        n_params_          = 0;
    PolicySite         site_              = PolicySite::Finalization;
    KnobMask           used_{};
    bool               ranks_by_reduction_ = false;

    void validate_();
    bool compute_ranks_by_reduction_() const;
};

} // namespace todd
