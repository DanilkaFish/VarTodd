#pragma once

#include "algorithms.hpp"
#include "nullspace.hpp"
#include "policy_expr.hpp"
#include "typedef.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace todd {
using Int = int;

namespace detail {
inline Int checked_int_from_index(index_t value, const char* what) {
    if (value > static_cast<index_t>(std::numeric_limits<Int>::max()))
        throw std::overflow_error(what);
    return static_cast<Int>(value);
}

inline Int checked_int_sum(Int a, Int b, const char* what) {
    if (b > std::numeric_limits<Int>::max() - a)
        throw std::overflow_error(what);
    return a + b;
}

inline Int checked_int_double(Int value, const char* what) {
    if (value > std::numeric_limits<Int>::max() / 2 || value < std::numeric_limits<Int>::min() / 2)
        throw std::overflow_error(what);
    return value * 2;
}
} // namespace detail

enum CandidateSource : Int {
    CandidateSourceUnknown     = 0,
    CandidateSourceTohpe       = 1,
    CandidateSourceTohpePrefix = 2,
    CandidateSourceTodd        = 3,
};

struct Candidate {
    static constexpr std::uint32_t no_bucket_id = std::numeric_limits<std::uint32_t>::max();

    Row           vec;
    Row           z;
    float         final_score            = 0.0f;
    float         pool_score             = 0.0f;
    Int           reduction              = 0;
    Int           k                      = k_single_sentinel<Int>();
    Int           l                      = k_single_sentinel<Int>();
    Int           basis_dim              = 0;
    Int           tohpe_dim              = 0;
    Int           bucket_size            = 0;
    Int           possible_max_reduction = 0;
    Int           num_better_red         = 0;
    Int           num_better_dim         = 0;
    Int           num_better_pool_score  = 0;
    Int           pool_size              = 0;
    Int           pool_tohpe_size        = 0;
    Int           pool_tohpeprefix_size  = 0;
    Int           pool_todd_size         = 0;
    Int           vec_weight             = 0;
    Int           z_weight               = 0;
    Int           z_size                 = 1;
    Int           source                 = CandidateSourceUnknown;
    std::uint32_t bucket_id              = no_bucket_id;

    Candidate() = default;

    Candidate(float s, Int r, Int kk, Int ll, Row v, Int basis_dim, Int bucket_size, Int z_weight, Int z_size,
              Int source = CandidateSourceUnknown)
        : vec(std::move(v)), pool_score(s), reduction(r), k(kk), l(ll), basis_dim(basis_dim), bucket_size(bucket_size),
          possible_max_reduction(detail::checked_int_double(bucket_size, "Candidate possible reduction overflow")),
          vec_weight(detail::checked_int_from_index(vec.count(), "Candidate vector weight overflow")),
          z_weight(z_weight), z_size(std::max<Int>(1, z_size)), source(source) {}

    Candidate(float s, Int r, Int kk, Int ll, Row&& v, Row&& zz, Int basis_dim, Int bucket_size,
              std::uint32_t bucket_id = no_bucket_id, Int source = CandidateSourceUnknown)
        : vec(std::move(v)), z(std::move(zz)), pool_score(s), reduction(r), k(kk), l(ll), basis_dim(basis_dim),
          bucket_size(bucket_size),
          possible_max_reduction(detail::checked_int_double(bucket_size, "Candidate possible reduction overflow")),
          vec_weight(detail::checked_int_from_index(vec.count(), "Candidate vector weight overflow")),
          z_weight(detail::checked_int_from_index(z.count(), "Candidate z weight overflow")),
          z_size(detail::checked_int_from_index(std::max<index_t>(1, z.size()), "Candidate z size overflow")),
          source(source), bucket_id(bucket_id) {}

    bool at_least_single() const;
    bool is_tohpe() const;
};

// Canonical fallback for candidates whose score and reduction are equal.
bool candidate_tie_preferred(const Candidate& a, const Candidate& b) noexcept;

struct SeenValues {
    Int max_red = 0;
    Int max_dim = 0;

    std::vector<index_t> red_freq;
    std::vector<index_t> dim_freq;
    std::vector<index_t> red_suf;
    std::vector<index_t> dim_suf;
    std::vector<float>   pool_scores_sorted;
    bool                 finalized = false;

    void reserve(std::size_t n);
    void observe(Int red, Int dim, float score);
    void finalize();
    Int  better_red(Int r) const;
    Int  better_dim(Int d) const;
    Int  better_score(float s) const;
    void merge_from(const SeenValues& other);
};

struct Stats {
    index_t total                  = 0;
    index_t z_researched           = 0;
    index_t nonzero                = 0;
    index_t evaluated              = 0;
    index_t accepted               = 0;
    index_t rejected               = 0;
    index_t accepted_tohpe         = 0;
    index_t accepted_tohpeprefix   = 0;
    index_t accepted_todd          = 0;
    float   mean_mr                = 0.0f;
    float   mean_basis             = 0.0f;
    float   mean_reduction         = 0.0f;
    float   mean_score             = 0.0f;
    float   max_final_tohpe_dim    = 0.0f;
    float   mean_final_tohpe_dim   = 0.0f;
    float   mean_final_score       = 0.0f;
    float   max_pool_score         = -static_cast<float>(1 << 10);
    float   max_final_score        = -static_cast<float>(1 << 10);
    Int     max_basis              = 0;
    Int     max_reduction          = 0;
    Int     max_bucket             = 0;
    index_t nonfinite_score_count  = 0;
};

struct CandidateExport {
    float final_score{};
    float pool_score{};
    Int   reduction{};
    Int   k{};
    Int   l{};
    Int   basis_dim{};
    Int   tohpe_dim{};
    Int   bucket_size{};
    Int   possible_max_reduction{};
    Int   num_better_dim{};
    Int   num_better_red{};
    Int   num_better_pool_score{};
    Int   pool_size{};
    Int   pool_tohpe_size{};
    Int   pool_tohpeprefix_size{};
    Int   pool_todd_size{};
    Int   source{CandidateSourceUnknown};
};

struct Result {
    std::vector<CandidateExport> chosen;
    std::vector<Matrix>          states;

    Stats         stats;
    std::uint64_t seed{};
};

static CandidateExport export_candidate(Candidate const& c) {
    return CandidateExport{
        .final_score            = c.final_score,
        .pool_score             = c.pool_score,
        .reduction              = c.reduction,
        .k                      = c.k,
        .l                      = c.l,
        .basis_dim              = c.basis_dim,
        .tohpe_dim              = c.tohpe_dim,
        .bucket_size            = c.bucket_size,
        .possible_max_reduction = c.possible_max_reduction,
        .num_better_dim         = c.num_better_dim,
        .num_better_red         = c.num_better_red,
        .num_better_pool_score  = c.num_better_pool_score,
        .pool_size              = c.pool_size,
        .pool_tohpe_size        = c.pool_tohpe_size,
        .pool_tohpeprefix_size  = c.pool_tohpeprefix_size,
        .pool_todd_size         = c.pool_todd_size,
        .source                 = c.source,
    };
}

constexpr Int k_all_one_hot_samples = -1;

// Scoring functor over a compiled policy expression. Holds the program and the
// bound free scalars by reference; both outlive the pools that score with them,
// which live for one policy iteration.
struct PolicyScorer {
    const PolicyProgram*   program = nullptr;
    std::span<const float> params{};
    const KnobFrame*       frame = nullptr; // per-iteration normalizers

    float evaluate(const Candidate& cand) const noexcept {
        KnobFrame f = *frame;
        f.red       = static_cast<float>(cand.reduction);
        f.dim       = static_cast<float>(cand.basis_dim);
        f.bucket    = static_cast<float>(cand.bucket_size);
        f.yw        = static_cast<float>(cand.vec_weight);
        f.zw        = static_cast<float>(cand.z_weight);
        f.zsize     = static_cast<float>(cand.z_size);
        f.max_red   = static_cast<float>(cand.possible_max_reduction);
        f.tohpe     = static_cast<float>(cand.tohpe_dim);
        f.rank_red  = static_cast<float>(cand.num_better_red);
        f.rank_dim  = static_cast<float>(cand.num_better_dim);
        f.rank_score = static_cast<float>(cand.num_better_pool_score);
        f.pool_size  = static_cast<float>(cand.pool_size);
        f.pool_tohpe = static_cast<float>(cand.pool_tohpe_size);
        f.pool_prefix = static_cast<float>(cand.pool_tohpeprefix_size);
        f.pool_todd  = static_cast<float>(cand.pool_todd_size);
        f.source     = static_cast<float>(cand.source);
        f.bucket_id  = static_cast<float>(cand.bucket_id);
        f.k_idx      = static_cast<float>(cand.k);
        f.l_idx      = static_cast<float>(cand.l);
        return program->eval(f, params);
    }
};

// Ranks candidates as each source pool is filled.
struct ExplorationScorer : PolicyScorer {
    auto operator()(Candidate& cand) const noexcept {
        cand.pool_score = evaluate(cand);
        return std::make_pair(cand.pool_score, 0);
    }
    bool ranks_fixed_y_by_reduction() const noexcept {
        return program->ranks_fixed_y_by_reduction();
    }
};

// Ranks the merged pool before selection.
struct FinalizationScorer : PolicyScorer {
    auto operator()(Candidate& cand) const noexcept {
        cand.final_score = evaluate(cand);
        return std::make_pair(cand.final_score, cand.tohpe_dim);
    }
    bool needs_tohpe_dim() const noexcept {
        return program->uses(Knob::tohpe) || program->uses(Knob::ntohpe);
    }
};

// A single `nred` instruction: score = reduction / bn / 2. This is the
// reduction-greedy default the previous ExplorationScore{1,0,0,0,0} and
// FinalizationScore{1,0,0,0,0,0} expressed, kept as the default so an
// unconfigured PolicyConfig still behaves as it always did.
inline PolicyProgram greedy_reduction_program(PolicySite site) {
    return PolicyProgram({Instr{Op::LoadKnob, static_cast<std::uint16_t>(Knob::nred)}}, {}, 0, site);
}

struct PolicyScores {
    PolicyProgram      exploration = greedy_reduction_program(PolicySite::ExplorationZ);
    PolicyProgram      final       = greedy_reduction_program(PolicySite::Finalization);
    std::vector<float> params;
};

struct ActionSelection {
    Int         count       = 1;
    std::string mode        = "best";
    float       temperature = 0.0f;
};

struct ActionPool {
    Int final_size = 16;
};

struct SamplingBudget {
    Int one_hot          = k_all_one_hot_samples;
    Int sparse           = 0;
    Int dense            = 32;
    Int sparse_max_weight = 2;
};

struct SourcePool {
    Int keep    = 1;
    Int reserve = 0;
};

struct ZBucketSearch {
    Int   min_buckets     = 32;
    Int   max_buckets     = 0;
    float temperature     = 0.0f;
    float random_fraction = 0.0f;
    Int   limit_bucket    = -1;
};

struct TohpeSearch {
    SamplingBudget sampling{};
    SourcePool     pool{2, 0};
    Int            z_choices = 8;
};

struct TohpePrefixSearch {
    SamplingBudget sampling{};
    SourcePool     pool{0, 0};
    Int            actions_per_bucket = 4;
    ZBucketSearch  buckets{};
};

struct ToddSearch {
    SamplingBudget sampling{};
    SourcePool     pool{16, 0};
    Int            actions_per_bucket = 4;
    ZBucketSearch  buckets{};
};

struct PolicyConfig {
    PolicyScores    scores{};
    ActionSelection selection{};
    ActionPool      pool{};
    TohpeSearch     tohpe{};
    ToddSearch      todd{};
    TohpePrefixSearch tohpeprefix{};
};

auto policy_iteration_impl(const std::shared_ptr<MatrixWithData>& data, PolicyConfig config, index_t seed = 1,
                           index_t add_seed = 1) -> Result;
} // namespace todd
