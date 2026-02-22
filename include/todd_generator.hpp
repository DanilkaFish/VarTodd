#pragma once

#include "algorithms.hpp"
#include "nullspace.hpp"
#include "typedef.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace todd {
using Int = int;

struct Candidate {
    Row   vec;
    Row   z;
    float final_score            = 0.0f;
    float pool_score             = 0.0f;
    Int   reduction              = 0;
    Int   k                      = k_single_sentinel<Int>();
    Int   l                      = k_single_sentinel<Int>();
    Int   basis_dim              = 0;
    Int   tohpe_dim              = 0;
    Int   bucket_size            = 0;
    Int   possible_max_reduction = 0;
    Int   num_better_red         = 0;
    Int   num_better_dim         = 0;
    Int   num_better_pool_score  = 0;

    std::shared_ptr<NullSpace> nsptr;

    Candidate() = default;

    Candidate(float s, Int r, Int kk, Int ll, Row v, Int basis_dim, Int bucket_size,
              const std::shared_ptr<NullSpace>& nsptr)
        : vec(std::move(v)), pool_score(s), reduction(r), k(kk), l(ll), basis_dim(basis_dim), bucket_size(bucket_size),
          possible_max_reduction(static_cast<Int>(bucket_size * 2)), nsptr(nsptr) {}

    Candidate(float s, Int r, Int kk, Int ll, Row&& v, Row&& zz, Int basis_dim, Int bucket_size,
              const std::shared_ptr<NullSpace>& nsptr)
        : vec(std::move(v)), z(std::move(zz)), pool_score(s), reduction(r), k(kk), l(ll), basis_dim(basis_dim),
          bucket_size(bucket_size), possible_max_reduction(static_cast<Int>(bucket_size * 2)), nsptr(nsptr) {}

    bool at_least_single() const;
    bool is_tohpe() const;
};

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
    index_t nonzero                = 0;
    index_t evaluated              = 0;
    index_t accepted               = 0;
    index_t rejected               = 0;
    index_t accepted_non_improving = 0;
    index_t accepted_tohpe         = 0;
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
    };
}
struct ExplorationScore {
    float   wred    = 0;
    float   wdim    = 0;
    float   wbucket = 0;
    float   wvw     = 0;
    index_t bn      = 1;
    index_t wvwn    = 1;
    index_t dn      = 1;

    explicit ExplorationScore() = default;
    ExplorationScore(index_t bn, index_t wvwn, index_t dn) : bn{bn}, wvwn{wvwn}, dn{dn} {};
    ExplorationScore(float wred, float wdim, float wbucket, float wvw)
        : wred{wred}, wdim{wdim}, wbucket{wbucket}, wvw{wvw} {}

    auto operator()(Candidate& cand) {
        cand.pool_score = cand.reduction / float(bn) / 2 * wred + cand.basis_dim / float(dn) * wdim +
                          cand.bucket_size / float(bn) * wbucket + cand.vec.count() * wvw / float(wvwn);
        return std::make_pair(cand.pool_score, 0);
    }
};

struct FinalizationScore {
    float   wred       = 0;
    float   wdim       = 0;
    float   wbucket    = 0;
    float   wvw        = 0;
    float   wtohpe_dim = 0;
    index_t bn         = 1;
    index_t wvwn       = 1;
    index_t dn         = 1;

    explicit FinalizationScore() = default;
    FinalizationScore(index_t bn, index_t wvwn, index_t dn) : bn{bn}, wvwn{wvwn}, dn{dn} {};
    FinalizationScore(float wred, float wdim, float wbucket, float wvw, float wtohpe_dim)
        : wred{wred}, wtohpe_dim{wtohpe_dim}, wdim{wdim}, wbucket{wbucket}, wvw{wvw} {}
    auto operator()(Candidate& cand) {
        cand.pool_score = wred * cand.reduction / float(bn) / 2 + wdim * cand.basis_dim / float(dn) +
                          wbucket * cand.bucket_size / float(bn) + wvw * cand.vec.count() / float(wvwn);
        if (wtohpe_dim != 0) {
            auto next_mat  = cand.nsptr->apply(cand.vec);
            auto tohpe_dim = get_tohpe_basis(next_mat).rows();
            cand.final_score += wtohpe_dim * tohpe_dim / dn;
            cand.tohpe_dim = tohpe_dim;
        }
        return std::make_pair(cand.final_score, cand.tohpe_dim);
    }
};

struct PolicyConfig {
    ExplorationScore  escore{1.0, 0., 0., 0.};
    FinalizationScore fscore{1.0, 0., 0., 0., 0.0};
    std::string       selection = "softmax";

    float  temperature                = 0.0f;
    float  non_improving_prob         = 0.0f;
    float  max_z_to_research_fraction = 1.0;
    Int    num_samples                = 64;
    Int    num_candidates             = 1;
    Int    top_pool                   = 1;
    Int    max_from_single_ns         = 100;
    Int    min_reduction              = 0;
    Int    max_reduction              = k_single_sentinel<Int>();
    Int max_z_to_research          = 1 << 20;
    Int    min_pool_size              = 1;
    Int    max_tohpe                  = 1;
    Int    threads                    = 1;
    Int    tohpe_sample               = 1;
    bool   try_only_tohpe             = true;
};

auto policy_iteration_impl(const std::shared_ptr<MatrixWithData>& data, PolicyConfig config, index_t seed = 1,
                           index_t add_seed = 1) -> Result;
} // namespace todd
