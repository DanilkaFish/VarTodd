#include "todd_generator.hpp"

#include "nullspace.hpp"
#include "random.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace todd {

bool candidate_tie_preferred(const Candidate& a, const Candidate& b) noexcept {
    if (a.source != b.source)
        return a.source < b.source;
    if (a.k != b.k)
        return a.k < b.k;
    if (a.l != b.l)
        return a.l < b.l;
    if (a.bucket_id != b.bucket_id)
        return a.bucket_id < b.bucket_id;
    if (!(a.z == b.z))
        return a.z < b.z;
    return a.vec < b.vec;
}

namespace {

bool scored_candidate_preferred(const Candidate& a, float a_score, const Candidate& b, float b_score) noexcept {
    if (a_score != b_score)
        return a_score > b_score;
    if (a.reduction != b.reduction)
        return a.reduction > b.reduction;
    return candidate_tie_preferred(a, b);
}

} // namespace

template <class Score> class TopKPool {
  public:
    explicit TopKPool(std::size_t limit, Score keyfn = Score{}) : limit_(limit), keyfn_(std::move(keyfn)) {}

    std::size_t size() const noexcept { return heap_.size(); }

    bool empty() const noexcept { return heap_.empty(); }
    void reserve(std::size_t n) { heap_.reserve(n); }

    std::pair<float, float> push(Candidate&& c) {
        if (limit_ == 0)
            return {0, 0};
        const auto [key, tohpe] = keyfn_(c);
        if (heap_.size() < limit_) {
            heap_.push_back(Node{key, std::move(c)});
            std::push_heap(heap_.begin(), heap_.end(), worse_node_);
            return {key, tohpe};
        }
        if (worse_than(heap_.front(), key, c))
            return {key, tohpe};
        std::pop_heap(heap_.begin(), heap_.end(), worse_node_);
        heap_.back() = Node{key, std::move(c)};
        std::push_heap(heap_.begin(), heap_.end(), worse_node_);
        return {key, tohpe};
    }

    void merge_from(TopKPool& other) {
        for (auto& n : other.heap_)
            push(std::move(n.cand));
        other.heap_.clear();
    }

    std::vector<Candidate> release_unsorted() {
        std::vector<Candidate> out;
        out.reserve(heap_.size());
        for (auto& n : heap_)
            out.push_back(std::move(n.cand));
        heap_.clear();
        return out;
    }

  private:
    struct Node {
        float     key;
        Candidate cand;
    };

    std::size_t       limit_;
    Score             keyfn_;
    std::vector<Node> heap_;
    static bool       worse_candidate_(float a_key, Candidate const& a, float b_key, Candidate const& b) noexcept {
        return scored_candidate_preferred(a, a_key, b, b_key);
    }
    static bool worse_node_(Node const& a, Node const& b) noexcept {
        return worse_candidate_(a.key, a.cand, b.key, b.cand);
    }
    static bool worse_than(Node const& a, float b_key, Candidate const& b) noexcept {
        return worse_candidate_(a.key, a.cand, b_key, b);
    }
};

template <class TohpeGenProvider, class FullGenProvider>
NullSpace make_candidate_nullspace(Candidate const& c, TohpeGenProvider& get_tohpe_gen, FullGenProvider& get_full_gen) {
    if (c.is_tohpe()) {
        if (c.bucket_id != Candidate::no_bucket_id)
            return get_tohpe_gen().make(c.z, c.bucket_id);
        return get_tohpe_gen().make(c.z);
    }
    if (c.l == k_single_sentinel<Int>())
        return get_full_gen().make(static_cast<index_t>(c.k));
    return get_full_gen().make(static_cast<index_t>(c.k), static_cast<index_t>(c.l));
}

std::uint64_t splitmix64_step(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

index_t mixed_seed(std::uint64_t base_seed, std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d) {
    std::uint64_t x = splitmix64_step(base_seed);
    x ^= splitmix64_step(a + 0x632be59bd9b4e019ULL);
    x ^= splitmix64_step(b + 0x85157af5ULL);
    x ^= splitmix64_step(c + 0x9e3779b97f4a7c15ULL);
    x ^= splitmix64_step(d + 0xbf58476d1ce4e5b9ULL);
    return static_cast<index_t>((x >> 32) ^ x);
}

template <class Mean, class Value, class Count> void update_mean_value(Mean& mean, Value value, Count n) {
    const auto denom = n > Count{0} ? n : Count{1};
    mean += (value - mean) / denom;
}

using FinalizationPool = TopKPool<FinalizationScore>;

struct NormalizedSamplingBudget {
    std::array<index_t, 3> vector_samples{};
    Int                   sparse_max_weight = 2;
};

struct NormalizedSourcePool {
    Int keep    = 0;
    Int reserve = 0;
};

struct NormalizedZBucketSearch {
    Int   min_buckets     = 0;
    Int   max_buckets     = 0;
    float temperature     = 0.0f;
    float random_fraction = 0.0f;
    Int   limit_bucket    = -1;
};

struct NormalizedTohpeSearch {
    NormalizedSamplingBudget sampling{};
    NormalizedSourcePool     pool{};
    Int                      z_choices = 1;
};

struct NormalizedTohpePrefixSearch {
    NormalizedSamplingBudget sampling{};
    NormalizedSourcePool     pool{};
    Int                      actions_per_bucket = 1;
    NormalizedZBucketSearch  buckets{};
};

struct NormalizedToddSearch {
    NormalizedSamplingBudget sampling{};
    NormalizedSourcePool     pool{};
    Int                      actions_per_bucket = 1;
    NormalizedZBucketSearch  buckets{};
};

struct NormalizedPolicyConfig {
    ExplorationScore      escore{};
    FinalizationScore     fscore{};
    ActionSelection       selection{};
    ActionPool            pool{};
    NormalizedTohpeSearch       tohpe{};
    NormalizedToddSearch        todd{};
    NormalizedTohpePrefixSearch tohpeprefix{};
};

NormalizedSamplingBudget normalize_sampling_budget(SamplingBudget sampling) {
    auto one_hot_count = static_cast<index_t>(std::numeric_limits<index_t>::max());
    if (sampling.one_hot >= 0)
        one_hot_count = static_cast<index_t>(sampling.one_hot);

    return {
        .vector_samples =
            {
                one_hot_count,
                static_cast<index_t>(std::max<Int>(sampling.sparse, 0)),
                static_cast<index_t>(std::max<Int>(sampling.dense, 0)),
            },
        .sparse_max_weight = std::max(sampling.sparse_max_weight, static_cast<Int>(2)),
    };
}

NormalizedSourcePool normalize_source_pool(SourcePool pool) {
    return {
        .keep    = std::max(pool.keep, static_cast<Int>(0)),
        .reserve = std::max(pool.reserve, static_cast<Int>(0)),
    };
}

NormalizedZBucketSearch normalize_bucket_search(ZBucketSearch search) {
    return {
        .min_buckets     = std::max(search.min_buckets, static_cast<Int>(0)),
        .max_buckets     = std::max(search.max_buckets, static_cast<Int>(0)),
        .temperature     = std::max(search.temperature, 0.0f),
        .random_fraction = std::clamp(search.random_fraction, 0.0f, 1.0f),
        .limit_bucket    = search.limit_bucket < 0 ? static_cast<Int>(-1)
                                                    : std::max(search.limit_bucket, static_cast<Int>(0)),
    };
}

NormalizedPolicyConfig normalize_policy_config(PolicyConfig config) {
    NormalizedPolicyConfig out{
        .escore    = std::move(config.scores.exploration),
        .fscore    = std::move(config.scores.final),
        .selection = std::move(config.selection),
        .pool      = config.pool,
        .tohpe =
            {
                .sampling  = normalize_sampling_budget(config.tohpe.sampling),
                .pool      = normalize_source_pool(config.tohpe.pool),
                .z_choices = std::max(config.tohpe.z_choices, static_cast<Int>(1)),
            },
        .todd =
            {
                .sampling           = normalize_sampling_budget(config.todd.sampling),
                .pool               = normalize_source_pool(config.todd.pool),
                .actions_per_bucket = std::max(config.todd.actions_per_bucket, static_cast<Int>(1)),
                .buckets            = normalize_bucket_search(config.todd.buckets),
            },
        .tohpeprefix =
            {
                .sampling           = normalize_sampling_budget(config.tohpeprefix.sampling),
                .pool               = normalize_source_pool(config.tohpeprefix.pool),
                .actions_per_bucket = std::max(config.tohpeprefix.actions_per_bucket, static_cast<Int>(1)),
                .buckets            = normalize_bucket_search(config.tohpeprefix.buckets),
            },
    };
    out.selection.count = std::max(out.selection.count, static_cast<Int>(1));
    const Int tohpe_reserve =
        detail::checked_int_sum(out.tohpe.pool.reserve, out.tohpeprefix.pool.reserve, "TOHPE pool reserve overflow");
    const Int reserved_sources = detail::checked_int_sum(tohpe_reserve, out.todd.pool.reserve,
                                                         "policy pool reserve overflow");
    out.pool.final_size = std::max({out.selection.count, out.pool.final_size, reserved_sources});
    return out;
}

struct PolicyIterationContext {
    std::shared_ptr<MatrixWithData>  data;
    std::uint64_t                    base_seed{};
    index_t                          tohpe_dim{};
    std::optional<TohpeGenerator>    tohpe_gen;
    std::optional<FullToddGenerator> full_gen;

    PolicyIterationContext(std::shared_ptr<MatrixWithData> d, std::uint64_t seed)
        : data(std::move(d)), base_seed(seed), tohpe_dim(data->tohpe_basis().rows()) {}

    FullToddGenerator& get_full_gen() {
        if (!full_gen)
            full_gen.emplace(data);
        return *full_gen;
    }

    TohpeGenerator& get_tohpe_gen() {
        if (!tohpe_gen)
            tohpe_gen.emplace(data);
        return *tohpe_gen;
    }
};

PolicyIterationContext make_policy_context(const std::shared_ptr<MatrixWithData>& data, NormalizedPolicyConfig& config,
                                           index_t seed, index_t add_seed) {
    PolicyIterationContext ctx(data, matrix_seed(data->P(), seed, add_seed));
    const auto             bucket_normalization = data->index().max_bucket();
    const auto             wvw_normalization    = data->P().rows();
    const auto             dim_normalization    = ctx.tohpe_dim + 5;
    config.escore.bn                         = bucket_normalization;
    config.escore.wvwn                       = wvw_normalization;
    config.escore.dn                         = dim_normalization;
    config.fscore.bn                         = bucket_normalization;
    config.fscore.wvwn                       = wvw_normalization;
    config.fscore.dn                         = dim_normalization;
    return ctx;
}

struct CandidatePools {
    TopKPool<ExplorationScore> tohpe_pool;
    TopKPool<ExplorationScore> tohpeprefix_pool;
    TopKPool<ExplorationScore> todd_pool;

    explicit CandidatePools(const NormalizedPolicyConfig& config)
        : tohpe_pool(config.tohpe.pool.keep, config.escore),
          tohpeprefix_pool(config.tohpeprefix.pool.keep, config.escore),
          todd_pool(config.todd.pool.keep, config.escore) {}
};

struct GenerationOutput {
    CandidatePools pools;
    SeenValues     svs;
    Stats          stats;

    explicit GenerationOutput(const NormalizedPolicyConfig& config) : pools(config) {}
};

template <class Buckets>
void diversify_buckets(Buckets& buckets, std::size_t max_count, float temperature, float random_fraction, PyRNG& rng) {
    if (buckets.empty() || max_count == 0) {
        buckets.clear();
        return;
    }

    const std::size_t N   = buckets.size();
    const std::size_t cap = std::min(N, max_count);
    temperature           = std::max(0.0f, temperature);
    random_fraction       = std::clamp(random_fraction, 0.0f, 1.0f);
    auto better = [&](std::size_t a, std::size_t b) {
        return buckets[a].second != buckets[b].second ? buckets[a].second > buckets[b].second
                                                      : buckets[a].first < buckets[b].first;
    };

    if (temperature == 0.0f && random_fraction == 0.0f) {
        std::vector<std::size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        if (cap < N) {
            std::nth_element(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(cap), idx.end(), better);
            idx.resize(cap);
        }
        std::sort(idx.begin(), idx.end(), better);

        Buckets out;
        out.reserve(idx.size());
        for (auto id : idx)
            out.push_back(std::move(buckets[id]));
        buckets = std::move(out);
        return;
    }

    std::vector<std::size_t> idx(N);
    std::vector<double>      score(N);
    std::iota(idx.begin(), idx.end(), 0);
    for (std::size_t i = 0; i < N; ++i) {
        double u = rng.rand_double(0.0, 1.0);
        if (u <= 0.0)
            u = 1e-12;
        if (u >= 1.0)
            u = 1.0 - 1e-12;
        const double gumbel = -std::log(-std::log(u));
        score[i] = std::log1p(static_cast<double>(buckets[i].second)) + static_cast<double>(temperature) * gumbel;
    }

    const std::size_t random_count = std::min<std::size_t>(cap, static_cast<std::size_t>(cap * random_fraction));
    const std::size_t top_count    = cap - random_count;

    std::vector<std::size_t> top_idx;
    std::vector<std::uint8_t> selected(N, 0);
    top_idx.reserve(top_count);

    if (top_count > 0) {
        if (top_count < N) {
            std::nth_element(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(top_count), idx.end(),
                             [&](std::size_t a, std::size_t b) {
                                 if (score[a] != score[b])
                                     return score[a] > score[b];
                                 return better(a, b);
                             });
            idx.resize(top_count);
        }
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            if (score[a] != score[b])
                return score[a] > score[b];
            return better(a, b);
        });
        for (auto id : idx) {
            selected[id] = 1;
            top_idx.push_back(id);
        }
    }

    std::vector<std::size_t> random_idx;
    random_idx.reserve(random_count);
    if (random_count > 0) {
        std::vector<std::size_t> remaining;
        remaining.reserve(N - top_idx.size());
        for (std::size_t i = 0; i < N; ++i) {
            if (!selected[i])
                remaining.push_back(i);
        }
        rng.shuffle(remaining);
        if (remaining.size() > random_count)
            remaining.resize(random_count);
        random_idx = std::move(remaining);
    }

    std::vector<std::size_t> ordered;
    ordered.reserve(top_idx.size() + random_idx.size());
    std::size_t top_i = 0;
    std::size_t rnd_i = 0;
    while (ordered.size() < cap && (top_i < top_idx.size() || rnd_i < random_idx.size())) {
        const bool due_random =
            rnd_i < random_idx.size() &&
            (top_i >= top_idx.size() || ordered.size() + 1 >= ((rnd_i + 1) * cap) / (random_idx.size() + 1));
        if (due_random) {
            ordered.push_back(random_idx[rnd_i++]);
        } else if (top_i < top_idx.size()) {
            ordered.push_back(top_idx[top_i++]);
        } else {
            ordered.push_back(random_idx[rnd_i++]);
        }
    }

    Buckets out;
    out.reserve(ordered.size());
    for (auto id : ordered)
        out.push_back(std::move(buckets[id]));
    buckets = std::move(out);
}

bool pool_preferred(Candidate const& a, Candidate const& b) {
    return scored_candidate_preferred(a, a.pool_score, b, b.pool_score);
}

// Move the reserved best prefix from one source; remaining candidates are merged and sorted later.
void take_source_quota(std::vector<Candidate>& out, std::vector<Candidate>& src, std::size_t desired,
                       std::size_t hard_limit) {
    if (desired == 0 || hard_limit == 0 || out.size() >= hard_limit || src.empty())
        return;
    const std::size_t n = std::min({desired, hard_limit - out.size(), src.size()});
    std::partial_sort(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(n), src.end(), pool_preferred);
    for (std::size_t i = 0; i < n; ++i)
        out.push_back(std::move(src[i]));
    src.erase(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(n));
}

std::vector<Candidate> merge_source_candidates(std::vector<Candidate> tohpe, std::vector<Candidate> tohpeprefix,
                                               std::vector<Candidate> todd, std::size_t final_size,
                                               std::size_t tohpe_reserve, std::size_t tohpeprefix_reserve,
                                               std::size_t todd_reserve) {
    std::vector<Candidate> out;
    out.reserve(final_size);
    take_source_quota(out, tohpe, tohpe_reserve, final_size);
    take_source_quota(out, tohpeprefix, tohpeprefix_reserve, final_size);
    take_source_quota(out, todd, todd_reserve, final_size);

    std::vector<Candidate> rest;
    rest.reserve(tohpe.size() + tohpeprefix.size() + todd.size());
    for (auto& c : tohpe)
        rest.push_back(std::move(c));
    for (auto& c : tohpeprefix)
        rest.push_back(std::move(c));
    for (auto& c : todd)
        rest.push_back(std::move(c));

    const std::size_t remaining = final_size - out.size();
    if (remaining < rest.size()) {
        std::partial_sort(rest.begin(), rest.begin() + static_cast<std::ptrdiff_t>(remaining), rest.end(),
                          pool_preferred);
        rest.resize(remaining);
    } else {
        std::sort(rest.begin(), rest.end(), pool_preferred);
    }
    for (auto& c : rest) {
        out.push_back(std::move(c));
    }
    return out;
}

Row linear_combination_from_basis(const Matrix& basis, RowCView coefs) {
    if (coefs.size() > basis.rows()) {
        throw std::runtime_error("Number of coefs more than basis size");
    }
    Row out(basis.cols());
    for (auto i = coefs.find_first(); i != Row::npos; i = coefs.find_next(i)) {
        if (static_cast<index_t>(i) >= basis.rows())
            break;
        out ^= basis[static_cast<index_t>(i)];
    }
    return out;
}

std::vector<Candidate> pick_best_view(FinalizationPool& pool, std::size_t n) {
    auto all = pool.release_unsorted();
    if (all.empty() || n == 0)
        return {};
    if (n > all.size())
        n = all.size();
    auto better = [&](const Candidate& a, const Candidate& b) {
        return scored_candidate_preferred(a, a.final_score, b, b.final_score);
    };
    if (n < all.size()) {
        std::ranges::nth_element(all, all.begin() + n, better);
        all.resize(n);
    }
    std::ranges::sort(all, better);
    return all;
}

std::vector<Candidate> pick_softmax_view(FinalizationPool& pool, std::size_t n, float temperature, PyRNG& rng) {
    auto all = pool.release_unsorted();
    if (all.empty() || n == 0)
        return {};
    if (n > all.size())
        n = all.size();
    auto better = [&](const Candidate& a, const Candidate& b) {
        return scored_candidate_preferred(a, a.final_score, b, b.final_score);
    };
    if (!(temperature > 0.0f) || !std::isfinite(temperature)) {
        if (n < all.size()) {
            std::ranges::nth_element(all, all.begin() + n, better);
            all.resize(n);
        }
        std::ranges::sort(all, better);
        return all;
    }

    const std::size_t        N = all.size();
    std::vector<float>       key(N);
    std::vector<std::size_t> idx(N);

    // Preserve the historical candidate-to-random-draw order for softmax selection.
    std::ranges::sort(all, better);
    for (std::size_t i = 0; i < N; ++i) {
        double u = rng.rand_double(0.0, 1.0);
        if (u <= 0.0)
            u = 1e-12;
        if (u >= 1.0)
            u = 1.0 - 1e-12;
        const float g = static_cast<float>(-std::log(-std::log(u)));
        key[i]        = static_cast<float>(all[i].final_score / temperature) + g;
        idx[i]        = i;
    }

    if (n < N) {
        std::nth_element(idx.begin(), idx.begin() + n, idx.end(),
                         [&](std::size_t a, std::size_t b) {
                             return key[a] != key[b] ? key[a] > key[b] : better(all[a], all[b]);
                         });
        idx.resize(n);
    }

    std::ranges::sort(idx, [&](std::size_t a, std::size_t b) {
        if (key[a] != key[b])
            return key[a] > key[b];
        return better(all[a], all[b]);
    });

    std::vector<Candidate> chosen;
    chosen.reserve(n);
    for (auto i : idx)
        chosen.push_back(std::move(all[i]));

    std::ranges::sort(chosen, better);
    return chosen;
}

template <class TohpeGenProvider, class FullGenProvider>
Result build_result(FinalizationPool& pool, bool is_best, std::size_t action_count, float temperature,
                    PyRNG& pick_rng, SeenValues const& svs, TohpeGenProvider& get_tohpe_gen,
                    FullGenProvider& get_full_gen, Stats const& global_stats, std::uint64_t base_seed) {
    auto chosen_view =
        is_best ? pick_best_view(pool, action_count) : pick_softmax_view(pool, action_count, temperature, pick_rng);

    Result out;
    out.stats = global_stats;
    out.seed  = base_seed;

    out.chosen.reserve(chosen_view.size());
    out.states.reserve(chosen_view.size());

    struct Acc {
        Result& out;
        void    add(Candidate const& c, decltype(out.states)::value_type state) {
            out.chosen.push_back(export_candidate(c));
            out.states.push_back(std::move(state));
        }
    } acc{out};

    std::vector<std::uint8_t> scratch_killed;
    for (Candidate& c : chosen_view) {
        c.num_better_dim        = svs.better_dim(c.basis_dim);
        c.num_better_red        = svs.better_red(c.reduction);
        c.num_better_pool_score = svs.better_score(c.pool_score);

        auto ns    = make_candidate_nullspace(c, get_tohpe_gen, get_full_gen);
        auto state = ns.apply(c.vec, scratch_killed);

        acc.add(c, std::move(state));
    }

    return out;
}

void generate_tohpe_candidates(PolicyIterationContext& ctx, const NormalizedPolicyConfig& config,
                               GenerationOutput& out) {
    if (config.tohpe.pool.keep <= 0)
        return;

    auto local_pool = TopKPool<ExplorationScore>(config.tohpe.pool.keep, config.escore);
    const index_t dim     = ctx.tohpe_dim;
    const Int     dim_int = detail::checked_int_from_index(dim, "TOHPE basis dimension overflow");
    out.stats.max_basis   = std::max(out.stats.max_basis, dim_int);

    PyRNG local_rng(mixed_seed(ctx.base_seed, 0, 0, 0, CandidateSourceTohpe));
    std::vector<TohpeZInfo> scratch_best_z;
    local_rng.for_each_capped_bitvector(dim, config.tohpe.sampling.vector_samples,
                                        config.tohpe.sampling.sparse_max_weight, [&](RowCView coefs) {
        assert(coefs.count() != 0);
        out.stats.evaluated++;
        auto vec = linear_combination_from_basis(ctx.data->tohpe_basis(), coefs);
        ctx.get_tohpe_gen().best_z_n_details_into(vec, config.tohpe.z_choices, scratch_best_z);
        for (std::size_t zi = 0; zi < scratch_best_z.size(); ++zi) {
            auto&      info = scratch_best_z[zi];
            const auto red  = info.reduction;
            if (red <= 0) {
                out.stats.rejected++;
                continue;
            }
            Row cand_vec = (zi + 1 == scratch_best_z.size()) ? std::move(vec) : Row(vec);
            Candidate candidate(0, detail::checked_int_from_index(red, "Candidate reduction overflow"),
                                k_single_sentinel<Int>(), k_single_sentinel<Int>(), std::move(cand_vec),
                                std::move(info.z), dim_int,
                                detail::checked_int_from_index(info.bucket_size, "Candidate bucket size overflow"),
                                info.bucket_id, CandidateSourceTohpe);
            out.stats.nonzero++;
            out.stats.accepted++;
            out.stats.accepted_tohpe++;
            auto [score, tohpe] = local_pool.push(std::move(candidate));
            (void)tohpe;
            out.svs.observe(detail::checked_int_from_index(red, "SeenValues reduction overflow"), dim_int, score);
            out.stats.max_pool_score = std::max(out.stats.max_pool_score, score);
            out.stats.max_reduction =
                std::max(out.stats.max_reduction, detail::checked_int_from_index(red, "Stats reduction overflow"));
            out.stats.max_bucket = std::max(
                out.stats.max_bucket, detail::checked_int_from_index(info.bucket_size, "Stats bucket size overflow"));
            update_mean_value(out.stats.mean_score, score, out.stats.nonzero);
            update_mean_value(out.stats.mean_reduction, red, out.stats.nonzero);
        }
    });
    out.pools.tohpe_pool.merge_from(local_pool);
}

struct SourceTraversalState {
    std::size_t limit          = 0;
    std::size_t min_buckets    = 0;
    std::size_t max_buckets    = 0;
    std::size_t reserve_target = 0;
    bool        has_max        = false;
    bool        enabled        = false;

    bool active(std::size_t bucket_index, std::size_t pool_size) const {
        if (!enabled || bucket_index >= limit)
            return false;
        if (bucket_index < min_buckets)
            return true;
        const bool reserve_met = pool_size >= reserve_target;
        const bool cap_reached = has_max && bucket_index >= max_buckets;
        return !(reserve_met || (cap_reached && pool_size != 0));
    }
};

template <class Search>
SourceTraversalState make_source_traversal_state(const Search& search, std::size_t bucket_count) {
    const bool enabled = search.pool.keep > 0;
    const auto limit =
        search.buckets.limit_bucket < 0
            ? bucket_count
            : std::min<std::size_t>(bucket_count, static_cast<std::size_t>(search.buckets.limit_bucket));
    return {
        .limit = enabled ? limit : 0,
        .min_buckets = std::min<std::size_t>(limit, static_cast<std::size_t>(search.buckets.min_buckets)),
        .max_buckets = search.buckets.max_buckets <= 0
                           ? limit
                           : std::min<std::size_t>(limit, static_cast<std::size_t>(search.buckets.max_buckets)),
        .reserve_target = static_cast<std::size_t>(
            std::min(search.pool.keep, std::max(search.pool.reserve, static_cast<Int>(1)))),
        .has_max = search.buckets.max_buckets > 0,
        .enabled = enabled,
    };
}

void generate_bucket_candidates(PolicyIterationContext& ctx, const NormalizedPolicyConfig& config,
                                GenerationOutput& out) {
    const ToddIndex& index        = ctx.data->index();
    const auto       bucket_count = static_cast<std::size_t>(index.buckets_num());
    const auto prefix_state = make_source_traversal_state(config.tohpeprefix, bucket_count);
    const auto todd_state   = make_source_traversal_state(config.todd, bucket_count);
    const auto traversal_limit = std::max(prefix_state.limit, todd_state.limit);
    if (traversal_limit == 0)
        return;

    auto& buckets = index.top_sum_bucket_id_sizes_scratch(traversal_limit);
    auto  local_prefix_pool = TopKPool<ExplorationScore>(config.tohpeprefix.pool.keep, config.escore);
    auto  local_todd_pool  = TopKPool<ExplorationScore>(config.todd.pool.keep, config.escore);
    auto  local_svs        = SeenValues{};
    Stats stats;
    local_prefix_pool.reserve(config.tohpeprefix.pool.keep);
    local_todd_pool.reserve(config.todd.pool.keep);

    std::vector<SumEntry> bucket_entries;
    bucket_entries.reserve(static_cast<std::size_t>(index.max_bucket()));
    auto& generator = ctx.get_full_gen();

    for (std::size_t i = 0; i < buckets.size(); ++i) {
        const bool need_prefix = prefix_state.active(i, local_prefix_pool.size());
        const bool need_todd   = todd_state.active(i, local_todd_pool.size());
        if (!need_prefix && !need_todd)
            break;

        const auto [bucket_id, bucket_size] = buckets[i];
        stats.total += 1;
        stats.z_researched += 1;
        if (!index.materialize_bucket(bucket_id, bucket_entries) || bucket_entries.empty())
            continue;

        const SumEntry* ptr       = bucket_entries.data();
        const index_t   len       = static_cast<index_t>(bucket_entries.size());
        const index_t k           = ptr[0].a;
        const bool    is_single   = !ptr[0].is_pair();
        const index_t l           = is_single ? k_single_sentinel<index_t>() : static_cast<index_t>(ptr[0].b);
        const Row     key         = index.key_of(bucket_id);
        const auto request = need_todd ? (need_prefix ? SolutionBasisRequest::Both : SolutionBasisRequest::ToddOnly)
                                       : SolutionBasisRequest::TohpeOnly;
        auto ns = generator.make(key.cview(), ptr, len, request);

        const index_t full_dim   = ns.basis().rows();
        const index_t prefix_dim = ns.tohpe_prefix_size();
        const auto    n          = stats.total;
        update_mean_value(stats.mean_basis, full_dim, n);
        update_mean_value(stats.mean_mr, bucket_size, n);
        stats.max_basis =
            std::max(stats.max_basis, detail::checked_int_from_index(full_dim, "Stats basis dimension overflow"));
        stats.max_bucket =
            std::max(stats.max_bucket, detail::checked_int_from_index(bucket_size, "Stats bucket size overflow"));
        if (full_dim == 0)
            continue;

        PyRNG local_rng(mixed_seed(ctx.base_seed, k, is_single ? 0 : l, i, CandidateSourceTodd));
        auto  bucket_prefix_pool = TopKPool<ExplorationScore>(config.tohpeprefix.actions_per_bucket, config.escore);
        auto  bucket_todd_pool  = TopKPool<ExplorationScore>(config.todd.actions_per_bucket, config.escore);
        const Int cand_l =
            is_single ? k_single_sentinel<Int>() : detail::checked_int_from_index(l, "Candidate l overflow");

        auto sample_into = [&](TopKPool<ExplorationScore>& target, CandidateSource source, Int basis_dim,
                               RowCView coefs) {
            stats.evaluated++;
            auto       vec = ns.linear_combination(coefs);
            const auto red = ns.rank_divergence(vec);
            if (red <= 0) {
                stats.rejected++;
                return;
            }
            Candidate candidate(0, red, detail::checked_int_from_index(k, "Candidate k overflow"), cand_l,
                                std::move(vec), basis_dim,
                                detail::checked_int_from_index(bucket_size, "Candidate bucket size overflow"),
                                detail::checked_int_from_index(key.count(), "Candidate z weight overflow"),
                                detail::checked_int_from_index(key.size(), "Candidate z size overflow"), source);
            stats.accepted++;
            stats.nonzero++;
            if (source == CandidateSourceTohpePrefix)
                stats.accepted_tohpeprefix++;
            else
                stats.accepted_todd++;
            auto [score, tohpe] = target.push(std::move(candidate));
            (void)tohpe;
            local_svs.observe(red, basis_dim, score);
            stats.max_pool_score = std::max(stats.max_pool_score, score);
            stats.max_reduction  = std::max(stats.max_reduction, red);
            update_mean_value(stats.mean_score, score, stats.nonzero);
            update_mean_value(stats.mean_reduction, red, stats.nonzero);
        };

        if (need_prefix && need_todd) {
            local_rng.for_each_capped_bitvector_regions(
                full_dim, prefix_dim, config.tohpeprefix.sampling.vector_samples,
                config.tohpeprefix.sampling.sparse_max_weight, config.todd.sampling.vector_samples,
                config.todd.sampling.sparse_max_weight,
                [&](RowCView coefs) {
                    sample_into(bucket_prefix_pool, CandidateSourceTohpePrefix,
                                detail::checked_int_from_index(prefix_dim, "TOHPE basis dimension overflow"), coefs);
                },
                [&](RowCView coefs) {
                    sample_into(bucket_todd_pool, CandidateSourceTodd,
                                detail::checked_int_from_index(full_dim, "Todd basis dimension overflow"), coefs);
                });
        } else if (need_prefix) {
            local_rng.for_each_capped_bitvector(
                prefix_dim, config.tohpeprefix.sampling.vector_samples, config.tohpeprefix.sampling.sparse_max_weight,
                [&](RowCView coefs) {
                    sample_into(bucket_prefix_pool, CandidateSourceTohpePrefix,
                                detail::checked_int_from_index(prefix_dim, "TOHPE basis dimension overflow"), coefs);
                });
        } else {
            local_rng.for_each_capped_bitvector(
                full_dim, config.todd.sampling.vector_samples, config.todd.sampling.sparse_max_weight,
                [&](RowCView coefs) {
                    sample_into(bucket_todd_pool, CandidateSourceTodd,
                                detail::checked_int_from_index(full_dim, "Todd basis dimension overflow"), coefs);
                });
        }
        local_prefix_pool.merge_from(bucket_prefix_pool);
        local_todd_pool.merge_from(bucket_todd_pool);
    }

    out.pools.tohpeprefix_pool.merge_from(local_prefix_pool);
    out.pools.todd_pool.merge_from(local_todd_pool);
    out.svs.merge_from(local_svs);

    if (stats.total > 0) {
        const auto global_total = out.stats.total;
        const auto local_total  = stats.total;
        out.stats.mean_basis =
            (out.stats.mean_basis * global_total + stats.mean_basis * local_total) / (global_total + local_total);
        out.stats.mean_mr = (out.stats.mean_mr * global_total + stats.mean_mr * local_total) /
                            (global_total + local_total);
    }
    if (stats.nonzero > 0) {
        const auto global_nonzero = out.stats.nonzero;
        const auto local_nonzero  = stats.nonzero;
        out.stats.mean_score = (out.stats.mean_score * global_nonzero + stats.mean_score * local_nonzero) /
                               (global_nonzero + local_nonzero);
        out.stats.mean_reduction =
            (out.stats.mean_reduction * global_nonzero + stats.mean_reduction * local_nonzero) /
            (global_nonzero + local_nonzero);
    }
    out.stats.rejected += stats.rejected;
    out.stats.accepted += stats.accepted;
    out.stats.accepted_tohpe += stats.accepted_tohpe;
    out.stats.accepted_tohpeprefix += stats.accepted_tohpeprefix;
    out.stats.accepted_todd += stats.accepted_todd;
    out.stats.evaluated += stats.evaluated;
    out.stats.total += stats.total;
    out.stats.z_researched += stats.z_researched;
    out.stats.nonzero += stats.nonzero;
    out.stats.max_basis = std::max(out.stats.max_basis, stats.max_basis);
    out.stats.max_pool_score = std::max(out.stats.max_pool_score, stats.max_pool_score);
    out.stats.max_reduction = std::max(out.stats.max_reduction, stats.max_reduction);
    out.stats.max_bucket = std::max(out.stats.max_bucket, stats.max_bucket);
}

FinalizationPool merge_and_finalize_candidates(PolicyIterationContext& ctx, const NormalizedPolicyConfig& config,
                                               GenerationOutput& out) {
    auto merged_candidates =
        merge_source_candidates(out.pools.tohpe_pool.release_unsorted(), out.pools.tohpeprefix_pool.release_unsorted(),
                                out.pools.todd_pool.release_unsorted(), config.pool.final_size,
                                static_cast<std::size_t>(config.tohpe.pool.reserve),
                                static_cast<std::size_t>(config.tohpeprefix.pool.reserve),
                                static_cast<std::size_t>(config.todd.pool.reserve));
    const Int pool_size =
        detail::checked_int_from_index(merged_candidates.size(), "Candidate pool size overflow");
    Int       pool_tohpe_size = 0;
    Int       pool_tohpeprefix_size = 0;
    Int       pool_todd_size  = 0;
    for (const auto& cand : merged_candidates) {
        if (cand.source == CandidateSourceTohpe)
            ++pool_tohpe_size;
        else if (cand.source == CandidateSourceTohpePrefix)
            ++pool_tohpeprefix_size;
        else if (cand.source == CandidateSourceTodd)
            ++pool_todd_size;
    }
    for (auto& cand : merged_candidates) {
        cand.pool_size       = pool_size;
        cand.pool_tohpe_size = pool_tohpe_size;
        cand.pool_tohpeprefix_size = pool_tohpeprefix_size;
        cand.pool_todd_size  = pool_todd_size;
    }
    const bool need_tohpe_dim = config.fscore.needs_tohpe_dim();
    if (!merged_candidates.empty()) {
        std::vector<std::uint8_t> scratch_killed;
        if (need_tohpe_dim) {
            auto get_tohpe_gen = [&]() -> TohpeGenerator& { return ctx.get_tohpe_gen(); };
            auto get_full_gen = [&]() -> FullToddGenerator& { return ctx.get_full_gen(); };
            for (auto& cand : merged_candidates) {
                auto ns        = make_candidate_nullspace(cand, get_tohpe_gen, get_full_gen);
                cand.tohpe_dim = detail::checked_int_from_index(get_tohpe_basis(ns.apply(cand.vec, scratch_killed)).rows(),
                                                                "Candidate TOHPE dimension overflow");
            }
        }
    }

    auto final_pool = FinalizationPool(config.pool.final_size, config.fscore);
    if (!merged_candidates.empty()) {
        index_t n = 0;
        for (auto& cand : merged_candidates) {
            auto [score, tohpe] = final_pool.push(std::move(cand));

            ++n;
            out.stats.max_final_tohpe_dim = std::max(out.stats.max_final_tohpe_dim, tohpe);
            out.stats.max_final_score     = std::max(out.stats.max_final_score, score);
            update_mean_value(out.stats.mean_final_tohpe_dim, tohpe, n);
            update_mean_value(out.stats.mean_final_score, score, n);
        }
    }
    return final_pool;
}

Result select_and_apply_candidates(PolicyIterationContext& ctx, const NormalizedPolicyConfig& config,
                                   FinalizationPool& final_pool, const SeenValues& svs, const Stats& stats,
                                   index_t seed) {
    PyRNG pick_rng(seed);
    bool  is_best = (config.selection.mode == "greedy" || config.selection.mode == "best");

    auto get_tohpe_gen = [&]() -> TohpeGenerator& { return ctx.get_tohpe_gen(); };
    auto get_full_gen = [&]() -> FullToddGenerator& { return ctx.get_full_gen(); };
    return build_result(final_pool, is_best, config.selection.count, config.selection.temperature, pick_rng,
                        svs, get_tohpe_gen, get_full_gen, stats, ctx.base_seed);
}
auto policy_iteration_impl(const std::shared_ptr<MatrixWithData>& data, PolicyConfig config, index_t seed,
                           index_t add_seed) -> Result {
    if (!data)
        throw std::runtime_error("data is null");

    auto config_norm = normalize_policy_config(std::move(config));
    auto ctx         = make_policy_context(data, config_norm, seed, add_seed);
    auto generated   = GenerationOutput(config_norm);

    generate_tohpe_candidates(ctx, config_norm, generated);
    generate_bucket_candidates(ctx, config_norm, generated);
    generated.svs.finalize();

    auto final_pool = merge_and_finalize_candidates(ctx, config_norm, generated);
    auto result     = select_and_apply_candidates(ctx, config_norm, final_pool, generated.svs, generated.stats, seed);
    ctx.full_gen.reset();
    return result;
}

bool Candidate::at_least_single() const { return (k != k_single_sentinel<decltype(l)>()); }
bool Candidate::is_tohpe() const {
    return (k == k_single_sentinel<decltype(k)>()) && (l == k_single_sentinel<decltype(l)>());
}

void SeenValues::reserve(std::size_t n) { pool_scores_sorted.reserve(n); }

void SeenValues::observe(Int red, Int dim, float score) {
    red     = std::max<Int>(red, -1);
    dim     = std::max<Int>(dim, 0);
    max_red = std::max(max_red, red);
    max_dim = std::max(max_dim, dim);

    if (max_red > std::numeric_limits<Int>::max() - 2)
        throw std::overflow_error("SeenValues reduction frequency overflow");
    if (max_dim == std::numeric_limits<Int>::max())
        throw std::overflow_error("SeenValues dimension frequency overflow");

    if (red_freq.size() <= static_cast<std::size_t>(max_red) + 1)
        red_freq.resize(static_cast<std::size_t>(max_red) + 2, 0);
    if (dim_freq.size() <= static_cast<std::size_t>(max_dim))
        dim_freq.resize(static_cast<std::size_t>(max_dim) + 1, 0);
    ++red_freq[static_cast<std::size_t>(red + 1)];
    ++dim_freq[static_cast<std::size_t>(dim)];

    pool_scores_sorted.push_back(score);
    finalized = false;
}

void SeenValues::finalize() {
    if (finalized)
        return;

    auto build_suffix = [](const auto& freq, auto& suf) {
        using T = typename std::remove_reference_t<decltype(suf)>::value_type;
        suf.assign(freq.size() + 1, T{});
        for (std::size_t i = freq.size(); i-- > 0;)
            suf[i] = suf[i + 1] + freq[i];
    };

    std::ranges::sort(pool_scores_sorted);
    build_suffix(red_freq, red_suf);
    build_suffix(dim_freq, dim_suf);

    finalized = true;
}

Int SeenValues::better_red(Int r) const {
    assert(finalized);
    const auto signed_idx = static_cast<std::int64_t>(r) + 1;
    if (signed_idx < 0)
        return red_suf.empty() ? 0 : detail::checked_int_from_index(red_suf[0], "SeenValues red count overflow");
    const auto idx = static_cast<std::size_t>(signed_idx);
    if (idx >= red_freq.size())
        return 0;
    return detail::checked_int_from_index(red_suf[idx], "SeenValues red count overflow"); // >= r
}

Int SeenValues::better_dim(Int d) const {
    assert(finalized);
    if (d < 0)
        return dim_suf.empty() ? 0 : detail::checked_int_from_index(dim_suf[0], "SeenValues dim count overflow");
    const auto idx = static_cast<std::size_t>(d);
    if (idx >= dim_freq.size())
        return 0;
    return detail::checked_int_from_index(dim_suf[idx], "SeenValues dim count overflow"); // >= d
}

Int SeenValues::better_score(float s) const {
    assert(finalized);
    auto it = std::lower_bound(pool_scores_sorted.begin(), pool_scores_sorted.end(), s);
    return detail::checked_int_from_index(static_cast<index_t>(pool_scores_sorted.end() - it),
                                          "SeenValues score count overflow"); // >= s
}

void SeenValues::merge_from(const SeenValues& other) {
    finalized            = false;
    max_red              = std::max(max_red, other.max_red);
    max_dim              = std::max(max_dim, other.max_dim);
    if (max_red > std::numeric_limits<Int>::max() - 2)
        throw std::overflow_error("SeenValues reduction frequency overflow");
    if (max_dim == std::numeric_limits<Int>::max())
        throw std::overflow_error("SeenValues dimension frequency overflow");
    auto add_vector_data = [](auto& init, const auto& other) {
        std::transform(other.begin(), other.end(), init.begin(), init.begin(), std::plus<>{});
    };
    dim_freq.resize(static_cast<std::size_t>(max_dim) + 1, 0);
    red_freq.resize(static_cast<std::size_t>(max_red) + 2, 0);
    add_vector_data(dim_freq, other.dim_freq);
    add_vector_data(red_freq, other.red_freq);

    pool_scores_sorted.insert(pool_scores_sorted.end(), other.pool_scores_sorted.begin(),
                              other.pool_scores_sorted.end());

    red_suf.clear();
    dim_suf.clear();
}

} // namespace todd
