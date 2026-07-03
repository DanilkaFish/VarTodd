#include "todd_generator.hpp"

#include "nullspace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random.hpp>
#include <stdexcept>

namespace todd {

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
        std::ranges::sort(heap_, worse_node_);
        out.reserve(heap_.size());
        for (auto& n : heap_)
            out.push_back(std::move(n.cand));
        heap_.clear();
        return out;
    }

    void set_keyfn(Score k) {
        keyfn_ = std::move(k);
        for (auto& n : heap_) {
            const auto [key, tohpe] = keyfn_(n.cand);
            (void)tohpe;
            n.key = key;
        }
        std::make_heap(heap_.begin(), heap_.end(), worse_node_);
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
        if (a_key != b_key)
            return a_key > b_key;
        if (a.reduction != b.reduction)
            return a.reduction > b.reduction;
        if (a.k != b.k)
            return a.k < b.k;
        if (a.l != b.l)
            return a.l < b.l;
        return a.vec < b.vec;
    }
    static bool worse_node_(Node const& a, Node const& b) noexcept {
        return worse_candidate_(a.key, a.cand, b.key, b.cand);
    }
    static bool worse_than(Node const& a, float b_key, Candidate const& b) noexcept {
        return worse_candidate_(a.key, a.cand, b_key, b);
    }
};

template <class FullGenProvider>
NullSpace make_candidate_nullspace(Candidate const& c, TohpeGenerator& tohpe_gen, FullGenProvider& get_full_gen) {
    if (c.is_tohpe()) {
        if (c.bucket_id != Candidate::no_bucket_id)
            return tohpe_gen.make(c.z, c.bucket_id);
        return tohpe_gen.make(c.z);
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

template <class Mean, class Value> void update_mean_value(Mean& mean, Value value, Int n) {
    mean += (value - mean) / std::max((Int)1, n);
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

struct NormalizedTohpeSearch {
    NormalizedSamplingBudget sampling{};
    NormalizedSourcePool     pool{};
    Int                      z_choices = 1;
};

struct NormalizedZBucketSearch {
    Int   min_buckets     = 0;
    Int   max_buckets     = 0;
    float temperature     = 0.0f;
    float random_fraction = 0.0f;
    Int   limit_bucket    = -1;
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
    NormalizedTohpeSearch tohpe{};
    NormalizedToddSearch  todd{};
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
        .sparse_max_weight = std::max(sampling.sparse_max_weight, (Int)2),
    };
}

NormalizedSourcePool normalize_source_pool(SourcePool pool) {
    return {
        .keep    = std::max(pool.keep, (Int)0),
        .reserve = std::max(pool.reserve, (Int)0),
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
                .z_choices = std::max(config.tohpe.z_choices, (Int)1),
            },
        .todd =
            {
                .sampling           = normalize_sampling_budget(config.todd.sampling),
                .pool               = normalize_source_pool(config.todd.pool),
                .actions_per_bucket = std::max(config.todd.actions_per_bucket, (Int)1),
                .buckets =
                    {
                        .min_buckets     = std::max(config.todd.buckets.min_buckets, (Int)0),
                        .max_buckets     = std::max(config.todd.buckets.max_buckets, (Int)0),
                        .temperature     = std::max(config.todd.buckets.temperature, 0.0f),
                        .random_fraction = std::clamp(config.todd.buckets.random_fraction, 0.0f, 1.0f),
                        .limit_bucket    = config.todd.buckets.limit_bucket < 0
                                               ? (Int)-1
                                               : std::max(config.todd.buckets.limit_bucket, (Int)0),
                    },
            },
    };
    out.selection.count = std::max(out.selection.count, (Int)1);
    out.pool.final_size = std::max({out.selection.count, out.pool.final_size, out.tohpe.pool.reserve + out.todd.pool.reserve});
    return out;
}

struct PolicyIterationContext {
    std::shared_ptr<MatrixWithData>  data;
    std::uint64_t                    base_seed{};
    index_t                          tohpe_dim{};
    const Matrix&                    tohpe_basis;
    TohpeGenerator                   tohpe_gen;
    std::optional<FullToddGenerator> full_gen;

    PolicyIterationContext(std::shared_ptr<MatrixWithData> d, std::uint64_t seed)
        : data(std::move(d)), base_seed(seed), tohpe_dim(data->tohpe_basis().rows()),
          tohpe_basis(data->tohpe_basis()), tohpe_gen(data) {}

    FullToddGenerator& get_full_gen() {
        if (!full_gen)
            full_gen.emplace(data);
        return *full_gen;
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
    TopKPool<ExplorationScore> todd_pool;

    explicit CandidatePools(const NormalizedPolicyConfig& config)
        : tohpe_pool(config.tohpe.pool.keep, config.escore), todd_pool(config.todd.pool.keep, config.escore) {}
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

    if (temperature == 0.0f && random_fraction == 0.0f) {
        std::vector<std::size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        auto better = [&](std::size_t a, std::size_t b) { return buckets[a].second > buckets[b].second; };
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
                                 return buckets[a].second > buckets[b].second;
                             });
            idx.resize(top_count);
        }
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            if (score[a] != score[b])
                return score[a] > score[b];
            return buckets[a].second > buckets[b].second;
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
        std::shuffle(remaining.begin(), remaining.end(), rng.get_engine());
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
    if (a.pool_score != b.pool_score)
        return a.pool_score > b.pool_score;
    if (a.reduction != b.reduction)
        return a.reduction > b.reduction;
    if (a.k != b.k)
        return a.k < b.k;
    if (a.l != b.l)
        return a.l < b.l;
    return a.vec < b.vec;
}

void take_source_quota(std::vector<Candidate>& out, std::vector<Candidate>& src, std::size_t desired,
                       std::size_t hard_limit) {
    if (desired == 0 || hard_limit == 0 || out.size() >= hard_limit || src.empty())
        return;
    std::sort(src.begin(), src.end(), pool_preferred);
    const std::size_t n = std::min({desired, hard_limit - out.size(), src.size()});
    for (std::size_t i = 0; i < n; ++i)
        out.push_back(std::move(src[i]));
    src.erase(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(n));
}

std::vector<Candidate> merge_source_candidates(std::vector<Candidate> tohpe, std::vector<Candidate> todd,
                                               std::size_t final_size, std::size_t tohpe_reserve,
                                               std::size_t todd_reserve) {
    std::vector<Candidate> out;
    out.reserve(final_size);
    take_source_quota(out, tohpe, tohpe_reserve, final_size);
    take_source_quota(out, todd, todd_reserve, final_size);

    std::vector<Candidate> rest;
    rest.reserve(tohpe.size() + todd.size());
    for (auto& c : tohpe)
        rest.push_back(std::move(c));
    for (auto& c : todd)
        rest.push_back(std::move(c));

    std::sort(rest.begin(), rest.end(), pool_preferred);
    for (auto& c : rest) {
        if (out.size() >= final_size)
            break;
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
        if ((index_t)i >= basis.rows())
            break;
        out ^= basis[(index_t)i];
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
        if (a.final_score != b.final_score)
            return a.final_score > b.final_score;
        if (a.reduction != b.reduction)
            return a.reduction > b.reduction;
        if (a.k != b.k)
            return a.k < b.k;
        if (a.l != b.l)
            return a.l < b.l;
        return a.vec < b.vec;
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
        if (a.final_score != b.final_score)
            return a.final_score > b.final_score;
        if (a.reduction != b.reduction)
            return a.reduction > b.reduction;
        if (a.k != b.k)
            return a.k < b.k;
        if (a.l != b.l)
            return a.l < b.l;
        return a.vec < b.vec;
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
                         [&](std::size_t a, std::size_t b) { return key[a] > key[b]; });
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

template <class FullGenProvider>
Result build_result(FinalizationPool& pool, bool is_best, std::size_t action_count, float temperature,
                    PyRNG& pick_rng, SeenValues const& svs, TohpeGenerator& tohpe_gen, FullGenProvider& get_full_gen,
                    Stats const& global_stats, std::uint64_t base_seed) {
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

        auto ns    = make_candidate_nullspace(c, tohpe_gen, get_full_gen);
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
    out.stats.max_basis = dim;

    PyRNG local_rng(mixed_seed(ctx.base_seed, 0, 0, 0, CandidateSourceTohpe));

    std::vector<TohpeZInfo> scratch_best_z;
    local_rng.for_each_capped_bitvector(dim, config.tohpe.sampling.vector_samples,
                                        config.tohpe.sampling.sparse_max_weight, [&](RowCView coefs) {
        assert(coefs.count() != 0);
        out.stats.evaluated++;
        auto vec = linear_combination_from_basis(ctx.tohpe_basis, coefs);
        ctx.tohpe_gen.best_z_n_details_into(vec, config.tohpe.z_choices, scratch_best_z);
        for (std::size_t zi = 0; zi < scratch_best_z.size(); ++zi) {
            auto&      info = scratch_best_z[zi];
            const auto red  = info.reduction;
            if (red <= 0) {
                out.stats.rejected++;
                continue;
            }
            Row       cand_vec = (zi + 1 == scratch_best_z.size()) ? std::move(vec) : Row(vec);
            Candidate c(0, (Int)red, k_single_sentinel<Int>(), k_single_sentinel<Int>(), std::move(cand_vec),
                        std::move(info.z), (Int)dim, (Int)info.bucket_size, info.bucket_id, CandidateSourceTohpe);
            out.stats.nonzero++;
            out.stats.accepted_tohpe++;
            auto [score, tohpe] = local_pool.push(std::move(c));
            (void)tohpe;
            out.svs.observe(red, dim, score);

            out.stats.max_pool_score = std::max(out.stats.max_pool_score, score);
            out.stats.max_reduction  = std::max(out.stats.max_reduction, (Int)red);
            out.stats.max_bucket     = std::max(out.stats.max_bucket, (Int)info.bucket_size);
            update_mean_value(out.stats.mean_score, score, out.stats.nonzero);
            update_mean_value(out.stats.mean_reduction, red, out.stats.nonzero);
        }
    });
    out.pools.tohpe_pool.merge_from(local_pool);
}

void generate_todd_candidates(PolicyIterationContext& ctx, const NormalizedPolicyConfig& config,
                              GenerationOutput& out) {
    if (config.todd.pool.keep <= 0)
        return;

    const ToddIndex& index   = ctx.data->index();
    auto&            buckets = index.sum_bucket_id_sizes_scratch();
    const auto bucket_search_limit =
        config.todd.buckets.limit_bucket < 0
            ? buckets.size()
            : std::min<std::size_t>(buckets.size(), static_cast<std::size_t>(config.todd.buckets.limit_bucket));
    const auto reserve_search_cap =
        config.todd.buckets.max_buckets <= 0
            ? bucket_search_limit
            : std::min<std::size_t>(bucket_search_limit, static_cast<std::size_t>(config.todd.buckets.max_buckets));
    const auto limit_seed_component = config.todd.buckets.limit_bucket < 0 ? (Int)0 : config.todd.buckets.limit_bucket;
    PyRNG bucket_rng(mixed_seed(ctx.base_seed, buckets.size(), config.todd.buckets.max_buckets,
                                limit_seed_component, CandidateSourceTodd));
    diversify_buckets(buckets, bucket_search_limit, config.todd.buckets.temperature,
                      config.todd.buckets.random_fraction, bucket_rng);

    const auto min_buckets_to_search =
        std::min<std::size_t>(bucket_search_limit, static_cast<std::size_t>(config.todd.buckets.min_buckets));

    Stats stats;

    const auto reserve_target =
        static_cast<std::size_t>(std::min(config.todd.pool.keep, std::max(config.todd.pool.reserve, (Int)1)));
    auto  local_pool = TopKPool<ExplorationScore>(config.todd.pool.keep, config.escore);
    auto& local_gen  = ctx.get_full_gen();
    auto  local_svs  = SeenValues{};
    local_pool.reserve(config.todd.pool.keep);

    const SumEntry* ptr = nullptr;
    auto            len = index_t{};

    for (std::size_t i = 0; i < bucket_search_limit; ++i) {
        if (i >= min_buckets_to_search) {
            const bool reserve_met = local_pool.size() >= reserve_target;
            const bool cap_reached = config.todd.buckets.max_buckets > 0 && i >= reserve_search_cap;
            if (reserve_met || (cap_reached && !local_pool.empty())) {
                break;
            }
        }

        auto& [bucket_id, bucket_size] = buckets[i];
        stats.total += 1;
        stats.z_researched += 1;
        if (!index.sum_bucket(bucket_id, ptr, len) || len == 0)
            continue;

        const index_t k         = ptr[0].a;
        const bool    is_single = !ptr[0].is_pair();
        const index_t l         = is_single ? k_single_sentinel<index_t>() : (index_t)ptr[0].b;
        const RowCView key      = index.key_of(bucket_id);
        auto          ns        = local_gen.make(key, ptr, static_cast<int>(len));

        const auto dim = ns.basis().rows();
        const auto n   = stats.total;
        update_mean_value(stats.mean_basis, dim, n);
        update_mean_value(stats.mean_mr, bucket_size, n);
        stats.max_basis = std::max(stats.max_basis, (Int)dim);

        if (dim == 0)
            continue;

        PyRNG local_rng(mixed_seed(ctx.base_seed, k, is_single ? 0 : l, i, CandidateSourceTodd));

        stats.max_bucket = std::max(stats.max_bucket, (Int)bucket_size);

        auto local_local_pool = TopKPool<ExplorationScore>(config.todd.actions_per_bucket, config.escore);
        local_rng.for_each_capped_bitvector(dim, config.todd.sampling.vector_samples,
                                            config.todd.sampling.sparse_max_weight,
                                            [&](RowCView coefs) {
                                                stats.evaluated++;
                                                auto       vec = ns.linear_combination(coefs);
                                                const auto red = ns.rank_divergence(vec);

                                                if (red <= 0) {
                                                    stats.rejected++;
                                                    return;
                                                }
                                                const Int cand_l =
                                                    is_single ? k_single_sentinel<Int>() : static_cast<Int>(l);
                                                Candidate c(0, (Int)red, static_cast<Int>(k), cand_l, std::move(vec),
                                                            (Int)dim, (Int)bucket_size, static_cast<Int>(key.count()),
                                                            static_cast<Int>(key.size()), CandidateSourceTodd);
                                                stats.accepted++;
                                                stats.nonzero++;
                                                auto [score, tohpe] = local_local_pool.push(std::move(c));
                                                (void)tohpe;
                                                local_svs.observe(red, dim, score);

                                                stats.max_pool_score = std::max(stats.max_pool_score, score);
                                                stats.max_reduction  = std::max(stats.max_reduction, (Int)red);
                                                stats.max_bucket     = std::max(stats.max_bucket, (Int)bucket_size);
                                                update_mean_value(stats.mean_score, score, stats.nonzero);
                                                update_mean_value(stats.mean_reduction, red, stats.nonzero);
                                            });
        local_pool.merge_from(local_local_pool);
    }

    out.pools.todd_pool.merge_from(local_pool);
    out.svs.merge_from(local_svs);

    if (stats.total > 0) {
        const auto gnz       = out.stats.total;
        const auto lnz       = stats.total;
        out.stats.mean_basis = (out.stats.mean_basis * gnz + stats.mean_basis * lnz) / (gnz + lnz);
        out.stats.mean_mr    = (out.stats.mean_mr * gnz + stats.mean_mr * lnz) / (gnz + lnz);
    }
    if (stats.nonzero > 0) {
        const auto gnz       = out.stats.nonzero;
        const auto lnz       = stats.nonzero;
        out.stats.mean_score = (out.stats.mean_score * gnz + stats.mean_score * lnz) / (gnz + lnz);
        out.stats.mean_reduction =
            (out.stats.mean_reduction * gnz + stats.mean_reduction * lnz) / (gnz + lnz);
    }
    out.stats.rejected += stats.rejected;
    out.stats.accepted += stats.accepted;
    out.stats.evaluated += stats.evaluated;
    out.stats.total += stats.total;
    out.stats.z_researched += stats.z_researched;
    out.stats.nonzero += stats.nonzero;
    out.stats.max_basis      = std::max(stats.max_basis, out.stats.max_basis);
    out.stats.max_pool_score = std::max(stats.max_pool_score, out.stats.max_pool_score);
    out.stats.max_reduction  = std::max(stats.max_reduction, out.stats.max_reduction);
    out.stats.max_bucket     = std::max(stats.max_bucket, out.stats.max_bucket);
}

FinalizationPool merge_and_finalize_candidates(PolicyIterationContext& ctx, const NormalizedPolicyConfig& config,
                                               GenerationOutput& out) {
    auto merged_candidates =
        merge_source_candidates(out.pools.tohpe_pool.release_unsorted(), out.pools.todd_pool.release_unsorted(),
                                config.pool.final_size, static_cast<std::size_t>(config.tohpe.pool.reserve),
                                static_cast<std::size_t>(config.todd.pool.reserve));
    const Int pool_size = static_cast<Int>(merged_candidates.size());
    Int       pool_tohpe_size = 0;
    Int       pool_todd_size  = 0;
    for (const auto& cand : merged_candidates) {
        if (cand.source == CandidateSourceTohpe)
            ++pool_tohpe_size;
        else if (cand.source == CandidateSourceTodd)
            ++pool_todd_size;
    }
    for (auto& cand : merged_candidates) {
        cand.pool_size       = pool_size;
        cand.pool_tohpe_size = pool_tohpe_size;
        cand.pool_todd_size  = pool_todd_size;
    }
    const bool need_tohpe_dim = config.fscore.needs_tohpe_dim();
    if (!merged_candidates.empty()) {
        std::vector<std::uint8_t> scratch_killed;
        if (need_tohpe_dim) {
            auto get_full_gen = [&]() -> FullToddGenerator& { return ctx.get_full_gen(); };
            for (auto& cand : merged_candidates) {
                auto ns        = make_candidate_nullspace(cand, ctx.tohpe_gen, get_full_gen);
                cand.tohpe_dim = static_cast<Int>(get_tohpe_basis(ns.apply(cand.vec, scratch_killed)).rows());
            }
        }
    }

    auto final_pool = FinalizationPool(config.pool.final_size, config.fscore);
    if (!merged_candidates.empty()) {
        int n = 0;
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

    auto get_full_gen = [&]() -> FullToddGenerator& { return ctx.get_full_gen(); };
    return build_result(final_pool, is_best, config.selection.count, config.selection.temperature, pick_rng,
                        svs, ctx.tohpe_gen, get_full_gen, stats, ctx.base_seed);
}
auto policy_iteration_impl(const std::shared_ptr<MatrixWithData>& data, PolicyConfig config, index_t seed,
                           index_t add_seed) -> Result {
    if (!data)
        throw std::runtime_error("data is null");

    auto config_norm = normalize_policy_config(std::move(config));
    auto ctx         = make_policy_context(data, config_norm, seed, add_seed);
    auto generated   = GenerationOutput(config_norm);

    generate_tohpe_candidates(ctx, config_norm, generated);
    generate_todd_candidates(ctx, config_norm, generated);
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
    max_red = std::max(max_red, red);
    max_dim = std::max(max_dim, dim);

    if (red_freq.size() <= max_red + 1)
        red_freq.resize(max_red + 2, 0);
    if (dim_freq.size() <= max_dim)
        dim_freq.resize(max_dim + 1, 0);
    ++red_freq[red + 1];
    ++dim_freq[dim];

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
    if (r + 1 < 0)
        return red_suf.empty() ? 0 : red_suf[0];
    if (r + 1 >= (Int)red_freq.size())
        return 0;
    return red_suf[r + 1]; // >= r
}

Int SeenValues::better_dim(Int d) const {
    assert(finalized);
    if (d < 0)
        return dim_suf.empty() ? 0 : dim_suf[0];
    if (d >= (Int)dim_freq.size())
        return 0;
    return dim_suf[d]; // >= d
}

Int SeenValues::better_score(float s) const {
    assert(finalized);
    auto it = std::lower_bound(pool_scores_sorted.begin(), pool_scores_sorted.end(), s);
    return (Int)(pool_scores_sorted.end() - it); // >= s
}

void SeenValues::merge_from(const SeenValues& other) {
    finalized            = false;
    max_red              = std::max(max_red, other.max_red);
    max_dim              = std::max(max_dim, other.max_dim);
    auto add_vector_data = [](auto& init, const auto& other) {
        std::transform(other.begin(), other.end(), init.begin(), init.begin(), std::plus<>{});
    };
    dim_freq.resize(max_dim + 1, 0);
    red_freq.resize(max_red + 2, 0);
    add_vector_data(dim_freq, other.dim_freq);
    add_vector_data(red_freq, other.red_freq);

    pool_scores_sorted.insert(pool_scores_sorted.end(), other.pool_scores_sorted.begin(),
                              other.pool_scores_sorted.end());

    red_suf.clear();
    dim_suf.clear();
}

} // namespace todd
