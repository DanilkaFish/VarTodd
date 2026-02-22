#include "todd_generator.hpp"

#include "algorithms.hpp"
#include "nullspace.hpp"

#include <algorithm>
#include <numeric>
#include <random.hpp>
#include <ranges>
#ifdef __OPENMP
#include <omp.h>
#endif
#include <atomic>

namespace todd {

template <class Score> class TopKPool {
  public:
    explicit TopKPool(std::size_t limit, Score keyfn = Score{}) : limit_(limit), keyfn_(std::move(keyfn)) {}

    void reserve(std::size_t n) { heap_.reserve(n); }

    std::size_t size() const noexcept { return heap_.size(); }
    bool        empty() const noexcept { return heap_.empty(); }

    std::pair<float, float> push(Candidate&& c) {
        if (limit_ == 0)
            return {0, 0};

        const auto [key, tohpe] = keyfn_(c);

        if (heap_.size() < limit_) {
            heap_.push_back(Node{key, std::move(c)});
            std::push_heap(heap_.begin(), heap_.end(), worse_node_);
            return {key, tohpe};
        }

        if (worse_node_(heap_.front(), Node{key, c}))
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
    static bool       worse_node_(Node const& a, Node const& b) noexcept {
        if (a.key != b.key)
            return a.key > b.key;
        if (a.cand.reduction != b.cand.reduction)
            return a.cand.reduction > b.cand.reduction;
        if (a.cand.k != b.cand.k)
            return a.cand.k < b.cand.k;
        if (a.cand.l != b.cand.l)
            return a.cand.l < b.cand.l;
        return a.cand.vec < b.cand.vec;
    }
};

std::vector<Candidate> pick_best_view(TopKPool<FinalizationScore>& pool, std::size_t n) {
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

std::vector<Candidate> pick_softmax_view(TopKPool<FinalizationScore>& pool, std::size_t n, float temperature,
                                         PyRNG& rng) {
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

    std::nth_element(idx.begin(), idx.begin() + n, idx.end(),
                     [&](std::size_t a, std::size_t b) { return key[a] > key[b]; });
    idx.resize(n);

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

Result build_result(TopKPool<FinalizationScore>& pool, bool is_best, std::size_t num_candidates, float temperature,
                    PyRNG& pick_rng, SeenValues const& svs, std::shared_ptr<MatrixWithData> data,
                    Stats const& global_stats, std::uint64_t base_seed) {
    auto chosen_view =
        is_best ? pick_best_view(pool, num_candidates) : pick_softmax_view(pool, num_candidates, temperature, pick_rng);

    Result out;
    out.stats = global_stats;
    out.seed  = base_seed;

    out.chosen.reserve(chosen_view.size());
    out.states.reserve(chosen_view.size());

    FullToddGenerator gen(data);

    struct Acc {
        Result& out;
        void    add(Candidate const& c, decltype(out.states)::value_type state) {
            out.chosen.push_back(export_candidate(c));
            out.states.push_back(std::move(state));
        }
    } acc{out};

    for (Candidate& c : chosen_view) {
        c.num_better_dim        = svs.better_dim(c.basis_dim);
        c.num_better_red        = svs.better_red(c.reduction);
        c.num_better_pool_score = svs.better_score(c.pool_score);

        auto state = c.nsptr->apply(c.vec);

        acc.add(c, std::move(state));
    }

    return out;
}
auto policy_iteration_impl(const std::shared_ptr<MatrixWithData>& data, PolicyConfig config, index_t seed,
                           index_t add_seed) -> Result {

    if (!data)
        throw std::runtime_error("data is null");
    auto       update_mean        = [](auto& mean, auto value, Int n) { mean += (value - mean) / std::max((Int)1, n); };
    auto       selection          = config.selection;
    auto       num_samples        = config.num_samples;
    auto       num_candidates     = config.num_candidates;
    auto       top_pool           = config.top_pool;
    auto       temperature        = config.temperature;
    auto       non_improving_prob = config.non_improving_prob;
    auto       min_reduction      = config.min_reduction;
    auto       max_reduction      = config.max_reduction;
    auto       min_pool           = std::max(1, config.min_pool_size);
    auto       escore             = config.escore;
    auto       fscore             = config.fscore;

    auto       max_z_to_research_fraction = config.max_z_to_research_fraction;
    auto       max_z_to_research          = index_t(std::max(0,config.max_z_to_research));
    auto       max_tohpe                  = config.max_tohpe;
    auto       max_from_single_ns         = config.max_from_single_ns;
    auto       tohpe_sample               = config.tohpe_sample;
    auto       try_only_tohpe             = config.try_only_tohpe;
    
    const auto threads                    = config.threads;
#ifdef __OPENMP
    if (threads > 0)
        omp_set_num_threads(threads);
#endif
    num_samples                = std::max(num_samples, (Int)1);
    num_candidates             = std::max(num_candidates, (Int)1);
    top_pool                   = std::max(num_candidates, top_pool);
    max_z_to_research_fraction = max_z_to_research_fraction < 0 ? 0 : max_z_to_research_fraction;
    max_reduction              = max_reduction > 0 ? max_reduction : k_single_sentinel<decltype(max_reduction)>();
    min_reduction              = std::max(min_reduction, (Int)0);
    non_improving_prob         = std::min(std::max(0.0f, non_improving_prob), 1.0f);

    const auto     base_seed            = matrix_seed(data->P(), seed, add_seed);
    const auto     nrows                = data->P().rows();
    const auto     tohpe_dim            = data->tohpe_basis().rows();
    auto           bucket_normalization = data->index().max_bucket();
    auto           wvw_normalization    = data->P().rows();
    TohpeGenerator gen(data);
    auto           dim_normalization = gen.make(0).basis().rows() + 5;
    escore.bn                        = bucket_normalization;
    escore.wvwn                      = wvw_normalization;
    escore.dn                        = dim_normalization;
    fscore.bn                        = bucket_normalization;
    fscore.wvwn                      = wvw_normalization;
    fscore.dn                        = dim_normalization;
    auto svs                         = SeenValues{};
    auto pool                        = TopKPool<ExplorationScore>(top_pool, escore);
    auto accepted_tohpe              = 0;

    Stats global_stats;
    {
        auto local_beyond_pool = TopKPool<ExplorationScore>(max_tohpe, escore);
        auto local_pool        = TopKPool<ExplorationScore>(max_tohpe, escore);

        if (max_tohpe > 0) {
            auto          ns       = std::make_shared<NullSpace>(gen.make(0));
            const index_t dim      = ns->basis().rows();
            global_stats.max_basis = dim;

            PyRNG local_rng(base_seed);
            auto  beyond        = [&](const auto red) { return (red < min_reduction || red > max_reduction); };
            // TODO
            auto  accept_anyway = [&](auto red) {
                return false && red <= 0 && non_improving_prob && local_rng.rand_double(0.0, 1.0) < non_improving_prob;
            };

            auto coefs_list = local_rng.sample_small_unique_bitvectors(dim, num_samples);
            for (auto const& coefs : coefs_list) {
                auto vec = ns->linear_combination(coefs);
                for (auto [z, red] : gen.best_z_n(vec, tohpe_sample)) {
                    auto nsptr = std::make_shared<NullSpace>(gen.make(z));
                    if (beyond(red)) {
                        if (accept_anyway(red) || (red > 0)) {
                            global_stats.accepted_non_improving++;
                        } else {
                            global_stats.rejected++;
                            continue;
                        }
                    }
                    global_stats.nonzero++;
                    global_stats.accepted_tohpe++;
                    auto      bucket_size = data->index().get_size_from_z(z);
                    Candidate c(0, (Int)red, k_single_sentinel<Int>(), k_single_sentinel<Int>(), vec, (Int)dim,
                                (Int)bucket_size, nsptr);
                    auto [score, tohpe] =
                        (beyond(red)) ? local_beyond_pool.push(std::move(c)) : local_pool.push(std::move(c));
                    svs.observe(red, dim, score);

                    global_stats.max_pool_score = std::max(global_stats.max_pool_score, score);
                    global_stats.max_reduction  = std::max(global_stats.max_reduction, (Int)red);
                    global_stats.max_bucket     = std::max(global_stats.max_bucket, (Int)bucket_size);
                    update_mean(global_stats.mean_score, score, global_stats.nonzero);
                    update_mean(global_stats.mean_reduction, red, global_stats.nonzero);
                }
            }
            pool.merge_from(local_pool);
            pool.merge_from(local_beyond_pool);
        }

        if (!try_only_tohpe || pool.size() < min_pool) {

            auto cmp = [](auto const& a, auto const& b) {
                return std::tie(a.second, b.first) > std::tie(b.second, a.first);
            };

            auto buckets = data->index().sum_key_sizes();
            // auto above_min = [&](auto const& kv) { return (min_reduction > 1 && 2 * kv.second < min_reduction); };
            // std::erase_if(buckets, above_min);
            max_z_to_research = static_cast<int>(std::min(buckets.size(), max_z_to_research));
            if (buckets.size() > max_z_to_research) {
                std::ranges::nth_element(buckets, buckets.begin() + max_z_to_research, cmp);
                // buckets.resize(max_z_to_research);
            }

#ifdef __OPENMP
#pragma omp parallel
#endif
            {
                Stats stats;

                auto local_beyond_pool = TopKPool<ExplorationScore>(min_pool, escore);
                auto local_pool        = TopKPool<ExplorationScore>(top_pool, escore);
                auto local_gen         = FullToddGenerator(data);
                auto local_svs         = SeenValues{};
                local_pool.reserve(top_pool);
                // #ifdef __OPENMP
                std::atomic<bool> stop{false};
                // #endif
                const SumEntry* ptr = nullptr;
                auto            len = index_t{};

#ifdef __OPENMP
#pragma omp for schedule(dynamic)
#endif
                for (std::size_t i = 0; (i < buckets.size()); ++i) {
                    if (i >= max_z_to_research &&
                        (local_pool.size() >= min_pool || local_beyond_pool.size() >= min_pool)) {
                        stop.store(true, std::memory_order_relaxed);
                    }
                    if (stop.load(std::memory_order_relaxed))
                        continue;
                    auto& [key, bucket_size] = buckets[i];
                    stats.total += 1;
                    if (!data->index().sum_bucket(key, ptr, len) || len == 0)
                        continue;
                    const auto k  = ptr[0].a;
                    const auto l  = ptr[0].b;
                    auto       ns = std::make_shared<NullSpace>(
                        (l == k_single_sentinel<decltype(l)>()) ? local_gen.make(k) : local_gen.make(k, l));

                    const auto dim = ns->basis().rows();
                    const auto n   = stats.total;
                    update_mean(stats.mean_basis, dim, n);
                    update_mean(stats.mean_mr, bucket_size, n);
                    stats.max_basis = std::max(stats.max_basis, (Int)dim);

                    if (dim == 0)
                        continue;

                    PyRNG local_rng(base_seed + k + l);
                    auto  beyond        = [&](const auto red) { return (red < min_reduction || red > max_reduction); };
                    // TODO
                    auto  accept_anyway = [&](auto red) {
                        return false && red <= 0 && non_improving_prob && local_rng.rand_double(0.0, 1.0) < non_improving_prob;
                    };

                    auto coefs_list = local_rng.sample_small_unique_bitvectors(dim, num_samples);
                    // auto coefs_list = (len > 1) ? local_rng.sample_small_unique_bitvectors(dim, num_samples):
                    // local_rng.sample_special_bitvec(ns->basis(), ptr->a, ptr->b, num_samples);
                    stats.max_bucket = std::max(stats.max_bucket, (Int)bucket_size);
                    auto counter     = 0;

                    auto local_local_pool = TopKPool<ExplorationScore>(max_from_single_ns, escore);
                    ;
                    for (auto& coefs : coefs_list) {
                        auto       vec = ns->linear_combination(coefs);
                        const auto red = ns->rank_divergence(vec);

                        if (beyond(red)) {
                            if (accept_anyway(red) || (red > 0)) {
                                stats.accepted_non_improving++;
                            } else {
                                stats.rejected++;
                                continue;
                            }
                        }
                        stats.accepted++;
                        stats.nonzero++;
                        Candidate c(0, (Int)red, k, l, std::move(vec), (Int)dim, (Int)bucket_size, ns);
                        auto [score, tohpe] =
                            (beyond(red)) ? local_beyond_pool.push(std::move(c)) : local_local_pool.push(std::move(c));
                        local_svs.observe(red, dim, score);

                        stats.max_pool_score = std::max(stats.max_pool_score, score);
                        stats.max_reduction  = std::max(stats.max_reduction, (Int)red);
                        stats.max_bucket     = std::max(stats.max_bucket, (Int)bucket_size);
                        update_mean(stats.mean_score, score, stats.nonzero);
                        update_mean(stats.mean_reduction, red, stats.nonzero);
                    }
                    local_pool.merge_from(local_local_pool);
                }

#ifdef __OPENMP
#pragma omp critical
#endif
                {
                    if (local_pool.size() > 0) {
                        pool.merge_from(local_pool);
                    } else {
                        pool.merge_from(local_beyond_pool);
                    }
                    pool.merge_from(local_pool);
                    svs.merge_from(local_svs);
                    auto gnz                = global_stats.total;
                    auto lnz                = stats.total;
                    global_stats.mean_basis = (global_stats.mean_basis * gnz + stats.mean_basis * lnz) / (gnz + lnz);
                    global_stats.mean_mr    = (global_stats.mean_mr * gnz + stats.mean_mr * lnz) / (gnz + lnz);
                    gnz                     = global_stats.nonzero;
                    lnz                     = stats.nonzero;
                    global_stats.mean_score = (global_stats.mean_score * gnz + stats.mean_score * lnz) / (gnz + lnz);
                    global_stats.mean_reduction =
                        (global_stats.mean_reduction * gnz + stats.mean_reduction * lnz) / (gnz + lnz);
                    global_stats.accepted_non_improving += stats.accepted_non_improving;
                    global_stats.rejected += stats.rejected;
                    global_stats.accepted += stats.accepted;
                    global_stats.total += stats.total;
                    global_stats.nonzero += stats.nonzero;
                    global_stats.max_basis      = std::max(stats.max_basis, global_stats.max_basis);
                    global_stats.max_pool_score = std::max(stats.max_pool_score, global_stats.max_pool_score);
                    global_stats.max_reduction  = std::max(stats.max_reduction, global_stats.max_reduction);
                }
            }
        }
    }

    svs.finalize();

    auto final_pool = TopKPool<FinalizationScore>(top_pool, fscore);
    if (!pool.empty()) {
        FullToddGenerator gen(data);
        std::size_t       limit = pool.size();

        int n = 0;
        for (auto& cand : pool.release_unsorted()) {
            auto [score, tohpe] = final_pool.push(std::move(cand));

            global_stats.max_final_tohpe_dim = std::max(global_stats.max_final_tohpe_dim, tohpe);
            global_stats.max_final_score     = std::max(global_stats.max_final_score, score);
            update_mean(global_stats.mean_final_tohpe_dim, tohpe_dim, n);
            update_mean(global_stats.mean_final_score, score, ++n);
        }
    }

    PyRNG pick_rng(seed);
    bool  is_best = (selection == "greedy" || selection == "best");

    return build_result(final_pool, is_best, num_candidates, temperature, pick_rng, svs, data, global_stats, base_seed);
}

bool Candidate::at_least_single() const { return (k != k_single_sentinel<decltype(l)>()); }
bool Candidate::is_tohpe() const { return (l == k_single_sentinel<decltype(k)>()); }

void SeenValues::reserve(std::size_t n) { pool_scores_sorted.reserve(n); }

void SeenValues::observe(Int red, Int dim, float score) {
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
        std::inclusive_scan(freq.begin(), freq.end(), suf.begin());
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