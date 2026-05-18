#include "todd_generator.hpp"

#include "nullspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random.hpp>
#include <stdexcept>
#include <unordered_map>

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

Int log2_bin(Int value) {
    if (value <= 0)
        return 0;
    Int bin = 0;
    while (value > 1) {
        value >>= 1;
        ++bin;
    }
    return bin;
}

std::uint64_t candidate_signature(Candidate const& c) {
    auto pack = [](std::uint64_t key, std::uint64_t value, unsigned bits) {
        const auto mask = (bits >= 64) ? ~0ULL : ((1ULL << bits) - 1ULL);
        return (key << bits) | (value & mask);
    };

    const Int red_bin = std::clamp<Int>(c.reduction + 64, 0, 127);
    std::uint64_t key = 0;
    key               = pack(key, static_cast<std::uint64_t>(c.source), 3);
    key               = pack(key, static_cast<std::uint64_t>(red_bin), 7);
    key               = pack(key, static_cast<std::uint64_t>(log2_bin(c.basis_dim)), 6);
    key               = pack(key, static_cast<std::uint64_t>(log2_bin(c.bucket_size)), 6);
    key               = pack(key, static_cast<std::uint64_t>(log2_bin(c.vec_weight)), 6);
    key               = pack(key, static_cast<std::uint64_t>(log2_bin(c.z_weight)), 6);
    return key;
}

std::array<index_t, 3> normalize_sample_caps(std::array<Int, 3> caps) {
    return {
        static_cast<index_t>(std::max<Int>(caps[0], 0)),
        static_cast<index_t>(std::max<Int>(caps[1], 0)),
        static_cast<index_t>(std::max<Int>(caps[2], 0)),
    };
}

class SignatureLimiter {
  public:
    explicit SignatureLimiter(Int limit) : limit_(std::max<Int>(0, limit)) {}

    bool accept(Candidate const& c) {
        if (limit_ == 0)
            return true;
        auto& seen = counts_[candidate_signature(c)];
        if (seen >= limit_)
            return false;
        ++seen;
        return true;
    }

  private:
    Int                                     limit_;
    std::unordered_map<std::uint64_t, Int> counts_;
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
                                               std::size_t top_pool, std::size_t min_tohpe_actions,
                                               std::size_t min_todd_actions) {
    std::vector<Candidate> out;
    out.reserve(top_pool);
    take_source_quota(out, tohpe, min_tohpe_actions, top_pool);
    take_source_quota(out, todd, min_todd_actions, top_pool);

    std::vector<Candidate> rest;
    rest.reserve(tohpe.size() + todd.size());
    for (auto& c : tohpe)
        rest.push_back(std::move(c));
    for (auto& c : todd)
        rest.push_back(std::move(c));

    std::sort(rest.begin(), rest.end(), pool_preferred);
    for (auto& c : rest) {
        if (out.size() >= top_pool)
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
Result build_result(TopKPool<FinalizationScore>& pool, bool is_best, std::size_t num_candidates, float temperature,
                    PyRNG& pick_rng, SeenValues const& svs, TohpeGenerator& tohpe_gen, FullGenProvider& get_full_gen,
                    Stats const& global_stats, std::uint64_t base_seed) {
    auto chosen_view =
        is_best ? pick_best_view(pool, num_candidates) : pick_softmax_view(pool, num_candidates, temperature, pick_rng);

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
auto policy_iteration_impl(const std::shared_ptr<MatrixWithData>& data, PolicyConfig config, index_t seed,
                           index_t add_seed) -> Result {

    if (!data)
        throw std::runtime_error("data is null");
    auto update_mean            = [](auto& mean, auto value, Int n) { mean += (value - mean) / std::max((Int)1, n); };
    auto selection              = config.selection;
    auto tohpe_vector_samples   = normalize_sample_caps(config.tohpe_vector_samples);
    auto todd_vector_samples    = normalize_sample_caps(config.todd_vector_samples);
    auto num_candidates         = config.num_candidates;
    auto top_pool               = config.top_pool;
    auto temperature            = config.temperature;
    auto min_reduction          = config.min_reduction;
    auto max_reduction          = config.max_reduction;
    auto min_pool               = std::max(1, config.min_pool_size);
    auto escore                 = config.escore;
    auto fscore                 = config.fscore;
    auto max_z_to_research      = config.max_z_to_research;
    auto min_z_to_research      = index_t(std::max(0, config.min_z_to_research));
    auto tohpe_pool_size        = config.tohpe_pool_size;
    auto todd_pool_size         = config.todd_pool_size;
    auto min_tohpe_actions      = config.min_tohpe_actions;
    auto min_todd_actions       = config.min_todd_actions;
    auto max_from_single_ns     = config.max_from_single_ns;
    auto tohpe_sample           = config.tohpe_sample;
    auto sparse_max_weight      = config.sparse_max_weight;
    auto bucket_temperature     = config.bucket_temperature;
    auto bucket_random_fraction = config.bucket_random_fraction;
    auto max_per_signature      = config.max_per_signature;
    num_candidates              = std::max(num_candidates, (Int)1);
    min_tohpe_actions           = std::max(min_tohpe_actions, (Int)0);
    min_todd_actions            = std::max(min_todd_actions, (Int)0);
    top_pool = std::max({num_candidates, top_pool, min_tohpe_actions + min_todd_actions});
    max_reduction           = max_reduction > 0 ? max_reduction : k_single_sentinel<decltype(max_reduction)>();
    min_reduction           = std::max(min_reduction, (Int)0);
    tohpe_pool_size         = std::max(tohpe_pool_size, (Int)0);
    todd_pool_size          = std::max(todd_pool_size, (Int)0);
    max_from_single_ns      = std::max(max_from_single_ns, (Int)1);
    tohpe_sample            = std::max(tohpe_sample, (Int)1);
    sparse_max_weight       = std::max(sparse_max_weight, (Int)2);
    bucket_temperature      = std::max(bucket_temperature, 0.0f);
    bucket_random_fraction  = std::clamp(bucket_random_fraction, 0.0f, 1.0f);
    max_per_signature       = std::max(max_per_signature, (Int)0);

    const auto                       base_seed            = matrix_seed(data->P(), seed, add_seed);
    const auto                       tohpe_dim            = data->tohpe_basis().rows();
    const Matrix&                    tohpe_basis          = data->tohpe_basis();
    auto                             bucket_normalization = data->index().max_bucket();
    auto                             wvw_normalization    = data->P().rows();
    TohpeGenerator                   tohpe_gen(data);
    std::optional<FullToddGenerator> full_gen;
    auto                             get_full_gen = [&]() -> FullToddGenerator& {
        if (!full_gen)
            full_gen.emplace(data);
        return *full_gen;
    };
    auto dim_normalization = tohpe_dim + 5;
    escore.bn              = bucket_normalization;
    escore.wvwn            = wvw_normalization;
    escore.dn              = dim_normalization;
    fscore.bn              = bucket_normalization;
    fscore.wvwn            = wvw_normalization;
    fscore.dn              = dim_normalization;
    auto svs               = SeenValues{};
    auto tohpe_pool        = TopKPool<ExplorationScore>(tohpe_pool_size, escore);
    auto todd_pool         = TopKPool<ExplorationScore>(todd_pool_size, escore);
    auto signatures        = SignatureLimiter(max_per_signature);

    Stats global_stats;
    {
        if (tohpe_pool_size > 0) {
            auto local_beyond_pool = TopKPool<ExplorationScore>(tohpe_pool_size, escore);
            auto local_pool        = TopKPool<ExplorationScore>(tohpe_pool_size, escore);

            const index_t dim      = tohpe_dim;
            global_stats.max_basis = dim;

            PyRNG local_rng(mixed_seed(base_seed, 0, 0, 0, CandidateSourceTohpe));
            auto  beyond = [&](const auto red) { return (red < min_reduction || red > max_reduction); };

            std::vector<TohpeZInfo> scratch_best_z;
            local_rng.for_each_capped_bitvector(dim, tohpe_vector_samples, sparse_max_weight, [&](RowCView coefs) {
                assert(coefs.count() != 0);
                global_stats.evaluated++;
                auto vec = linear_combination_from_basis(tohpe_basis, coefs);
                tohpe_gen.best_z_n_details_into(vec, tohpe_sample, scratch_best_z);
                for (std::size_t zi = 0; zi < scratch_best_z.size(); ++zi) {
                    auto&      info = scratch_best_z[zi];
                    const auto red  = info.reduction;
                    if (beyond(red) && red <= 0) {
                        global_stats.rejected++;
                        continue;
                    }
                    Row       cand_vec = (zi + 1 == scratch_best_z.size()) ? std::move(vec) : Row(vec);
                    Candidate c(0, (Int)red, k_single_sentinel<Int>(), k_single_sentinel<Int>(), std::move(cand_vec),
                                std::move(info.z), (Int)dim, (Int)info.bucket_size, info.bucket_id,
                                CandidateSourceTohpe);
                    if (!signatures.accept(c)) {
                        global_stats.rejected++;
                        continue;
                    }
                    global_stats.nonzero++;
                    global_stats.accepted_tohpe++;
                    auto [score, tohpe] =
                        (beyond(red)) ? local_beyond_pool.push(std::move(c)) : local_pool.push(std::move(c));
                    svs.observe(red, dim, score);

                    global_stats.max_pool_score = std::max(global_stats.max_pool_score, score);
                    global_stats.max_reduction  = std::max(global_stats.max_reduction, (Int)red);
                    global_stats.max_bucket     = std::max(global_stats.max_bucket, (Int)info.bucket_size);
                    update_mean(global_stats.mean_score, score, global_stats.nonzero);
                    update_mean(global_stats.mean_reduction, red, global_stats.nonzero);
                }
            });
            tohpe_pool.merge_from(local_pool);
            tohpe_pool.merge_from(local_beyond_pool);
        }

        const bool run_todd_stage = todd_pool_size > 0;

        if (run_todd_stage) {

            auto& buckets = data->index().sum_key_sizes_scratch();
            const auto max_bucket_research =
                std::min<std::size_t>(buckets.size(), static_cast<std::size_t>(std::max<Int>(0, max_z_to_research)));
            PyRNG bucket_rng(mixed_seed(base_seed, buckets.size(), max_bucket_research, 0, CandidateSourceTodd));
            diversify_buckets(buckets, max_bucket_research, bucket_temperature, bucket_random_fraction, bucket_rng);

            min_z_to_research = std::min<std::size_t>(buckets.size(), min_z_to_research);

            {
                Stats stats;

                const auto required_todd_actions = static_cast<std::size_t>(std::max<Int>(min_pool, min_todd_actions));
                auto       local_beyond_pool     = TopKPool<ExplorationScore>(required_todd_actions, escore);
                auto       local_pool            = TopKPool<ExplorationScore>(todd_pool_size, escore);
                auto& local_gen         = get_full_gen();
                auto  local_svs         = SeenValues{};
                local_pool.reserve(todd_pool_size);

                const SumEntry* ptr = nullptr;
                auto            len = index_t{};

                for (std::size_t i = 0; i < buckets.size(); ++i) {
                    if (i >= min_z_to_research &&
                        (local_pool.size() >= required_todd_actions ||
                         local_beyond_pool.size() >= required_todd_actions)) {
                        break;
                    }

                    auto& [key, bucket_size] = buckets[i];
                    stats.total += 1;
                    stats.z_researched += 1;
                    if (!data->index().sum_bucket(key, ptr, len) || len == 0)
                        continue;

                    const index_t k         = ptr[0].a;
                    const bool    is_single = !ptr[0].is_pair();
                    const index_t l         = is_single ? k_single_sentinel<index_t>() : (index_t)ptr[0].b;
                    auto          ns        = is_single ? local_gen.make(k) : local_gen.make(k, l);

                    const auto dim = ns.basis().rows();
                    const auto n   = stats.total;
                    update_mean(stats.mean_basis, dim, n);
                    update_mean(stats.mean_mr, bucket_size, n);
                    stats.max_basis = std::max(stats.max_basis, (Int)dim);

                    if (dim == 0)
                        continue;

                    PyRNG local_rng(mixed_seed(base_seed, k, is_single ? 0 : l, i, CandidateSourceTodd));
                    auto  beyond = [&](const auto red) { return (red < min_reduction || red > max_reduction); };

                    stats.max_bucket = std::max(stats.max_bucket, (Int)bucket_size);

                    auto local_local_pool = TopKPool<ExplorationScore>(max_from_single_ns, escore);
                    local_rng.for_each_capped_bitvector(dim, todd_vector_samples, sparse_max_weight, [&](RowCView coefs) {
                        stats.evaluated++;
                        auto       vec = ns.linear_combination(coefs);
                        const auto red = ns.rank_divergence(vec);

                        if (beyond(red) && red <= 0) {
                            stats.rejected++;
                            return;
                        }
                        const Int cand_l = is_single ? k_single_sentinel<Int>() : static_cast<Int>(l);
                        Candidate c(0, (Int)red, static_cast<Int>(k), cand_l, std::move(vec), (Int)dim,
                                    (Int)bucket_size, static_cast<Int>(key.count()), static_cast<Int>(key.size()),
                                    CandidateSourceTodd);
                        if (!signatures.accept(c)) {
                            stats.rejected++;
                            return;
                        }
                        stats.accepted++;
                        stats.nonzero++;
                        auto [score, tohpe] =
                            (beyond(red)) ? local_beyond_pool.push(std::move(c)) : local_local_pool.push(std::move(c));
                        (void)tohpe;
                        local_svs.observe(red, dim, score);

                        stats.max_pool_score = std::max(stats.max_pool_score, score);
                        stats.max_reduction  = std::max(stats.max_reduction, (Int)red);
                        stats.max_bucket     = std::max(stats.max_bucket, (Int)bucket_size);
                        update_mean(stats.mean_score, score, stats.nonzero);
                        update_mean(stats.mean_reduction, red, stats.nonzero);
                    });
                    local_pool.merge_from(local_local_pool);
                }

                if (local_pool.size() > 0) {
                    todd_pool.merge_from(local_pool);
                } else {
                    todd_pool.merge_from(local_beyond_pool);
                }
                svs.merge_from(local_svs);

                if (stats.total > 0) {
                    const auto gnz          = global_stats.total;
                    const auto lnz          = stats.total;
                    global_stats.mean_basis = (global_stats.mean_basis * gnz + stats.mean_basis * lnz) / (gnz + lnz);
                    global_stats.mean_mr    = (global_stats.mean_mr * gnz + stats.mean_mr * lnz) / (gnz + lnz);
                }
                if (stats.nonzero > 0) {
                    const auto gnz          = global_stats.nonzero;
                    const auto lnz          = stats.nonzero;
                    global_stats.mean_score = (global_stats.mean_score * gnz + stats.mean_score * lnz) / (gnz + lnz);
                    global_stats.mean_reduction =
                        (global_stats.mean_reduction * gnz + stats.mean_reduction * lnz) / (gnz + lnz);
                }
                global_stats.rejected += stats.rejected;
                global_stats.accepted += stats.accepted;
                global_stats.evaluated += stats.evaluated;
                global_stats.total += stats.total;
                global_stats.z_researched += stats.z_researched;
                global_stats.nonzero += stats.nonzero;
                global_stats.max_basis      = std::max(stats.max_basis, global_stats.max_basis);
                global_stats.max_pool_score = std::max(stats.max_pool_score, global_stats.max_pool_score);
                global_stats.max_reduction  = std::max(stats.max_reduction, global_stats.max_reduction);
                global_stats.max_bucket     = std::max(stats.max_bucket, global_stats.max_bucket);
            }
        }
    }

    svs.finalize();

    auto final_pool = TopKPool<FinalizationScore>(top_pool, fscore);
    auto merged_candidates =
        merge_source_candidates(tohpe_pool.release_unsorted(), todd_pool.release_unsorted(), top_pool,
                                static_cast<std::size_t>(min_tohpe_actions), static_cast<std::size_t>(min_todd_actions));
    if (!merged_candidates.empty()) {
        const bool                need_tohpe_dim = fscore.needs_tohpe_dim();
        std::vector<std::uint8_t> scratch_killed;
        int                       n = 0;
        for (auto& cand : merged_candidates) {
            if (need_tohpe_dim) {
                auto ns        = make_candidate_nullspace(cand, tohpe_gen, get_full_gen);
                cand.tohpe_dim = static_cast<Int>(get_tohpe_basis(ns.apply(cand.vec, scratch_killed)).rows());
            }
            auto [score, tohpe] = final_pool.push(std::move(cand));

            ++n;
            global_stats.max_final_tohpe_dim = std::max(global_stats.max_final_tohpe_dim, tohpe);
            global_stats.max_final_score     = std::max(global_stats.max_final_score, score);
            update_mean(global_stats.mean_final_tohpe_dim, tohpe, n);
            update_mean(global_stats.mean_final_score, score, n);
        }
    }

    PyRNG pick_rng(seed);
    bool  is_best = (selection == "greedy" || selection == "best");

    auto result = build_result(final_pool, is_best, num_candidates, temperature, pick_rng, svs, tohpe_gen, get_full_gen,
                               global_stats, base_seed);
    full_gen.reset();
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
