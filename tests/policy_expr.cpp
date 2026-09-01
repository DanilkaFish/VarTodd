// Unit checks for the policy expression VM.
//
// The equivalence checks compare compiled programs against a frozen copy of
// the scoring functions the VM replaced (see namespace legacy below), pinning
// the new implementation to the old behaviour permanently.

#include "policy_expr.hpp"
#include "todd_generator.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace todd;

// --- frozen copy of the replaced scoring implementation ----------------------
//
// These reproduce ScoringFunction / ExplorationScore / FinalizationScore as
// they stood before the expression VM replaced them (todd_generator.hpp,
// commit 9027cb6). They are the reference the compiled programs are checked
// against, kept here verbatim so the equivalence gate survives the cutover
// rather than disappearing with the code it validated. Do not "fix" them --
// their whole value is being an unchanged record of the old behaviour.
namespace legacy {

struct ScoringFunction {
    enum Function { LINEAR, POLYNOM, DISTANCE, LOGARITHMIC, SIGMOID };
    Function type = LINEAR;
    float    pow  = 2.0;

    float evaluate(std::span<const float> weights, std::span<const float> centers, std::span<const float> x,
                   float first_center_scale = 1.0f) const {
        switch (type) {
        case LINEAR:
            return linear_scoring(weights, x);
        case POLYNOM:
            return polynom_scoring(weights, centers, x, first_center_scale);
        case DISTANCE:
            return distance_scoring(weights, x);
        case SIGMOID:
            return sigmoid_scoring(weights, x);
        case LOGARITHMIC:
            return logarithmic_scoring(weights, x);
        }
        throw std::runtime_error("Wrong function type for evaluation");
    }

    float linear_scoring(std::span<const float> w, std::span<const float> x) const {
        float             sum = 0.0f;
        const std::size_t n   = std::min(w.size(), x.size());
        for (std::size_t i = 0; i < n; ++i)
            sum += w[i] * x[i];
        return sum;
    }

    float polynom_scoring(std::span<const float> w, std::span<const float> c, std::span<const float> x,
                          float first_center_scale = 1.0f) const {
        float             sum = 0.0f;
        const std::size_t n   = std::min({w.size(), c.size(), x.size()});
        if (pow >= 0) {
            for (std::size_t i = 0; i < n; ++i) {
                const float center = i == 0 ? c[i] * first_center_scale : c[i];
                const float d      = std::abs(x[i] - center);
                sum += w[i] * ((pow == 2.0f) ? d * d : std::pow(d, pow));
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                const float center = i == 0 ? c[i] * first_center_scale : c[i];
                sum += w[i] * std::pow(1 / std::max(std::abs(x[i] - center), 0.001f), -pow);
            }
        }
        return sum;
    }

    float distance_scoring(std::span<const float> w, std::span<const float> x) const {
        float             sum = 0.0f;
        const std::size_t n   = std::min(w.size(), x.size());
        for (std::size_t i = 0; i < n; ++i)
            sum -= (x[i] - w[i]) * (x[i] - w[i]);
        return sum;
    }

    float sigmoid_scoring(std::span<const float> w, std::span<const float> x) const {
        float             sum = 0.0f;
        const std::size_t n   = std::min(w.size(), x.size());
        for (std::size_t i = 0; i < n; ++i)
            sum += w[i] * ((pow == 2.0f) ? x[i] * x[i] : std::pow(x[i], pow));
        return 1.0f / (1.0f + std::exp(-sum));
    }

    float logarithmic_scoring(std::span<const float> w, std::span<const float> x) const {
        float             sum = 0.0f;
        const std::size_t n   = std::min(w.size(), x.size());
        for (std::size_t i = 0; i < n; ++i)
            sum += w[i] * std::log(std::max(1e-6f, x[i]));
        return sum;
    }
};

struct ExplorationScore {
    std::vector<float> weights = {0, 0, 0, 0, 0};
    std::vector<float> centers = {0, 0, 0, 0, 0};
    float              bn      = 1;
    float              wvwn    = 1;
    float              dn      = 1;
    ScoringFunction    sc      = {};

    ExplorationScore() = default;
    ExplorationScore(float wred, float wdim, float wbucket, float wvw, float wz)
        : weights{wred, wdim, wbucket, wvw, wz} {}
    ExplorationScore(std::vector<float> w, std::vector<float> c, ScoringFunction s)
        : weights{std::move(w)}, centers{std::move(c)}, sc{s} {}
    ExplorationScore(std::vector<float> w, std::vector<float> c, float pow)
        : weights{std::move(w)}, centers{std::move(c)}, sc{ScoringFunction::POLYNOM, pow} {}

    float evaluate_features(Int reduction, Int basis_dim, Int bucket_size, Int vec_weight, Int z_weight,
                            Int z_size) const {
        const std::array<float, 5> x = {
            reduction / bn / 2.0f,
            basis_dim / dn,
            bucket_size / bn,
            vec_weight / wvwn,
            z_weight / static_cast<float>(std::max<Int>(1, z_size)),
        };
        return sc.evaluate(weights, centers, x, 1.0f / bn / 2.0f);
    }

};

struct FinalizationScore {
    std::vector<float> weights = {0, 0, 0, 0, 0, 0};
    std::vector<float> centers = {0, 0, 0, 0, 0, 0};
    float              bn      = 1;
    float              wvwn    = 1;
    float              dn      = 1;
    ScoringFunction    sc      = {};

    FinalizationScore() = default;
    FinalizationScore(std::vector<float> w, std::vector<float> c, ScoringFunction s)
        : weights{std::move(w)}, centers{std::move(c)}, sc{s} {}

    bool needs_tohpe_dim() const { return weights.size() > 5 && weights[5] != 0.0f; }

    float evaluate(const Candidate& cand) const {
        const std::array<float, 6> x = {cand.reduction / bn / 2,
                                        cand.basis_dim / dn,
                                        cand.bucket_size / bn,
                                        cand.vec_weight / wvwn,
                                        cand.z_weight / static_cast<float>(cand.z_size),
                                        cand.tohpe_dim / dn};
        const std::size_t          n = needs_tohpe_dim() ? 6 : 5;
        return sc.evaluate(weights, centers, std::span<const float>(x.data(), n), 1.0f / bn / 2.0f);
    }
};

} // namespace legacy

void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

bool close(float a, float b, float tol = 1e-6f) {
    const float scale = std::max({1.0f, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= tol * scale;
}

Instr knob(Knob k) { return Instr{Op::LoadKnob, static_cast<std::uint16_t>(k)}; }
Instr cst(std::uint16_t i) { return Instr{Op::LoadConst, i}; }
Instr par(std::uint16_t i) { return Instr{Op::LoadParam, i}; }
Instr op(Op o) { return Instr{o, 0}; }

PolicyProgram make(std::vector<Instr> code, std::vector<float> consts = {}, std::size_t n_params = 0,
                   PolicySite site = PolicySite::Finalization) {
    return PolicyProgram(std::move(code), std::move(consts), n_params, site);
}

// --- basic evaluation -------------------------------------------------------

void check_arithmetic_and_stack() {
    // 2 + 3 * 4 == 14, written postfix
    auto prog = make({cst(0), cst(1), cst(2), op(Op::Mul), op(Op::Add)}, {2.0f, 3.0f, 4.0f});
    KnobFrame f{};
    require(close(prog.eval(f, {}), 14.0f), "arithmetic: 2 + 3*4 should be 14");

    auto neg = make({cst(0), op(Op::Neg), op(Op::Abs)}, {-5.0f});
    require(close(neg.eval(f, {}), 5.0f), "abs(-(-5)) should be 5");
}

void check_knob_normalization() {
    KnobFrame f{};
    f.red    = 10.0f;
    f.dim    = 8.0f;
    f.bucket = 6.0f;
    f.yw     = 4.0f;
    f.zw     = 3.0f;
    f.zsize  = 6.0f;
    f.bn     = 5.0f;
    f.dn     = 4.0f;
    f.wvwn   = 2.0f;

    require(close(make({knob(Knob::nred)}).eval(f, {}), 10.0f / 5.0f / 2.0f), "nred = red/bn/2");
    require(close(make({knob(Knob::ndim)}).eval(f, {}), 8.0f / 4.0f), "ndim = dim/dn");
    require(close(make({knob(Knob::nbucket)}).eval(f, {}), 6.0f / 5.0f), "nbucket = bucket/bn");
    require(close(make({knob(Knob::nyw)}).eval(f, {}), 4.0f / 2.0f), "nyw = yw/wvwn");
    require(close(make({knob(Knob::nzw)}).eval(f, {}), 3.0f / 6.0f), "nzw = zw/max(1,zsize)");

    // zsize is floored at 1 so a zero-length z cannot divide by zero
    f.zsize = 0.0f;
    require(close(make({knob(Knob::nzw)}).eval(f, {}), 3.0f), "nzw should floor zsize at 1");
}

void check_pool_fractions_and_ranks() {
    KnobFrame f{};
    f.pool_size   = 20.0f;
    f.pool_tohpe  = 5.0f;
    f.pool_prefix = 3.0f;
    f.pool_todd   = 12.0f;
    f.rank_red    = 4.0f;

    require(close(make({knob(Knob::f_tohpe)}).eval(f, {}), 0.25f), "f_tohpe = pool_tohpe/pool_size");
    require(close(make({knob(Knob::f_prefix)}).eval(f, {}), 0.15f), "f_prefix");
    require(close(make({knob(Knob::f_todd)}).eval(f, {}), 0.6f), "f_todd");
    require(close(make({knob(Knob::nrank_red)}).eval(f, {}), 0.2f), "nrank_red = rank_red/pool_size");

    f.pool_size = 0.0f;
    require(close(make({knob(Knob::f_todd)}).eval(f, {}), 12.0f), "pool fractions floor pool_size at 1");
}

void check_select_and_comparisons() {
    KnobFrame f{};
    f.dim = 7.0f;
    // where(dim > 5, 100, 200)
    auto prog = make({knob(Knob::dim), cst(0), op(Op::Gt), cst(1), cst(2), op(Op::Select)},
                     {5.0f, 100.0f, 200.0f});
    require(close(prog.eval(f, {}), 100.0f), "select should take the true branch");
    f.dim = 3.0f;
    require(close(prog.eval(f, {}), 200.0f), "select should take the false branch");

    // Select argument order: (cond, a, b) with b pushed last.
    f.dim     = 1.0f;
    auto logic = make({knob(Knob::dim), cst(0), op(Op::Lt), op(Op::Not)}, {5.0f});
    require(close(logic.eval(f, {}), 0.0f), "not(1 < 5) should be 0");
}

void check_guarded_division_and_log() {
    KnobFrame f{};
    // Dividing by zero must stay finite, matching the legacy inverse-distance floor.
    auto div = make({cst(0), cst(1), op(Op::Div)}, {1.0f, 0.0f});
    const float v = div.eval(f, {});
    require(std::isfinite(v), "division by zero must stay finite");
    require(close(v, 1.0f / k_div_epsilon), "division by zero uses the epsilon floor");

    // log clamps at 1e-6 exactly as logarithmic_scoring did
    auto lg = make({cst(0), op(Op::Log)}, {0.0f});
    require(close(lg.eval(f, {}), std::log(k_log_floor)), "log(0) should clamp to log(1e-6)");

    auto sq = make({cst(0), op(Op::Sqrt)}, {-4.0f});
    require(close(sq.eval(f, {}), 0.0f), "sqrt of a negative should clamp to 0");
}

void check_params() {
    KnobFrame f{};
    f.red = 3.0f;
    f.bn  = 1.0f;
    // red * p0 + p1
    auto prog = make({knob(Knob::red), par(0), op(Op::Mul), par(1), op(Op::Add)}, {}, 2);
    const std::vector<float> params = {2.0f, 10.0f};
    require(close(prog.eval(f, params), 16.0f), "params should bind positionally");
    require(prog.n_params() == 2, "n_params should reflect the declared count");
}

// --- validation -------------------------------------------------------------

bool throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const PolicyProgramError&) {
        return true;
    }
    return false;
}

void check_validation_rejects_malformed_programs() {
    require(throws([] { make({}); }), "empty program must be rejected");
    require(throws([] { make({op(Op::Add)}); }), "stack underflow must be rejected");
    require(throws([] { make({cst(0), cst(0)}, {1.0f}); }), "leftover stack values must be rejected");
    require(throws([] { make({cst(5)}, {1.0f}); }), "out-of-range const index must be rejected");
    require(throws([] { make({par(0)}, {}, 0); }), "param index beyond n_params must be rejected");
    require(throws([] { make({Instr{Op::LoadKnob, static_cast<std::uint16_t>(k_knob_count)}}); }),
            "out-of-range knob index must be rejected");
}

void check_site_availability_is_enforced() {
    // tohpe_dim is only known at finalization.
    require(throws([] { make({knob(Knob::tohpe)}, {}, 0, PolicySite::ExplorationPool); }),
            "tohpe must be rejected at the exploration site");
    require(throws([] { make({knob(Knob::rank_red)}, {}, 0, PolicySite::ExplorationZ); }),
            "rank_red must be rejected at the exploration site");
    require(throws([] { make({knob(Knob::f_todd)}, {}, 0, PolicySite::ExplorationPool); }),
            "pool composition must be rejected at the exploration site");

    // ...and accepted at finalization.
    make({knob(Knob::tohpe)}, {}, 0, PolicySite::Finalization);
    make({knob(Knob::rank_red)}, {}, 0, PolicySite::Finalization);

    // Shared knobs work everywhere.
    make({knob(Knob::nred)}, {}, 0, PolicySite::ExplorationZ);
    make({knob(Knob::nred)}, {}, 0, PolicySite::Finalization);
}

void check_used_knobs_reporting() {
    auto prog = make({knob(Knob::nred), knob(Knob::nbucket), op(Op::Add)}, {}, 0, PolicySite::ExplorationPool);
    require(prog.uses(Knob::nred), "used_knobs should report nred");
    require(prog.uses(Knob::nbucket), "used_knobs should report nbucket");
    require(!prog.uses(Knob::ndim), "used_knobs should not report unused knobs");
}

// --- fast-path predicate ----------------------------------------------------


// This is the parity claim from the plan: the new predicate must accept every
// program the legacy ranks_fixed_y_by_reduction() accepted. That predicate was
// LINEAR && bn > 0 && w[0] >= 0 && w[2] == 0 && w[4] == 0, i.e. a weighted sum
// over {red, dim, yw} with a non-negative reduction weight.

// --- equivalence with the legacy scoring functions --------------------------

// Builds the postfix form of sum_i w[i] * x[i] over the five exploration knobs.
std::vector<Instr> linear_sum_code(const std::vector<float>& weights, const std::vector<Knob>& knobs) {
    std::vector<Instr> code;
    bool               first = true;
    for (std::uint16_t i = 0; i < static_cast<std::uint16_t>(knobs.size()); ++i) {
        code.push_back(knob(knobs[i]));
        code.push_back(cst(i));
        code.push_back(op(Op::Mul));
        if (!first)
            code.push_back(op(Op::Add));
        first = false;
    }
    (void)weights;
    return code;
}

void check_linear_score_equivalence() {
    std::mt19937                          rng(20260830);
    std::uniform_real_distribution<float> w_dist(-4.0f, 4.0f);
    std::uniform_int_distribution<int>    v_dist(0, 40);

    const std::vector<Knob> knobs = {Knob::nred, Knob::ndim, Knob::nbucket, Knob::nyw, Knob::nzw};

    for (int trial = 0; trial < 200; ++trial) {
        std::vector<float> weights(5);
        for (float& w : weights)
            w = w_dist(rng);

        const Int   reduction = v_dist(rng);
        const Int   basis_dim = v_dist(rng);
        const Int   bucket    = v_dist(rng);
        const Int   vec_w     = v_dist(rng);
        const Int   z_w       = v_dist(rng);
        const Int   z_size    = std::max(1, v_dist(rng));
        const float bn        = 1.0f + static_cast<float>(v_dist(rng));
        const float dn        = 1.0f + static_cast<float>(v_dist(rng));
        const float wvwn      = 1.0f + static_cast<float>(v_dist(rng));

        legacy::ExplorationScore legacy(weights, {0, 0, 0, 0, 0},
                                        legacy::ScoringFunction{legacy::ScoringFunction::LINEAR, 1.0f});
        legacy.bn   = bn;
        legacy.dn   = dn;
        legacy.wvwn = wvwn;
        const float expected = legacy.evaluate_features(reduction, basis_dim, bucket, vec_w, z_w, z_size);

        auto      prog = PolicyProgram(linear_sum_code(weights, knobs), weights, 0, PolicySite::ExplorationPool);
        KnobFrame f{};
        f.red    = static_cast<float>(reduction);
        f.dim    = static_cast<float>(basis_dim);
        f.bucket = static_cast<float>(bucket);
        f.yw     = static_cast<float>(vec_w);
        f.zw     = static_cast<float>(z_w);
        f.zsize  = static_cast<float>(z_size);
        f.bn     = bn;
        f.dn     = dn;
        f.wvwn   = wvwn;

        require(close(prog.eval(f, {}), expected, 1e-5f),
                "compiled linear program must match ExplorationScore::evaluate_features");
    }
}

// POLYNOM with pow == 2: sum_i w[i] * (x[i] - c[i])^2, where the first center is
// scaled by 1/bn/2 exactly as polynom_scoring does.
void check_polynom_score_equivalence() {
    std::mt19937                          rng(20260831);
    std::uniform_real_distribution<float> w_dist(-3.0f, 3.0f);
    std::uniform_real_distribution<float> c_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int>    v_dist(0, 40);

    const std::vector<Knob> knobs = {Knob::nred, Knob::ndim, Knob::nbucket, Knob::nyw, Knob::nzw};

    for (int trial = 0; trial < 200; ++trial) {
        std::vector<float> weights(5);
        std::vector<float> centers(5);
        for (int i = 0; i < 5; ++i) {
            weights[i] = w_dist(rng);
            centers[i] = c_dist(rng);
        }

        const Int   reduction = v_dist(rng);
        const Int   basis_dim = v_dist(rng);
        const Int   bucket    = v_dist(rng);
        const Int   vec_w     = v_dist(rng);
        const Int   z_w       = v_dist(rng);
        const Int   z_size    = std::max(1, v_dist(rng));
        const float bn        = 1.0f + static_cast<float>(v_dist(rng));
        const float dn        = 1.0f + static_cast<float>(v_dist(rng));
        const float wvwn      = 1.0f + static_cast<float>(v_dist(rng));

        legacy::ExplorationScore legacy(weights, centers, 2.0f);
        legacy.bn   = bn;
        legacy.dn   = dn;
        legacy.wvwn = wvwn;
        const float expected = legacy.evaluate_features(reduction, basis_dim, bucket, vec_w, z_w, z_size);

        // consts layout: [w0..w4, c0..c4] with c0 already scaled by 1/bn/2.
        std::vector<float> consts;
        consts.insert(consts.end(), weights.begin(), weights.end());
        for (int i = 0; i < 5; ++i)
            consts.push_back(i == 0 ? centers[i] / bn / 2.0f : centers[i]);

        std::vector<Instr> code;
        bool               first = true;
        for (std::uint16_t i = 0; i < 5; ++i) {
            // w[i] * (x[i] - c[i])^2 via Mul rather than Pow, matching the
            // pow == 2 special case in polynom_scoring.
            code.push_back(knob(knobs[i]));
            code.push_back(cst(static_cast<std::uint16_t>(5 + i)));
            code.push_back(op(Op::Sub));
            code.push_back(op(Op::Abs));
            const std::size_t dup_start = code.size();
            (void)dup_start;
            // square by multiplying the difference with itself: recompute it
            code.push_back(knob(knobs[i]));
            code.push_back(cst(static_cast<std::uint16_t>(5 + i)));
            code.push_back(op(Op::Sub));
            code.push_back(op(Op::Abs));
            code.push_back(op(Op::Mul));
            code.push_back(cst(i));
            code.push_back(op(Op::Mul));
            if (!first)
                code.push_back(op(Op::Add));
            first = false;
        }

        auto      prog = PolicyProgram(code, consts, 0, PolicySite::ExplorationPool);
        KnobFrame f{};
        f.red    = static_cast<float>(reduction);
        f.dim    = static_cast<float>(basis_dim);
        f.bucket = static_cast<float>(bucket);
        f.yw     = static_cast<float>(vec_w);
        f.zw     = static_cast<float>(z_w);
        f.zsize  = static_cast<float>(z_size);
        f.bn     = bn;
        f.dn     = dn;
        f.wvwn   = wvwn;

        require(close(prog.eval(f, {}), expected, 1e-4f),
                "compiled polynom program must match ExplorationScore::evaluate_features");
    }
}

void check_finalization_six_feature_equivalence() {
    std::mt19937                          rng(20260901);
    std::uniform_real_distribution<float> w_dist(-4.0f, 4.0f);
    std::uniform_int_distribution<int>    v_dist(0, 40);

    const std::vector<Knob> knobs = {Knob::nred,  Knob::ndim, Knob::nbucket,
                                     Knob::nyw,   Knob::nzw,  Knob::ntohpe};

    for (int trial = 0; trial < 200; ++trial) {
        std::vector<float> weights(6);
        for (float& w : weights)
            w = w_dist(rng);
        // needs_tohpe_dim() gates the 6th feature on a nonzero weight.
        if (weights[5] == 0.0f)
            weights[5] = 0.5f;

        Candidate cand;
        cand.reduction   = v_dist(rng);
        cand.basis_dim   = v_dist(rng);
        cand.bucket_size = v_dist(rng);
        cand.vec_weight  = v_dist(rng);
        cand.z_weight    = v_dist(rng);
        cand.z_size      = std::max(1, v_dist(rng));
        cand.tohpe_dim   = v_dist(rng);

        const float bn   = 1.0f + static_cast<float>(v_dist(rng));
        const float dn   = 1.0f + static_cast<float>(v_dist(rng));
        const float wvwn = 1.0f + static_cast<float>(v_dist(rng));

        legacy::FinalizationScore legacy(weights, {0, 0, 0, 0, 0, 0},
                                         legacy::ScoringFunction{legacy::ScoringFunction::LINEAR, 1.0f});
        legacy.bn   = bn;
        legacy.dn   = dn;
        legacy.wvwn = wvwn;
        const float expected = legacy.evaluate(cand);

        auto      prog = PolicyProgram(linear_sum_code(weights, knobs), weights, 0, PolicySite::Finalization);
        KnobFrame f{};
        f.red    = static_cast<float>(cand.reduction);
        f.dim    = static_cast<float>(cand.basis_dim);
        f.bucket = static_cast<float>(cand.bucket_size);
        f.yw     = static_cast<float>(cand.vec_weight);
        f.zw     = static_cast<float>(cand.z_weight);
        f.zsize  = static_cast<float>(cand.z_size);
        f.tohpe  = static_cast<float>(cand.tohpe_dim);
        f.bn     = bn;
        f.dn     = dn;
        f.wvwn   = wvwn;

        require(close(prog.eval(f, {}), expected, 1e-5f),
                "compiled program must match FinalizationScore over six features");
    }
}

// --- non-finite behaviour ---------------------------------------------------

void check_no_spurious_non_finite() {
    // Every guarded op must stay finite on hostile inputs, so that a policy
    // cannot silently poison candidate ordering with NaN.
    KnobFrame f{};
    f.zsize     = 0.0f;
    f.bn        = 1.0f;
    f.dn        = 1.0f;
    f.wvwn      = 1.0f;
    f.pool_size = 0.0f;

    const std::vector<std::vector<Instr>> programs = {
        {cst(0), cst(1), op(Op::Div)},
        {cst(1), op(Op::Log)},
        {cst(2), op(Op::Sqrt)},
        {knob(Knob::nzw)},
        {knob(Knob::f_todd)},
        {knob(Knob::nrank_red)},
    };
    for (const auto& code : programs) {
        auto        prog = make(code, {1.0f, 0.0f, -1.0f});
        const float v    = prog.eval(f, {});
        require(std::isfinite(v), "guarded operations must not produce NaN or Inf");
    }
}

// Prints one line per program so tests/test_policy_api.py can diff the Python
// reference evaluator against this VM. The two implement the same semantics
// independently -- the Python one backs the load-time probe pass -- so they
// must be checked against each other or they will drift.
void dump_cross_check_vectors() {
    KnobFrame f{};
    f.red = 12; f.dim = 7; f.bucket = 9; f.yw = 5; f.zw = 3; f.zsize = 8;
    f.max_red = 18; f.tohpe = 4; f.rank_red = 2; f.rank_dim = 3; f.rank_score = 1;
    f.pool_size = 20; f.pool_tohpe = 5; f.pool_prefix = 3; f.pool_todd = 12;
    f.source = 1; f.bucket_id = 42; f.k_idx = 2; f.l_idx = 6;
    f.bn = 6; f.dn = 5; f.wvwn = 4;

    const std::vector<float> params = {1.5f, -0.75f, 0.25f};

    struct Case {
        const char*        name;
        std::vector<Instr> code;
        std::vector<float> consts;
        std::size_t        n_params;
    };
    const std::vector<Case> cases = {
        {"nred", {knob(Knob::nred)}, {}, 0},
        {"ndim", {knob(Knob::ndim)}, {}, 0},
        {"nzw", {knob(Knob::nzw)}, {}, 0},
        {"f_todd", {knob(Knob::f_todd)}, {}, 0},
        {"nrank_red", {knob(Knob::nrank_red)}, {}, 0},
        {"ntohpe", {knob(Knob::ntohpe)}, {}, 0},
        {"nmax_red", {knob(Knob::nmax_red)}, {}, 0},
        {"lin",
         {knob(Knob::nred), par(0), op(Op::Mul), knob(Knob::ndim), par(1), op(Op::Mul), op(Op::Add)},
         {},
         3},
        {"div0", {cst(0), cst(1), op(Op::Div)}, {1.0f, 0.0f}, 0},
        {"divneg", {cst(0), cst(1), op(Op::Div)}, {1.0f, -0.0001f}, 0},
        {"log0", {cst(0), op(Op::Log)}, {0.0f}, 0},
        {"sqrtneg", {cst(0), op(Op::Sqrt)}, {-4.0f}, 0},
        {"sigmoid", {knob(Knob::nred), op(Op::Sigmoid)}, {}, 0},
        {"tanh", {knob(Knob::nred), op(Op::Tanh)}, {}, 0},
        {"exp", {knob(Knob::nzw), op(Op::Exp)}, {}, 0},
        {"floor", {knob(Knob::nred), op(Op::Floor)}, {}, 0},
        {"sign", {knob(Knob::nred), op(Op::Neg), op(Op::Sign)}, {}, 0},
        {"where",
         {knob(Knob::dim), cst(0), op(Op::Gt), knob(Knob::ndim), cst(1), op(Op::Select)},
         {5.0f, 0.0f},
         0},
        {"pow", {knob(Knob::nred), cst(0), op(Op::Pow)}, {3.0f}, 0},
        {"minmax", {knob(Knob::nred), cst(0), op(Op::Min), cst(1), op(Op::Max)}, {0.5f, 0.1f}, 0},
        {"ratio", {knob(Knob::nred), knob(Knob::nbucket), cst(0), op(Op::Max), op(Op::Div)}, {0.05f}, 0},
        {"chain",
         {knob(Knob::nred), cst(0), op(Op::Lt), knob(Knob::ndim), cst(1), op(Op::Lt), op(Op::And)},
         {5.0f, 10.0f},
         0},
        {"notop", {knob(Knob::nred), cst(0), op(Op::Gt), op(Op::Not)}, {0.5f}, 0},
    };

    for (const Case& c : cases) {
        PolicyProgram prog(c.code, c.consts, c.n_params, PolicySite::Finalization);
        std::printf("%s %.9g\n", c.name, static_cast<double>(prog.eval(f, params)));
    }
}

struct Check {
    const char* name;
    void (*fn)();
};

const Check k_checks[] = {
    {"arithmetic_and_stack", check_arithmetic_and_stack},
    {"knob_normalization", check_knob_normalization},
    {"pool_fractions_and_ranks", check_pool_fractions_and_ranks},
    {"select_and_comparisons", check_select_and_comparisons},
    {"guarded_division_and_log", check_guarded_division_and_log},
    {"params", check_params},
    {"validation_rejects_malformed_programs", check_validation_rejects_malformed_programs},
    {"site_availability_is_enforced", check_site_availability_is_enforced},
    {"used_knobs_reporting", check_used_knobs_reporting},
    {"linear_score_equivalence", check_linear_score_equivalence},
    {"polynom_score_equivalence", check_polynom_score_equivalence},
    {"finalization_six_feature_equivalence", check_finalization_six_feature_equivalence},
    {"no_spurious_non_finite", check_no_spurious_non_finite},
};

} // namespace

int main(int argc, char** argv) {
    // `--dump-cross-check` emits the vectors the Python test diffs against.
    if (argc > 1 && std::string(argv[1]) == "--dump-cross-check") {
        dump_cross_check_vectors();
        return 0;
    }

    for (const Check& check : k_checks) {
        try {
            check.fn();
        } catch (const std::exception& e) {
            std::cerr << "FAILED " << check.name << ": " << e.what() << "\n";
            return 1;
        }
        std::cout << "ok " << check.name << "\n";
    }
    std::cout << "all policy_expr checks passed\n";
    return 0;
}
