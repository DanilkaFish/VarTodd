#include "policy_expr.hpp"

#include <algorithm>
#include <optional>

namespace todd {
namespace {

struct KnobInfo {
    Knob        knob;
    const char* name;
};

// Indexed by Knob value; kept in enum order.
constexpr std::array<KnobInfo, k_knob_count> k_knob_table = {{
    {Knob::red, "red"},
    {Knob::nred, "nred"},
    {Knob::dim, "dim"},
    {Knob::ndim, "ndim"},
    {Knob::bucket, "bucket"},
    {Knob::nbucket, "nbucket"},
    {Knob::yw, "yw"},
    {Knob::nyw, "nyw"},
    {Knob::zw, "zw"},
    {Knob::nzw, "nzw"},
    {Knob::zsize, "zsize"},
    {Knob::max_red, "max_red"},
    {Knob::nmax_red, "nmax_red"},
    {Knob::tohpe, "tohpe"},
    {Knob::ntohpe, "ntohpe"},
    {Knob::rank_red, "rank_red"},
    {Knob::nrank_red, "nrank_red"},
    {Knob::rank_dim, "rank_dim"},
    {Knob::nrank_dim, "nrank_dim"},
    {Knob::rank_score, "rank_score"},
    {Knob::nrank_score, "nrank_score"},
    {Knob::pool_size, "pool_size"},
    {Knob::pool_tohpe, "pool_tohpe"},
    {Knob::pool_prefix, "pool_prefix"},
    {Knob::pool_todd, "pool_todd"},
    {Knob::f_tohpe, "f_tohpe"},
    {Knob::f_prefix, "f_prefix"},
    {Knob::f_todd, "f_todd"},
    {Knob::source, "source"},
    {Knob::bucket_id, "bucket_id"},
    {Knob::k_idx, "k_idx"},
    {Knob::l_idx, "l_idx"},
    {Knob::bn, "bn"},
    {Knob::dn, "dn"},
    {Knob::wvwn, "wvwn"},
}};

// Divide with the same floor the legacy polynom scoring used, so a zero
// denominator produces a large finite number rather than an infinity that
// would poison candidate ordering.
inline float guarded_div(float a, float b) noexcept {
    const float d = std::abs(b) < k_div_epsilon ? std::copysign(k_div_epsilon, b == 0.0f ? 1.0f : b) : b;
    return a / d;
}

inline float knob_value(const KnobFrame& f, Knob knob) noexcept {
    switch (knob) {
    case Knob::red:
        return f.red;
    case Knob::nred:
        return f.red / f.bn / 2.0f;
    case Knob::dim:
        return f.dim;
    case Knob::ndim:
        return f.dim / f.dn;
    case Knob::bucket:
        return f.bucket;
    case Knob::nbucket:
        return f.bucket / f.bn;
    case Knob::yw:
        return f.yw;
    case Knob::nyw:
        return f.yw / f.wvwn;
    case Knob::zw:
        return f.zw;
    case Knob::nzw:
        return f.zw / std::max(1.0f, f.zsize);
    case Knob::zsize:
        return f.zsize;
    case Knob::max_red:
        return f.max_red;
    case Knob::nmax_red:
        return f.max_red / f.bn / 2.0f;
    case Knob::tohpe:
        return f.tohpe;
    case Knob::ntohpe:
        return f.tohpe / f.dn;
    case Knob::rank_red:
        return f.rank_red;
    case Knob::nrank_red:
        return f.rank_red / std::max(1.0f, f.pool_size);
    case Knob::rank_dim:
        return f.rank_dim;
    case Knob::nrank_dim:
        return f.rank_dim / std::max(1.0f, f.pool_size);
    case Knob::rank_score:
        return f.rank_score;
    case Knob::nrank_score:
        return f.rank_score / std::max(1.0f, f.pool_size);
    case Knob::pool_size:
        return f.pool_size;
    case Knob::pool_tohpe:
        return f.pool_tohpe;
    case Knob::pool_prefix:
        return f.pool_prefix;
    case Knob::pool_todd:
        return f.pool_todd;
    case Knob::f_tohpe:
        return f.pool_tohpe / std::max(1.0f, f.pool_size);
    case Knob::f_prefix:
        return f.pool_prefix / std::max(1.0f, f.pool_size);
    case Knob::f_todd:
        return f.pool_todd / std::max(1.0f, f.pool_size);
    case Knob::source:
        return f.source;
    case Knob::bucket_id:
        return f.bucket_id;
    case Knob::k_idx:
        return f.k_idx;
    case Knob::l_idx:
        return f.l_idx;
    case Knob::bn:
        return f.bn;
    case Knob::dn:
        return f.dn;
    case Knob::wvwn:
        return f.wvwn;
    case Knob::count_:
        break;
    }
    return 0.0f;
}

// Net stack effect of an opcode: pushes minus pops.
struct StackEffect {
    int pops;
    int pushes;
};

constexpr StackEffect op_stack_effect(Op op) noexcept {
    switch (op) {
    case Op::LoadKnob:
    case Op::LoadConst:
    case Op::LoadParam:
        return {0, 1};
    case Op::Add:
    case Op::Sub:
    case Op::Mul:
    case Op::Div:
    case Op::Min:
    case Op::Max:
    case Op::Pow:
    case Op::Lt:
    case Op::Le:
    case Op::Gt:
    case Op::Ge:
    case Op::Eq:
    case Op::And:
    case Op::Or:
        return {2, 1};
    case Op::Neg:
    case Op::Abs:
    case Op::Exp:
    case Op::Log:
    case Op::Sqrt:
    case Op::Tanh:
    case Op::Sigmoid:
    case Op::Floor:
    case Op::Sign:
    case Op::Not:
        return {1, 1};
    case Op::Select:
        return {3, 1};
    }
    return {0, 0};
}

KnobMask make_mask(std::initializer_list<Knob> knobs) {
    KnobMask mask;
    for (Knob k : knobs)
        mask.set(static_cast<std::size_t>(k));
    return mask;
}


} // namespace

const char* knob_name(Knob knob) noexcept {
    const auto index = static_cast<std::size_t>(knob);
    if (index >= k_knob_count)
        return "<invalid>";
    return k_knob_table[index].name;
}

const char* policy_site_name(PolicySite site) noexcept {
    switch (site) {
    case PolicySite::ExplorationZ:
        return "exploration (inner z)";
    case PolicySite::ExplorationPool:
        return "exploration (pool)";
    case PolicySite::Finalization:
        return "final";
    }
    return "<invalid>";
}

KnobMask site_available_knobs(PolicySite site) noexcept {
    // Knobs known before finalization at either exploration site.
    static const KnobMask exploration = make_mask({
        Knob::red, Knob::nred, Knob::dim, Knob::ndim, Knob::bucket, Knob::nbucket, Knob::yw, Knob::nyw,
        Knob::zw, Knob::nzw, Knob::zsize, Knob::max_red, Knob::nmax_red, Knob::source, Knob::bucket_id,
        Knob::bn, Knob::dn, Knob::wvwn,
    });
    // Finalization additionally sees tohpe_dim, the SeenValues ranks, the pool
    // composition and the k/l indices.
    static const KnobMask finalization = exploration | make_mask({
        Knob::tohpe, Knob::ntohpe, Knob::rank_red, Knob::nrank_red, Knob::rank_dim, Knob::nrank_dim,
        Knob::rank_score, Knob::nrank_score, Knob::pool_size, Knob::pool_tohpe, Knob::pool_prefix,
        Knob::pool_todd, Knob::f_tohpe, Knob::f_prefix, Knob::f_todd, Knob::k_idx, Knob::l_idx,
    });

    switch (site) {
    case PolicySite::ExplorationZ:
    case PolicySite::ExplorationPool:
        return exploration;
    case PolicySite::Finalization:
        return finalization;
    }
    return KnobMask{};
}

PolicyProgram::PolicyProgram(std::vector<Instr> code, std::vector<float> consts, std::size_t n_params,
                             PolicySite site)
    : code_(std::move(code)), consts_(std::move(consts)), n_params_(n_params), site_(site) {
    validate_();
}


void PolicyProgram::validate_() {
    if (code_.empty())
        throw PolicyProgramError("policy program is empty");

    const KnobMask available = site_available_knobs(site_);

    int depth     = 0;
    int max_depth = 0;
    for (std::size_t i = 0; i < code_.size(); ++i) {
        const Instr& in = code_[i];

        switch (in.op) {
        case Op::LoadKnob: {
            if (in.arg >= k_knob_count)
                throw PolicyProgramError("instruction " + std::to_string(i) + ": knob index " +
                                         std::to_string(in.arg) + " out of range");
            const auto knob = static_cast<Knob>(in.arg);
            if (!available.test(in.arg))
                throw PolicyProgramError(std::string("knob '") + knob_name(knob) + "' is not available at the " +
                                         policy_site_name(site_) + " site");
            used_.set(in.arg);
            break;
        }
        case Op::LoadConst:
            if (in.arg >= consts_.size())
                throw PolicyProgramError("instruction " + std::to_string(i) + ": const index " +
                                         std::to_string(in.arg) + " out of range");
            break;
        case Op::LoadParam:
            if (in.arg >= n_params_)
                throw PolicyProgramError("instruction " + std::to_string(i) + ": param index " +
                                         std::to_string(in.arg) + " exceeds declared parameter count " +
                                         std::to_string(n_params_));
            break;
        default:
            break;
        }

        const auto [pops, pushes] = op_stack_effect(in.op);
        if (depth < pops)
            throw PolicyProgramError("instruction " + std::to_string(i) + ": stack underflow (needs " +
                                     std::to_string(pops) + ", has " + std::to_string(depth) + ")");
        depth += pushes - pops;
        max_depth = std::max(max_depth, depth);
    }

    if (depth != 1)
        throw PolicyProgramError("policy program must leave exactly one value on the stack, left " +
                                 std::to_string(depth));
    if (static_cast<std::size_t>(max_depth) > k_max_stack_depth)
        throw PolicyProgramError("policy program exceeds the maximum stack depth of " +
                                 std::to_string(k_max_stack_depth));
}

float PolicyProgram::eval(const KnobFrame& f, std::span<const float> params) const noexcept {
    std::array<float, k_max_stack_depth> stack{};
    std::size_t                          sp = 0;

    // Bounds were proven in validate_(), so the loop carries no checks.
    for (const Instr& in : code_) {
        switch (in.op) {
        case Op::LoadKnob:
            stack[sp++] = knob_value(f, static_cast<Knob>(in.arg));
            break;
        case Op::LoadConst:
            stack[sp++] = consts_[in.arg];
            break;
        case Op::LoadParam:
            stack[sp++] = in.arg < params.size() ? params[in.arg] : 0.0f;
            break;

        case Op::Add:
            stack[sp - 2] = stack[sp - 2] + stack[sp - 1];
            --sp;
            break;
        case Op::Sub:
            stack[sp - 2] = stack[sp - 2] - stack[sp - 1];
            --sp;
            break;
        case Op::Mul:
            stack[sp - 2] = stack[sp - 2] * stack[sp - 1];
            --sp;
            break;
        case Op::Div:
            stack[sp - 2] = guarded_div(stack[sp - 2], stack[sp - 1]);
            --sp;
            break;
        case Op::Min:
            stack[sp - 2] = std::min(stack[sp - 2], stack[sp - 1]);
            --sp;
            break;
        case Op::Max:
            stack[sp - 2] = std::max(stack[sp - 2], stack[sp - 1]);
            --sp;
            break;
        case Op::Pow:
            stack[sp - 2] = std::pow(stack[sp - 2], stack[sp - 1]);
            --sp;
            break;

        case Op::Neg:
            stack[sp - 1] = -stack[sp - 1];
            break;
        case Op::Abs:
            stack[sp - 1] = std::abs(stack[sp - 1]);
            break;
        case Op::Exp:
            stack[sp - 1] = std::exp(stack[sp - 1]);
            break;
        case Op::Log:
            stack[sp - 1] = std::log(std::max(k_log_floor, stack[sp - 1]));
            break;
        case Op::Sqrt:
            stack[sp - 1] = std::sqrt(std::max(0.0f, stack[sp - 1]));
            break;
        case Op::Tanh:
            stack[sp - 1] = std::tanh(stack[sp - 1]);
            break;
        case Op::Sigmoid:
            stack[sp - 1] = 1.0f / (1.0f + std::exp(-stack[sp - 1]));
            break;
        case Op::Floor:
            stack[sp - 1] = std::floor(stack[sp - 1]);
            break;
        case Op::Sign:
            stack[sp - 1] = stack[sp - 1] > 0.0f ? 1.0f : (stack[sp - 1] < 0.0f ? -1.0f : 0.0f);
            break;

        case Op::Lt:
            stack[sp - 2] = stack[sp - 2] < stack[sp - 1] ? 1.0f : 0.0f;
            --sp;
            break;
        case Op::Le:
            stack[sp - 2] = stack[sp - 2] <= stack[sp - 1] ? 1.0f : 0.0f;
            --sp;
            break;
        case Op::Gt:
            stack[sp - 2] = stack[sp - 2] > stack[sp - 1] ? 1.0f : 0.0f;
            --sp;
            break;
        case Op::Ge:
            stack[sp - 2] = stack[sp - 2] >= stack[sp - 1] ? 1.0f : 0.0f;
            --sp;
            break;
        case Op::Eq:
            stack[sp - 2] = stack[sp - 2] == stack[sp - 1] ? 1.0f : 0.0f;
            --sp;
            break;
        case Op::And:
            stack[sp - 2] = (stack[sp - 2] != 0.0f && stack[sp - 1] != 0.0f) ? 1.0f : 0.0f;
            --sp;
            break;
        case Op::Or:
            stack[sp - 2] = (stack[sp - 2] != 0.0f || stack[sp - 1] != 0.0f) ? 1.0f : 0.0f;
            --sp;
            break;
        case Op::Not:
            stack[sp - 1] = stack[sp - 1] == 0.0f ? 1.0f : 0.0f;
            break;

        case Op::Select:
            // (cond, a, b) with b on top
            stack[sp - 3] = stack[sp - 3] != 0.0f ? stack[sp - 2] : stack[sp - 1];
            sp -= 2;
            break;
        }
    }

    return stack[0];
}

} // namespace todd
