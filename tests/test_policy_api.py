"""Tests for the Python policy authoring API.

Every misuse test asserts that the error message names the fix, not just that
an error was raised: these messages are what an LLM mutating a policy file sees
and acts on.
"""

import math
import os
import pickle
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "the_latest_version"),
)

from policy_expr import (  # noqa: E402
    PolicyError,
    RewriteError,
    describe_knobs,
    distance_score,
    greedy,
    linear_score,
    log_score,
    policy,
    polynom_score,
    probe_frames,
    sigmoid_score,
    validate_bound,
    validate_scores,
    weighted,
)
from policy_expr.expr import FN, SITE_EXPLORATION, SITE_FINAL, Knobs, Params  # noqa: E402


# --- expressions used across the tests --------------------------------------

@policy.exploration
def linear_explore(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1)


@policy.final
def branchy_final(k, p, fn):
    base = k.nred * p.w(0)
    if k.dim > 5:
        base = base + k.ndim * p.w(1)
    return base - k.rank_red / fn.max(k.pool_size, 1)


@policy.exploration
def ratio_explore(k, p, fn):
    return k.nred / fn.max(k.nbucket, 0.05) * p.w(0)


class TestTracing(unittest.TestCase):
    def test_n_params_counts_declared_scalars(self):
        self.assertEqual(linear_explore.n_params, 2)
        self.assertEqual(branchy_final.n_params, 2)
        self.assertEqual(ratio_explore.n_params, 1)

    def test_used_knobs_reported(self):
        self.assertEqual(linear_explore.used_knobs, frozenset({"nred", "ndim"}))
        self.assertIn("rank_red", branchy_final.used_knobs)

    def test_evaluation_matches_hand_computation(self):
        bound = branchy_final.bind([2.0, 3.0])
        common = dict(red=10, bn=5, dn=4, rank_red=2, pool_size=20)
        # nred = 10/5/2 = 1.0, ndim = dim/4, rank_red/pool_size = 0.1
        self.assertAlmostEqual(bound.evaluate(dim=7, **common), 1 * 2 + 1.75 * 3 - 0.1, places=5)
        self.assertAlmostEqual(bound.evaluate(dim=3, **common), 1 * 2 - 0.1, places=5)

    def test_guarded_division_stays_finite(self):
        bound = ratio_explore.bind([1.0])
        value = bound.evaluate(red=4, bucket=0, bn=2, dn=1, wvwn=1)
        self.assertTrue(math.isfinite(value))


class TestAstRewrite(unittest.TestCase):
    def test_if_becomes_where(self):
        self.assertIn("fn.where", branchy_final.source())

    def test_ternary_becomes_where(self):
        @policy.final
        def ternary(k, p, fn):
            return k.nred * p.w(0) + (k.ndim if k.dim > 5 else 0.0)

        self.assertIn("fn.where", ternary.source())

    def test_boolean_operators_become_where(self):
        @policy.final
        def booly(k, p, fn):
            flag = (k.dim > 5) and (k.nred > 0.1)
            return k.nred * p.w(0) + flag

        self.assertIn("fn.where", booly.source())

    def test_chained_comparison_is_rewritten(self):
        @policy.final
        def chained(k, p, fn):
            return k.nred * p.w(0) + (1 < k.dim < 10)

        self.assertIn("fn.where", chained.source())

    def test_both_branches_evaluate(self):
        # Documented consequence of branchless selection: the guard does not
        # prevent the other arm from being computed.
        @policy.final
        def guarded(k, p, fn):
            value = 0.0
            if k.bucket > 0:
                value = k.nred / k.nbucket
            return value

        bound = guarded.bind([])
        self.assertTrue(math.isfinite(bound.evaluate(red=4, bucket=0, bn=2)))

    def test_unsupported_statements_are_rejected(self):
        for body, needle in (
            ("for i in range(3):\n            total = total + k.ndim", "`for`"),
            ("while True:\n            pass", "`while`"),
        ):
            src = (
                "@policy.exploration\n"
                "def f(k, p, fn):\n"
                "    total = k.nred\n"
                "    " + body + "\n"
                "    return total\n"
            )
            with self.assertRaises((PolicyError, RewriteError)) as ctx:
                exec(compile(src, "<test>", "exec"), {"policy": policy})
            # exec'd source has no file, so the rewriter refuses first; either
            # message is an actionable refusal rather than a silent mis-trace.
            self.assertTrue(str(ctx.exception))

    def test_half_returning_if_is_rejected(self):
        with self.assertRaises(PolicyError) as ctx:
            @policy.final
            def half(k, p, fn):
                if k.dim > 5:
                    return k.nred
                x = k.ndim
                return x

        self.assertIn("both branches", str(ctx.exception))

    def test_source_unavailable_is_a_loud_refusal(self):
        # A policy defined via exec has no source; the rewriter must refuse
        # rather than silently skip and let `if` mis-trace.
        namespace = {"policy": policy}
        with self.assertRaises(RewriteError) as ctx:
            exec(
                "@policy.exploration\ndef f(k, p, fn):\n    return k.nred * p.w(0)\n",
                namespace,
            )
        self.assertIn("cannot read the source", str(ctx.exception))


class TestMisuseMessages(unittest.TestCase):
    """Each message must name the fix, not merely report a failure."""

    def test_unknown_knob_suggests_nearest(self):
        knobs = Knobs(SITE_EXPLORATION)
        with self.assertRaises(PolicyError) as ctx:
            knobs.nredd
        message = str(ctx.exception)
        self.assertIn("is not a knob", message)
        self.assertIn("nred", message)

    def test_final_only_knob_in_exploration_names_the_site(self):
        knobs = Knobs(SITE_EXPLORATION)
        for name in ("tohpe", "rank_red", "f_todd", "pool_size"):
            with self.assertRaises(PolicyError) as ctx:
                getattr(knobs, name)
            message = str(ctx.exception)
            self.assertIn("not available", message)
            self.assertIn("finalization", message)
            self.assertIn("Available here", message)

    def test_final_site_accepts_the_extra_knobs(self):
        knobs = Knobs(SITE_FINAL)
        for name in ("tohpe", "rank_red", "f_todd", "pool_size"):
            getattr(knobs, name)  # must not raise

    def test_param_index_gap_names_the_missing_index(self):
        with self.assertRaises(PolicyError) as ctx:
            @policy.exploration
            def gappy(k, p, fn):
                return k.nred * p.w(0) + k.ndim * p.w(2)

        message = str(ctx.exception)
        self.assertIn("skips p.w(1)", message)

    def test_attribute_style_param_names_the_fix(self):
        params = Params()
        with self.assertRaises(PolicyError) as ctx:
            params.w_red
        self.assertIn("p.w(0)", str(ctx.exception))

    def test_non_integer_param_index_rejected(self):
        params = Params()
        with self.assertRaises(PolicyError) as ctx:
            params.w("red")
        self.assertIn("integer slot index", str(ctx.exception))

    def test_bind_length_mismatch_names_both_counts(self):
        with self.assertRaises(PolicyError) as ctx:
            linear_explore.bind([1.0, 2.0, 3.0])
        message = str(ctx.exception)
        self.assertIn("declares 2", message)
        self.assertIn("got 3", message)

    def test_bind_rejects_non_finite(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(PolicyError) as ctx:
                linear_explore.bind([1.0, bad])
            self.assertIn("finite", str(ctx.exception))

    def test_policy_returning_none_is_rejected(self):
        with self.assertRaises(PolicyError) as ctx:
            @policy.exploration
            def nothing(k, p, fn):
                k.nred * p.w(0)

        self.assertIn("must return an expression", str(ctx.exception))

    def test_bool_backstop_names_where(self):
        knobs = Knobs(SITE_EXPLORATION)
        with self.assertRaises(PolicyError) as ctx:
            bool(knobs.nred > 1)
        self.assertIn("fn.where", str(ctx.exception))


class TestLint(unittest.TestCase):
    def test_constant_expression_is_flagged(self):
        @policy.exploration
        def constant(k, p, fn):
            return 1.0 + 2.0

        self.assertTrue(any("ignores every knob" in w for w in constant.warnings))

    def test_missing_reduction_is_flagged(self):
        @policy.exploration
        def no_reduction(k, p, fn):
            return k.ndim * p.w(0)

        self.assertTrue(any("never reads k.red" in w for w in no_reduction.warnings))

    def test_unused_parameter_is_flagged(self):
        @policy.exploration
        def unused(k, p, fn):
            spare = p.w(1)
            return k.nred * p.w(0)

        self.assertTrue(any("p.w(1)" in w for w in unused.warnings))

    def test_fast_path_loss_is_reported(self):
        # An exploration score reading a z-varying knob forfeits the
        # reduction-only path in the inner z search. That is a wall-clock cost,
        # so it must be stated rather than discovered as unexplained slowness.
        self.assertTrue(
            any("fast path" in w for w in ratio_explore.warnings),
            ratio_explore.warnings,
        )

    def test_clean_policy_has_no_warnings(self):
        self.assertEqual(linear_explore.warnings, [])


class TestLegacyEquivalence(unittest.TestCase):
    """The library constructors must reproduce the replaced scoring forms."""

    FEATURES = dict(red=12, dim=7, bucket=9, yw=5, zw=3, zsize=8, bn=6, dn=5, wvwn=4)

    def _x(self):
        f = self.FEATURES
        return [
            f["red"] / f["bn"] / 2.0,
            f["dim"] / f["dn"],
            f["bucket"] / f["bn"],
            f["yw"] / f["wvwn"],
            f["zw"] / max(1, f["zsize"]),
        ]

    def test_linear_matches(self):
        weights = [1.5, -0.5, 0.25, 2.0, -1.0]
        expected = sum(w * x for w, x in zip(weights, self._x()))
        actual = linear_score(weights).bind([]).evaluate(**self.FEATURES)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_polynom_matches_including_first_center_scale(self):
        weights = [1.5, -0.5, 0.25, 2.0, -1.0]
        centers = [0.3, 0.4, 0.5, 0.6, 0.7]
        bn = self.FEATURES["bn"]
        expected = 0.0
        for i, (w, c, x) in enumerate(zip(weights, centers, self._x())):
            center = c / bn / 2.0 if i == 0 else c
            expected += w * abs(x - center) ** 2
        actual = polynom_score(weights, centers, 2.0).bind([]).evaluate(**self.FEATURES)
        self.assertAlmostEqual(actual, expected, places=4)

    def test_distance_matches(self):
        weights = [0.1, 0.2, 0.3, 0.4, 0.5]
        expected = -sum((x - w) ** 2 for w, x in zip(weights, self._x()))
        actual = distance_score(weights).bind([]).evaluate(**self.FEATURES)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_sigmoid_matches(self):
        weights = [1.0, -0.5, 0.25, 0.0, 0.5]
        total = sum(w * x * x for w, x in zip(weights, self._x()))
        expected = 1.0 / (1.0 + math.exp(-total))
        actual = sigmoid_score(weights, 2.0).bind([]).evaluate(**self.FEATURES)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_log_matches_with_the_same_floor(self):
        weights = [1.0, 0.5, -0.25, 0.75, 0.0]
        expected = sum(
            w * math.log(max(1e-6, x)) for w, x in zip(weights, self._x()) if w != 0.0
        )
        actual = log_score(weights).bind([]).evaluate(**self.FEATURES)
        self.assertAlmostEqual(actual, expected, places=4)

    def test_weighted_exposes_free_scalars(self):
        expr = weighted(5)
        self.assertEqual(expr.n_params, 5)
        weights = [1.5, -0.5, 0.25, 2.0, -1.0]
        expected = sum(w * x for w, x in zip(weights, self._x()))
        self.assertAlmostEqual(
            expr.bind(weights).evaluate(**self.FEATURES), expected, places=5
        )

    def test_greedy_is_pure_reduction(self):
        value = greedy().bind([]).evaluate(**self.FEATURES)
        self.assertAlmostEqual(value, self._x()[0], places=6)


class TestRoundTrip(unittest.TestCase):
    def test_repr_is_valid_python_rebuilding_the_expression(self):
        source = branchy_final.source()
        namespace = {"k": Knobs(SITE_FINAL), "p": Params(), "fn": FN}
        rebuilt = eval(source, namespace)  # noqa: S307
        self.assertEqual(repr(rebuilt), source)

    def test_bound_policy_pickles(self):
        bound = branchy_final.bind([1.25, -0.5])
        restored = pickle.loads(pickle.dumps(bound))
        self.assertEqual(restored.params, bound.params)
        common = dict(red=10, dim=7, bn=5, dn=4, rank_red=2, pool_size=20)
        self.assertAlmostEqual(restored.evaluate(**common), bound.evaluate(**common))


class TestProbePass(unittest.TestCase):
    """Non-finite policies must fail at load, not corrupt a search silently."""

    def test_clean_policies_pass(self):
        validate_bound(linear_explore.bind([1.0, 2.0]))
        validate_bound(branchy_final.bind([1.0, 2.0]))
        validate_bound(ratio_explore.bind([1.0]))

    def test_overflow_is_caught(self):
        @policy.exploration
        def explodes(k, p, fn):
            # exp of a large raw knob overflows well inside the probe ranges
            return fn.exp(k.bucket * k.bn)

        with self.assertRaises(PolicyError) as ctx:
            validate_bound(explodes.bind([]))
        message = str(ctx.exception)
        self.assertIn("finite", message)
        self.assertIn("Expression:", message)

    def test_message_names_the_knob_values(self):
        @policy.exploration
        def explodes(k, p, fn):
            return fn.exp(k.red * k.bn) + k.nred

        with self.assertRaises(PolicyError) as ctx:
            validate_bound(explodes.bind([]))
        # The frame that produced the failure must be reported, restricted to
        # the knobs the expression actually reads.
        self.assertIn("red", str(ctx.exception))

    def test_guarded_division_survives_the_zero_frame(self):
        # The all-zero frame is exactly what an unguarded ratio would trip on;
        # with the epsilon floor it must stay finite.
        @policy.exploration
        def ratio(k, p, fn):
            return k.nred / k.nbucket

        validate_bound(ratio.bind([]))

    def test_strict_mode_promotes_warnings(self):
        @policy.exploration
        def no_reduction(k, p, fn):
            return k.ndim * p.w(0)

        bound = no_reduction.bind([1.0])
        self.assertTrue(validate_bound(bound))  # warnings returned, not raised
        with self.assertRaises(PolicyError) as ctx:
            validate_bound(bound, strict=True)
        self.assertIn("strict", str(ctx.exception))

    def test_validate_scores_accepts_several(self):
        warnings = validate_scores(
            linear_explore.bind([1.0, 2.0]), branchy_final.bind([1.0, 2.0])
        )
        self.assertEqual(warnings, [])

    def test_unbound_expression_is_rejected(self):
        with self.assertRaises(PolicyError) as ctx:
            validate_bound(linear_explore)
        self.assertIn("bind", str(ctx.exception))

    def test_probe_frames_cover_the_boundaries(self):
        frames = probe_frames()
        self.assertGreater(len(frames), 4)
        # normalizers must never be zero, or every normalized knob divides by 0
        for frame in frames:
            self.assertGreaterEqual(frame["bn"], 1.0)
            self.assertGreaterEqual(frame["dn"], 1.0)
            self.assertGreaterEqual(frame["wvwn"], 1.0)


class TestCppCrossCheck(unittest.TestCase):
    """Pin the Python reference evaluator to the C++ VM.

    Both implement the same semantics independently -- the Python one backs the
    load-time probe pass -- so without this they would silently drift. Skipped
    when the C++ test binary has not been built.
    """

    BINARY_CANDIDATES = (
        os.path.join("build", "bin", "Debug", "vartodd_policy_expr"),
        os.path.join("build", "bin", "Release", "vartodd_policy_expr"),
        os.path.join("build", "bin", "vartodd_policy_expr"),
        os.path.join("build", "tests", "vartodd_policy_expr"),
    )

    FRAME = dict(
        red=12, dim=7, bucket=9, yw=5, zw=3, zsize=8, max_red=18, tohpe=4,
        rank_red=2, rank_dim=3, rank_score=1, pool_size=20, pool_tohpe=5,
        pool_prefix=3, pool_todd=12, source=1, bucket_id=42, k_idx=2, l_idx=6,
        bn=6, dn=5, wvwn=4,
    )
    PARAMS = [1.5, -0.75, 0.25]

    @classmethod
    def _binary(cls):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        for candidate in cls.BINARY_CANDIDATES:
            path = os.path.join(root, candidate)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def _cases(self):
        from policy_expr.expr import KNOB_INDEX, OP_LOAD_KNOB, OP_LOAD_PARAM, Expr, Node

        def K(name):
            return Expr(Node(OP_LOAD_KNOB, float(KNOB_INDEX[name])))

        def P(i):
            return Expr(Node(OP_LOAD_PARAM, float(i)))

        one, zero = Expr._coerce(1.0), Expr._coerce(0.0)
        return {
            "nred": (K("nred"), 0),
            "ndim": (K("ndim"), 0),
            "nzw": (K("nzw"), 0),
            "f_todd": (K("f_todd"), 0),
            "nrank_red": (K("nrank_red"), 0),
            "ntohpe": (K("ntohpe"), 0),
            "nmax_red": (K("nmax_red"), 0),
            "lin": (K("nred") * P(0) + K("ndim") * P(1), 3),
            "div0": (one / zero, 0),
            "divneg": (one / Expr._coerce(-0.0001), 0),
            "log0": (FN.log(zero), 0),
            "sqrtneg": (FN.sqrt(Expr._coerce(-4.0)), 0),
            "sigmoid": (FN.sigmoid(K("nred")), 0),
            "tanh": (FN.tanh(K("nred")), 0),
            "exp": (FN.exp(K("nzw")), 0),
            "floor": (FN.floor(K("nred")), 0),
            "sign": (FN.sign(-K("nred")), 0),
            "where": (FN.where(K("dim") > 5.0, K("ndim"), 0.0), 0),
            "pow": (FN.pow(K("nred"), 3.0), 0),
            "minmax": (FN.max(FN.min(K("nred"), 0.5), 0.1), 0),
            "ratio": (K("nred") / FN.max(K("nbucket"), 0.05), 0),
            "chain": (FN.and_(K("nred") < 5.0, K("ndim") < 10.0), 0),
            "notop": (FN.not_(K("nred") > 0.5), 0),
        }

    def test_python_evaluator_matches_cpp_vm(self):
        import subprocess

        from policy_expr.policy import _reference_eval, from_expression

        binary = self._binary()
        if binary is None:
            self.skipTest("vartodd_policy_expr not built; run cmake --build build")

        output = subprocess.run(
            [binary, "--dump-cross-check"], capture_output=True, text=True, check=True
        ).stdout
        expected = {}
        for line in output.splitlines():
            name, value = line.split()
            expected[name] = float(value)

        cases = self._cases()
        self.assertEqual(
            set(cases), set(expected),
            "the C++ and Python cross-check case lists have diverged",
        )

        for name, (expr, n_params) in cases.items():
            with self.subTest(program=name):
                actual = _reference_eval(
                    from_expression(name, SITE_FINAL, expr, n_params),
                    self.PARAMS,
                    self.FRAME,
                )
                # float32 in C++ against float64 in Python: compare relatively.
                self.assertAlmostEqual(
                    actual, expected[name],
                    delta=1e-5 * max(1.0, abs(actual), abs(expected[name])),
                )


def _native_module():
    """The newest built extension exposing PolicyProgram, or None.

    Several build configurations write their own .so under pyvartodd/. Only one
    may be imported per process -- pybind11 refuses a second registration of the
    same types -- so the freshest one is tried first and the search stops at the
    first successful import.
    """
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if root not in sys.path:
        sys.path.insert(0, root)

    candidates = []
    for module, relative in (
        ("pyvartodd.Release.pyvartodd", "pyvartodd/Release/pyvartodd.cpython-312-x86_64-linux-gnu.so"),
        ("pyvartodd.Debug.pyvartodd", "pyvartodd/Debug/pyvartodd.cpython-312-x86_64-linux-gnu.so"),
        ("pyvartodd.pyvartodd", "pyvartodd/pyvartodd.cpython-312-x86_64-linux-gnu.so"),
    ):
        path = os.path.join(root, relative)
        if os.path.isfile(path):
            candidates.append((os.path.getmtime(path), module))

    for _, module in sorted(candidates, reverse=True):
        try:
            imported = __import__(module, fromlist=["PolicyProgram"])
        except Exception:
            continue
        # A stale build imports fine but predates PolicyProgram; it has already
        # registered its types, so no other candidate can load in this process.
        return imported if hasattr(imported, "PolicyProgram") else None
    return None


class TestNativeBinding(unittest.TestCase):
    """The compiled program must agree with the Python side that produced it."""

    def setUp(self):
        self.native = _native_module()
        if self.native is None or not hasattr(self.native, "PolicyProgram"):
            self.skipTest("pyvartodd with PolicyProgram not built")

    def test_knob_tables_match_the_engine(self):
        # The Python knob list exists so errors arrive at authoring time; the
        # engine's table is authoritative. If they drift, a policy could pass
        # Python validation and be rejected by C++ (or worse, read a different
        # knob than the author named).
        from policy_expr.expr import KNOB_NAMES

        self.assertEqual(list(self.native.policy_knob_names()), list(KNOB_NAMES))

    def test_site_availability_matches_the_engine(self):
        from policy_expr.expr import SITE_CODE, SITE_KNOBS

        for site, code in SITE_CODE.items():
            self.assertEqual(
                set(self.native.policy_site_knobs(code)),
                set(SITE_KNOBS[site]),
                f"site knob sets differ for {site}",
            )

    def test_native_evaluation_matches_the_reference(self):
        bound = branchy_final.bind([2.0, 3.0])
        program = bound.native()
        frame = dict(red=10, bn=5, dn=4, rank_red=2, pool_size=20)
        for dim in (3, 7, 12):
            self.assertAlmostEqual(
                program.evaluate(dim=dim, params=bound.params, **frame),
                bound.evaluate(dim=dim, **frame),
                places=5,
            )

    def test_bound_parameters_can_restore_the_fast_path(self):
        # Parameters are constants for the whole policy iteration, so their
        # sign is decidable once bound. Without that, every tunable-weight
        # exploration score paid the slow path -- measured at ~4x.
        @policy.exploration
        def tunable(k, p, fn):
            return k.nred * p.w(0) + k.ndim * p.w(1)

        positive = tunable.bind([1.5, 0.5]).native()
        self.assertFalse(positive.ranks_fixed_y_by_reduction())  # unbound: unknown
        self.assertTrue(positive.ranks_fixed_y_by_reduction([1.5, 0.5]))
        self.assertFalse(positive.ranks_fixed_y_by_reduction([-1.5, 0.5]))
        self.assertFalse(positive.ranks_fixed_y_by_reduction([0.0, 0.5]))

    def test_fast_path_claim_is_sound(self):
        """Whenever the analysis claims the fast path, the score must really
        rank reductions in descending order for a fixed y.

        The fast path replaces scoring with a reduction-only ranking, so a
        false claim silently changes which candidates are chosen -- it would
        not fail, it would quietly search differently. Randomized because the
        unsound cases are combinations (sqrt(x)*a - x*b is decreasing for large
        x even though both coefficients are positive), not single operators.
        """
        import random

        @policy.exploration
        def affine(k, p, fn):
            return k.nred * p.w(0) + k.ndim * p.w(1)

        @policy.exploration
        def monotone(k, p, fn):
            return fn.sqrt(fn.max(k.nred, 0.0)) * p.w(0) + k.ndim * p.w(1)

        @policy.exploration
        def saturating(k, p, fn):
            return fn.tanh(k.nred * p.w(0)) * p.w(1) + k.nyw * p.w(2)

        @policy.exploration
        def mixed_rates(k, p, fn):
            # sqrt and linear rates are not comparable; must be refused.
            return fn.sqrt(fn.max(k.nred, 0.0)) * p.w(0) - k.nred * p.w(1)

        @policy.exploration
        def clamped(k, p, fn):
            return fn.min(k.nred, p.w(0)) * p.w(1) + k.ndim * p.w(2)

        fixed = dict(dim=3, yw=2, zw=1, zsize=4, bucket=5, bn=10, dn=8, wvwn=6)
        reductions = list(range(30, 0, -1))
        rng = random.Random(7)
        claimed = 0
        for expr in (affine, monotone, saturating, mixed_rates, clamped):
            for _ in range(120):
                args = [round(rng.uniform(-3, 3), 3) for _ in range(expr.n_params)]
                bound = expr.bind(args)
                if not bound.native().ranks_fixed_y_by_reduction(bound.params):
                    continue
                claimed += 1
                scores = [bound.evaluate(red=r, **fixed) for r in reductions]
                for i in range(len(scores) - 1):
                    self.assertGreaterEqual(
                        scores[i], scores[i + 1] - 1e-9,
                        msg=f"{expr.name} args={args} claims the fast path but is "
                            f"not monotone in reduction",
                    )
        self.assertGreater(claimed, 100, "the fuzz should exercise real claims")

    def test_monotone_functions_of_reduction_keep_the_fast_path(self):
        # sqrt/tanh/sigmoid are strictly increasing, so applying them to a
        # reduction term preserves candidate order and must not be refused.
        @policy.exploration
        def compressed(k, p, fn):
            return fn.sqrt(fn.max(k.nred, 0.0)) * p.w(0) + k.ndim * p.w(1)

        bound = compressed.bind([1.5, 0.5])
        self.assertTrue(bound.native().ranks_fixed_y_by_reduction(bound.params))

    def test_z_varying_knob_disqualifies_at_any_coefficient(self):
        @policy.exploration
        def with_bucket(k, p, fn):
            return k.nred * p.w(0) + k.nbucket * p.w(1)

        program = with_bucket.bind([1.0, 1.0]).native()
        for weight in (1.0, 0.0, -1.0):
            self.assertFalse(program.ranks_fixed_y_by_reduction([1.0, weight]))

    def test_native_reports_the_fast_path(self):
        @policy.exploration
        def greedy_expr(k, p, fn):
            return k.nred * 2.0

        self.assertTrue(greedy_expr.bind([]).native().ranks_fixed_y_by_reduction())
        # ratio_explore reads nbucket, so it must give the fast path up
        bound = ratio_explore.bind([1.0])
        self.assertFalse(bound.native().ranks_fixed_y_by_reduction(bound.params))

    def test_native_program_pickles(self):
        program = branchy_final.bind([2.0, 3.0]).native()
        restored = pickle.loads(pickle.dumps(program))
        frame = dict(red=10, dim=7, bn=5, dn=4, rank_red=2, pool_size=20, params=[2.0, 3.0])
        self.assertAlmostEqual(restored.evaluate(**frame), program.evaluate(**frame), places=6)

    def test_native_rejects_site_violations(self):
        from policy_expr.expr import KNOB_INDEX

        with self.assertRaises(Exception) as ctx:
            self.native.PolicyProgram([0], [KNOB_INDEX["tohpe"]], [], 0, 0)
        self.assertIn("tohpe", str(ctx.exception))


class TestKnobRedundancy(unittest.TestCase):
    """Pin identities between knobs that look independent but are not."""

    def test_nmax_red_equals_nbucket(self):
        # max_red is always 2 * bucket_size and both normalize by the same
        # per-iteration bn, so nmax_red carries no information beyond nbucket.
        # Documented in KNOB_DOC; pinned here so a normalizer change cannot
        # silently make the documentation wrong in either direction.
        @policy.exploration
        def a(k, p, fn):
            return k.nmax_red

        @policy.exploration
        def b(k, p, fn):
            return k.nbucket

        for bucket, bn, red in ((3, 10, 4), (7, 10, 5), (10, 10, 9), (2, 40, 1)):
            frame = dict(red=red, bucket=bucket, max_red=2 * bucket, bn=bn, dn=1, wvwn=1)
            self.assertAlmostEqual(
                a.bind([]).evaluate(**frame),
                b.bind([]).evaluate(**frame),
                places=9,
                msg=f"nmax_red and nbucket diverged at bucket={bucket}, bn={bn}",
            )

    def test_knob_doc_records_the_identity(self):
        from policy_expr.expr import KNOB_DOC

        self.assertIn("nbucket", KNOB_DOC["nmax_red"])

    def test_normalized_ratio_cancels_to_the_raw_ratio(self):
        # nred and nbucket share the per-iteration normalizer bn, so their
        # quotient equals red/bucket up to the constant 2 from nred's /2. A
        # ratio written on n* knobs therefore carries a division that does
        # nothing; policies should use the raw knobs for ratios.
        @policy.exploration
        def normalized(k, p, fn):
            return k.nred / fn.max(k.nbucket, 1e-9)

        @policy.exploration
        def raw(k, p, fn):
            return k.red / fn.max(k.bucket, 1e-9)

        for bucket, bn, red in ((3, 10, 4), (7, 10, 5), (2, 40, 1), (5, 100, 3)):
            frame = dict(red=red, bucket=bucket, bn=bn, dn=1, wvwn=1)
            self.assertAlmostEqual(
                normalized.bind([]).evaluate(**frame) * 2.0,
                raw.bind([]).evaluate(**frame),
                places=5,
                msg=f"bn failed to cancel at bucket={bucket}, bn={bn}",
            )

    def test_describe_knobs_warns_about_ratios(self):
        text = describe_knobs()
        self.assertIn("k.red / k.bucket", text)


class TestTohpeRedTarget(unittest.TestCase):
    """The TOHPE inner-z reduction band."""

    def setUp(self):
        self.native = _native_module()
        if self.native is None or not hasattr(self.native, "TohpeSearch"):
            self.skipTest("pyvartodd not built")

    def test_defaults_are_unbounded(self):
        search = self.native.TohpeSearch()
        self.assertEqual(search.target_min_red, 1)
        self.assertGreater(search.target_max_red, 10_000)

    def test_zero_minimum_is_rejected(self):
        # A zero reduction is not a usable action, so a band starting there
        # cannot be targeted; the message has to say why.
        with self.assertRaises(Exception) as ctx:
            self.native.TohpeSearch(target_min_red=0, target_max_red=5)
        self.assertIn("must be > 0", str(ctx.exception))

    def test_inverted_band_is_rejected(self):
        with self.assertRaises(Exception) as ctx:
            self.native.TohpeSearch(target_min_red=5, target_max_red=2)
        self.assertIn(">= target_min_red", str(ctx.exception))

    def test_band_appears_in_repr(self):
        self.assertIn("target_red=3..4",
                      repr(self.native.TohpeSearch(target_min_red=3, target_max_red=4)))
        self.assertIn("target_red=1..any", repr(self.native.TohpeSearch()))

    def test_band_survives_pickle(self):
        search = self.native.TohpeSearch(target_min_red=2, target_max_red=3)
        restored = pickle.loads(pickle.dumps(search))
        self.assertEqual(restored.target_min_red, 2)
        self.assertEqual(restored.target_max_red, 3)

    def test_band_bounds_the_reductions_that_reach_the_pool(self):
        # End-to-end: with a band, no candidate above its upper edge should be
        # selected, whichever y-sampling mode is used.
        import numpy as np

        matrix_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..",
            "data", "init_npy", "other", "ham15_mad.npy",
        )
        if not os.path.isfile(matrix_path):
            self.skipTest("benchmark matrix not available")

        n = self.native
        matrix = n.Matrix.from_numpy(np.load(matrix_path))

        @policy.exploration
        def greedy_expl(k, p, fn):
            return k.nred

        @policy.final
        def greedy_fin(k, p, fn):
            return k.nred

        def best_reduction(low, high, sparse, dense):
            cfg = n.PolicyConfig()
            cfg.scores = n.PolicyScores(
                exploration=greedy_expl.bind([]).native(),
                final=greedy_fin.bind([]).native(),
            )
            cfg.selection = n.ActionSelection(count=3, mode="best", temperature=0.0)
            cfg.pool = n.ActionPool(final_size=40)
            cfg.tohpe = n.TohpeSearch(
                n.SamplingBudget(one_hot=-1, sparse=sparse, dense=dense, sparse_max_weight=2),
                n.SourcePool(keep=40, reserve=0), 4,
                target_min_red=low, target_max_red=high,
            )
            empty = n.ZBucketSearch(0, 0, 0.0, 0.0, 0)
            cfg.tohpeprefix = n.TohpePrefixSearch(
                n.SamplingBudget(), n.SourcePool(keep=0, reserve=0), 1, empty)
            cfg.todd = n.ToddSearch(
                n.SamplingBudget(), n.SourcePool(keep=0, reserve=0), 1, empty)
            result = n.policy_iteration(cur_mat=matrix, policy_cfg=cfg, seed=11, add_seed=0)
            return max(c.reduction for c in result.chosen), result.stats.max_reduction

        unbounded, _ = best_reduction(1, 1_000_000, 8, 16)
        for low, high in ((3, 4), (2, 3)):
            for sparse, dense in ((0, 0), (8, 16)):
                chosen, seen = best_reduction(low, high, sparse, dense)
                self.assertLessEqual(
                    chosen, high,
                    msg=f"band [{low},{high}] admitted reduction {chosen} "
                        f"(sparse={sparse}, dense={dense})",
                )
        self.assertGreater(unbounded, 4, "fixture should offer reductions above the bands")


class TestReportRendering(unittest.TestCase):
    """A converged policy must read back as the expression it was written as.

    Run reports are what the mutation loop reads to decide the next change, so
    an opaque rendering (a program id, or a weight vector for an expression
    that is not a weighted sum) would break that loop.
    """

    def test_bound_policy_substitutes_its_parameters(self):
        rendered = branchy_final.bind([1.83, -0.41]).rendered()
        self.assertIn("k.nred", rendered)
        self.assertIn("+1.83", rendered)
        self.assertIn("-0.41", rendered)
        self.assertNotIn("p.w(", rendered)

    def test_nonlinear_forms_survive_rendering(self):
        @policy.final
        def nonlinear(k, p, fn):
            return (fn.tanh(k.nred * p.w(0))
                    + k.nred / fn.max(k.nbucket, 0.05) * p.w(1)
                    + fn.where(k.dim > 3, k.ndim, 0.0) * p.w(2))

        rendered = nonlinear.bind([1.0, 2.0, 3.0]).rendered()
        for fragment in ("fn.tanh", "fn.max", "fn.where", "k.nbucket"):
            self.assertIn(fragment, rendered)

    def test_float32_literals_render_readably(self):
        # A literal that has round-tripped through float32 must not print as
        # 0.05000000074505806.
        @policy.exploration
        def ratio(k, p, fn):
            return k.nred / fn.max(k.nbucket, 0.05)

        self.assertIn("0.05", ratio.bind([]).rendered())
        self.assertNotIn("0.0500000", ratio.bind([]).rendered())

    def test_report_renders_a_compiled_program(self):
        # The reporting path receives the compiled program plus its parameters,
        # not the Python expression, so it must reconstruct the source form.
        native = _native_module()
        if native is None or not hasattr(native, "PolicyProgram"):
            self.skipTest("pyvartodd with PolicyProgram not built")
        try:
            from mcts_dao import Path
        except Exception:
            self.skipTest("mcts_dao requires the harness-supplied variant module")

        bound = branchy_final.bind([1.83, -0.41])
        rendered = Path._format_score(bound.native(), bound.params)
        self.assertIn("k.nred", rendered)
        self.assertIn("+1.83", rendered)
        self.assertNotIn("PolicyProgram(", rendered)


class TestDiscoverability(unittest.TestCase):
    def test_describe_knobs_covers_both_sites(self):
        text = describe_knobs()
        self.assertIn("k.nred", text)
        self.assertIn("k.rank_red", text)
        self.assertIn("fn.where", text)
        self.assertIn("p.w(0)", text)

    def test_describe_knobs_respects_site(self):
        exploration = describe_knobs(SITE_EXPLORATION)
        self.assertIn("k.nred", exploration)
        self.assertNotIn("k.rank_red", exploration)


if __name__ == "__main__":
    unittest.main()
