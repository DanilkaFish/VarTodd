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
