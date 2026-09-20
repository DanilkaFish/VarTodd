import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "the_latest_version"))

from policy_expr import PolicyError, describe_knobs, policy
from policy_expr.expr import Knobs


class TestTracing(unittest.TestCase):
    def test_reduction_interface_uses_raw_features(self):
        for name in ("bn", "nred", "nbucket", "nmax_red"):
            self.assertNotIn(name, describe_knobs())

        @policy.exploration
        def raw_features(k, p, fn):
            return k.red * p.w(0) + k.bucket * p.w(1)

        self.assertEqual(raw_features.used_knobs, frozenset({"red", "bucket"}))
        self.assertAlmostEqual(raw_features.bind([2.0, 3.0]).evaluate(red=4, bucket=5), 23.0)

    def test_removed_knobs_fail_at_authoring(self):
        for name in ("bn", "nred", "nbucket", "nmax_red"):
            with self.assertRaises(PolicyError):
                getattr(Knobs("exploration"), name)

    def test_other_normalized_features_remain(self):
        @policy.exploration
        def score(k, p, fn):
            return k.ndim + k.nyw + k.nzw
        value = score.bind([]).evaluate(dim=8, dn=4, yw=6, wvwn=3, zw=2, zsize=4)
        self.assertAlmostEqual(value, 4.5)

    def test_explicit_normalized_overrides_and_ysize(self):
        @policy.final
        def score(k, p, fn):
            return k.ndim + k.nyw + k.nzw + k.ntohpe + k.ysize + k.max_red
        bound = score.bind([])
        derived = dict(dim=8, dn=4, yw=6, ysize=3, zw=2, zsize=4,
                       tohpe=12, max_red=5)
        overrides = dict(derived, ndim=10, nyw=20, nzw=30, ntohpe=40)
        for evaluate in (bound.evaluate, bound.native().evaluate):
            self.assertAlmostEqual(evaluate(**derived), 15.5)
            self.assertAlmostEqual(evaluate(**overrides), 108.0)
            self.assertAlmostEqual(evaluate(wvwn=7), 7.0)
            with self.assertRaises(ValueError):
                evaluate(ysize=3, wvwn=4)
            with self.assertRaises(ValueError):
                evaluate(nred=1)

    def test_ysize_preserves_knob_wire_order(self):
        from policy_expr.expr import KNOB_NAMES
        self.assertEqual(KNOB_NAMES[30:32], ("wvwn", "ysize"))
        self.assertEqual(KNOB_NAMES[32:], ("population_size",))
        self.assertIn("ysize", describe_knobs())

    def test_finite_probe(self):
        @policy.exploration
        def score(k, p, fn):
            return k.red / fn.max(k.bucket, 1.0)
        self.assertTrue(math.isfinite(score.bind([]).evaluate(red=4, bucket=0)))


if __name__ == "__main__":
    unittest.main()
