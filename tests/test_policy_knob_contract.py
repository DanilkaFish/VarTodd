"""The expression API's wire mapping, independently specified feature values."""
import pickle
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "the_latest_version"))

from policy_expr import policy, PolicyError
from policy_expr.expr import KNOB_NAMES, SITE_KNOBS
from node import _ext as native


def score(k, p, fn):
    return getattr(k, _AUDIT_KNOB)


def weighted(k, p, fn):
    return getattr(k, _AUDIT_KNOB) * p.w(0) + p.w(1)


def knob_expression(name, site="final", parameterized=False):
    global _AUDIT_KNOB
    _AUDIT_KNOB = name
    return getattr(policy, site)(weighted if parameterized else score)


class TestPolicyKnobContract(unittest.TestCase):
    values = dict(red=7, dim=11, bucket=13, yw=17, zw=19, zsize=23,
                  max_red=26, tohpe=29, rank_red=31, rank_dim=37, rank_score=41,
                  pool_size=43, pool_tohpe=5, pool_todd=38, source=3,
                  bucket_id=47, k_idx=53, l_idx=59, dn=61, ysize=67, population_size=101)
    expected = dict(values, ndim=11/61, nyw=17/67, nzw=19/23, ntohpe=29/61,
                    nrank_red=31/101, nrank_dim=37/101, nrank_score=41/101,
                    f_tohpe=5/43, f_todd=38/43, wvwn=67)

    def test_all_wire_names_and_sites(self):
        self.assertEqual(tuple(native.policy_knob_names()), KNOB_NAMES)
        self.assertEqual(set(self.expected), set(KNOB_NAMES) - {"pool_prefix", "f_prefix"})
        for site, code in (("exploration", 0), ("exploration", 1), ("final", 2)):
            self.assertEqual(set(native.policy_site_knobs(code)), SITE_KNOBS[site])

    def test_every_knob_reference_native_and_pickle(self):
        for name, expected in self.expected.items():
            with self.subTest(knob=name):
                expr = knob_expression(name, parameterized=True)
                bound = expr.bind([2.0, 3.0])
                for current in (bound, pickle.loads(pickle.dumps(bound))):
                    self.assertAlmostEqual(current.evaluate(**self.values), expected * 2 + 3, places=5)
                    program = pickle.loads(pickle.dumps(current.native()))
                    self.assertAlmostEqual(program.evaluate(**self.values, params=current.params),
                                           expected * 2 + 3, places=5)

    def test_every_normalized_override(self):
        for name in ("ndim", "nyw", "nzw", "ntohpe", "nrank_red", "nrank_dim",
                     "nrank_score", "f_tohpe", "f_todd"):
            with self.subTest(knob=name):
                bound = knob_expression(name).bind([])
                values = dict(self.values, **{name: 123.25})
                self.assertEqual(bound.evaluate(**values), 123.25)
                self.assertEqual(bound.native().evaluate(**values), 123.25)

    def test_rank_normalization_is_independent_of_pool_size(self):
        for name in ("nrank_red", "nrank_dim", "nrank_score"):
            bound = knob_expression(name).bind([])
            for size in (0, 1, 16, 1000):
                values = dict(self.values, pool_size=size)
                for evaluate in (bound.evaluate, bound.native().evaluate):
                    self.assertAlmostEqual(evaluate(**values), self.expected[name], places=6)
                    self.assertEqual(evaluate(rank_red=0, rank_dim=0, rank_score=0, population_size=0), 0)

    def test_final_only_knobs_are_rejected_during_authoring(self):
        for name in self.expected:
            with self.subTest(knob=name):
                if name in SITE_KNOBS["exploration"]:
                    expr = knob_expression(name, "exploration")
                    self.assertEqual(expr.used_knobs, frozenset({name}))
                    expr.bind([]).native()
                else:
                    with self.assertRaises(PolicyError):
                        knob_expression(name, "exploration")
                    with self.assertRaises(Exception):
                        native.PolicyProgram([0], [KNOB_NAMES.index(name)], [], 0, 0)


if __name__ == "__main__":
    unittest.main()
