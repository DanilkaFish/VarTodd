"""Reference policies and exact re-expressions of the legacy scoring forms.

The `*_score` builders reproduce the arithmetic of the five ScoringFunction
variants that PolicyExpr replaces, so an existing policy can be ported without
changing its behaviour, and so the C++ regression tests have something to
compare against.
"""

from __future__ import annotations

from typing import Sequence

from .expr import FN, Expr, Knobs, PolicyError, SITE_EXPLORATION, SITE_FINAL
from .policy import PolicyExpr, from_expression

# Feature order of the legacy scores, from ExplorationScore::evaluate_features
# and FinalizationScore::operator() in include/todd_generator.hpp.
EXPLORATION_FEATURES = ("nred", "ndim", "nbucket", "nyw", "nzw")
FINALIZATION_FEATURES = EXPLORATION_FEATURES + ("ntohpe",)


def _features(site: str, count: int):
    knobs = Knobs(site)
    names = FINALIZATION_FEATURES if site == SITE_FINAL else EXPLORATION_FEATURES
    if count > len(names):
        raise PolicyError(
            f"a {site} score has at most {len(names)} features, got {count}"
        )
    return [getattr(knobs, name) for name in names[:count]]


# The library constructors assemble their trees programmatically -- a
# comprehension over weights cannot be traced through the AST rewriter, which
# rejects comprehensions on purpose -- so they build a PolicyExpr directly.
_LiteralExpr = from_expression


def linear_score(weights: Sequence[float], *, site: str = SITE_EXPLORATION) -> PolicyExpr:
    """sum_i w[i] * x[i] -- ScoringFunction::LINEAR."""
    xs = _features(site, len(weights))
    total = None
    for w, x in zip(weights, xs):
        if w == 0.0:
            continue  # a zero weight contributes nothing; keep the tree small
        term = x * float(w)
        total = term if total is None else total + term
    if total is None:
        total = Expr._coerce(0.0)
    return _LiteralExpr("linear_score", site, total)


def polynom_score(
    weights: Sequence[float],
    centers: Sequence[float],
    pow: float = 2.0,
    *,
    site: str = SITE_EXPLORATION,
    bn: float = None,
) -> PolicyExpr:
    """sum_i w[i] * |x[i] - c[i]|^pow -- ScoringFunction::POLYNOM.

    The legacy form scales the *first* center by ``1 / bn / 2`` so that a center
    expressed in raw reduction units lands in normalized space. `bn` varies per
    iteration, so it is read as a knob rather than baked in.
    """
    n = min(len(weights), len(centers))
    xs = _features(site, n)
    total = None
    knobs = Knobs(site)
    for i in range(n):
        w = float(weights[i])
        if w == 0.0:
            continue
        c = float(centers[i])
        if i == 0:
            # first_center_scale = 1 / bn / 2
            center = c / knobs.bn / 2.0 if bn is None else Expr._coerce(c / bn / 2.0)
        else:
            center = Expr._coerce(c)
        d = FN.abs(xs[i] - center)
        if pow >= 0:
            term = (d * d) if pow == 2.0 else FN.pow(d, float(pow))
        else:
            # Inverse-distance reward: w * (1 / max(d, 0.001)) ** -pow
            term = FN.pow(1.0 / FN.max(d, 0.001), float(-pow))
        term = term * w
        total = term if total is None else total + term
    if total is None:
        total = Expr._coerce(0.0)
    return _LiteralExpr("polynom_score", site, total)


def distance_score(weights: Sequence[float], *, site: str = SITE_EXPLORATION) -> PolicyExpr:
    """-sum_i (x[i] - w[i])^2 -- ScoringFunction::DISTANCE."""
    xs = _features(site, len(weights))
    total = None
    for w, x in zip(weights, xs):
        d = x - float(w)
        term = -(d * d)
        total = term if total is None else total + term
    if total is None:
        total = Expr._coerce(0.0)
    return _LiteralExpr("distance_score", site, total)


def sigmoid_score(weights: Sequence[float], pow: float = 2.0, *,
                  site: str = SITE_EXPLORATION) -> PolicyExpr:
    """sigmoid(sum_i w[i] * x[i]^pow) -- ScoringFunction::SIGMOID."""
    xs = _features(site, len(weights))
    total = None
    for w, x in zip(weights, xs):
        if w == 0.0:
            continue
        powered = (x * x) if pow == 2.0 else FN.pow(x, float(pow))
        term = powered * float(w)
        total = term if total is None else total + term
    if total is None:
        total = Expr._coerce(0.0)
    return _LiteralExpr("sigmoid_score", site, FN.sigmoid(total))


def log_score(weights: Sequence[float], *, site: str = SITE_EXPLORATION) -> PolicyExpr:
    """sum_i w[i] * log(max(1e-6, x[i])) -- ScoringFunction::LOGARITHMIC."""
    xs = _features(site, len(weights))
    total = None
    for w, x in zip(weights, xs):
        if w == 0.0:
            continue
        term = FN.log(x) * float(w)
        total = term if total is None else total + term
    if total is None:
        total = Expr._coerce(0.0)
    return _LiteralExpr("log_score", site, total)


# --- named reference policies --------------------------------------------------

def greedy(site: str = SITE_EXPLORATION) -> PolicyExpr:
    """Pure reduction-greedy: the fixed policy the benchmarks mimic."""
    return _LiteralExpr("greedy", site, Knobs(site).nred)


def anti_greedy(site: str = SITE_EXPLORATION) -> PolicyExpr:
    """Reduction-averse; the counterpart used to probe the search space."""
    return _LiteralExpr("anti_greedy", site, -Knobs(site).nred)


def weighted(n: int, site: str = SITE_EXPLORATION) -> PolicyExpr:
    """The classic tunable linear score: sum_i p.w(i) * feature_i.

    Reproduces what `ExplorationScore([...5 tunables...], pow=1)` expressed,
    but with the weights as free scalars rather than baked-in constants.
    """
    xs = _features(site, n)
    from .expr import Node, OP_LOAD_PARAM

    total = None
    for i, x in enumerate(xs):
        term = x * Expr(Node(OP_LOAD_PARAM, float(i)))
        total = term if total is None else total + term
    if total is None:
        raise PolicyError("weighted() needs at least one feature")
    return _LiteralExpr("weighted", site, total, n_params=n)
