"""Python-authored policy scoring expressions for the TODD backend.

A policy is a plain function over named knobs::

    from helper import policy

    @policy.exploration
    def explore(k, p, fn):
        return k.nred * p.w(0) + k.nbucket * p.w(1)

The decorator traces the function once, compiles it to a flat program, and
returns a value. Free scalars declared with ``p.w(i)`` are supplied in
``policy_mapping()`` from ordinary ``map_par`` calls::

    weights = [self.float_range(-4, 4) for _ in range(explore.n_params)]
    self.set_scores(PolicyScores(explore.bind(weights), final.bind(...)))

Call ``describe_knobs()`` for the list of knobs available at each site.
"""

from .expr import (
    KNOB_DOC,
    KNOB_NAMES,
    SITE_EXPLORATION,
    SITE_FINAL,
    Expr,
    PolicyError,
    describe_knobs,
)
from .library import (
    anti_greedy,
    distance_score,
    greedy,
    linear_score,
    log_score,
    polynom_score,
    sigmoid_score,
    weighted,
)
from .policy import BoundPolicy, PolicyExpr, policy
from .rewrite import RewriteError
from .validate import check_finite, probe_frames, validate_bound, validate_scores

__all__ = [
    "policy",
    "PolicyExpr",
    "BoundPolicy",
    "PolicyError",
    "RewriteError",
    "Expr",
    "describe_knobs",
    "KNOB_NAMES",
    "KNOB_DOC",
    "SITE_EXPLORATION",
    "SITE_FINAL",
    "linear_score",
    "polynom_score",
    "distance_score",
    "sigmoid_score",
    "log_score",
    "greedy",
    "anti_greedy",
    "weighted",
    "validate_bound",
    "validate_scores",
    "check_finite",
    "probe_frames",
]
