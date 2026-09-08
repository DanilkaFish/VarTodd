"""Load-time probe pass: catch non-finite policies before spending search budget.

A NaN score does not crash -- it sorts unpredictably and produces a
plausible-looking but meaningless search result. So every bound policy is
evaluated over a spread of boundary and random knob frames at load time, and a
non-finite result fails loudly with the values that produced it.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Sequence

from .expr import PolicyError
from .policy import BoundPolicy, PolicyExpr, _reference_eval

# Raw knob ranges seen in practice. `bn`, `dn` and `wvwn` are the per-iteration
# normalizers and are always >= 1, which the engine guarantees.
_RANGES: Dict[str, tuple] = {
    "red": (0, 400),
    "dim": (0, 200),
    "bucket": (0, 5000),
    "yw": (0, 400),
    "zw": (0, 400),
    "zsize": (1, 400),
    "max_red": (0, 10000),
    "tohpe": (0, 200),
    "rank_red": (0, 100),
    "rank_dim": (0, 100),
    "rank_score": (0, 100),
    "pool_size": (0, 64),
    "pool_tohpe": (0, 64),
    "pool_prefix": (0, 64),
    "pool_todd": (0, 64),
    "source": (0, 3),
    "bucket_id": (0, 100000),
    "k_idx": (0, 400),
    "l_idx": (0, 400),
    "bn": (1, 5000),
    "dn": (1, 200),
    "wvwn": (1, 400),
}


def probe_frames(seed: int = 20260830, random_frames: int = 64) -> List[Dict[str, float]]:
    """Boundary frames plus randomized ones.

    The boundary frames are the interesting cases: an all-zero candidate, the
    smallest legal normalizers, and the range maxima. Most non-finite policies
    are caught by those; the random frames cover interactions between knobs.
    """
    frames: List[Dict[str, float]] = []

    # Everything at its floor, with normalizers at their minimum of 1. This is
    # the frame that exposes unguarded division and log of zero.
    zero_frame = {name: float(lo) for name, (lo, _) in _RANGES.items()}
    zero_frame.update(bn=1.0, dn=1.0, wvwn=1.0, zsize=1.0)
    frames.append(zero_frame)

    # Everything at its ceiling: exposes overflow in exp() and pow().
    frames.append({name: float(hi) for name, (_, hi) in _RANGES.items()})

    # Large values divided by minimal normalizers -- the worst case for the
    # normalized knobs.
    skewed = {name: float(hi) for name, (_, hi) in _RANGES.items()}
    skewed.update(bn=1.0, dn=1.0, wvwn=1.0, zsize=1.0, pool_size=0.0)
    frames.append(skewed)

    # A typical mid-range candidate.
    frames.append({name: float((lo + hi) // 2) for name, (lo, hi) in _RANGES.items()})

    rng = random.Random(seed)
    for _ in range(random_frames):
        frames.append(
            {name: float(rng.randint(lo, hi)) for name, (lo, hi) in _RANGES.items()}
        )
    return frames


def check_finite(bound: BoundPolicy, *, frames: Iterable[Dict[str, float]] = None) -> None:
    """Raise if `bound` evaluates to NaN or infinity on any probe frame."""
    name = bound.expr.name
    for frame in frames if frames is not None else probe_frames():
        value = _reference_eval(bound.expr, bound.params, frame)
        if not math.isfinite(value):
            interesting = {
                k: v for k, v in sorted(frame.items()) if k in bound.expr.used_knobs
            }
            raise PolicyError(
                f"policy {name!r} evaluated to {value} on a probe frame; "
                f"scores must stay finite or candidate ordering becomes "
                f"meaningless. Knob values that produced it: {interesting}. "
                f"Expression: {bound.expr.source()}"
            )


def validate_bound(bound: BoundPolicy, *, strict: bool = False) -> List[str]:
    """Full load-time validation of one bound policy.

    Returns the lint warnings. With ``strict=True`` warnings are raised as
    errors instead, which is the mode to use in an automated mutation loop that
    should not waste budget on a degenerate policy.
    """
    if not isinstance(bound, BoundPolicy):
        raise PolicyError(
            f"expected a bound policy (call .bind(params) on the decorated "
            f"function), got {type(bound).__name__}"
        )
    check_finite(bound)
    warnings = bound.expr.warnings
    if strict and warnings:
        raise PolicyError(
            f"policy {bound.expr.name!r} has lint warnings and strict=True:\n  "
            + "\n  ".join(warnings)
        )
    return warnings


def validate_scores(*bounds: BoundPolicy, strict: bool = False) -> List[str]:
    """Validate several bound policies, returning their combined warnings."""
    report: List[str] = []
    for bound in bounds:
        if bound is None:
            continue
        report.extend(validate_bound(bound, strict=strict))
    return report
