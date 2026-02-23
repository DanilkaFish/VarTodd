
from __future__ import annotations

from typing import List
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple

from node import Node, Matrix


def build_context() -> Any:
    raise RuntimeError("build_context is not configured; pass _build_context to ToddMCTS")


def _get(obj, *names, default=None):
    """First existing attribute in names, else default."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def _q_ge(num_better: int, total: int) -> float:
    if not total:
        return 0.0
    return 1.0 - (num_better / total)

def format_path_stats_tiny(path: List['Node'], tdepth) -> str:
    _out = []
    # for node in out
    tohpe_zero = 0
    for index, node in enumerate(path[1:]):

        cand = node.incoming.cand
        s = node.incoming.global_info

        n = int(_get(s, "nonzero", default=0) or 0)

        r = int(_get(cand, "reduction", default=0) or 0)
        rmax = int(_get(s, "max_reduction", "best_reduction", default=r) or r)
        mean_dim = s.mean_basis
        max_dim = s.max_basis
        dim = cand.basis_dim
        rd = r - rmax
        rbetter = int(_get(cand, "num_better_red", "num_better_reduction", default=0) or 0)
        rq = _q_ge(rbetter, n)

        is_beyond = n == s.accepted_non_improving
        if not is_beyond:
            n = n - s.accepted_non_improving 
                
        if tohpe_zero == 0 and s.accepted_tohpe == 0:
            tohpe_zero = index
        _out.append(
            f"bd{dim}/d{dim - max_dim}/m{mean_dim:.3g};"
            f"r{r}/d{rd:+d}/q{rq:.2f}%;tha{s.accepted_tohpe}/tda{s.accepted}/b{int(is_beyond)}"
        )

    out = []
    for index, node in enumerate(path[:-1]):
        rank = node.state.rows
        out.append(f"{rank}:{_out[index]}")
    out.append(f"{path[-1].state.rows}:final")
    l = len(out)
    if len(out) > 15:
        if tohpe_zero <= 7:
            first_part = out[0:7]
        else:
            first_part = out[0:5]
        if len(out) - tohpe_zero <= 7:
            second_part = out[-7:]
        else:
            second_part = out[-5:]
        if tohpe_zero > 7 and len(out) - tohpe_zero > 7:
            return "\n".join(first_part + ["..."] + out[tohpe_zero - 1 : tohpe_zero + 3] + ["..."] + second_part) 
        return "\n".join(first_part + ["..."] + second_part) 
    return "\n".join(out)

def make_report(
    *,
    root: Node,
    best_node: Node,
    tdepth: int
) -> Tuple[str, List[Matrix]]:
    s = format_path_stats_tiny(best_node.path_from_root(), tdepth=tdepth)
    res = best_node.path_from_root()
    return s, res
