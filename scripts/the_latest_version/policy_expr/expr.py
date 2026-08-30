"""Symbolic expression tree for policy scoring functions.

An expression is built by calling a decorated policy function once with
symbolic ``k`` (knobs), ``p`` (free scalars) and ``fn`` (math) arguments. Every
operator on the resulting objects records a node instead of computing a value;
the tree is then lowered to the flat postfix program the C++ VM evaluates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# Opcodes -- must match the Op enum in include/policy_expr.hpp exactly.
OP_LOAD_KNOB = 0
OP_LOAD_CONST = 1
OP_LOAD_PARAM = 2
OP_ADD = 3
OP_SUB = 4
OP_MUL = 5
OP_DIV = 6
OP_MIN = 7
OP_MAX = 8
OP_POW = 9
OP_NEG = 10
OP_ABS = 11
OP_EXP = 12
OP_LOG = 13
OP_SQRT = 14
OP_TANH = 15
OP_SIGMOID = 16
OP_FLOOR = 17
OP_SIGN = 18
OP_LT = 19
OP_LE = 20
OP_GT = 21
OP_GE = 22
OP_EQ = 23
OP_AND = 24
OP_OR = 25
OP_NOT = 26
OP_SELECT = 27

# Knob names in the order of the Knob enum in include/policy_expr.hpp.
KNOB_NAMES: Tuple[str, ...] = (
    "red",
    "nred",
    "dim",
    "ndim",
    "bucket",
    "nbucket",
    "yw",
    "nyw",
    "zw",
    "nzw",
    "zsize",
    "max_red",
    "nmax_red",
    "tohpe",
    "ntohpe",
    "rank_red",
    "nrank_red",
    "rank_dim",
    "nrank_dim",
    "rank_score",
    "nrank_score",
    "pool_size",
    "pool_tohpe",
    "pool_prefix",
    "pool_todd",
    "f_tohpe",
    "f_prefix",
    "f_todd",
    "source",
    "bucket_id",
    "k_idx",
    "l_idx",
    "bn",
    "dn",
    "wvwn",
)

KNOB_INDEX: Dict[str, int] = {name: i for i, name in enumerate(KNOB_NAMES)}

# One-line meaning per knob, surfaced by describe_knobs() so the prompt an LLM
# sees is generated from this file rather than maintained separately.
KNOB_DOC: Dict[str, str] = {
    "red": "raw T-count reduction this action achieves (the primary objective signal)",
    "nred": "reduction normalized by the max bucket size: red / bn / 2",
    "dim": "raw dimension of the solution basis at this bucket",
    "ndim": "basis dimension normalized: dim / dn",
    "bucket": "raw size of the collision bucket the candidate came from",
    "nbucket": "bucket size normalized: bucket / bn",
    "yw": "raw Hamming weight of the candidate vector y",
    "nyw": "y weight normalized by the number of parity rows: yw / wvwn",
    "zw": "raw Hamming weight of the z vector",
    "nzw": "density of z: zw / max(1, zsize)",
    "zsize": "length of the z vector",
    "max_red": "theoretical reduction ceiling for this bucket (2 * bucket)",
    "nmax_red": "reduction ceiling normalized: max_red / bn / 2",
    "tohpe": "TOHPE dimension of the resulting state (finalization only)",
    "ntohpe": "TOHPE dimension normalized: tohpe / dn (finalization only)",
    "rank_red": "how many candidates in the population beat this reduction (finalization only)",
    "nrank_red": "rank_red as a fraction of the pool (finalization only)",
    "rank_dim": "how many candidates beat this basis dimension (finalization only)",
    "nrank_dim": "rank_dim as a fraction of the pool (finalization only)",
    "rank_score": "how many candidates beat this pool score (finalization only)",
    "nrank_score": "rank_score as a fraction of the pool (finalization only)",
    "pool_size": "number of candidates in the finalized pool (finalization only)",
    "pool_tohpe": "how many pool candidates came from the standalone TOHPE source",
    "pool_prefix": "how many pool candidates came from the TOHPE-prefix source",
    "pool_todd": "how many pool candidates came from the full TODD source",
    "f_tohpe": "fraction of the pool from the TOHPE source",
    "f_prefix": "fraction of the pool from the TOHPE-prefix source",
    "f_todd": "fraction of the pool from the TODD source",
    "source": "provenance tag: 1=tohpe, 2=tohpe-prefix, 3=todd",
    "bucket_id": "identifier of the originating bucket",
    "k_idx": "first index of the action pair (finalization only)",
    "l_idx": "second index of the action pair (finalization only)",
    "bn": "normalizer: the largest bucket size this iteration",
    "dn": "normalizer: TOHPE dimension plus five",
    "wvwn": "normalizer: number of rows in the parity matrix",
}

# Knobs readable at each site. Mirrors site_available_knobs() in
# src/core/policy_expr.cpp; the C++ side re-validates, this copy exists so the
# error arrives at authoring time with a useful message.
SITE_EXPLORATION = "exploration"
SITE_FINAL = "final"

_EXPLORATION_KNOBS = frozenset(
    {
        "red", "nred", "dim", "ndim", "bucket", "nbucket", "yw", "nyw",
        "zw", "nzw", "zsize", "max_red", "nmax_red", "source", "bucket_id",
        "bn", "dn", "wvwn",
    }
)
_FINAL_ONLY_KNOBS = frozenset(
    {
        "tohpe", "ntohpe", "rank_red", "nrank_red", "rank_dim", "nrank_dim",
        "rank_score", "nrank_score", "pool_size", "pool_tohpe", "pool_prefix",
        "pool_todd", "f_tohpe", "f_prefix", "f_todd", "k_idx", "l_idx",
    }
)
SITE_KNOBS: Dict[str, frozenset] = {
    SITE_EXPLORATION: _EXPLORATION_KNOBS,
    SITE_FINAL: _EXPLORATION_KNOBS | _FINAL_ONLY_KNOBS,
}

# PolicySite values in include/policy_expr.hpp. Exploration expressions are
# validated against the stricter inner-z site, since the same program is used
# for both z ranking and pool ranking.
SITE_CODE: Dict[str, int] = {SITE_EXPLORATION: 0, SITE_FINAL: 2}


class PolicyError(Exception):
    """Raised for any misuse of the policy authoring API."""


def _suggest(name: str, candidates) -> str:
    """Closest valid names, so a typo reports its fix rather than a bare error."""
    import difflib

    matches = difflib.get_close_matches(name, sorted(candidates), n=3, cutoff=0.5)
    if matches:
        return "did you mean " + " or ".join(repr(m) for m in matches) + "?"
    return "valid names: " + ", ".join(repr(c) for c in sorted(candidates))


@dataclass(frozen=True)
class Node:
    """One node of the expression tree."""

    op: int
    arg: float = 0.0
    children: Tuple["Node", ...] = field(default_factory=tuple)


class Expr:
    """A symbolic value. Operators build nodes instead of computing numbers."""

    __slots__ = ("_node",)
    # Beat numpy's array operators: `2.0 * k.nred` with a numpy scalar on the
    # left must build a node, not an elementwise array.
    __array_priority__ = 1000.0

    def __init__(self, node: Node):
        self._node = node

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _coerce(value) -> "Expr":
        if isinstance(value, Expr):
            return value
        if isinstance(value, bool):
            return Expr(Node(OP_LOAD_CONST, 1.0 if value else 0.0))
        if isinstance(value, (int, float)):
            return Expr(Node(OP_LOAD_CONST, float(value)))
        raise PolicyError(
            f"cannot use {type(value).__name__} in a policy expression; "
            "only knobs, parameters and real numbers are allowed"
        )

    @staticmethod
    def _binary(op: int, a, b) -> "Expr":
        return Expr(Node(op, 0.0, (Expr._coerce(a)._node, Expr._coerce(b)._node)))

    @staticmethod
    def _unary(op: int, a) -> "Expr":
        return Expr(Node(op, 0.0, (Expr._coerce(a)._node,)))

    # -- arithmetic -----------------------------------------------------------

    def __add__(self, other): return Expr._binary(OP_ADD, self, other)
    def __radd__(self, other): return Expr._binary(OP_ADD, other, self)
    def __sub__(self, other): return Expr._binary(OP_SUB, self, other)
    def __rsub__(self, other): return Expr._binary(OP_SUB, other, self)
    def __mul__(self, other): return Expr._binary(OP_MUL, self, other)
    def __rmul__(self, other): return Expr._binary(OP_MUL, other, self)
    def __truediv__(self, other): return Expr._binary(OP_DIV, self, other)
    def __rtruediv__(self, other): return Expr._binary(OP_DIV, other, self)
    def __pow__(self, other): return Expr._binary(OP_POW, self, other)
    def __rpow__(self, other): return Expr._binary(OP_POW, other, self)
    def __neg__(self): return Expr._unary(OP_NEG, self)
    def __pos__(self): return self
    def __abs__(self): return Expr._unary(OP_ABS, self)

    # -- comparisons ----------------------------------------------------------

    def __lt__(self, other): return Expr._binary(OP_LT, self, other)
    def __le__(self, other): return Expr._binary(OP_LE, self, other)
    def __gt__(self, other): return Expr._binary(OP_GT, self, other)
    def __ge__(self, other): return Expr._binary(OP_GE, self, other)
    def __eq__(self, other): return Expr._binary(OP_EQ, self, other)  # type: ignore[override]
    def __ne__(self, other):  # type: ignore[override]
        return Expr._unary(OP_NOT, Expr._binary(OP_EQ, self, other))

    __hash__ = None  # type: ignore[assignment]

    def __bool__(self):
        # Backstop for control flow the AST rewriter could not reach (lambdas,
        # comprehensions, exec'd source). Without this, Python would silently
        # pick one branch at trace time and bake it in.
        raise PolicyError(
            "cannot use a policy expression in `if`, `and`, `or`, or `not` here; "
            "use fn.where(cond, a, b) instead. (Inside a decorated policy "
            "function, plain `if` is rewritten automatically -- this error means "
            "the expression escaped into a lambda, comprehension or generator.)"
        )

    # -- discoverable method forms of the fn.* helpers ------------------------

    def exp(self): return Expr._unary(OP_EXP, self)
    def log(self): return Expr._unary(OP_LOG, self)
    def sqrt(self): return Expr._unary(OP_SQRT, self)
    def tanh(self): return Expr._unary(OP_TANH, self)
    def sigmoid(self): return Expr._unary(OP_SIGMOID, self)
    def floor(self): return Expr._unary(OP_FLOOR, self)
    def sign(self): return Expr._unary(OP_SIGN, self)

    def __repr__(self) -> str:
        return _render(self._node)


# --- rendering: repr(expr) must be valid Python that rebuilds the expression ---

_BINARY_SYMBOL = {
    OP_ADD: "+", OP_SUB: "-", OP_MUL: "*", OP_DIV: "/",
    OP_LT: "<", OP_LE: "<=", OP_GT: ">", OP_GE: ">=", OP_EQ: "==",
}
_UNARY_FN = {
    OP_EXP: "fn.exp", OP_LOG: "fn.log", OP_SQRT: "fn.sqrt", OP_TANH: "fn.tanh",
    OP_SIGMOID: "fn.sigmoid", OP_FLOOR: "fn.floor", OP_SIGN: "fn.sign",
    OP_ABS: "fn.abs", OP_NOT: "fn.not_",
}
_BINARY_FN = {OP_MIN: "fn.min", OP_MAX: "fn.max", OP_POW: "fn.pow",
              OP_AND: "fn.and_", OP_OR: "fn.or_"}


def _render(node: Node) -> str:
    op = node.op
    if op == OP_LOAD_KNOB:
        return f"k.{KNOB_NAMES[int(node.arg)]}"
    if op == OP_LOAD_CONST:
        value = node.arg
        return repr(int(value)) if value == int(value) else repr(value)
    if op == OP_LOAD_PARAM:
        return f"p.w({int(node.arg)})"
    if op == OP_NEG:
        return f"(-{_render(node.children[0])})"
    if op in _UNARY_FN:
        return f"{_UNARY_FN[op]}({_render(node.children[0])})"
    if op in _BINARY_SYMBOL:
        left, right = (_render(c) for c in node.children)
        return f"({left} {_BINARY_SYMBOL[op]} {right})"
    if op in _BINARY_FN:
        left, right = (_render(c) for c in node.children)
        return f"{_BINARY_FN[op]}({left}, {right})"
    if op == OP_SELECT:
        cond, a, b = (_render(c) for c in node.children)
        return f"fn.where({cond}, {a}, {b})"
    return f"<op {op}>"


# --- lowering: tree -> flat postfix program -----------------------------------

def _constant_value(node: Node):
    """Value of a node if it folds to a constant, else None."""
    return node.arg if node.op == OP_LOAD_CONST else None


def _fold(node: Node) -> Node:
    """Constant-fold a subtree so literal arithmetic costs nothing at runtime."""
    if not node.children:
        return node

    children = tuple(_fold(c) for c in node.children)
    values = [_constant_value(c) for c in children]

    if all(v is not None for v in values):
        import math

        a = values[0]
        b = values[1] if len(values) > 1 else None
        try:
            if node.op == OP_ADD:
                return Node(OP_LOAD_CONST, a + b)
            if node.op == OP_SUB:
                return Node(OP_LOAD_CONST, a - b)
            if node.op == OP_MUL:
                return Node(OP_LOAD_CONST, a * b)
            if node.op == OP_NEG:
                return Node(OP_LOAD_CONST, -a)
            if node.op == OP_ABS:
                return Node(OP_LOAD_CONST, abs(a))
            if node.op == OP_MIN:
                return Node(OP_LOAD_CONST, min(a, b))
            if node.op == OP_MAX:
                return Node(OP_LOAD_CONST, max(a, b))
            if node.op == OP_SQRT:
                return Node(OP_LOAD_CONST, math.sqrt(max(0.0, a)))
            if node.op == OP_EXP:
                return Node(OP_LOAD_CONST, math.exp(a))
            if node.op == OP_TANH:
                return Node(OP_LOAD_CONST, math.tanh(a))
        except (ValueError, OverflowError):
            # Leave it to the VM's guarded ops rather than failing to fold.
            pass

    return Node(node.op, node.arg, children)


def lower(expr: Expr, n_params: int) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Flatten an expression into (code, consts).

    ``code`` is a list of ``(op, arg)`` pairs in postfix order; ``consts`` holds
    the literal pool that ``OP_LOAD_CONST`` indexes into.
    """
    code: List[Tuple[int, int]] = []
    consts: List[float] = []
    const_index: Dict[float, int] = {}

    def intern(value: float) -> int:
        key = float(value)
        if key not in const_index:
            const_index[key] = len(consts)
            consts.append(key)
        return const_index[key]

    def emit(node: Node) -> None:
        for child in node.children:
            emit(child)
        if node.op == OP_LOAD_CONST:
            code.append((OP_LOAD_CONST, intern(node.arg)))
        elif node.op in (OP_LOAD_KNOB, OP_LOAD_PARAM):
            code.append((node.op, int(node.arg)))
        else:
            code.append((node.op, 0))

    emit(_fold(expr._node))
    if not code:
        raise PolicyError("policy expression compiled to an empty program")
    return code, consts


def collect(expr: Expr) -> Tuple[frozenset, frozenset]:
    """Knob names and parameter indices an expression actually reads."""
    knobs = set()
    params = set()

    def walk(node: Node) -> None:
        if node.op == OP_LOAD_KNOB:
            knobs.add(KNOB_NAMES[int(node.arg)])
        elif node.op == OP_LOAD_PARAM:
            params.add(int(node.arg))
        for child in node.children:
            walk(child)

    walk(expr._node)
    return frozenset(knobs), frozenset(params)


# --- the symbolic arguments handed to a policy function -----------------------

class Knobs:
    """The ``k`` argument: attribute access yields a knob leaf."""

    __slots__ = ("_site",)

    def __init__(self, site: str):
        object.__setattr__(self, "_site", site)

    def __getattr__(self, name: str) -> Expr:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in KNOB_INDEX:
            raise PolicyError(f"{name!r} is not a knob; {_suggest(name, KNOB_NAMES)}")
        allowed = SITE_KNOBS[self._site]
        if name not in allowed:
            raise PolicyError(
                f"knob {name!r} is not available in a {self._site} expression "
                f"(it is only known during finalization). "
                f"Available here: {', '.join(sorted(allowed))}"
            )
        return Expr(Node(OP_LOAD_KNOB, float(KNOB_INDEX[name])))

    def __setattr__(self, name, value):
        raise PolicyError("knobs are read-only")

    def __dir__(self):
        return sorted(SITE_KNOBS[self._site])


class Params:
    """The ``p`` argument: ``p.w(i)`` marks the i-th free scalar."""

    __slots__ = ("_seen",)

    def __init__(self):
        object.__setattr__(self, "_seen", set())

    def w(self, index) -> Expr:
        if isinstance(index, bool) or not isinstance(index, int):
            raise PolicyError(
                f"p.w(...) needs an integer slot index, got {index!r}"
            )
        if index < 0:
            raise PolicyError(f"p.w(...) index must be non-negative, got {index}")
        self._seen.add(index)
        return Expr(Node(OP_LOAD_PARAM, float(index)))

    # A tunable read as an attribute is a common slip; name the fix.
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        raise PolicyError(
            f"p.{name} is not valid; free scalars are addressed positionally as "
            f"p.w(0), p.w(1), ... and supplied in that order by policy_mapping()"
        )

    @property
    def indices(self) -> frozenset:
        return frozenset(self._seen)


class _Fn:
    """The ``fn`` argument: math that traces into opcodes."""

    @staticmethod
    def exp(x): return Expr._unary(OP_EXP, x)
    @staticmethod
    def log(x): return Expr._unary(OP_LOG, x)
    @staticmethod
    def sqrt(x): return Expr._unary(OP_SQRT, x)
    @staticmethod
    def tanh(x): return Expr._unary(OP_TANH, x)
    @staticmethod
    def sigmoid(x): return Expr._unary(OP_SIGMOID, x)
    @staticmethod
    def floor(x): return Expr._unary(OP_FLOOR, x)
    @staticmethod
    def sign(x): return Expr._unary(OP_SIGN, x)
    @staticmethod
    def abs(x): return Expr._unary(OP_ABS, x)
    @staticmethod
    def not_(x): return Expr._unary(OP_NOT, x)

    @staticmethod
    def min(a, b): return Expr._binary(OP_MIN, a, b)
    @staticmethod
    def max(a, b): return Expr._binary(OP_MAX, a, b)
    @staticmethod
    def pow(a, b): return Expr._binary(OP_POW, a, b)
    @staticmethod
    def and_(a, b): return Expr._binary(OP_AND, a, b)
    @staticmethod
    def or_(a, b): return Expr._binary(OP_OR, a, b)

    @staticmethod
    def where(cond, a, b) -> Expr:
        """Branchless conditional. Both arms are always evaluated."""
        return Expr(
            Node(
                OP_SELECT,
                0.0,
                (
                    Expr._coerce(cond)._node,
                    Expr._coerce(a)._node,
                    Expr._coerce(b)._node,
                ),
            )
        )

    @staticmethod
    def clip(x, lo, hi) -> Expr:
        return _Fn.min(_Fn.max(x, lo), hi)


FN = _Fn()


def describe_knobs(site: str = None) -> str:
    """Human-readable knob reference, generated from this module.

    Used to build the prompt an LLM sees, so the documentation cannot drift
    from the implementation.
    """
    sites = [site] if site else [SITE_EXPLORATION, SITE_FINAL]
    lines = []
    for s in sites:
        lines.append(f"knobs available in a `{s}` expression:")
        for name in KNOB_NAMES:
            if name in SITE_KNOBS[s]:
                lines.append(f"  k.{name:<12} {KNOB_DOC[name]}")
        lines.append("")
    lines.append("math: fn.exp fn.log fn.sqrt fn.tanh fn.sigmoid fn.floor fn.sign")
    lines.append("      fn.abs fn.min fn.max fn.pow fn.clip fn.where fn.not_")
    lines.append("free scalars: p.w(0), p.w(1), ... supplied by policy_mapping()")
    return "\n".join(lines)
