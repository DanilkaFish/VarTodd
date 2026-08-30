"""The @policy decorators: turn a plain function into a compiled expression.

A decorated function is a *value*, not a schedule. It carries no ranks, no
parameter groups and no optimizer wiring -- those stay in policy_mapping()
exactly as they were. Binding free scalars produces the object that occupies
the slot ExplorationScore / FinalizationScore occupied before.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence

from .expr import (
    FN,
    KNOB_NAMES,
    OP_ABS,
    OP_ADD,
    OP_AND,
    OP_DIV,
    OP_EQ,
    OP_EXP,
    OP_FLOOR,
    OP_GE,
    OP_GT,
    OP_LE,
    OP_LOAD_CONST,
    OP_LOAD_KNOB,
    OP_LOAD_PARAM,
    OP_LOG,
    OP_LT,
    OP_MAX,
    OP_MIN,
    OP_MUL,
    OP_NEG,
    OP_NOT,
    OP_OR,
    OP_POW,
    OP_SELECT,
    OP_SIGMOID,
    OP_SIGN,
    OP_SQRT,
    OP_SUB,
    OP_TANH,
    SITE_CODE,
    SITE_EXPLORATION,
    SITE_FINAL,
    SITE_KNOBS,
    Expr,
    Knobs,
    Params,
    PolicyError,
    collect,
    lower,
)
from .rewrite import rewrite_policy_function


def _load_native():
    """The compiled PolicyProgram type, or None before the bindings are built."""
    for loader in (
        lambda: __import__("node", fromlist=["PolicyProgram"]).PolicyProgram,
        lambda: __import__(
            "pyvartodd.Release.pyvartodd", fromlist=["PolicyProgram"]
        ).PolicyProgram,
        lambda: __import__(
            "pyvartodd.Debug.pyvartodd", fromlist=["PolicyProgram"]
        ).PolicyProgram,
        lambda: __import__("pyvartodd.pyvartodd", fromlist=["PolicyProgram"]).PolicyProgram,
    ):
        try:
            return loader()
        except Exception:
            continue
    return None


class BoundPolicy:
    """A compiled expression with its free scalars supplied.

    This is what `PolicyScores(...)` and `set_scores(...)` receive. It is
    picklable by value so the Dao deep-copy used for worker processes keeps
    working.
    """

    __slots__ = ("_expr", "_params", "_native")

    def __init__(self, expr: "PolicyExpr", params: Sequence[float]):
        self._expr = expr
        self._params = [float(v) for v in params]
        self._native = None

    @property
    def expr(self) -> "PolicyExpr":
        return self._expr

    @property
    def params(self) -> List[float]:
        return list(self._params)

    @property
    def site(self) -> str:
        return self._expr.site

    def native(self):
        """The C++ PolicyProgram, built on first use and cached."""
        if self._native is None:
            program_type = _load_native()
            if program_type is None:
                raise PolicyError(
                    "the compiled PolicyProgram type is unavailable; rebuild the "
                    "pyvartodd extension to run expression-based policies"
                )
            code, consts = self._expr.lowered
            self._native = program_type(
                [op for op, _ in code],
                [arg for _, arg in code],
                consts,
                self._expr.n_params,
                SITE_CODE[self._expr.site],
            )
        return self._native

    def evaluate(self, **knob_values) -> float:
        """Evaluate on explicit knob values. For tests and the probe pass."""
        return _reference_eval(self._expr, self._params, knob_values)

    def __repr__(self) -> str:
        return f"{self._expr.name}({', '.join(f'{v:.4g}' for v in self._params)})"

    def __reduce__(self):
        return (_rebuild_bound, (self._expr, self._params))

    def __eq__(self, other):
        return (
            isinstance(other, BoundPolicy)
            and other._expr is self._expr
            and other._params == self._params
        )

    def __hash__(self):
        return hash((id(self._expr), tuple(self._params)))


def _rebuild_bound(expr, params):
    return BoundPolicy(expr, params)


def from_expression(name: str, site: str, expr: Expr, n_params: int = 0,
                    warnings: List[str] = None) -> "PolicyExpr":
    """Build a PolicyExpr from an expression tree instead of by tracing.

    Used by the library constructors, which assemble their trees
    programmatically, and by unpickling.
    """
    obj = PolicyExpr.__new__(PolicyExpr)
    obj._fn = _NamedStub(name)
    obj._site = site
    obj._expr = expr
    obj._n_params = n_params
    obj._code, obj._consts = lower(expr, n_params)
    obj._used_knobs = frozenset(
        KNOB_NAMES[arg] for op, arg in obj._code if op == OP_LOAD_KNOB
    )
    obj._warnings = list(warnings or [])
    return obj


def _rebuild_expr(name, site, expr, n_params, warnings):
    return from_expression(name, site, expr, n_params, warnings)


class _NamedStub:
    """Stands in for the traced function once only its name is needed."""

    __slots__ = ("__name__",)

    def __init__(self, name: str):
        self.__name__ = name


class PolicyExpr:
    """A traced, compiled policy expression awaiting its free scalars."""

    __slots__ = ("_fn", "_site", "_expr", "_code", "_consts", "_n_params",
                 "_used_knobs", "_warnings")

    def __init__(self, func: Callable, site: str):
        self._fn = func
        self._site = site

        rewritten = rewrite_policy_function(func)

        knobs = Knobs(site)
        params = Params()
        try:
            result = rewritten(knobs, params, FN)
        except PolicyError:
            raise
        except Exception as exc:  # pragma: no cover - surfaced with context
            raise PolicyError(
                f"tracing policy {func.__name__!r} failed: {exc}"
            ) from exc

        if result is None:
            raise PolicyError(
                f"policy {func.__name__!r} returned None; it must return an "
                "expression built from the knobs"
            )
        if not isinstance(result, Expr):
            # A policy that returns a plain number is almost always a mistake,
            # but it is representable, so accept it after saying so.
            if isinstance(result, (int, float)):
                result = Expr._coerce(result)
            else:
                raise PolicyError(
                    f"policy {func.__name__!r} returned {type(result).__name__}; "
                    "it must return an expression built from the knobs"
                )

        self._expr = result

        indices = sorted(params.indices)
        if indices:
            expected = list(range(len(indices)))
            if indices != expected:
                missing = sorted(set(range(max(indices) + 1)) - set(indices))
                raise PolicyError(
                    f"policy {func.__name__!r} uses p.w({', '.join(map(str, indices))}) "
                    f"but skips p.w({', '.join(map(str, missing))}); free scalar "
                    "indices must run 0..n-1 with no gaps"
                )
        self._n_params = len(indices)

        self._code, self._consts = lower(result, self._n_params)

        # Read what the expression uses off the *lowered* program, not the
        # source tree: constant folding can drop a term entirely (`0.0 * p.w(1)`
        # collapses to 0), and a parameter that vanished still consumes a
        # map_par slot, so it must be reported.
        used_knobs = frozenset(
            KNOB_NAMES[arg] for op, arg in self._code if op == OP_LOAD_KNOB
        )
        used_params = frozenset(arg for op, arg in self._code if op == OP_LOAD_PARAM)
        self._used_knobs = used_knobs

        self._warnings = _lint(func.__name__, self, used_knobs, used_params, indices)

    # -- identity -------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._fn.__name__

    @property
    def site(self) -> str:
        return self._site

    @property
    def n_params(self) -> int:
        """How many free scalars policy_mapping() must supply."""
        return self._n_params

    @property
    def used_knobs(self) -> frozenset:
        return self._used_knobs

    @property
    def lowered(self):
        return self._code, self._consts

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    def __len__(self) -> int:
        return self._n_params

    # -- binding --------------------------------------------------------------

    def bind(self, params: Sequence[float]) -> BoundPolicy:
        values = list(params)
        if len(values) != self._n_params:
            raise PolicyError(
                f"{self.name} declares {self._n_params} free scalar"
                f"{'' if self._n_params == 1 else 's'} via p.w(i), got "
                f"{len(values)}; supply exactly {self.name}.n_params values"
            )
        for i, v in enumerate(values):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise PolicyError(
                    f"{self.name}: parameter {i} must be a real number, got {v!r}"
                )
            if not math.isfinite(float(v)):
                raise PolicyError(
                    f"{self.name}: parameter {i} is {v!r}; parameters must be finite"
                )
        return BoundPolicy(self, values)

    # Calling the expression is the same as binding it.
    __call__ = bind

    def __repr__(self) -> str:
        return f"<policy {self._site} {self.name}: {self._expr!r}>"

    def source(self) -> str:
        """The expression as valid Python, for reports and round-tripping."""
        return repr(self._expr)

    # A PolicyExpr is pickled by its compiled program rather than by reference
    # to the traced function: the decorator replaces that function's module
    # name with this object, so pickle could not find it again. Worker
    # processes therefore receive the expression itself, with no need to
    # re-import or re-trace the defining module.
    def __reduce__(self):
        return (
            _rebuild_expr,
            (self.name, self._site, self._expr, self._n_params, self._warnings),
        )


# --- lint ---------------------------------------------------------------------

# Knobs whose value changes with the choice of z for a fixed y. Reading any of
# them in an exploration expression forfeits the reduction-only fast path in the
# inner z search, which is a wall-clock cost rather than a score change -- so it
# is reported rather than left to be discovered as unexplained slowness.
_Z_VARYING = frozenset(
    {"bucket", "nbucket", "zw", "nzw", "zsize", "bucket_id", "max_red", "nmax_red"}
)
_REDUCTION_KNOBS = frozenset({"red", "nred"})


def _lint(name, expr: "PolicyExpr", used_knobs, used_params, declared_indices) -> List[str]:
    warnings: List[str] = []

    if not used_knobs:
        warnings.append(
            f"{name}: expression ignores every knob, so all candidates score "
            "the same and selection falls back to tie-breaking"
        )
    elif not (used_knobs & _REDUCTION_KNOBS):
        warnings.append(
            f"{name}: expression never reads k.red or k.nred, so it does not "
            "reward reduction at all -- usually a mistake"
        )

    unused = set(declared_indices) - set(used_params)
    if unused:
        warnings.append(
            f"{name}: p.w({', '.join(map(str, sorted(unused)))}) declared but "
            "never used in the expression"
        )

    if expr.site == SITE_EXPLORATION and (used_knobs & _Z_VARYING):
        touched = ", ".join(sorted(used_knobs & _Z_VARYING))
        warnings.append(
            f"{name}: reads {touched}, so the inner z search must score every "
            "touched bucket instead of taking the reduction-only fast path. "
            "This is a wall-clock cost, not a scoring change."
        )

    return warnings


# --- reference evaluator (probe pass and tests) --------------------------------

DIV_EPSILON = 0.001  # matches k_div_epsilon in include/policy_expr.hpp
LOG_FLOOR = 1e-6     # matches k_log_floor


def _guarded_div(a: float, b: float) -> float:
    if abs(b) < DIV_EPSILON:
        b = math.copysign(DIV_EPSILON, b if b != 0.0 else 1.0)
    return a / b


def _knob_value(values: dict, name: str) -> float:
    """Mirror of knob_value() in src/core/policy_expr.cpp."""
    raw = lambda n, default=0.0: float(values.get(n, default))  # noqa: E731
    bn = raw("bn", 1.0)
    dn = raw("dn", 1.0)
    wvwn = raw("wvwn", 1.0)
    pool = max(1.0, raw("pool_size"))
    if name == "nred":
        return raw("red") / bn / 2.0
    if name == "ndim":
        return raw("dim") / dn
    if name == "nbucket":
        return raw("bucket") / bn
    if name == "nyw":
        return raw("yw") / wvwn
    if name == "nzw":
        return raw("zw") / max(1.0, raw("zsize", 1.0))
    if name == "nmax_red":
        return raw("max_red") / bn / 2.0
    if name == "ntohpe":
        return raw("tohpe") / dn
    if name == "nrank_red":
        return raw("rank_red") / pool
    if name == "nrank_dim":
        return raw("rank_dim") / pool
    if name == "nrank_score":
        return raw("rank_score") / pool
    if name == "f_tohpe":
        return raw("pool_tohpe") / pool
    if name == "f_prefix":
        return raw("pool_prefix") / pool
    if name == "f_todd":
        return raw("pool_todd") / pool
    if name == "zsize":
        return raw("zsize", 1.0)
    if name in ("bn", "dn", "wvwn"):
        return {"bn": bn, "dn": dn, "wvwn": wvwn}[name]
    return raw(name)


def _reference_eval(expr: "PolicyExpr", params: Sequence[float], knob_values: dict) -> float:
    """Pure-Python evaluation of a lowered program.

    Used by the load-time probe pass and by tests, so a policy can be checked
    for non-finite results before any C++ evaluation happens.
    """
    code, consts = expr.lowered
    stack: List[float] = []
    for op, arg in code:
        if op == OP_LOAD_KNOB:
            stack.append(_knob_value(knob_values, KNOB_NAMES[arg]))
        elif op == OP_LOAD_CONST:
            stack.append(consts[arg])
        elif op == OP_LOAD_PARAM:
            stack.append(float(params[arg]) if arg < len(params) else 0.0)
        elif op == OP_ADD:
            b = stack.pop(); stack[-1] = stack[-1] + b
        elif op == OP_SUB:
            b = stack.pop(); stack[-1] = stack[-1] - b
        elif op == OP_MUL:
            b = stack.pop(); stack[-1] = stack[-1] * b
        elif op == OP_DIV:
            b = stack.pop(); stack[-1] = _guarded_div(stack[-1], b)
        elif op == OP_MIN:
            b = stack.pop(); stack[-1] = min(stack[-1], b)
        elif op == OP_MAX:
            b = stack.pop(); stack[-1] = max(stack[-1], b)
        elif op == OP_POW:
            b = stack.pop()
            try:
                stack[-1] = float(stack[-1]) ** b
            except (ValueError, OverflowError, ZeroDivisionError):
                stack[-1] = float("nan")
        elif op == OP_NEG:
            stack[-1] = -stack[-1]
        elif op == OP_ABS:
            stack[-1] = abs(stack[-1])
        elif op == OP_EXP:
            try:
                stack[-1] = math.exp(stack[-1])
            except OverflowError:
                stack[-1] = float("inf")
        elif op == OP_LOG:
            stack[-1] = math.log(max(LOG_FLOOR, stack[-1]))
        elif op == OP_SQRT:
            stack[-1] = math.sqrt(max(0.0, stack[-1]))
        elif op == OP_TANH:
            stack[-1] = math.tanh(stack[-1])
        elif op == OP_SIGMOID:
            stack[-1] = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, stack[-1]))))
        elif op == OP_FLOOR:
            stack[-1] = math.floor(stack[-1])
        elif op == OP_SIGN:
            stack[-1] = 1.0 if stack[-1] > 0 else (-1.0 if stack[-1] < 0 else 0.0)
        elif op == OP_LT:
            b = stack.pop(); stack[-1] = 1.0 if stack[-1] < b else 0.0
        elif op == OP_LE:
            b = stack.pop(); stack[-1] = 1.0 if stack[-1] <= b else 0.0
        elif op == OP_GT:
            b = stack.pop(); stack[-1] = 1.0 if stack[-1] > b else 0.0
        elif op == OP_GE:
            b = stack.pop(); stack[-1] = 1.0 if stack[-1] >= b else 0.0
        elif op == OP_EQ:
            b = stack.pop(); stack[-1] = 1.0 if stack[-1] == b else 0.0
        elif op == OP_AND:
            b = stack.pop(); stack[-1] = 1.0 if (stack[-1] != 0.0 and b != 0.0) else 0.0
        elif op == OP_OR:
            b = stack.pop(); stack[-1] = 1.0 if (stack[-1] != 0.0 or b != 0.0) else 0.0
        elif op == OP_NOT:
            stack[-1] = 1.0 if stack[-1] == 0.0 else 0.0
        elif op == OP_SELECT:
            b = stack.pop(); a = stack.pop(); cond = stack.pop()
            stack.append(a if cond != 0.0 else b)
        else:  # pragma: no cover
            raise PolicyError(f"unknown opcode {op}")
    return stack[0]


# --- the decorator namespace ---------------------------------------------------

class _PolicyNamespace:
    """`policy.exploration` and `policy.final` decorators."""

    def exploration(self, func: Callable) -> PolicyExpr:
        """Score used to rank y-vectors into pools and z buckets within a y."""
        return PolicyExpr(func, SITE_EXPLORATION)

    def final(self, func: Callable) -> PolicyExpr:
        """Score used to rank the merged candidate pool before selection."""
        return PolicyExpr(func, SITE_FINAL)


policy = _PolicyNamespace()
