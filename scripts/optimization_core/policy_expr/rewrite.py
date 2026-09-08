"""Rewrites ordinary Python control flow in a policy function into fn.where.

Tracing cannot see through `if cond:` -- Python would call __bool__ on the
condition and bake one branch in at trace time. Rather than forbidding `if`, the
decorator re-parses the function source and rewrites the branching forms into
branchless `fn.where` calls before tracing.

Both arms of a rewritten `if` are always evaluated. That is inherent to the
approach and is what the lint pass warns about when an arm divides by something
the condition was guarding.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable, List, Optional, Set

from .expr import PolicyError


class RewriteError(PolicyError):
    """Raised when a policy function uses a construct that cannot be traced."""


def _err(fn_name: str, node: ast.AST, message: str) -> RewriteError:
    line = getattr(node, "lineno", "?")
    return RewriteError(f"{fn_name}, line {line}: {message}")


class _Rewriter(ast.NodeTransformer):
    """Turns branching statements and boolean operators into fn.where calls."""

    def __init__(self, fn_name: str, fn_arg: str, allowed_names: Set[str]):
        self.fn_name = fn_name
        self.fn_arg = fn_arg
        self.allowed_names = allowed_names

    # -- helpers -------------------------------------------------------------

    def _fn_call(self, attr: str, args: List[ast.expr]) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.fn_arg, ctx=ast.Load()), attr=attr, ctx=ast.Load()
            ),
            args=args,
            keywords=[],
        )

    # -- expression-level rewrites -------------------------------------------

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        # `a and b` -> where(a, b, a); `a or b` -> where(a, a, b).
        # Folded left-to-right so n-ary chains keep Python's associativity.
        values = [self.visit(v) for v in node.values]
        result = values[0]
        for nxt in values[1:]:
            if isinstance(node.op, ast.And):
                result = self._fn_call("where", [result, nxt, result])
            else:
                result = self._fn_call("where", [result, result, nxt])
        return ast.copy_location(result, node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if isinstance(node.op, ast.Not):
            operand = self.visit(node.operand)
            return ast.copy_location(self._fn_call("not_", [operand]), node)
        return self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        # `a if cond else b`
        return ast.copy_location(
            self._fn_call(
                "where",
                [self.visit(node.test), self.visit(node.body), self.visit(node.orelse)],
            ),
            node,
        )

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        # `a < b < c` -> where(a < b, b < c, 0): Python's chained comparison
        # would otherwise short-circuit through __bool__.
        node = self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        pairs: List[ast.expr] = []
        operands = [node.left] + list(node.comparators)
        for i, op in enumerate(node.ops):
            pairs.append(
                ast.Compare(left=operands[i], ops=[op], comparators=[operands[i + 1]])
            )
        result = pairs[-1]
        for cmp_node in reversed(pairs[:-1]):
            result = self._fn_call(
                "where", [cmp_node, result, ast.Constant(value=0)]
            )
        return ast.copy_location(result, node)

    # -- statement-level rewrites --------------------------------------------

    def _rewrite_if(self, node: ast.If) -> List[ast.stmt]:
        """Rewrite an if statement whose arms only assign or return."""
        test = self.visit(node.test)
        body = [self.visit(s) for s in node.body]
        orelse = [self.visit(s) for s in node.orelse]

        test_name = f"__policy_cond_{id(node) & 0xFFFF}"
        stmts: List[ast.stmt] = [
            ast.Assign(
                targets=[ast.Name(id=test_name, ctx=ast.Store())], value=test
            )
        ]

        def arm_map(arm: List[ast.stmt]):
            """Map each assigned name to its value; None if the arm returns."""
            assigns = {}
            ret = None
            for stmt in arm:
                if isinstance(stmt, ast.Assign):
                    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                        raise _err(
                            self.fn_name,
                            stmt,
                            "only simple `name = expression` assignments can be "
                            "rewritten inside an `if`; use fn.where(...) directly",
                        )
                    assigns[stmt.targets[0].id] = stmt.value
                elif isinstance(stmt, ast.AugAssign):
                    if not isinstance(stmt.target, ast.Name):
                        raise _err(self.fn_name, stmt, "unsupported augmented assignment target")
                    assigns[stmt.target.id] = ast.BinOp(
                        left=ast.Name(id=stmt.target.id, ctx=ast.Load()),
                        op=stmt.op,
                        right=stmt.value,
                    )
                elif isinstance(stmt, ast.Return):
                    ret = stmt.value
                elif isinstance(stmt, ast.Pass):
                    continue
                else:
                    raise _err(
                        self.fn_name,
                        stmt,
                        f"`{type(stmt).__name__}` cannot appear inside an `if` in a "
                        "policy function; keep branches to assignments or a return",
                    )
            return assigns, ret

        body_assigns, body_ret = arm_map(body)
        else_assigns, else_ret = arm_map(orelse)

        if (body_ret is None) != (else_ret is None):
            raise _err(
                self.fn_name,
                node,
                "an `if` in a policy function must either return in both "
                "branches or in neither; a bare `if cond: return x` has no "
                "value for the other branch",
            )

        if body_ret is not None:
            stmts.append(
                ast.Return(
                    value=self._fn_call(
                        "where",
                        [ast.Name(id=test_name, ctx=ast.Load()), body_ret, else_ret],
                    )
                )
            )
            return stmts

        # Every name touched by either arm becomes a where() over its two values.
        for name in list(body_assigns) + [n for n in else_assigns if n not in body_assigns]:
            true_value = body_assigns.get(name, ast.Name(id=name, ctx=ast.Load()))
            false_value = else_assigns.get(name, ast.Name(id=name, ctx=ast.Load()))
            stmts.append(
                ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=self._fn_call(
                        "where",
                        [ast.Name(id=test_name, ctx=ast.Load()), true_value, false_value],
                    ),
                )
            )
        return stmts

    def visit_If(self, node: ast.If) -> List[ast.stmt]:
        return [ast.copy_location(s, node) for s in self._rewrite_if(node)]

    # -- rejected constructs --------------------------------------------------

    def _reject(self, node, what: str):
        raise _err(
            self.fn_name,
            node,
            f"`{what}` is not supported in a policy function; a policy must be a "
            "straight-line expression over knobs (use fn.where for conditionals)",
        )

    def visit_For(self, node): self._reject(node, "for")
    def visit_While(self, node): self._reject(node, "while")
    def visit_Try(self, node): self._reject(node, "try")
    def visit_With(self, node): self._reject(node, "with")
    def visit_Lambda(self, node): self._reject(node, "lambda")
    def visit_ListComp(self, node): self._reject(node, "list comprehension")
    def visit_SetComp(self, node): self._reject(node, "set comprehension")
    def visit_DictComp(self, node): self._reject(node, "dict comprehension")
    def visit_GeneratorExp(self, node): self._reject(node, "generator expression")
    def visit_Yield(self, node): self._reject(node, "yield")
    def visit_YieldFrom(self, node): self._reject(node, "yield from")
    def visit_Await(self, node): self._reject(node, "await")
    def visit_Assert(self, node): self._reject(node, "assert")
    def visit_Raise(self, node): self._reject(node, "raise")
    def visit_Import(self, node): self._reject(node, "import")
    def visit_ImportFrom(self, node): self._reject(node, "import")


def rewrite_policy_function(func: Callable) -> Callable:
    """Return `func` with its control flow rewritten into fn.where calls.

    Raises RewriteError when the source is unavailable -- silently skipping the
    rewrite would let a plain `if` mis-trace into a constant.
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as exc:
        raise RewriteError(
            f"cannot read the source of {func.__name__!r}, so its control flow "
            "cannot be rewritten. Define policy functions at module level in a "
            "real file (not in a REPL, exec() or lambda)."
        ) from exc

    tree = ast.parse(textwrap.dedent(source))
    func_def = tree.body[0]
    if not isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise RewriteError(f"{func.__name__!r} is not a plain function definition")
    if isinstance(func_def, ast.AsyncFunctionDef):
        raise RewriteError(f"{func.__name__!r} must not be async")

    arg_names = [a.arg for a in func_def.args.args]
    if len(arg_names) < 3:
        raise RewriteError(
            f"{func.__name__!r} must take three arguments (knobs, params, math), "
            f"e.g. `def {func.__name__}(k, p, fn):` -- got {arg_names or 'none'}"
        )
    fn_arg = arg_names[2]

    rewriter = _Rewriter(func.__name__, fn_arg, set(arg_names))
    new_body = []
    for stmt in func_def.body:
        result = rewriter.visit(stmt)
        if isinstance(result, list):
            new_body.extend(result)
        elif result is not None:
            new_body.append(result)
    func_def.body = new_body

    # Drop decorators so re-executing the definition does not recurse.
    func_def.decorator_list = []
    ast.fix_missing_locations(tree)

    namespace = dict(getattr(func, "__globals__", {}))
    filename = getattr(func, "__code__", None)
    filename = filename.co_filename if filename else "<policy>"
    exec(compile(tree, filename, "exec"), namespace)  # noqa: S102
    rewritten = namespace[func_def.name]
    rewritten.__doc__ = func.__doc__
    return rewritten
