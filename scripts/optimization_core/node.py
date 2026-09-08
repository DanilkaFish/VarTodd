from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

def _load_extension():
    """Import the built extension, preferring the most recently built one.

    Only one configuration's module may load per process -- pybind11 refuses a
    second registration of the same types -- so a stale build left over from an
    earlier configuration must not shadow the current one.
    """
    import os
    import sys

    relatives = (
        ("pyvartodd.Release.pyvartodd", "pyvartodd/Release"),
        ("pyvartodd.Debug.pyvartodd", "pyvartodd/Debug"),
        ("pyvartodd.RelWithDebInfo.pyvartodd", "pyvartodd/RelWithDebInfo"),
        ("pyvartodd.pyvartodd", "pyvartodd"),
    )

    # This file is symlinked into per-matrix overlay directories, so the repo
    # root is found relative to the real file, not the symlink, and the overlay
    # is not always three levels deep.  Walk up until a directory holding a
    # built extension turns up.
    def _find_root(start):
        directory = start
        while True:
            for _, relative in relatives:
                if os.path.isdir(os.path.join(directory, relative)):
                    return directory
            parent = os.path.dirname(directory)
            if parent == directory:
                return None
            directory = parent

    here = os.path.dirname(os.path.realpath(__file__))
    root = _find_root(here) or _find_root(os.path.dirname(os.path.abspath(__file__)))
    if root is None:
        root = os.path.dirname(os.path.dirname(here))
    # The extension is built into the repo root, which is not necessarily on
    # sys.path when a program is run from inside scripts/.
    if root not in sys.path:
        sys.path.insert(0, root)
    candidates = []
    for module, relative in relatives:
        directory = os.path.join(root, relative)
        if not os.path.isdir(directory):
            continue
        for entry in os.listdir(directory):
            if entry.startswith("pyvartodd.") and entry.endswith(".so"):
                candidates.append((os.path.getmtime(os.path.join(directory, entry)), module))
                break

    errors = []
    for _, module in sorted(candidates, reverse=True):
        try:
            return __import__(module, fromlist=["policy_iteration"])
        except Exception as exc:  # a stale or ABI-mismatched build
            errors.append(f"{module}: {exc}")
    raise ImportError(
        "could not import the pyvartodd extension; build it with "
        "`cmake --build build`. Tried:\n  " + "\n  ".join(errors or ["no built module found"])
    )


_ext = _load_extension()

ActionPool = _ext.ActionPool
ActionSelection = _ext.ActionSelection
CandidateExport = _ext.CandidateExport
Matrix = _ext.Matrix
PolicyConfig = _ext.PolicyConfig
PolicyProgram = _ext.PolicyProgram
PolicyScores = _ext.PolicyScores
Result = _ext.Result
SamplingBudget = _ext.SamplingBudget
SourcePool = _ext.SourcePool
Stats = _ext.Stats
ToddSearch = _ext.ToddSearch
Tensor3D = _ext.Tensor3D
TohpePrefixSearch = _ext.TohpePrefixSearch
TohpeSearch = _ext.TohpeSearch
ZBucketSearch = _ext.ZBucketSearch
policy_iteration = _ext.policy_iteration


@dataclass(slots=True)
class ActionInfo:
    cand: CandidateExport
    global_info: Stats
    source: str = ""

    @staticmethod
    def from_candidate(
        cand: CandidateExport,
        *,
        global_info: Optional[Stats] = None,
        source: str = "",
    ) -> "ActionInfo":
        return ActionInfo(
            cand=cand,
            global_info=global_info,
            source=source,
        )

    @property
    def reduction(self):
        return self.cand.reduction

    @property
    def total(self) -> int:
        if self.global_info is None:
            return 0
        return int(getattr(self.global_info, "total", 0) or 0)

@dataclass(slots=True)
class Node:
    state: Matrix
    parent: Optional["Node"] = None
    incoming: Optional[ActionInfo] = None
    depth: int = 0

    def add_child(
        self,
        *,
        state: Matrix,
        incoming: ActionInfo
    ) -> "Node":
        child = Node(
            state=state,
            parent=self,
            incoming=incoming,
            depth=self.depth + 1,
        )
        return child

    def path_from_root(self) -> List["Node"]:
        out: List["Node"] = []
        cur: Optional["Node"] = self
        while cur is not None:
            out.append(cur)
            cur = cur.parent
        out.reverse()
        return out
