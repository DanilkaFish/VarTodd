from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pyvartodd.Release.pyvartodd import (  # type: ignore
    ActionPool,
    ActionSelection,
    CandidateExport,
    ExplorationScore,
    FinalizationScore,
    Matrix,
    PolicyConfig,
    PolicyScores,
    Result,
    SamplingBudget,
    ScoringFunction,
    SourcePool,
    Stats,
    ToddSearch,
    Tensor3D,
    TohpePrefixSearch,
    TohpeSearch,
    ZBucketSearch,
    policy_iteration,
)


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
