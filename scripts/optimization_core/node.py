from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional
from pyvartodd.Release.pyvartodd import Matrix, CandidateExport, Stats, Result,  PolicyConfig, ExplorationScore, FinalizationScore, policy_iteration, Tensor3D


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

@dataclass(slots=True)
class Node:
    state: Matrix
    parent: Optional["Node"] = None
    incoming: Optional[ActionInfo] = None
    depth: int = 0

    expanded: bool = False
    exhausted: bool = False
    active: bool = True

    frozen_until: int = 0

    # MCTS stats
    visits: int = 0
    value_sum: float = 0.0

    # prior used by PUCT (set at expansion time)
    prior: float = 1.0

    children: List["Node"] = field(default_factory=list)

    @property
    def value_mean(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def record(self, value: float) -> None:
        self.visits += 1
        self.value_sum += float(value)

    def backpropagate(self, *, value: float, discount: float = 1.0) -> None:
        v = float(value)
        cur: Optional["Node"] = self
        while cur is not None:
            cur.record(v)
            v *= discount
            cur = cur.parent

    def is_frozen(self, it: int) -> bool:
        return it < self.frozen_until

    def iter_selectable_children(self, it: int) -> Iterator["Node"]:
        for c in self.children:
            if c.active and (not c.exhausted) and (not c.is_frozen(it)):
                yield c

    def add_child(
        self,
        *,
        state: Matrix,
        incoming: ActionInfo,
        prior: float,
        frozen_until: int = 0,
        active: bool = True,
    ) -> "Node":
        child = Node(
            state=state,
            parent=self,
            incoming=incoming,
            depth=self.depth + 1,
            active=active,
            frozen_until=frozen_until,
            prior=float(prior),
        )
        self.children.append(child)
        return child

    def deactivate(self) -> None:
        self.active = False

    def recompute_exhausted_upwards(self, it: int) -> None:
        """
        If a node has no selectable children, mark it exhausted; then check parent, etc.
        """
        cur: Optional["Node"] = self
        while cur is not None:
            if cur.exhausted:
                cur = cur.parent
                continue
            if cur.expanded and not any(True for _ in cur.iter_selectable_children(it)):
                cur.exhausted = True
            else:
                break
            cur = cur.parent

    @property
    def is_terminal(self) -> bool:
        # "search terminal"
        return self.exhausted

    def path_from_root(self) -> List["Node"]:
        out: List["Node"] = []
        cur: Optional["Node"] = self
        while cur is not None:
            out.append(cur)
            cur = cur.parent
        out.reverse()
        return out