
from __future__ import annotations

import heapq
from typing import Any

from .mcts_dao import Dao, Path
from .node import ActionInfo, Node, Result, policy_iteration


class Todd:
    def __init__(self, dao: Dao, depth: int):
        self.dao: Dao = dao
        self.depth = int(depth)

    @staticmethod
    def _beam_key(node: Node) -> tuple[int, float]:
        cand = node.incoming.cand if node.incoming is not None else None
        if cand is None:
            return (-node.state.rows, 0.0)
        return (-node.state.rows, float(cand.final_score))

    def run(
        self,
        path: Path,
        *args: Any,
        with_report: bool = False,
        seed: int = 1,
        **kwargs: Any,
    ):
        if "with_report" in kwargs:
            with_report = bool(kwargs.pop("with_report"))
        if "seed" in kwargs:
            seed = int(kwargs.pop("seed"))
        for legacy_key in ("width", "bs_width", "beamwidth", "todd_width"):
            kwargs.pop(legacy_key, None)
        if kwargs:
            raise TypeError(f"unexpected Todd.run keyword arguments: {sorted(kwargs)}")

        if args:
            if len(args) == 1:
                with_report = bool(args[0])
            elif len(args) == 2:
                if isinstance(args[0], bool):
                    with_report = bool(args[0])
                    seed = int(args[1])
            elif len(args) in (3, 4):
                with_report = bool(args[2])
                if len(args) == 4:
                    seed = int(args[3])
            else:
                raise TypeError(
                    "Todd.run expected at most 4 positional compatibility "
                    f"arguments, got {len(args)}"
                )

        root = path.final_node
        if root is None:
            raise ValueError("cannot run TODD from an empty path")

        best_node = root
        counter = 0
        best_counter = 0
        nodes = [root]
        for _ in range(self.depth):
            new_nodes = []
            next_width = 1
            counter = max(counter, len(nodes))
            for node in nodes:
                pcfg = self.dao.policy_config_at(depth=node.state.rows, mode="default")
                next_width = max(next_width, max(1, int(pcfg.selection.count)))
                out: Result = policy_iteration(cur_mat=node.state, policy_cfg=pcfg, seed=seed, add_seed=0)
                chosen = out.chosen
                states = out.states
                if not chosen or not states:
                    continue
                if len(states) == len(chosen) + 1:
                    states = states[1:]

                for cand, state in zip(chosen, states):
                    info = ActionInfo.from_candidate(cand, global_info=out.stats, source="rollout")
                    child = node.add_child(
                        state=state,
                        incoming=info,
                    )
                    if child.state.rows < best_node.state.rows:
                        best_counter = 0
                        best_node = child
                    if child.state.rows == best_node.state.rows:
                        best_counter += 1
                    new_nodes.append(child)
            if not new_nodes:
                break 
            nodes = heapq.nlargest(next_width, new_nodes, self._beam_key)

        if with_report:
            best_counter = min(counter, best_counter)
            return best_node, (counter, best_counter)

        return best_node.state.to_numpy()
