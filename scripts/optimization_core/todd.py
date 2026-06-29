
from __future__ import annotations

import heapq
from typing import Any

try:
    from .mcts_dao import Dao, Path
    from .node import ActionInfo, Node, Result, policy_iteration
except ImportError:  # kept for generated scripts that import helper as a flat module
    from mcts_dao import Dao, Path
    from node import ActionInfo, Node, Result, policy_iteration


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
        width: Any,
        todd_width: Any,
        with_report: bool = False,
        seed: int = 1,
    ):
        root = path.final_node
        if root is None:
            raise ValueError("cannot run TODD from an empty path")

        best_node = root
        counter = 0
        best_counter = 0
        nodes = [root]
        for _ in range(self.depth):
            new_nodes = []
            counter = max(counter, len(nodes))
            for node in nodes:
                pcfg = self.dao.policy_config_at(depth=node.state.rows, mode="default", action_count=todd_width.at(node.state.rows))
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
            nodes = heapq.nlargest(width.at(best_node.state.rows), new_nodes, self._beam_key)

        if with_report:
            best_counter = min(counter, best_counter)
            return best_node, (counter, best_counter)

        return best_node.state.to_numpy()
