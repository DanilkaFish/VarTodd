
from __future__ import annotations

from collections.abc import Callable
import heapq
import time
from typing import Any

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
        *args: Any,
        with_report: bool = False,
        with_timing: bool = False,
        seed: int = 1,
        stop_requested: Callable[[], bool] | None = None,
        **kwargs: Any,
    ):
        if "with_report" in kwargs:
            with_report = bool(kwargs.pop("with_report"))
        if "seed" in kwargs:
            seed = int(kwargs.pop("seed"))
        if "with_timing" in kwargs:
            with_timing = bool(kwargs.pop("with_timing"))
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
        best_discovered_at = time.perf_counter() if with_report and with_timing else None
        counter = 0
        best_counter = 0
        nodes = [root]
        for _ in range(self.depth):
            merged_nodes: list[tuple[Node, float | None]] = []
            next_width = 1
            counter = max(counter, len(nodes))
            for node in nodes:
                parent_nodes: list[tuple[Node, float | None]] = []
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
                    seen_at = (
                        time.perf_counter()
                        if with_report and with_timing
                        else None
                    )
                    parent_nodes.append((child, seen_at))

                # policy_iteration guarantees distinct states within one
                # parent batch.  Only compare against states produced from
                # earlier parent matrices during the global beam merge.
                earlier_parent_count = len(merged_nodes)
                for child, seen_at in parent_nodes:
                    duplicate_index = next(
                        (
                            index
                            for index, (previous, _seen_at) in enumerate(
                                merged_nodes[:earlier_parent_count]
                            )
                            if child.state == previous.state
                        ),
                        None,
                    )
                    if duplicate_index is None:
                        merged_nodes.append((child, seen_at))
                        continue

                    previous, seen_at = merged_nodes[duplicate_index]
                    child_score = float(child.incoming.cand.final_score)
                    previous_score = float(previous.incoming.cand.final_score)
                    if child_score > previous_score:
                        merged_nodes[duplicate_index] = (child, seen_at)
            if not merged_nodes:
                break 
            new_nodes = [child for child, _seen_at in merged_nodes]
            for child, seen_at in merged_nodes:
                if child.state.rows < best_node.state.rows:
                    best_counter = 0
                    best_node = child
                    if with_report and with_timing:
                        best_discovered_at = seen_at
                if child.state.rows == best_node.state.rows:
                    best_counter += 1
            nodes = heapq.nlargest(next_width, new_nodes, self._beam_key)
            if stop_requested is not None and stop_requested():
                break

        if with_report:
            best_counter = min(counter, best_counter)
            if with_timing:
                return best_node, (counter, best_counter), best_discovered_at
            return best_node, (counter, best_counter)

        return best_node.state.to_numpy()
