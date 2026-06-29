from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path as FsPath
from typing import Any, Iterator, List, Optional, Sequence
import json
import os
import pickle
from copy import deepcopy
try:
    import fcntl
except ImportError:  # pragma: no cover - Linux in production, fallback for portability
    fcntl = None

import numpy as np

try:
    from .mcts_dao import Path as MctsPath, Dao
    from .node import ActionInfo, Matrix, Node
except ImportError:  # kept for generated scripts that import helper as a flat module
    from mcts_dao import Path as MctsPath, Dao
    from node import ActionInfo, Matrix, Node

X0_LENGTH = 200
USAGE_INDEX_NAME = "_usage_index.json"
USAGE_LOCK_NAME = "_usage_index.lock"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

@dataclass
class PathStore:
    root_dir: str = "data/path_backups"

    def _resolve_dir(self, name: str) -> FsPath:
        if not name:
            raise ValueError("name must be a non-empty string")
        base = FsPath(self.root_dir) / name
        return base

    @property
    def root_path(self) -> FsPath:
        return FsPath(self.root_dir)

    def _usage_index_path(self) -> FsPath:
        return self.root_path / USAGE_INDEX_NAME

    def _usage_lock_path(self) -> FsPath:
        return self.root_path / USAGE_LOCK_NAME

    @contextmanager
    def _locked_usage_index(self) -> Iterator[dict[str, Any]]:
        self.root_path.mkdir(parents=True, exist_ok=True)
        lock_path = self._usage_lock_path()
        index_path = self._usage_index_path()
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    with open(index_path, "r", encoding="utf-8") as f:
                        index = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    index = {"version": 1, "paths": {}}

                if not isinstance(index, dict):
                    index = {"version": 1, "paths": {}}
                if not isinstance(index.get("paths"), dict):
                    index["paths"] = {}

                yield index

                tmp_path = index_path.with_suffix(".json.tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2, sort_keys=True)
                os.replace(tmp_path, index_path)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _ensure_usage_record(self, index: dict[str, Any], name: str) -> dict[str, Any]:
        paths = index.setdefault("paths", {})
        record = paths.setdefault(
            name,
            {
                "used_count": 0,
                "improved_count": 0,
                "best_child_rank": None,
                "last_child_rank": None,
                "last_used_at": None,
                "last_improved_at": None,
                "init_rank_thr_counts": {},
            },
        )
        record.setdefault("used_count", 0)
        record.setdefault("improved_count", 0)
        record.setdefault("best_child_rank", None)
        record.setdefault("last_child_rank", None)
        record.setdefault("last_used_at", None)
        record.setdefault("last_improved_at", None)
        record.setdefault("init_rank_thr_counts", {})
        return record

    def record_load(self, name: str, *, init_rank_thr: Optional[int] = None) -> None:
        with self._locked_usage_index() as index:
            record = self._ensure_usage_record(index, name)
            record["used_count"] = int(record.get("used_count") or 0) + 1
            record["last_used_at"] = _utc_now()
            if init_rank_thr is not None:
                counts = record.setdefault("init_rank_thr_counts", {})
                key = str(int(init_rank_thr))
                counts[key] = int(counts.get(key) or 0) + 1

    def record_result(
        self,
        name: str,
        *,
        loaded_rank: int,
        child_rank: int,
        init_rank_thr: Optional[int] = None,
    ) -> None:
        with self._locked_usage_index() as index:
            record = self._ensure_usage_record(index, name)
            child_rank = int(child_rank)
            record["last_child_rank"] = child_rank
            best_child = record.get("best_child_rank")
            if best_child is None or child_rank < int(best_child):
                record["best_child_rank"] = child_rank
            if child_rank < int(loaded_rank):
                record["improved_count"] = int(record.get("improved_count") or 0) + 1
                record["last_improved_at"] = _utc_now()
            if init_rank_thr is not None:
                counts = record.setdefault("init_rank_thr_counts", {})
                counts.setdefault(str(int(init_rank_thr)), 0)

    def read_usage_index(self) -> dict[str, Any]:
        try:
            with open(self._usage_index_path(), "r", encoding="utf-8") as f:
                index = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"version": 1, "paths": {}}
        if not isinstance(index, dict) or not isinstance(index.get("paths"), dict):
            return {"version": 1, "paths": {}}
        return index

    def save(
        self,
        name: str,
        paths: Sequence[MctsPath],
        *,
        store_daos: bool = True,
        parent_info: Optional[dict[str, Any]] = None,
    ) -> FsPath:
        base = self._resolve_dir(name)
        base.mkdir(parents=True, exist_ok=True)

        matrices_path = base / "matrices.npz"
        meta_path = base / "meta.json"
        daos_path = base / "daos.pkl"
        incoming_path = base / "incoming.pkl"
        widths_path = base / "widths.pkl"

        matrices: dict[str, np.ndarray] = {}
        meta = {
            "version": 3,
            "paths": [],
        }
        if parent_info:
            meta["parent"] = dict(parent_info)
        incoming_payload: List[List[Optional[tuple]]] = []

        for p_idx, path in enumerate(paths):
            if path.final_node is None:
                raise ValueError(f"path at index {p_idx} has no final_node")

            nodes: List[Node] = []
            cur = path.final_node
            while cur is not None:
                nodes.append(cur)
                cur = cur.parent
            nodes.reverse()

            matrix_keys: List[str] = []
            incoming_info: List[Optional[tuple]] = []
            for s_idx, node in enumerate(nodes):
                key = f"p{p_idx}_s{s_idx}"
                matrices[key] = node.state.to_numpy()
                matrix_keys.append(key)
                if node.incoming is None:
                    incoming_info.append(None)
                else:
                    incoming_info.append((node.incoming.cand, node.incoming.global_info, node.incoming.source))
            x0s = [list(x0) for x0 in path.x0s]
            for x0 in x0s:
                while x0 and x0[-1] == 0:
                    x0.pop()
            limit_buckets = self._path_limit_buckets({}, path.daos)
            meta["paths"].append(
                {
                    "matrix_keys": matrix_keys,
                    "ranks_thr": list(path.ranks_thr),
                    "x0s": x0s,
                    "limit_buckets": limit_buckets,
                }
            )
            incoming_payload.append(incoming_info)

        np.savez_compressed(matrices_path, **matrices)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if store_daos:
            daos_payload = [path.daos for path in paths]
            with open(daos_path, "wb") as f:
                pickle.dump(daos_payload, f)
            widths_payload = [
                {
                    "bs_widths": path.bs_widths,
                    "todd_widths": path.todd_widths,
                }
                for path in paths
            ]
            with open(widths_path, "wb") as f:
                pickle.dump(widths_payload, f)
        with open(incoming_path, "wb") as f:
            pickle.dump(incoming_payload, f)

        return base

    @staticmethod
    def _path_kind(_name: str, init_rank: Optional[int], mid_rank: Optional[int]) -> str:
        if init_rank is None or mid_rank is None:
            return "unknown"
        return "full" if init_rank == mid_rank else "partial"

    @staticmethod
    def _restart_band(record: dict[str, Any]) -> str:
        init_rank = record.get("init_rank")
        mid_rank = record.get("init_rank_thr")
        final_rank = record.get("rank")
        if init_rank is None or mid_rank is None or final_rank is None:
            return "unknown"
        init_rank = int(init_rank)
        mid_rank = int(mid_rank)
        final_rank = int(final_rank)
        if mid_rank == init_rank:
            return "from_init"
        midpoint = (init_rank + final_rank) / 2.0
        if mid_rank > midpoint:
            return "first_half"
        return "near_final"

    @staticmethod
    def _record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
        mid_rank = record.get("init_rank_thr")
        mid_rank_key = int(mid_rank) if mid_rank is not None else -1
        return (
            int(record["rank"]),
            PathStore._limit_buckets_sort_key(record.get("limit_buckets")),
            -int(record.get("improved_count") or 0),
            -mid_rank_key,
            int(record.get("used_count") or 0),
            record["name"],
        )

    @staticmethod
    def _limit_buckets_sort_key(value: Any) -> int:
        if value is None:
            return 10**18 - 1
        value = int(value)
        if value < 0:
            return 10**18
        return value

    @staticmethod
    def _format_limit_buckets(value: Any) -> str:
        if value is None:
            return "unknown"
        value = int(value)
        if value < 0:
            return "-1(full)"
        return str(value)

    @staticmethod
    def _path_limit_buckets(path_meta: dict[str, Any], daos: Optional[Sequence[Dao]]) -> Optional[int]:
        stored = path_meta.get("limit_buckets")
        if stored is not None:
            return int(stored)
        if not daos:
            return None

        limits: list[int] = []
        for dao in daos:
            try:
                for _rank, todd in dao.mode.todd.points:
                    limits.append(int(todd.buckets.limit_bucket))
            except Exception:
                continue
        if not limits:
            return None
        if any(limit < 0 for limit in limits):
            return -1
        return max(limits)

    def iter_path_records(self) -> list[dict[str, Any]]:
        import re

        root = self.root_path
        if not root.exists() or not root.is_dir():
            return []

        usage = self.read_usage_index().get("paths", {})
        name_re = re.compile(r"i(?P<init>\d+)_m(?P<thr>\d+)_f(?P<final>\d+)")
        records: list[dict[str, Any]] = []

        for backup_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            meta_path = backup_dir / "meta.json"
            matrices_path = backup_dir / "matrices.npz"
            if not meta_path.exists() or not matrices_path.exists():
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                paths_meta = meta.get("paths", [])
                if not paths_meta:
                    continue

                best_rank = None
                best_depth = None
                best_path_idx = None
                with np.load(matrices_path) as data:
                    for p_idx, p in enumerate(paths_meta):
                        keys = p.get("matrix_keys", [])
                        if not keys:
                            continue
                        last_key = keys[-1]
                        if last_key not in data:
                            continue
                        rank = int(data[last_key].shape[0])
                        depth = max(0, len(keys) - 1)
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_depth = depth
                            best_path_idx = p_idx

                if best_rank is None:
                    continue

                best_path_meta = (
                    paths_meta[best_path_idx]
                    if best_path_idx is not None and best_path_idx < len(paths_meta)
                    else {}
                )
                daos_for_best: Optional[Sequence[Dao]] = None
                if self._path_limit_buckets(best_path_meta, None) is None:
                    daos_path = backup_dir / "daos.pkl"
                    if daos_path.exists():
                        try:
                            with open(daos_path, "rb") as f:
                                daos_payload = pickle.load(f)
                            if (
                                isinstance(daos_payload, list)
                                and best_path_idx is not None
                                and best_path_idx < len(daos_payload)
                            ):
                                daos_for_best = daos_payload[best_path_idx]
                        except Exception:
                            daos_for_best = None
                limit_buckets = self._path_limit_buckets(best_path_meta, daos_for_best)

                init_rank = None
                init_thr_rank = None
                match = name_re.search(backup_dir.name)
                if match:
                    init_rank = int(match.group("init"))
                    init_thr_rank = int(match.group("thr"))

                usage_record = usage.get(backup_dir.name, {})
                parent_info = meta.get("parent", {})
                if not isinstance(parent_info, dict):
                    parent_info = {}
                parent_path_name = parent_info.get("parent_path_name")
                parent_usage = (
                    usage.get(parent_path_name, {})
                    if isinstance(parent_path_name, str)
                    else {}
                )
                parent_best_child_rank = parent_usage.get("best_child_rank")
                is_stale_improved_child = False
                if (
                    parent_info.get("child_improved_loaded") is True
                    and parent_best_child_rank is not None
                    and best_rank > int(parent_best_child_rank)
                ):
                    is_stale_improved_child = True
                records.append(
                    {
                        "name": backup_dir.name,
                        "rank": best_rank,
                        "depth": best_depth,
                        "limit_buckets": limit_buckets,
                        "init_rank": init_rank,
                        "init_rank_thr": init_thr_rank,
                        "kind": self._path_kind(backup_dir.name, init_rank, init_thr_rank),
                        "restart_band": self._restart_band(
                            {
                                "init_rank": init_rank,
                                "init_rank_thr": init_thr_rank,
                                "rank": best_rank,
                            }
                        ),
                        "used_count": int(usage_record.get("used_count") or 0),
                        "improved_count": int(usage_record.get("improved_count") or 0),
                        "best_child_rank": usage_record.get("best_child_rank"),
                        "last_child_rank": usage_record.get("last_child_rank"),
                        "last_used_at": usage_record.get("last_used_at"),
                        "last_improved_at": usage_record.get("last_improved_at"),
                        "parent_path_name": parent_path_name,
                        "parent_loaded_rank": parent_info.get(
                            "loaded_path_rank", parent_info.get("loaded_rank")
                        ),
                        "parent_loaded_start_rank": parent_info.get("loaded_rank"),
                        "parent_init_rank_thr": parent_info.get("init_rank_thr"),
                        "parent_best_child_rank": parent_best_child_rank,
                        "child_improved_loaded": parent_info.get("child_improved_loaded"),
                        "is_stale_improved_child": is_stale_improved_child,
                    }
                )
            except Exception:
                continue

        records.sort(key=self._record_sort_key)
        return records

    @staticmethod
    def _nonimproved_reuse_count(record: dict[str, Any]) -> int:
        return max(
            0,
            int(record.get("used_count") or 0) - int(record.get("improved_count") or 0),
        )

    @classmethod
    def _is_live_record(
        cls,
        record: dict[str, Any],
        *,
        max_nonimproved_reuse: int,
    ) -> bool:
        if record.get("child_improved_loaded") is False:
            return False
        if record.get("is_stale_improved_child"):
            return False
        return cls._nonimproved_reuse_count(record) <= int(max_nonimproved_reuse)

    @staticmethod
    def _format_summary_record(record: dict[str, Any]) -> str:
        best_child = record["best_child_rank"]
        best_child_text = "none" if best_child is None else str(best_child)
        parent_text = ""
        if record.get("parent_path_name"):
            start_rank = record.get("parent_loaded_start_rank")
            start_text = f" start={start_rank}" if start_rank is not None else ""
            parent_text = (
                f" parent={record['parent_path_name']}"
                f" loaded={record.get('parent_loaded_rank')}"
                f"{start_text}"
                f" thr={record.get('parent_init_rank_thr')}"
            )
        return (
            f"{record['name']} rank={record['rank']} depth={record['depth']} "
            f"limit_buckets={PathStore._format_limit_buckets(record.get('limit_buckets'))} "
            f"u/i={record['used_count']}/{record['improved_count']} "
            f"child={best_child_text} kind={record['kind']} "
            f"band={record.get('restart_band', 'unknown')}"
            f"{parent_text}"
        )

    def summarize(
        self,
        *,
        top_k: int = 6,
        per_rank_cap: int = 1,
        max_nonimproved_reuse: int = 4,
        dead_end_after: Optional[int] = None,
        dead_end_top_k: int = 2,
        nonimproved_child_top_k: int = 2,
        improved_dead_end_top_k: int = 1,
    ) -> str:
        records = self.iter_path_records()
        if not records:
            return (
                "best_paths:\n"
                "- none\n"
                "rule=use_only_names_under_best_paths\n"
                "balance=from_init:0,first_half:0,near_final:0,unknown:0; "
                "full:0,partial:0; live:0,total:0,best_rank_count:0"
            )

        top_k = min(6, max(1, int(top_k)))
        per_rank_cap = max(1, int(per_rank_cap))
        if dead_end_after is not None:
            max_nonimproved_reuse = dead_end_after
        max_nonimproved_reuse = max(0, int(max_nonimproved_reuse))
        count_full = sum(1 for r in records if r["kind"] == "full")
        count_partial = sum(1 for r in records if r["kind"] == "partial")
        best_rank = min(r["rank"] for r in records)
        best_rank_count = sum(1 for r in records if r["rank"] == best_rank)
        nonimproved_child_count = sum(1 for r in records if r.get("child_improved_loaded") is False)
        stale_improved_child_count = sum(1 for r in records if r.get("is_stale_improved_child"))
        over_failed_reuse_count = sum(
            1 for r in records if self._nonimproved_reuse_count(r) > max_nonimproved_reuse
        )
        live_records = [
            r
            for r in records
            if self._is_live_record(r, max_nonimproved_reuse=max_nonimproved_reuse)
        ]
        improved_dead_end_records = [
            r
            for r in records
            if self._nonimproved_reuse_count(r) > max_nonimproved_reuse
            and int(r.get("improved_count") or 0) > 0
            and r.get("child_improved_loaded") is not False
            and not r.get("is_stale_improved_child")
        ]
        revived_records = improved_dead_end_records[: max(0, int(improved_dead_end_top_k))]

        live_band_counts = {"from_init": 0, "first_half": 0, "near_final": 0, "unknown": 0}
        for record in live_records + revived_records:
            band = str(record.get("restart_band") or "unknown")
            live_band_counts[band] = live_band_counts.get(band, 0) + 1

        selected = []
        rank_counts: dict[int, int] = {}
        for record in live_records:
            if len(selected) >= top_k:
                break
            rank = int(record["rank"])
            if rank_counts.get(rank, 0) >= per_rank_cap:
                continue
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            selected.append(self._format_summary_record(record))

        selected_names = {line.split(" ", 1)[0] for line in selected if line and line != "none"}
        for record in revived_records:
            if record["name"] in selected_names:
                continue
            selected.append(
                f"{self._format_summary_record(record)} "
                f"revived=1 failed_reuse={self._nonimproved_reuse_count(record)}"
            )
            selected_names.add(record["name"])

        if not selected:
            selected.append("none")

        dead_end_records = [
            r
            for r in records
            if self._nonimproved_reuse_count(r) > max_nonimproved_reuse
            and not r.get("is_stale_improved_child")
        ][: max(0, int(dead_end_top_k))]
        nonimproved_child_records = [
            r for r in records if r.get("child_improved_loaded") is False
        ][: max(0, int(nonimproved_child_top_k))]
        dead_end_lines = [
            f"{self._format_summary_record(record)} "
            f"failed_reuse={self._nonimproved_reuse_count(record)}"
            for record in dead_end_records
        ]
        nonimproved_child_lines = [
            self._format_summary_record(record)
            for record in nonimproved_child_records
        ]

        lines = ["best_paths:", *[f"- {line}" for line in selected]]
        if dead_end_lines:
            lines += ["dead_end_paths:", *[f"- {line}" for line in dead_end_lines]]
        if nonimproved_child_lines:
            lines += [
                "best_nonimproved_child_paths:",
                *[f"- {line}" for line in nonimproved_child_lines],
            ]
        lines += [
            "rule=use_only_names_under_best_paths",
            (
                "balance="
                f"from_init:{live_band_counts.get('from_init', 0)},"
                f"first_half:{live_band_counts.get('first_half', 0)},"
                f"near_final:{live_band_counts.get('near_final', 0)},"
                f"unknown:{live_band_counts.get('unknown', 0)}; "
                f"full:{count_full},partial:{count_partial}; "
                f"live:{len(live_records)},total:{len(records)},best_rank_count:{best_rank_count}"
            ),
        ]
        if (
            nonimproved_child_count
            or stale_improved_child_count
            or over_failed_reuse_count
            or improved_dead_end_records
        ):
            lines.append(
                "omitted="
                f"nonimproved_child:{nonimproved_child_count},"
                f"stale_improved_child:{stale_improved_child_count},"
                f"over_failed_reuse:{over_failed_reuse_count},"
                f"improved_dead_end:{len(improved_dead_end_records)}"
            )
        return "\n".join(lines)

    def load(self, name: str, *, dao_fallback: Optional[Dao] = None) -> List[MctsPath]:
        base = self._resolve_dir(name)
        matrices_path = base / "matrices.npz"
        meta_path = base / "meta.json"
        daos_path = base / "daos.pkl"
        incoming_path = base / "incoming.pkl"
        widths_path = base / "widths.pkl"

        if not matrices_path.exists():
            raise FileNotFoundError(f"missing matrices file: {matrices_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"missing meta file: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        daos_payload: Optional[List[List[Dao]]] = None
        if daos_path.exists():
            try:
                with open(daos_path, "rb") as f:
                    daos_payload = pickle.load(f)
            except Exception:
                daos_payload = None

        widths_payload: Optional[List[dict[str, Any]]] = None
        if widths_path.exists():
            try:
                with open(widths_path, "rb") as f:
                    widths_payload = pickle.load(f)
            except Exception:
                widths_payload = None

        incoming_payload: Optional[List[List[Optional[tuple]]]] = None
        if incoming_path.exists():
            try:
                with open(incoming_path, "rb") as f:
                    incoming_payload = pickle.load(f)
            except Exception:
                incoming_payload = None

        out: List[MctsPath] = []
        with np.load(matrices_path) as data:
            for idx, p in enumerate(meta.get("paths", [])):
                keys = p.get("matrix_keys", [])
                if not keys:
                    raise ValueError(f"path at index {idx} has no matrix_keys")

                nodes: List[Node] = []
                prev: Optional[Node] = None
                incoming_entries: Optional[List[Optional[tuple]]] = None
                if isinstance(incoming_payload, list) and idx < len(incoming_payload):
                    per_path = incoming_payload[idx]
                    if isinstance(per_path, list) and len(per_path) == len(keys):
                        incoming_entries = per_path
                for key in keys:
                    incoming: Optional[ActionInfo] = None
                    if incoming_entries is not None:
                        entry = incoming_entries[len(nodes)]
                        if isinstance(entry, ActionInfo):
                            incoming = entry
                        elif isinstance(entry, tuple) and len(entry) == 3:
                            cand, global_info, source = entry
                            incoming = ActionInfo(cand=cand, global_info=global_info, source=source)
                    if key not in data:
                        raise KeyError(f"matrix key not found in npz: {key}")
                    mat = Matrix.from_numpy(data[key])
                    node = Node(state=mat, parent=prev, incoming=incoming, depth=0 if prev is None else prev.depth + 1)
                    nodes.append(node)
                    prev = node

                path = MctsPath(
                    final_node=nodes[-1],
                    ranks_thr=list(p.get("ranks_thr", [])),
                    daos=[],
                    bs_widths=[],
                    todd_widths=[],
                    x0s=[list(x0) + [0]*(X0_LENGTH - len(x0)) for x0 in p.get("x0s", [])],
                )

                if daos_payload is not None:
                    if idx >= len(daos_payload):
                        raise ValueError("daos payload length mismatch with paths")
                    path.daos = daos_payload[idx]
                elif dao_fallback is not None:
                    path.daos = [deepcopy(dao_fallback)]

                if isinstance(widths_payload, list) and idx < len(widths_payload):
                    width_info = widths_payload[idx]
                    if isinstance(width_info, dict):
                        path.bs_widths = list(width_info.get("bs_widths") or [])
                        path.todd_widths = list(width_info.get("todd_widths") or [])

                if not path.daos:
                    raise ValueError(
                        "loaded path has no dao snapshots; pass dao_fallback or save with store_daos=True"
                    )

                out.append(path)

        return out
