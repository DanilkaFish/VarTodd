from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path as FsPath
from typing import Any, Callable, Iterator, List, Optional, Sequence
import json
import os
import pickle
from copy import deepcopy
try:
    import fcntl
except ImportError:  # pragma: no cover - Linux in production, fallback for portability
    fcntl = None

import numpy as np

from mcts_dao import Dao, Path as MctsPath
from node import ActionInfo, Matrix, Node
from variant import resolve_variant

X0_LENGTH = 200
USAGE_INDEX_NAME = "_usage_index.json"
USAGE_LOCK_NAME = "_usage_index.lock"
# Evaluated programs can execute from a Hydra/worker directory.  The logical
# symlink path identifies the active GF variant and yields an absolute store.
DATA_PATH = str(resolve_variant(__file__).data_path)


class _PathStoreUnpickler(pickle.Unpickler):
    _MODULE_ALIASES = {
        # Saved path DAOs belong to this problem package.  Older backups may
        # carry the former scripts.optimization_core module name.
        "mcts_dao": "mcts_dao",
        "node": "node",
        "scripts.optimization_core.mcts_dao": "mcts_dao",
        "scripts.optimization_core.node": "node",
    }

    def find_class(self, module: str, name: str):
        return super().find_class(self._MODULE_ALIASES.get(module, module), name)


def _load_pickle(path: FsPath) -> Any:
    with open(path, "rb") as f:
        return _PathStoreUnpickler(f).load()


def _path_widths_from_payload(payload: Any, idx: int) -> tuple[List[Any], List[Any]]:
    if not isinstance(payload, list) or idx >= len(payload):
        return [], []

    item = payload[idx]
    if not isinstance(item, dict):
        return [], []

    bs_widths = item.get("bs_widths") or []
    todd_widths = item.get("todd_widths") or []
    return list(bs_widths), list(todd_widths)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

@dataclass
class PathStore:
    root_dir: str = DATA_PATH

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
        widths_payload: List[dict[str, Any]] = []

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
            #TODO
            for x0 in x0s:
                while x0 and x0[-1] == 0:
                    x0.pop()
            limit_buckets = self._path_todd_limit_buckets({}, path.daos)
            max_z_researched = self._path_max_z_researched(path)
            meta["paths"].append(
                {
                    "matrix_keys": matrix_keys,
                    "ranks_thr": list(path.ranks_thr),
                    "x0s": x0s,
                    "limit_buckets": limit_buckets,
                    "max_z_researched": max_z_researched,
                }
            )
            incoming_payload.append(incoming_info)
            widths_payload.append(
                {
                    "bs_widths": list(getattr(path, "bs_widths", []) or []),
                    "todd_widths": list(getattr(path, "todd_widths", []) or []),
                }
            )

        np.savez_compressed(matrices_path, **matrices)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if store_daos:
            daos_payload = [path.daos for path in paths]
            with open(daos_path, "wb") as f:
                pickle.dump(daos_payload, f)
        with open(incoming_path, "wb") as f:
            pickle.dump(incoming_payload, f)
        with open(widths_path, "wb") as f:
            pickle.dump(widths_payload, f)

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
    def _path_todd_limit_buckets(
        path_meta: dict[str, Any], daos: Optional[Sequence[Dao]]
    ) -> Optional[int]:
        """Return the current tail's TODD cap, excluding loaded-path history.

        A reopened path retains every parent DAO for reconstruction, but its
        live-store name and retention class describe the policy that produced
        the current tail.  Looking across the complete history incorrectly
        carries an ancestor's full ``limit_bucket=-1`` into a capped child.
        """
        stored = path_meta.get("limit_buckets")
        if daos:
            limits: list[int] = []
            try:
                current_dao = daos[-1]
                for _rank, todd in current_dao.mode.todd.points:
                    keep = int(todd.pool.keep)
                    limits.append(0 if keep == 0 else int(todd.buckets.limit_bucket))
            except Exception:
                pass
            if limits:
                if any(limit < 0 for limit in limits):
                    return -1
                return max(limits)
        if stored is not None:
            return int(stored)
        return None

    @staticmethod
    def _path_max_z_researched(path: MctsPath) -> Optional[int]:
        """Return the largest aggregate z-researched count on this path."""
        values: list[int] = []
        node = path.final_node
        while node is not None:
            incoming = getattr(node, "incoming", None)
            stats = getattr(incoming, "global_info", None)
            value = getattr(stats, "z_researched", None)
            if value is not None:
                try:
                    values.append(int(value))
                except (TypeError, ValueError):
                    pass
            node = getattr(node, "parent", None)
        return max(values) if values else None

    def iter_path_records(self) -> list[dict[str, Any]]:
        import re

        root = self.root_path
        if not root.exists() or not root.is_dir():
            return []

        usage = self.read_usage_index().get("paths", {})
        legacy_name_re = re.compile(r"i(?P<init>\d+)_m(?P<thr>\d+)_f(?P<final>\d+)")
        short_name_re = re.compile(
            r"f(?P<final>\d+)_i(?P<thr>\d+)_[^_]+_"
            r"(?:"
            r"lim(?P<legacy_limit>-?\d+|unknown)"
            r"(?:_z(?P<legacy_z>\d+|unknown))?"
            r"|"
            r"z(?P<researched_z>\d+|unknown)of(?P<limit>-?\d+|unknown)"
            r")"
        )
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
                best_init_rank = None
                with np.load(matrices_path) as data:
                    for p_idx, p in enumerate(paths_meta):
                        keys = p.get("matrix_keys", [])
                        if not keys:
                            continue
                        last_key = keys[-1]
                        first_key = keys[0]
                        if last_key not in data or first_key not in data:
                            continue
                        rank = int(data[last_key].shape[0])
                        init_rank = int(data[first_key].shape[0])
                        depth = max(0, len(keys) - 1)
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_depth = depth
                            best_path_idx = p_idx
                            best_init_rank = init_rank

                if best_rank is None:
                    continue

                best_path_meta = (
                    paths_meta[best_path_idx]
                    if best_path_idx is not None and best_path_idx < len(paths_meta)
                    else {}
                )
                daos_for_best: Optional[Sequence[Dao]] = None
                daos_path = backup_dir / "daos.pkl"
                if daos_path.exists():
                    try:
                        daos_payload = _load_pickle(daos_path)
                        if (
                            isinstance(daos_payload, list)
                            and best_path_idx is not None
                            and best_path_idx < len(daos_payload)
                        ):
                            daos_for_best = daos_payload[best_path_idx]
                    except Exception:
                        daos_for_best = None
                limit_buckets = self._path_todd_limit_buckets(best_path_meta, daos_for_best)
                max_z_researched = best_path_meta.get("max_z_researched")
                if max_z_researched is None:
                    max_z_researched = best_path_meta.get(
                        "max_todd_z_researched"
                    )
                if max_z_researched is None:
                    name_match = short_name_re.search(backup_dir.name)
                    if name_match:
                        name_z = (
                            name_match.group("researched_z")
                            or name_match.group("legacy_z")
                        )
                        if name_z not in (None, "unknown"):
                            max_z_researched = int(name_z)

                init_rank = None
                init_thr_rank = None
                match = legacy_name_re.search(backup_dir.name)
                if match:
                    init_rank = int(match.group("init"))
                    init_thr_rank = int(match.group("thr"))
                else:
                    match = short_name_re.search(backup_dir.name)
                    if match:
                        init_rank = best_init_rank
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
                        "max_z_researched": max_z_researched,
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

    @staticmethod
    def _saturated_reuse_count(record: dict[str, Any]) -> int:
        """Reuses where the best child equalled the loaded rank (path mined out).

        A saturated reuse means the path was reopened and searched but produced
        no rank below where it started: the path's tail is exhausted. This is
        the signal that should retire a path, unlike a path simply never tried
        (used=0) or one that regressed once by seed noise.
        """
        used = int(record.get("used_count") or 0)
        improved = int(record.get("improved_count") or 0)
        best_child = record.get("best_child_rank")
        rank = int(record["rank"])
        # Only count as saturated when we have evidence the child matched (not
        # beat) the path rank; otherwise fall back to plain non-improving reuse.
        if best_child is not None and int(best_child) >= rank:
            return max(0, used - improved)
        return 0

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
        # Symmetric retirement: apply the reuse cutoff to capped paths too, not
        # only full-search ones. A capped scout reopened many times without
        # improving is as mined-out as a full-search path. Retire on saturated
        # reuse (child matched the rank) rather than on raw non-improving reuse,
        # so a path that is merely untried or regressed once stays live.
        saturated = cls._saturated_reuse_count(record)
        if saturated > int(max_nonimproved_reuse):
            return False
        if not cls._is_full_limit_record(record):
            return True
        return cls._nonimproved_reuse_count(record) <= int(max_nonimproved_reuse)

    @classmethod
    def _promote_children_over_parents(
        cls,
        live_records: list[dict[str, Any]],
        *,
        child_qualifies: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
        """Hide a parent when a live child improved over it, migrating counters.

        An improved child's node chain contains the parent's full prefix (same
        higher-parity matrices; see mcts_dao.Path.branch_path), so the child can
        be reopened at any rank the parent could and also carries the improved
        tail. Keeping the parent as a separate selectable entry is redundant.

        `child_qualifies`, if given, restricts which improving children may
        supersede their parent in the returned view. Used to keep a capped
        near-tail parent visible even when a full-search child improved it —
        that child isn't near-tail material itself, so it shouldn't hide the
        capped path in a near-tail-only view.

        Returns the filtered live records plus (parent, child) name pairs for the
        footer. Usage counters stay attached to their own path records so the
        rendered u/i values match `_usage_index.json` exactly.
        """
        by_name = {r["name"]: r for r in live_records}
        # child that improved over its loaded parent, best improver per parent
        best_child_for_parent: dict[str, dict[str, Any]] = {}
        for child in live_records:
            parent = child.get("parent_path_name")
            loaded = child.get("parent_loaded_rank")
            if not parent or parent not in by_name or loaded is None:
                continue
            if int(child["rank"]) >= int(loaded):
                continue  # child did not improve the parent's loaded rank
            if child_qualifies is not None and not child_qualifies(child):
                continue
            cur = best_child_for_parent.get(parent)
            if cur is None or int(child["rank"]) < int(cur["rank"]):
                best_child_for_parent[parent] = child

        # Process a lineage chain A->B->C parent-first so hidden parents do not
        # reappear through intermediate children. Counters are not propagated:
        # rendered u/i is always the record's own usage.
        def parent_depth(name: str, _seen: Optional[set[str]] = None) -> int:
            seen = _seen or set()
            if name in seen or name not in best_child_for_parent:
                return 0
            seen.add(name)
            parent_of = by_name.get(name, {}).get("parent_path_name")
            return 1 + parent_depth(parent_of, seen) if parent_of in best_child_for_parent else 0

        promoted: list[tuple[str, str]] = []
        hidden: set[str] = set()
        for parent_name in sorted(best_child_for_parent, key=parent_depth):
            child = best_child_for_parent[parent_name]
            hidden.add(parent_name)
            promoted.append((parent_name, child["name"]))

        filtered = [r for r in live_records if r["name"] not in hidden]
        return filtered, promoted

    @staticmethod
    def _is_full_limit_record(record: dict[str, Any]) -> bool:
        value = record.get("limit_buckets")
        return value is not None and int(value) < 0

    @staticmethod
    def _children_full_search_count(records: Sequence[dict[str, Any]]) -> dict[str, int]:
        """How many children of each path already used a full (-1) search.

        Historical fact over all recorded children, live or not: it answers
        "has a full search already been tried from here", which shouldn't
        reset just because that child was later retired or superseded.
        """
        counts: dict[str, int] = {}
        for record in records:
            parent = record.get("parent_path_name")
            if parent and PathStore._is_full_limit_record(record):
                counts[parent] = counts.get(parent, 0) + 1
        return counts

    @staticmethod
    def _is_near_tail_eligible(
        record: dict[str, Any],
        *,
        limit_cutoff: int,
        full_children: dict[str, int],
        max_full_children: int,
    ) -> bool:
        """Capped path, not yet proven exhausted by repeated full search.

        Near-tail exhaustion is signalled by full-search children (someone
        already spent the expensive unrestricted search on this branch), not
        by raw reopen count — a capped scout can be reopened cheaply many
        times and still be worth another near-tail pass.
        """
        value = record.get("limit_buckets")
        if value is None or int(value) < 0 or int(value) >= int(limit_cutoff):
            return False
        return full_children.get(record["name"], 0) < int(max_full_children)

    @staticmethod
    def _select_top(records: Sequence[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
        by_rank: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            by_rank.setdefault(int(record["rank"]), []).append(record)
        representatives = [
            min(
                candidates,
                key=lambda record: (
                    -int(record.get("init_rank") or -1),
                    PathStore._record_sort_key(record),
                ),
            )
            for _rank, candidates in sorted(by_rank.items())
        ]
        return sorted(
            representatives, key=PathStore._record_sort_key
        )[: max(0, int(top_k))]

    @staticmethod
    def _has_better_child(record: dict[str, Any]) -> bool:
        best_child = record.get("best_child_rank")
        return best_child is not None and int(best_child) < int(record["rank"])

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
            f"u/i={record['used_count']}/{record['improved_count']} "
            f"child={best_child_text} kind={record['kind']} "
            f"band={record.get('restart_band', 'unknown')}"
            f"{parent_text}"
        )

    def summarize(
        self,
        *,
        top_k: int = 6,
        near_tail_top_k: int = 4,
        near_tail_limit_cutoff: int = 100_000,
        near_tail_max_full_children: int = 2,
        wide_margin_top_k: int = 4,
        wide_margin_max_used: int = 7,
        max_nonimproved_reuse: int = 10,
        dead_end_after: Optional[int] = 4,
        dead_end_top_k: int = 2,
        nonimproved_child_top_k: int = 2,
    ) -> str:
        """Render the two selectable groups the mutation prompt loads from.

        near_tail_paths: best paths with limit_buckets < near_tail_limit_cutoff
        that have fewer than near_tail_max_full_children children which used a
        full (-1) search — i.e. not yet proven exhausted by an expensive full
        search, so more near-tail refinement is still the natural next step.
        A full-search child that improves such a path does NOT hide it here
        (that child isn't near-tail material itself); only a better-ranked
        near-tail-qualifying child replaces it.

        wide_margin_paths: best remaining paths that were not selected in
        near_tail_paths, have no recorded better child, and have own
        used_count < wide_margin_max_used.

        `top_k` is ignored; the two groups are sized independently via
        near_tail_top_k / wide_margin_top_k.
        """
        del top_k
        records = self.iter_path_records()
        if not records:
            return (
                "best_paths:\n"
                "near_tail_paths:\n"
                "- none\n"
                "wide_margin_paths:\n"
                "- none\n"
                "rule=load_only_names_under_best_paths.near_tail_paths,"
                "wide_margin_paths\n"
                "balance=from_init:0,first_half:0,near_final:0,unknown:0; "
                "full:0,partial:0; live:0,total:0,best_rank_count:0"
            )

        near_tail_top_k = max(1, int(near_tail_top_k))
        wide_margin_top_k = max(1, int(wide_margin_top_k))
        max_nonimproved_reuse = max(0, int(max_nonimproved_reuse))
        dead_end_cutoff = (
            max_nonimproved_reuse if dead_end_after is None else max(0, int(dead_end_after))
        )

        count_full = sum(1 for r in records if r["kind"] == "full")
        count_partial = sum(1 for r in records if r["kind"] == "partial")
        best_rank = min(r["rank"] for r in records)
        best_rank_count = sum(1 for r in records if r["rank"] == best_rank)
        nonimproved_child_count = sum(1 for r in records if r.get("child_improved_loaded") is False)
        stale_improved_child_count = sum(1 for r in records if r.get("is_stale_improved_child"))
        over_failed_reuse_count = sum(
            1
            for r in records
            if self._is_full_limit_record(r)
            and self._nonimproved_reuse_count(r) > dead_end_cutoff
        )

        live_records_raw = [
            r
            for r in records
            if self._is_live_record(r, max_nonimproved_reuse=max_nonimproved_reuse)
        ]
        live_records, promoted_pairs = self._promote_children_over_parents(live_records_raw)

        full_children = self._children_full_search_count(records)

        def near_tail_ok(record: dict[str, Any]) -> bool:
            return self._is_near_tail_eligible(
                record,
                limit_cutoff=near_tail_limit_cutoff,
                full_children=full_children,
                max_full_children=near_tail_max_full_children,
            )

        near_tail_live_records_raw = [
            r
            for r in live_records_raw
            if near_tail_ok(r)
        ]
        near_tail_source, _ = self._promote_children_over_parents(
            near_tail_live_records_raw,
            child_qualifies=near_tail_ok,
        )
        near_tail_records = self._select_top(
            [r for r in near_tail_source if near_tail_ok(r)],
            top_k=near_tail_top_k,
        )
        near_tail_names = {r["name"] for r in near_tail_records}
        near_tail_ranks = {int(r["rank"]) for r in near_tail_records}

        wide_margin_records = self._select_top(
            [
                r
                for r in live_records
                if r["name"] not in near_tail_names
                and int(r["rank"]) not in near_tail_ranks
                and not self._has_better_child(r)
                and int(r.get("used_count") or 0) < int(wide_margin_max_used)
            ],
            top_k=wide_margin_top_k,
        )

        live_band_counts = {"from_init": 0, "first_half": 0, "near_final": 0, "unknown": 0}
        for record in live_records:
            band = str(record.get("restart_band") or "unknown")
            live_band_counts[band] = live_band_counts.get(band, 0) + 1

        near_tail_lines = [self._format_summary_record(r) for r in near_tail_records]
        wide_margin_lines = [self._format_summary_record(r) for r in wide_margin_records]
        hidden_nonimproved_child_count = nonimproved_child_count

        dead_end_records = [
            r
            for r in records
            if self._is_full_limit_record(r)
            and self._nonimproved_reuse_count(r) > dead_end_cutoff
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

        lines = [
            "best_paths:",
            "near_tail_paths:",
            *[f"- {line}" for line in (near_tail_lines or ["none"])],
            "wide_margin_paths:",
            *[f"- {line}" for line in (wide_margin_lines or ["none"])],
        ]
        if dead_end_lines:
            lines += ["dead_end_paths:", *[f"- {line}" for line in dead_end_lines]]
        if nonimproved_child_lines:
            lines += [
                "best_nonimproved_child_paths:",
                *[f"- {line}" for line in nonimproved_child_lines],
            ]
        band_from_init = live_band_counts.get("from_init", 0)
        band_near_final = live_band_counts.get("near_final", 0)
        underrep = self._underrepresented_hint(
            from_init=band_from_init,
            near_final=band_near_final,
            count_full=count_full,
            count_partial=count_partial,
            near_tail_live=len(near_tail_records),
        )
        lines += [
            "rule=load_only_names_under_best_paths.near_tail_paths,wide_margin_paths; "
            "nonimproved children are evidence only; "
            "wide_margin_paths are the best remaining paths not chosen in "
            "near_tail_paths, with no better child and own used_count<"
            f"{int(wide_margin_max_used)}; dead_end_paths and "
            "best_nonimproved_child_paths are evidence only",
            (
                "population: "
                f"live={len(live_records)}/total={len(records)}  "
                f"bands from_init={band_from_init}/near_final={band_near_final}  "
                f"modes full={count_full}/partial={count_partial}  "
                f"near_tail_live={len(near_tail_records)}/{near_tail_top_k}  "
                f"wide_margin_live={len(wide_margin_records)}/{wide_margin_top_k}"
            ),
        ]
        if underrep:
            lines.append(f"suggest: {underrep}")
        if promoted_pairs:
            lines.append(
                "promoted(child replaces parent, shares prefix): "
                + ", ".join(f"{parent}->{child}" for parent, child in promoted_pairs)
            )
        if (
            hidden_nonimproved_child_count
            or stale_improved_child_count
            or over_failed_reuse_count
        ):
            lines.append(
                "hidden(not selectable): "
                f"nonimproved_child={hidden_nonimproved_child_count}, "
                f"stale_improved_child={stale_improved_child_count}, "
                f"over_failed_reuse={over_failed_reuse_count}"
            )
        return "\n".join(lines)

    @staticmethod
    def _underrepresented_hint(
        *,
        from_init: int,
        near_final: int,
        count_full: int,
        count_partial: int,
        near_tail_live: int,
    ) -> str:
        """One actionable line: which regime the population is short on.

        The task wants roughly balanced coverage of ab-initio builders and
        saved-path refiners, plus a few live near-tail paths to exploit.
        Surface the gap so the mutation operator can restore balance instead of
        eyeballing counts.
        """
        if near_final == 0 and from_init > 0:
            return "no near-final refiners live; add a saved-path reopening"
        if from_init == 0 and near_final > 0:
            return "no ab-initio builders live; add a path_name='init' child"
        if near_tail_live == 0:
            return "no near-tail paths live; a cheap capped near-tail scout can seed new paths"
        if from_init >= 4 * max(near_final, 1):
            return "ab-initio dominates; prefer a saved-path refiner this round"
        if near_final >= 4 * max(from_init, 1):
            return "refiners dominate; prefer an ab-initio builder this round"
        return ""

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
                daos_payload = _load_pickle(daos_path)
            except Exception:
                daos_payload = None

        incoming_payload: Optional[List[List[Optional[tuple]]]] = None
        if incoming_path.exists():
            try:
                incoming_payload = _load_pickle(incoming_path)
            except Exception:
                incoming_payload = None

        widths_payload: Optional[List[dict[str, Any]]] = None
        if widths_path.exists():
            try:
                widths_payload = _load_pickle(widths_path)
            except Exception:
                widths_payload = None

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

                bs_widths, todd_widths = _path_widths_from_payload(widths_payload, idx)
                path = MctsPath(
                    final_node=nodes[-1],
                    ranks_thr=list(p.get("ranks_thr", [])),
                    daos=[],
                    bs_widths=bs_widths,
                    todd_widths=todd_widths,
                    x0s=[list(x0) + [0]*(X0_LENGTH - len(x0)) for x0 in p.get("x0s", [])],
                )

                if daos_payload is not None:
                    if idx >= len(daos_payload):
                        raise ValueError("daos payload length mismatch with paths")
                    path.daos = daos_payload[idx]
                elif dao_fallback is not None:
                    path.daos = [deepcopy(dao_fallback)]

                if not path.daos:
                    path.daos = [Dao()]

                out.append(path)

        return out
