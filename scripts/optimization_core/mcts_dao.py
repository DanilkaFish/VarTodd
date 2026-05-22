
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

from node import Node, PolicyConfig, ExplorationScore, FinalizationScore, Stats, CandidateExport
from copy import deepcopy

def _q_ge(num_better: int, total: int) -> float:
    if not total:
        return 0.0
    return 1.0 - (num_better / total)

def _fmt_float(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")

@dataclass(slots=True)
class Path:
    final_node: Node = None
    ranks_thr: List[int] = field(default_factory=list)
    daos: List[Dao] = field(default_factory=list)
    bs_widths: List[Any] = field(default_factory=list)
    todd_widths: List[Any] = field(default_factory=list)
    active_params: List[List[float]] = field(default_factory=list)
    x0s: List[List[float]] = field(default_factory=list)

    def branch_path(
        self,
        node: Node,
        dao: Dao,
        x0: List[float],
        bs_width: Any = None,
        todd_width: Any = None,
    ):
        new_path = Path()
        new_path.final_node = node
        new_path.daos = self.daos + [deepcopy(dao)]
        new_path.bs_widths = self.bs_widths + ([deepcopy(bs_width)] if bs_width is not None else [])
        new_path.todd_widths = self.todd_widths + ([deepcopy(todd_width)] if todd_width is not None else [])
        new_path.x0s = self.x0s + [deepcopy(x0)]
        new_path.ranks_thr = deepcopy(self.ranks_thr)
        return new_path
        
    def branch_path_at(self, rank_thr: int):
        node = self.final_node
        if node is None:
            return None
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        path = list(reversed(path))
        candidate = None
        for node in path:
            if node.state.rows > rank_thr:
                candidate = node
        if candidate is None:
            return None
        candidate_rank = int(candidate.state.rows)
        kept_ranks = [rank for rank in self.ranks_thr if int(rank) > candidate_rank]
        keep_count = min(len(self.daos), len(kept_ranks) + 1)
        new_path = Path()
        new_path.final_node = candidate
        new_path.ranks_thr = kept_ranks + [candidate_rank]
        new_path.daos = deepcopy(self.daos[:keep_count])
        new_path.bs_widths = deepcopy(self.bs_widths[:keep_count])
        new_path.todd_widths = deepcopy(self.todd_widths[:keep_count])
        new_path.x0s = deepcopy(self.x0s[:keep_count])
        return new_path

    def format_path_stats_tiny(self) -> str:
        return self.format_path_stats()

    def _nodes(self) -> List[Node]:
        path = []
        node = self.final_node
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def _segment_value_for_step(self, values: List[Any], parent_rank: int) -> Optional[Any]:
        if not values:
            return None
        if not self.ranks_thr:
            return values[-1]
        idx = 0
        for rank_thr in self.ranks_thr:
            if parent_rank <= int(rank_thr):
                idx += 1
            else:
                break
        return values[min(idx, len(values) - 1)]

    def _dao_for_step(self, parent_rank: int) -> Optional["Dao"]:
        return self._segment_value_for_step(self.daos, parent_rank)

    def _schedule_value_at(self, schedules: List[Any], parent_rank: int, default: Any) -> Any:
        schedule = self._segment_value_for_step(schedules, parent_rank)
        if schedule is None:
            return default
        if hasattr(schedule, "at"):
            return schedule.at(parent_rank)
        return schedule

    @staticmethod
    def _format_score(score: Any) -> str:
        try:
            size = len(score)
        except Exception:
            return str(score)

        weights = []
        centers = []
        for i in range(size):
            try:
                weights.append(float(score[i]))
            except Exception:
                break
        for i in range(size):
            try:
                centers.append(float(score[i + size]))
            except Exception:
                break
        try:
            power = score.pow()
        except Exception:
            power = getattr(score, "power", 1.0)

        parts = []
        if weights:
            parts.append("weights=[" + ",".join(_fmt_float(v, 3) for v in weights) + "]")
        if centers and any(centers):
            parts.append("centers=[" + ",".join(_fmt_float(v, 3) for v in centers) + "]")
        parts.append(f"power={_fmt_float(power, 3)}")
        return "(" + ",".join(parts) + ")"

    @staticmethod
    def _format_policy_value(value: Any) -> str:
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, float):
            return _fmt_float(value, 3)
        return str(value)

    def _policy_snapshot_at(self, parent_rank: int) -> Tuple[Tuple[str, str], ...]:
        dao = self._dao_for_step(parent_rank)
        if dao is None:
            return (
                ("beam_width", "1"),
                ("todd_width", "1"),
                ("tohpe_pool_size", "1"),
                ("todd_pool_size", "1"),
            )

        mode = dao.mode
        def value(name: str, default: Any) -> Any:
            schedule = getattr(mode, name, None)
            if schedule is None:
                return default
            if hasattr(schedule, "at"):
                return schedule.at(parent_rank)
            return schedule

        fields = [
            ("beam_width", _as_int(self._schedule_value_at(self.bs_widths, parent_rank, 1))),
            ("todd_width", _as_int(self._schedule_value_at(self.todd_widths, parent_rank, 1))),
            ("tohpe_vector_samples", _as_sample_caps(value("tohpe_vector_samples", [16, 32, 16]))),
            ("todd_vector_samples", _as_sample_caps(value("todd_vector_samples", [16, 32, 16]))),
            ("top_pool", _as_int(value("top_pool", 1))),
            ("tohpe_pool_size", _as_int(value("tohpe_pool_size", 1))),
            ("todd_pool_size", _as_int(value("todd_pool_size", 1))),
            ("min_tohpe_actions", _as_int(value("min_tohpe_actions", 0))),
            ("min_todd_actions", _as_int(value("min_todd_actions", 0))),
            ("tohpe_sample", _as_int(value("tohpe_sample", 1))),
            ("min_z_to_research", _as_int(value("min_z_to_research", 0))),
            ("max_z_to_research", _as_int(value("max_z_to_research", 0))),
            ("max_from_single_ns", _as_int(value("max_from_single_ns", 1))),
            ("min_reduction", _as_int(value("min_reduction", 0))),
            ("max_reduction", _as_int(value("max_reduction", 0))),
            ("sparse_max_weight", _as_int(value("sparse_max_weight", 8))),
            ("bucket_temperature", _as_float(value("bucket_temperature", 0.0))),
            ("bucket_random_fraction", _as_float(value("bucket_random_fraction", 0.0))),
            ("max_per_signature", _as_int(value("max_per_signature", 2))),
            ("pool_scores", self._format_score(value("pool_scores", ""))),
            ("final_scores", self._format_score(value("final_scores", ""))),
        ]
        return tuple((key, self._format_policy_value(value)) for key, value in fields)

    @staticmethod
    def _format_policy_snapshot(snapshot: Tuple[Tuple[str, str], ...]) -> str:
        return "{" + ", ".join(f"{key}={value}" for key, value in snapshot) + "}"

    @staticmethod
    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return float(num) / float(den) if den else 0.0

    @staticmethod
    def _format_group(group: Dict[str, Any], group_num: int) -> str:
        steps = int(group["steps"])
        accepted_tohpe = group["accepted_tohpe"]
        accepted_todd = group["accepted_todd"]
        researched_z = group["researched_z"]
        accepted_total = [
            float(tohpe) + float(todd)
            for tohpe, todd in zip(accepted_tohpe, accepted_todd)
        ]
        productive_indices = [idx for idx, accepted in enumerate(accepted_total) if accepted > 0]
        productive_tohpe = [accepted_tohpe[idx] for idx in productive_indices]
        productive_todd = [accepted_todd[idx] for idx in productive_indices]
        accepted_per_z = [
            Path._safe_div(accepted_total[idx], researched_z[idx])
            for idx in productive_indices
            if researched_z[idx] > 0
        ]

        parts = [
            f"  group={group_num}",
            f"ranks={group['start_rank']}->{group['end_rank']}",
            f"steps={steps}",
        ]
        if steps and group["red"]:
            parts.extend(
                [
                    f"red_mean={_fmt_float(Path._mean(group['red']))}",
                    f"red_max_mean={_fmt_float(Path._mean(group['red_max']))}",
                    f"basis_dim_mean={_fmt_float(Path._mean(group['basis_dim']))}",
                    f"basis_dim_max={int(max(group['basis_dim']))}",
                    f"bucket_mean={_fmt_float(Path._mean(group['bucket']))}",
                    f"bucket_max={int(max(group['bucket']))}",
                    (
                        "accepted_mean="
                        f"tohpe:{_fmt_float(Path._mean(accepted_tohpe))}/"
                        f"todd:{_fmt_float(Path._mean(accepted_todd))}"
                    ),
                    (
                        "accepted_min="
                        f"tohpe:{int(min(productive_tohpe or accepted_tohpe))}/"
                        f"todd:{int(min(productive_todd or accepted_todd))}"
                    ),
                    (
                        "researched_z_"
                        f"mean={_fmt_float(Path._mean(researched_z))}/"
                        f"max={int(max(researched_z))}"
                    ),
                ]
            )
            if accepted_per_z:
                parts.append(
                    "accepted_per_z_"
                    f"mean={_fmt_float(Path._mean(accepted_per_z), 4)}/"
                    f"min={_fmt_float(min(accepted_per_z), 4)}/"
                    f"max={_fmt_float(max(accepted_per_z), 4)}"
                )
            else:
                parts.append("accepted_per_z=n/a")
        else:
            parts.append("action_stats=unavailable")

        if group["missing_stats"]:
            parts.append(f"missing_stats={group['missing_stats']}")
        parts.append(f"policy={Path._format_policy_snapshot(group['policy'])}")
        return " ".join(parts)

    def _path_policy_groups(self, path: List[Node], ranks: List[int]) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for idx in range(1, len(path)):
            parent_rank = ranks[idx - 1]
            rank = ranks[idx]
            node = path[idx]
            policy = self._policy_snapshot_at(parent_rank)
            if not groups or groups[-1]["policy"] != policy:
                groups.append(
                    {
                        "policy": policy,
                        "start_rank": parent_rank,
                        "end_rank": rank,
                        "steps": 0,
                        "red": [],
                        "red_max": [],
                        "basis_dim": [],
                        "bucket": [],
                        "accepted_tohpe": [],
                        "accepted_todd": [],
                        "researched_z": [],
                        "missing_stats": 0,
                    }
                )

            group = groups[-1]
            group["steps"] += 1
            group["end_rank"] = rank

            if node.incoming is None or node.incoming.global_info is None:
                group["missing_stats"] += 1
                continue

            cand: CandidateExport = node.incoming.cand
            stats: Stats = node.incoming.global_info
            group["red"].append(float(cand.reduction))
            group["red_max"].append(float(stats.max_reduction))
            group["basis_dim"].append(float(cand.basis_dim))
            group["bucket"].append(float(cand.bucket_size))
            group["accepted_tohpe"].append(float(stats.accepted_tohpe))
            group["accepted_todd"].append(float(stats.accepted))
            group["researched_z"].append(float(getattr(stats, "z_researched", 0) or 0))

        return groups

    def format_path_stats(self, max_lines: int = 14, start_rank: Optional[int] = None) -> str:
        path = self._nodes()
        if not path:
            return "path unavailable"

        omitted_prefix_depth = 0
        if start_rank is not None:
            full_path = path
            for idx, node in enumerate(full_path):
                if int(node.state.rows) <= int(start_rank):
                    path = full_path[idx:]
                    omitted_prefix_depth = idx
                    break

        ranks = [int(node.state.rows) for node in path]
        depth = max(0, len(path) - 1)
        total_reduction = ranks[0] - ranks[-1]
        trajectory = " -> ".join(str(r) for r in ranks)
        if len(ranks) > 12:
            trajectory = " -> ".join(str(r) for r in ranks[:4] + ["..."] + ranks[-4:])

        header = [
            "path_summary:",
            f"  depth={depth} init_rank={ranks[0]} final_rank={ranks[-1]} total_reduction={total_reduction}",
            f"  rank_trajectory={trajectory}",
            "path_policy_groups:",
        ]
        if omitted_prefix_depth:
            header.insert(
                3,
                f"  omitted_prefix_depth={omitted_prefix_depth} shown_from_started_rank={ranks[0]}",
            )

        if depth == 0:
            return "\n".join(header + ["  <root only>"])

        groups = self._path_policy_groups(path, ranks)
        lines = [self._format_group(group, idx + 1) for idx, group in enumerate(groups)]
        if groups and all(group["missing_stats"] == group["steps"] for group in groups):
            lines.append("  note=loaded paths may lack incoming action stats if saved by an older version")

        return "\n".join(header + lines)
T = TypeVar("T")


def _as_int(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    return int(x)


def _as_float(x: Any) -> float:
    if isinstance(x, bool):
        return float(int(x))
    return float(x)


def _as_sample_caps(x: Any) -> List[int]:
    if isinstance(x, bool) or isinstance(x, (int, float)):
        return [max(0, int(x)), 0, 0]
    values = list(x)
    if len(values) != 3:
        raise ValueError(f"sample caps must have exactly 3 values: [one_hot, sparse, dense], got {x!r}")
    return [max(0, int(v)) for v in values]


def _as_bool(x: Any) -> bool:
    if isinstance(x, str):
        lx = x.strip().lower()
        if lx in {"1", "true", "yes", "y", "on"}:
            return True
        if lx in {"0", "false", "no", "n", "off"}:
            return False
    return bool(x)


@dataclass(slots=True)
class DepthSchedule(Generic[T]):
    points: List[Tuple[int, T]] = field(default_factory=list)

    @staticmethod
    def constant(value: T) -> "DepthSchedule[T]":
        return DepthSchedule(points=[(0, value)])

    @staticmethod
    def from_any(value: Union["DepthSchedule[T]", Sequence[Tuple[int, T]], T]) -> "DepthSchedule[T]":
        if isinstance(value, DepthSchedule):
            return value
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], tuple) and len(value[0]) == 2:
                pts = [(int(d), v) for d, v in value]  # type: ignore[misc]
                pts.sort(key=lambda x: x[0])
                return DepthSchedule(points=pts)
        return DepthSchedule.constant(value)  # type: ignore[arg-type]

    def at(self, depth: int) -> T:
        if not self.points:
            raise ValueError("empty schedule")
        d = int(depth)
        cur = self.points[0][1]
        for dd, vv in self.points:
            if dd <= d:
                cur = vv
            else:
                break
        return cur

@dataclass(slots=True)
class RankSchedule(Generic[T]):
    points: List[Tuple[int, T]] = field(default_factory=list)

    @staticmethod
    def constant(value: T) -> "RankSchedule[T]":
        return RankSchedule(points=[(0, value)])

    @staticmethod
    def from_any(value: Union["RankSchedule[T]", Sequence[Tuple[int, T]], T]) -> "RankSchedule[T]":
        if isinstance(value, RankSchedule):
            return value
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], tuple) and len(value[0]) == 2:
                pts = [(int(d), v) for d, v in value]  # type: ignore[misc]
                pts.sort(key=lambda x: x[0], reverse=True)
                return RankSchedule(points=pts)
        return RankSchedule.constant(value)  # type: ignore[arg-type]

    def at(self, rank: int) -> T:
        if not self.points:
            raise ValueError("empty schedule")
        r = int(rank)
        cur = self.points[0][1]
        for rr, vv in self.points:
            if r <= rr:
                cur = vv
            else:
                cur = vv
                break
        return cur


@dataclass(slots=True)
class UctDao:
    name: str = "puct"
    c: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(2.5))
    fn: Optional[Callable[..., float]] = None

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "UctDao":
        return UctDao(
            name=str(d.get("name", "puct")),
            c=DepthSchedule.from_any(d.get("c", 2.5)),
            fn=d.get("fn", None),
        )

    def c_at(self, depth: int) -> float:
        return _as_float(self.c.at(depth))


@dataclass(slots=True)
class TreeDao:
    rollout_add: bool = True
    rollout_active: bool = False
    rollout_frozen_until: int = 0

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "TreeDao":
        return TreeDao(
            rollout_add=_as_bool(d.get("rollout_add", True)),
            rollout_active=_as_bool(d.get("rollout_active", False)),
            rollout_frozen_until=_as_int(d.get("rollout_frozen_until", 0)),
        )


@dataclass(slots=True)
class ModeDao:
    tohpe_vector_samples: DepthSchedule[List[int]] = field(default_factory=lambda: DepthSchedule.constant([16, 32, 16]))
    todd_vector_samples: DepthSchedule[List[int]] = field(default_factory=lambda: DepthSchedule.constant([16, 32, 16]))
    sparse_max_weight: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(8))
    top_pool: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(1))
    selection: DepthSchedule[str] = field(default_factory=lambda: DepthSchedule.constant("softmax"))
    temperature: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(0.0))
    tohpe_pool_size: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(1))
    todd_pool_size: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(1))
    min_tohpe_actions: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(0))
    min_todd_actions: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(0))
    min_z_to_research: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(5000))
    max_z_to_research: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(10000000))
    pool_scores: DepthSchedule[ExplorationScore] = field(default_factory=lambda: DepthSchedule.constant(ExplorationScore([0.5, 0.5, 0.0, 0.0, 0])))
    final_scores: DepthSchedule[FinalizationScore] = field(default_factory=lambda: DepthSchedule.constant(FinalizationScore([0.5, 0.5, 0.0, 0.0, 0, 0])))
    max_from_single_ns: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(5))
    min_reduction: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(0))
    max_reduction: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(100))
    min_pool_size: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(0))
    tohpe_sample: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(1))
    bucket_temperature: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(0.0))
    bucket_random_fraction: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(0.0))
    max_per_signature: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(2))
    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ModeDao":
        return ModeDao(
            tohpe_vector_samples=DepthSchedule.from_any(d.get("tohpe_vector_samples", [16, 32, 16])),
            todd_vector_samples=DepthSchedule.from_any(d.get("todd_vector_samples", [16, 32, 16])),
            sparse_max_weight=DepthSchedule.from_any(d.get("sparse_max_weight", 8)),
            top_pool=DepthSchedule.from_any(d.get("top_pool", 96)),
            selection=DepthSchedule.from_any(d.get("selection", "best")),
            temperature=DepthSchedule.from_any(d.get("temperature", 0.0)),
            tohpe_pool_size=DepthSchedule.from_any(d.get("tohpe_pool_size", 5)),
            todd_pool_size=DepthSchedule.from_any(d.get("todd_pool_size", 5)),
            min_tohpe_actions=DepthSchedule.from_any(d.get("min_tohpe_actions", 0)),
            min_todd_actions=DepthSchedule.from_any(d.get("min_todd_actions", 0)),
            min_z_to_research=DepthSchedule.from_any(d.get("min_z_to_research", 5000)),
            max_z_to_research=DepthSchedule.from_any(d.get("max_z_to_research", 100000)),
            pool_scores=DepthSchedule.from_any(d.get("pool_scores", ExplorationScore(0.5, 0.5, 0.0,  0.0))),
            final_scores=DepthSchedule.from_any(d.get("final_scores", FinalizationScore(0.5, 0.5, 0.0,  0.0, 1.0))),
            max_from_single_ns=DepthSchedule.from_any(d.get("max_from_single_ns", 100)),
            max_reduction=DepthSchedule.from_any(d.get("max_reduction", 100)),
            min_reduction=DepthSchedule.from_any(d.get("min_reduction", 0)),
            min_pool_size=DepthSchedule.from_any(d.get("min_pool_size", 0)),
            tohpe_sample=DepthSchedule.from_any(d.get("tohpe_sample", 1)),
            bucket_temperature=DepthSchedule.from_any(d.get("bucket_temperature", 0.0)),
            bucket_random_fraction=DepthSchedule.from_any(d.get("bucket_random_fraction", 0.0)),
            max_per_signature=DepthSchedule.from_any(d.get("max_per_signature", 2)),
        )

    def policy_kwargs(self, *, depth: int, num_candidates: int,) -> Dict[str, Any]:
        return {
            "tohpe_vector_samples": _as_sample_caps(self.tohpe_vector_samples.at(depth)),
            "todd_vector_samples": _as_sample_caps(self.todd_vector_samples.at(depth)),
            "sparse_max_weight": _as_int(self.sparse_max_weight.at(depth)),
            "num_candidates": _as_int(num_candidates),
            "top_pool": _as_int(self.top_pool.at(depth)),
            "selection": str(self.selection.at(depth)),
            "temperature": _as_float(self.temperature.at(depth)),
            "tohpe_pool_size": _as_int(self.tohpe_pool_size.at(depth)),
            "todd_pool_size": _as_int(self.todd_pool_size.at(depth)),
            "min_tohpe_actions": _as_int(self.min_tohpe_actions.at(depth)),
            "min_todd_actions": _as_int(self.min_todd_actions.at(depth)),
            "min_z_to_research": _as_int(self.min_z_to_research.at(depth)),
            "max_z_to_research": _as_int(self.max_z_to_research.at(depth)),
            "min_pool_size": _as_int(self.min_pool_size.at(depth)),
            "ExplorationScore": self.pool_scores.at(depth),
            "FinalizationScore": self.final_scores.at(depth),
            "max_from_single_ns": _as_int(self.max_from_single_ns.at(depth)),
            "min_reduction": _as_int(self.min_reduction.at(depth)),
            "max_reduction": _as_int(self.max_reduction.at(depth)),
            "tohpe_sample": _as_int(self.tohpe_sample.at(depth)),
            "bucket_temperature": _as_float(self.bucket_temperature.at(depth)),
            "bucket_random_fraction": _as_float(self.bucket_random_fraction.at(depth)),
            "max_per_signature": _as_int(self.max_per_signature.at(depth)),
        }


@dataclass(slots=True)
class Dao:
    iterations: int = 3600
    discount: float = 0.995
    max_depth: int = 32
    threads: int = 4

    branching: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(4))
    rollout_depth: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(10))

    uct: UctDao = field(default_factory=UctDao)
    tree: TreeDao = field(default_factory=TreeDao)

    modes: Dict[str, ModeDao] = field(default_factory=dict)

    unfrozen_top: int = 0
    freeze_spread: int = 0
    def __post_init__(self):
        if "default" not in self.modes:
            self.modes["default"] = ModeDao()
    @staticmethod
    def from_dict(cfg: Mapping[str, Any]) -> "Dao":
        modes_in = cfg.get("modes", {}) or {}
        modes: Dict[str, ModeDao] = {}
        if isinstance(modes_in, Mapping):
            for k, v in modes_in.items():
                if isinstance(v, Mapping):
                    modes[str(k)] = ModeDao.from_dict(v)

        freeze = cfg.get("freeze", {}) or {}
        return Dao(
            iterations=_as_int(cfg.get("iterations", 3600)),
            discount=_as_float(cfg.get("discount", 0.995)),
            max_depth=_as_int(cfg.get("max_depth", 32)),
            threads=_as_int(cfg.get("threads", 4)),
            branching=DepthSchedule.from_any(cfg.get("branching", 1)),
            rollout_depth=DepthSchedule.from_any(cfg.get("rollout_depth", 0)),
            uct=UctDao.from_dict(cfg.get("uct", {}) or {}),
            tree=TreeDao.from_dict(cfg.get("tree", {}) or {}),
            modes=modes,
            unfrozen_top=_as_int(freeze.get("unfrozen_top", cfg.get("unfrozen_top", 0))),
            freeze_spread=_as_int(freeze.get("freeze_spread", cfg.get("freeze_spread", 0)))
        )
    @property
    def mode(self):
        return self.modes["default"]
    
    def branching_at(self, depth: int) -> int:
        return _as_int(self.branching.at(depth))

    def rollout_depth_at(self, depth: int) -> int:
        return _as_int(self.rollout_depth.at(depth))

    def policy_config_at(self, depth: int, mode: str="default", num_candidates: int = 1) -> PolicyConfig:
        m = self.modes.get(mode)
        if m is None:
            raise KeyError(f"unknown mode: {mode}")
        # num_candidates = self.branching_at(depth) if mode == "expand" else 1
        # num
        return PolicyConfig(**m.policy_kwargs(depth=depth, num_candidates=num_candidates))
