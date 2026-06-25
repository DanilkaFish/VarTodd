
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

from node import (
    ActionPool,
    ActionSelection,
    CandidateExport,
    ExplorationScore,
    FinalizationScore,
    Node,
    PolicyConfig,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    Stats,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
)
from copy import deepcopy

RANK_SCHEDULE_SENTINEL = 10**9

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
    def _format_signed(value: float) -> str:
        value = float(value)
        if abs(value) < 1e-12:
            return "0"
        return f"{value:+.3f}".rstrip("0").rstrip(".")

    @staticmethod
    def _format_score(score: Any, labels: Sequence[str]) -> str:
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

        labels = list(labels)[: len(weights)]
        if len(labels) < len(weights):
            labels.extend(f"w{i}" for i in range(len(labels), len(weights)))

        nonzero_weights = [
            (label, weight)
            for label, weight in zip(labels, weights)
            if abs(float(weight)) >= 1e-12
        ]
        if not nonzero_weights and weights:
            nonzero_weights = list(zip(labels, weights))

        weight_text = ",".join(
            f"{label}:{Path._format_signed(weight)}"
            for label, weight in nonzero_weights
        )
        if not weight_text:
            weight_text = "none"

        parts = [weight_text]
        nonzero_centers = [
            (label, center)
            for label, center in zip(labels, centers)
            if abs(float(center)) >= 1e-12
        ]
        if nonzero_centers:
            parts.append(
                "centers="
                + ",".join(
                    f"{label}:{Path._format_signed(center)}"
                    for label, center in nonzero_centers
                )
            )
        parts.append(f"p={_fmt_float(power, 3)}")
        return "(" + ";".join(parts) + ")"

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
                ("action_count", "1"),
                ("tohpe_pool", "2/0"),
                ("todd_pool", "16/0"),
            )

        mode = dao.mode
        def value(name: str, default: Any) -> Any:
            schedule = getattr(mode, name, None)
            if schedule is None:
                return default
            if hasattr(schedule, "at"):
                return schedule.at(parent_rank)
            return schedule

        def fmt_sampling(sampling: Any) -> str:
            sampling = _as_sampling_budget(sampling)
            return (
                f"oh:{sampling.one_hot}/sp:{_as_int(sampling.sparse)}/"
                f"de:{_as_int(sampling.dense)}/w:{_as_int(sampling.sparse_max_weight)}"
            )

        def fmt_source_pool(pool: Any) -> str:
            pool = _as_source_pool(pool)
            return f"{_as_int(pool.keep)}/{_as_int(pool.reserve)}"

        selection = _as_action_selection(value("selection", _default_action_selection()))
        action_count = _as_int(self._schedule_value_at(self.todd_widths, parent_rank, selection.count))
        action_pool = _as_action_pool(value("pool", _default_action_pool()))
        tohpe = _as_tohpe_search(value("tohpe", _default_tohpe_search()))
        todd = _as_todd_search(value("todd", _default_todd_search()))
        buckets = _as_z_bucket_search(todd.buckets)
        scores = _as_policy_scores(value("scores", _default_policy_scores()))

        fields = [
            ("beam_width", _as_int(self._schedule_value_at(self.bs_widths, parent_rank, 1))),
            ("action_count", action_count),
            ("selection_mode", selection.mode),
            ("selection_temperature", _as_float(selection.temperature)),
            ("action_pool_final_size", _as_int(action_pool.final_size)),
            ("tohpe_sampling", fmt_sampling(tohpe.sampling)),
            ("todd_sampling", fmt_sampling(todd.sampling)),
            ("tohpe_pool", fmt_source_pool(tohpe.pool)),
            ("todd_pool", fmt_source_pool(todd.pool)),
            ("tohpe_z_choices", _as_int(tohpe.z_choices)),
            ("todd_actions_per_bucket", _as_int(todd.actions_per_bucket)),
            ("z_min_buckets", _as_int(buckets.min_buckets)),
            ("z_max_buckets", _as_int(buckets.max_buckets)),
            ("pool_scores", self._format_score(scores.exploration, ["red", "dim", "bucket", "vw", "z"])),
            ("final_scores", self._format_score(scores.final, ["red", "dim", "bucket", "vw", "z", "tohpe"])),
        ]
        return tuple((key, self._format_policy_value(value)) for key, value in fields)

    @staticmethod
    def _format_policy_snapshot(snapshot: Tuple[Tuple[str, str], ...]) -> str:
        return "{" + ", ".join(f"{key}={value}" for key, value in snapshot) + "}"

    @staticmethod
    def _profile_region(profile_groups: List[Dict[str, Any]], all_groups: List[Dict[str, Any]]) -> str:
        if len(profile_groups) != 1:
            return ",".join(f"{g['start_rank']}->{g['end_rank']}" for g in profile_groups)

        group = profile_groups[0]
        idx = all_groups.index(group)
        if len(all_groups) > 1 and idx == 0:
            return f"above_{group['end_rank']}"
        if len(all_groups) > 1 and idx == len(all_groups) - 1:
            return f"below_{group['start_rank']}"
        return f"{group['start_rank']}->{group['end_rank']}"

    @staticmethod
    def _format_policy_profile(
        profile_id: str,
        snapshot: Tuple[Tuple[str, str], ...],
        profile_groups: List[Dict[str, Any]],
        all_groups: List[Dict[str, Any]],
    ) -> str:
        values = dict(snapshot)
        beam = values.get("beam_width", "1")
        action_count = values.get("action_count", "1")
        selection_mode = values.get("selection_mode", "best")
        selection_temp = values.get("selection_temperature", "0")
        final_pool = values.get("action_pool_final_size", "16")
        tohpe_pool = values.get("tohpe_pool", "2/0")
        todd_pool = values.get("todd_pool", "16/0")
        tohpe_sampling_label = values.get("tohpe_sampling", "oh:all/sp:0/de:32/w:2")
        todd_samples = values.get("todd_sampling", "oh:all/sp:0/de:32/w:2")
        tohpe_z_choices = values.get("tohpe_z_choices", "8")
        min_z = values.get("z_min_buckets", "32")
        max_z = values.get("z_max_buckets", "0")
        actions_per_bucket = values.get("todd_actions_per_bucket", "4")
        pool_scores = values.get("pool_scores", "(none;p=1)")
        final_scores = values.get("final_scores", "(none;p=1)")

        diversity_parts = [
            f"actions_per_bucket:{actions_per_bucket}",
        ]

        return (
            f"  {profile_id} "
            f"rank_region={Path._profile_region(profile_groups, all_groups)} "
            f"search_shape=beam:{beam}/actions:{action_count}/selection:{selection_mode}@{selection_temp} "
            f"pool=final:{final_pool}/tohpe:{tohpe_pool}/todd:{todd_pool} "
            f"samples=tohpe:{tohpe_sampling_label}/todd:{todd_samples}/tohpe_z:{tohpe_z_choices} "
            f"z_buckets=min:{min_z}/reserve_cap:{max_z} "
            f"diversity={'/'.join(diversity_parts)} "
            f"scores=pool{pool_scores} final{final_scores}"
        )

    @staticmethod
    def _evidence_notes(
        *,
        start_rank: int,
        accepted_tohpe: List[float],
        accepted_todd: List[float],
        researched_z: List[float],
        accepted_per_z: List[float],
        basis_dim: List[float],
    ) -> List[str]:
        notes: List[str] = []
        tohpe_mean = Path._mean(accepted_tohpe)
        todd_mean = Path._mean(accepted_todd)
        z_mean = Path._mean(researched_z)
        acc_z_mean = Path._mean(accepted_per_z)
        basis_mean = Path._mean(basis_dim)

        if todd_mean == 0 and tohpe_mean > 0:
            notes.append("tohpe_only")
        if todd_mean > max(1.0, 3.0 * max(tohpe_mean, 0.0)):
            notes.append("todd_dominant")
        if accepted_per_z and acc_z_mean >= 1.0 and z_mean <= 1000:
            notes.append("high_acceptance_low_z")
        if accepted_per_z and z_mean >= 5000 and acc_z_mean < 1.0:
            notes.append("hard_refinement_high_z")
        if accepted_per_z and acc_z_mean <= 0.01:
            notes.append("low_acceptance_per_z")
        if basis_mean >= 30 and int(start_rank) >= 500:
            notes.append("high_dim_early_region")
        return notes

    @staticmethod
    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return float(num) / float(den) if den else 0.0

    @staticmethod
    def _safe_float_attr(obj: Any, name: str) -> Optional[float]:
        try:
            value = getattr(obj, name)
        except Exception:
            return None
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

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
            f"profile={group.get('profile_id', '?')}",
        ]
        if group.get("split_reasons"):
            parts.append(f"split={','.join(group['split_reasons'])}")
        parts.append(f"steps={steps}")
        if steps and group["red"]:
            red_mean = Path._mean(group["red"])
            red_max_mean = Path._mean(group["red_max"])
            basis_mean = Path._mean(group["basis_dim"])
            basis_max_values = group.get("basis_max") or group["basis_dim"]
            bucket_mean = Path._mean(group["bucket"])
            parts.extend(
                [
                    (
                        "outcome="
                        f"red_mean:{_fmt_float(red_mean)}/"
                        f"red_max_mean:{_fmt_float(red_max_mean)} "
                        f"dim_mean:{_fmt_float(basis_mean)}/"
                        f"dim_max:{int(max(basis_max_values))} "
                        f"bucket_mean:{_fmt_float(bucket_mean)}/"
                        f"bucket_max:{int(max(group['bucket']))}"
                    ),
                    (
                        "source="
                        f"tohpe:{_fmt_float(Path._mean(accepted_tohpe))}/"
                        f"todd:{_fmt_float(Path._mean(accepted_todd))}"
                    ),
                ]
            )
            pool_sizes = group.get("pool_size") or []
            pool_tohpe = group.get("pool_tohpe_size") or []
            pool_todd = group.get("pool_todd_size") or []
            if pool_sizes or pool_tohpe or pool_todd:
                pool_parts = []
                if pool_sizes:
                    pool_parts.append(
                        f"total_mean:{_fmt_float(Path._mean(pool_sizes))}/"
                        f"total_max:{int(max(pool_sizes))}"
                    )
                if pool_tohpe:
                    pool_parts.append(f"tohpe_mean:{_fmt_float(Path._mean(pool_tohpe))}")
                if pool_todd:
                    pool_parts.append(f"todd_mean:{_fmt_float(Path._mean(pool_todd))}")
                parts.append("pool=" + "/".join(pool_parts))
            parts.extend(
                [
                    (
                        "accepted_min="
                        f"tohpe:{int(min(productive_tohpe or accepted_tohpe))}/"
                        f"todd:{int(min(productive_todd or accepted_todd))}"
                    ),
                    (
                        "z="
                        f"mean:{_fmt_float(Path._mean(researched_z))}/"
                        f"max:{int(max(researched_z))}"
                    ),
                ]
            )
            if accepted_per_z:
                parts.append(
                    "accepted_per_z="
                    f"mean:{_fmt_float(Path._mean(accepted_per_z), 4)}/"
                    f"min:{_fmt_float(min(accepted_per_z), 4)}/"
                    f"max:{_fmt_float(max(accepted_per_z), 4)}"
                )
            else:
                parts.append("accepted_per_z=n/a")
            notes = Path._evidence_notes(
                start_rank=int(group["start_rank"]),
                accepted_tohpe=accepted_tohpe,
                accepted_todd=accepted_todd,
                researched_z=researched_z,
                accepted_per_z=accepted_per_z,
                basis_dim=group["basis_dim"],
            )
            if notes:
                parts.append(f"note={','.join(notes)}")
        else:
            parts.append("action_stats=unavailable")

        if group["missing_stats"]:
            parts.append(f"missing_stats={group['missing_stats']}")
        return " ".join(parts)

    def _path_policy_groups(self, path: List[Node], ranks: List[int]) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        dim_lt5_split_seen = False
        todd_pool_split_seen = False

        def make_group(
            policy: Tuple[Tuple[str, str], ...],
            start_rank: int,
            end_rank: int,
            split_reasons: List[str],
        ) -> Dict[str, Any]:
            return {
                "policy": policy,
                "split_reasons": split_reasons,
                "start_rank": start_rank,
                "end_rank": end_rank,
                "steps": 0,
                "red": [],
                "red_max": [],
                "basis_dim": [],
                "basis_max": [],
                "bucket": [],
                "accepted_tohpe": [],
                "accepted_todd": [],
                "researched_z": [],
                "pool_size": [],
                "pool_tohpe_size": [],
                "pool_todd_size": [],
                "missing_stats": 0,
            }

        for idx in range(1, len(path)):
            parent_rank = ranks[idx - 1]
            rank = ranks[idx]
            node = path[idx]
            policy = self._policy_snapshot_at(parent_rank)
            cand = node.incoming.cand if node.incoming is not None else None
            stats = node.incoming.global_info if node.incoming is not None else None
            split_reasons: List[str] = []

            if cand is not None and stats is not None:
                basis_for_split = self._safe_float_attr(stats, "max_basis")
                if basis_for_split is None:
                    basis_for_split = self._safe_float_attr(cand, "basis_dim")
                if (
                    not dim_lt5_split_seen
                    and basis_for_split is not None
                    and basis_for_split < 5
                ):
                    split_reasons.append("dim_lt5")
                    dim_lt5_split_seen = True

                pool_tohpe_for_split = self._safe_float_attr(cand, "pool_tohpe_size")
                pool_todd_for_split = self._safe_float_attr(cand, "pool_todd_size")
                if (
                    not todd_pool_split_seen
                    and pool_tohpe_for_split is not None
                    and pool_todd_for_split is not None
                    and pool_todd_for_split > pool_tohpe_for_split
                ):
                    split_reasons.append("todd_pool_gt_tohpe")
                    todd_pool_split_seen = True

            if not groups or groups[-1]["policy"] != policy or split_reasons:
                groups.append(make_group(policy, parent_rank, rank, split_reasons))

            group = groups[-1]
            group["steps"] += 1
            group["end_rank"] = rank

            if cand is None or stats is None:
                group["missing_stats"] += 1
                continue

            group["red"].append(float(cand.reduction))
            group["red_max"].append(float(stats.max_reduction))
            group["basis_dim"].append(float(cand.basis_dim))
            max_basis = self._safe_float_attr(stats, "max_basis")
            if max_basis is not None:
                group["basis_max"].append(max_basis)
            group["bucket"].append(float(cand.bucket_size))
            for attr_name in ("pool_size", "pool_tohpe_size", "pool_todd_size"):
                value = self._safe_float_attr(cand, attr_name)
                if value is not None:
                    group[attr_name].append(value)
            accepted = float(stats.accepted)
            accepted_tohpe = float(stats.accepted_tohpe)
            group["accepted_tohpe"].append(accepted_tohpe)
            group["accepted_todd"].append(max(0.0, accepted - accepted_tohpe))
            researched_z = getattr(node.incoming, "total", None)
            if researched_z is None:
                researched_z = getattr(stats, "z_researched", None)
            if researched_z is None:
                researched_z = getattr(stats, "total", 0)
            group["researched_z"].append(float(researched_z or 0))

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
        profile_ids: Dict[Tuple[Tuple[str, str], ...], str] = {}
        profile_groups: Dict[str, List[Dict[str, Any]]] = {}
        profile_snapshots: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        for group in groups:
            snapshot = group["policy"]
            if snapshot not in profile_ids:
                profile_id = f"P{len(profile_ids) + 1}"
                profile_ids[snapshot] = profile_id
                profile_snapshots[profile_id] = snapshot
            profile_id = profile_ids[snapshot]
            group["profile_id"] = profile_id
            profile_groups.setdefault(profile_id, []).append(group)
        lines = [self._format_group(group, idx + 1) for idx, group in enumerate(groups)]
        if groups and all(group["missing_stats"] == group["steps"] for group in groups):
            lines.append("  note=loaded paths may lack incoming action stats if saved by an older version")
        lines.append("converged_policy_profiles:")
        lines.extend(
            self._format_policy_profile(
                profile_id,
                profile_snapshots[profile_id],
                profile_groups[profile_id],
                groups,
            )
            for profile_id in profile_snapshots
        )

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


def _as_sampling_budget(x: Any) -> SamplingBudget:
    if isinstance(x, SamplingBudget):
        return x
    if isinstance(x, Mapping):
        return SamplingBudget(
            one_hot=x.get("one_hot", "all"),
            sparse=_as_int(x.get("sparse", 0)),
            dense=_as_int(x.get("dense", 32)),
            sparse_max_weight=_as_int(x.get("sparse_max_weight", 2)),
        )
    raise TypeError(f"expected SamplingBudget or mapping, got {type(x).__name__}")


def _as_source_pool(x: Any) -> SourcePool:
    if isinstance(x, SourcePool):
        return x
    if isinstance(x, Mapping):
        return SourcePool(keep=_as_int(x.get("keep", 1)), reserve=_as_int(x.get("reserve", 0)))
    raise TypeError(f"expected SourcePool or mapping, got {type(x).__name__}")


def _as_z_bucket_search(x: Any) -> ZBucketSearch:
    if isinstance(x, ZBucketSearch):
        return x
    if isinstance(x, Mapping):
        return ZBucketSearch(
            min_buckets=_as_int(x.get("min_buckets", 32)),
            max_buckets=_as_int(x.get("max_buckets", 0)),
            temperature=_as_float(x.get("temperature", 0.0)),
            random_fraction=_as_float(x.get("random_fraction", 0.0)),
        )
    raise TypeError(f"expected ZBucketSearch or mapping, got {type(x).__name__}")


def _as_tohpe_search(x: Any) -> TohpeSearch:
    if isinstance(x, TohpeSearch):
        return x
    if isinstance(x, Mapping):
        return TohpeSearch(
            sampling=_as_sampling_budget(x.get("sampling", SamplingBudget())),
            pool=_as_source_pool(x.get("pool", SourcePool(keep=2, reserve=0))),
            z_choices=_as_int(x.get("z_choices", 8)),
        )
    raise TypeError(f"expected TohpeSearch or mapping, got {type(x).__name__}")


def _as_todd_search(x: Any) -> ToddSearch:
    if isinstance(x, ToddSearch):
        return x
    if isinstance(x, Mapping):
        return ToddSearch(
            sampling=_as_sampling_budget(x.get("sampling", SamplingBudget())),
            pool=_as_source_pool(x.get("pool", SourcePool(keep=16, reserve=0))),
            actions_per_bucket=_as_int(x.get("actions_per_bucket", 4)),
            buckets=_as_z_bucket_search(x.get("buckets", ZBucketSearch())),
        )
    raise TypeError(f"expected ToddSearch or mapping, got {type(x).__name__}")


def _as_policy_scores(x: Any) -> PolicyScores:
    if isinstance(x, PolicyScores):
        return x
    if isinstance(x, Mapping):
        return PolicyScores(exploration=x.get("exploration", ExplorationScore()), final=x.get("final", FinalizationScore()))
    raise TypeError(f"expected PolicyScores or mapping, got {type(x).__name__}")


def _as_action_selection(x: Any) -> ActionSelection:
    if isinstance(x, ActionSelection):
        return x
    if isinstance(x, Mapping):
        return ActionSelection(
            count=_as_int(x.get("count", 1)),
            mode=str(x.get("mode", "best")),
            temperature=_as_float(x.get("temperature", 0.0)),
        )
    raise TypeError(f"expected ActionSelection or mapping, got {type(x).__name__}")


def _as_action_pool(x: Any) -> ActionPool:
    if isinstance(x, ActionPool):
        return x
    if isinstance(x, Mapping):
        return ActionPool(final_size=_as_int(x.get("final_size", 16)))
    raise TypeError(f"expected ActionPool or mapping, got {type(x).__name__}")


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

    def __post_init__(self) -> None:
        if len(self.points) > 1:
            self.points.sort(key=lambda x: x[0], reverse=True)
            self.points[0] = (RANK_SCHEDULE_SENTINEL, self.points[0][1])

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
                if len(pts) == 1:
                    pts[0] = (RANK_SCHEDULE_SENTINEL, pts[0][1])
                return RankSchedule(points=pts)
        return RankSchedule.constant(value)  # type: ignore[arg-type]

    def at(self, rank: int) -> T:
        """Return the value active at this rank.

        Points are sorted from high rank to low rank. The first value is the
        fallback for ranks above the highest threshold; lower thresholds
        override it when rank <= threshold.
        """
        if not self.points:
            raise ValueError("empty schedule")
        r = int(rank)
        cur = self.points[0][1]
        for rr, vv in self.points:
            if r <= rr:
                cur = vv
            else:
                break
        return cur


def _converted_rank_schedule(value: Any, convert: Callable[[Any], T]) -> RankSchedule[T]:
    if isinstance(value, RankSchedule):
        return RankSchedule.from_any([(rank, convert(v)) for rank, v in value.points])
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], tuple) and len(value[0]) == 2:
            return RankSchedule.from_any([(int(rank), convert(v)) for rank, v in value])
    return RankSchedule.constant(convert(value))


def _default_policy_scores() -> PolicyScores:
    return PolicyScores(
        exploration=ExplorationScore([0.5, 0.5, 0.0, 0.0, 0.0]),
        final=FinalizationScore([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
    )


def _default_action_selection() -> ActionSelection:
    return ActionSelection(count=1, mode="best", temperature=0.0)


def _default_action_pool() -> ActionPool:
    return ActionPool(final_size=16)


def _default_tohpe_search() -> TohpeSearch:
    return TohpeSearch(
        sampling=SamplingBudget(one_hot="all", sparse=0, dense=32, sparse_max_weight=2),
        pool=SourcePool(keep=2, reserve=0),
        z_choices=8,
    )


def _default_todd_search() -> ToddSearch:
    return ToddSearch(
        sampling=SamplingBudget(one_hot="all", sparse=0, dense=32, sparse_max_weight=2),
        pool=SourcePool(keep=16, reserve=0),
        actions_per_bucket=4,
        buckets=ZBucketSearch(min_buckets=32, max_buckets=0, temperature=0.0, random_fraction=0.0),
    )


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
    scores: RankSchedule[PolicyScores] = field(
        default_factory=lambda: RankSchedule.constant(_default_policy_scores())
    )
    selection: RankSchedule[ActionSelection] = field(default_factory=lambda: RankSchedule.constant(_default_action_selection()))
    pool: RankSchedule[ActionPool] = field(default_factory=lambda: RankSchedule.constant(_default_action_pool()))
    tohpe: RankSchedule[TohpeSearch] = field(
        default_factory=lambda: RankSchedule.constant(_default_tohpe_search())
    )
    todd: RankSchedule[ToddSearch] = field(
        default_factory=lambda: RankSchedule.constant(_default_todd_search())
    )

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ModeDao":
        supported_keys = {"scores", "selection", "pool", "tohpe", "todd"}
        unsupported_keys = sorted(key for key in d if key not in supported_keys)
        if unsupported_keys:
            raise ValueError(
                "flat policy keys are no longer supported; use grouped policy objects instead: "
                + ", ".join(unsupported_keys)
            )
        return ModeDao(
            scores=_converted_rank_schedule(d.get("scores", _default_policy_scores()), _as_policy_scores),
            selection=_converted_rank_schedule(d.get("selection", _default_action_selection()), _as_action_selection),
            pool=_converted_rank_schedule(d.get("pool", _default_action_pool()), _as_action_pool),
            tohpe=_converted_rank_schedule(d.get("tohpe", _default_tohpe_search()), _as_tohpe_search),
            todd=_converted_rank_schedule(d.get("todd", _default_todd_search()), _as_todd_search),
        )

    def policy_kwargs(self, *, depth: int, action_count: Optional[int] = None,) -> Dict[str, Any]:
        selection = _as_action_selection(self.selection.at(depth))
        if action_count is not None:
            selection = ActionSelection(
                count=_as_int(action_count),
                mode=str(selection.mode),
                temperature=_as_float(selection.temperature),
            )
        return {
            "scores": _as_policy_scores(self.scores.at(depth)),
            "selection": selection,
            "pool": _as_action_pool(self.pool.at(depth)),
            "tohpe": _as_tohpe_search(self.tohpe.at(depth)),
            "todd": _as_todd_search(self.todd.at(depth)),
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

    def policy_config_at(self, depth: int, mode: str="default", action_count: int = 1) -> PolicyConfig:
        m = self.modes.get(mode)
        if m is None:
            raise KeyError(f"unknown mode: {mode}")
        return PolicyConfig(**m.policy_kwargs(depth=depth, action_count=action_count))
