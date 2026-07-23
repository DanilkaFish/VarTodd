import hashlib
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .mcts_dao import RANK_SCHEDULE_SENTINEL, Dao, Path, RankSchedule
from .node import (
    ActionPool,
    ActionSelection,
    ExplorationScore,
    FinalizationScore,
    Matrix,
    Node,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    Tensor3D,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
)
from .path_store import PathStore, X0_LENGTH
from .todd import Todd

MIN_SAVED_PATH_MARGIN = 1
DEFAULT_MATRIX_PATH = os.getenv(
    "VARTODD_DEFAULT_MATRIX",
    "data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^16_1612310.npy",
)
EXECUTOR_KIND = os.getenv("EVO_SEED_EXECUTOR_KIND", "thread")
PROGRAM_ID: Optional[str] = os.getenv("GIGAEVO_PROGRAM_ID")
ACTIVE_EVALUATOR: Optional["BaseEvaluator"] = None
DEFAULT_SEED_WORKERS = min(2, os.cpu_count() or 1)
DEFAULT_MAX_DEPTH = 10_000
TARGET_FINAL_RANK = int(os.getenv("VARTODD_TARGET_FINAL_RANK", "380"))


class GracefulEvaluationTimeout(RuntimeError):
    """Raised inside evaluated programs when the executor soft deadline is reached."""


def _register_active_evaluator(evaluator: "BaseEvaluator") -> None:
    global ACTIVE_EVALUATOR
    ACTIVE_EVALUATOR = evaluator


def _soft_deadline_epoch() -> Optional[float]:
    raw = os.getenv("GIGAEVO_EXEC_SOFT_DEADLINE_EPOCH")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _soft_deadline_reached() -> bool:
    deadline = _soft_deadline_epoch()
    return deadline is not None and time.time() >= deadline


def get_active_evaluator_best():
    evaluator = ACTIVE_EVALUATOR
    if evaluator is None or not evaluator.best_paths:
        return None
    return evaluator.get_best(timeout_salvage=True)


def trim_trailing_zero_cols(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape={arr.shape}")
    if arr.shape[1] == 0:
        return arr

    active = np.any(arr != 0, axis=0)
    if not np.any(active):
        return arr[:, :0]

    width = int(np.flatnonzero(active)[-1]) + 1
    if width == arr.shape[1]:
        return arr
    return arr[:, :width]


def normalize_matrix_array(arr: np.ndarray) -> np.ndarray:
    return trim_trailing_zero_cols(np.asarray(arr, dtype=bool))


def load_matrix_array(path: str | os.PathLike[str]) -> np.ndarray:
    return normalize_matrix_array(np.load(path))

def _worker_run_one_from_template(
    seed: int,
    path: Path,
    todd: Todd,
):
    # Todd.run only reads the incoming path and creates new child nodes.
    # Deep-copying a GF(2^64) path can recurse through ~1000 Node.parent links.
    node, counters = todd.run(path, True, seed)
    return seed, node, counters


def _copy_path_header(path: Path) -> Path:
    new_path = Path()
    new_path.final_node = path.final_node
    new_path.ranks_thr = list(path.ranks_thr)
    new_path.daos = deepcopy(path.daos)
    new_path.bs_widths = deepcopy(getattr(path, "bs_widths", []))
    new_path.todd_widths = deepcopy(getattr(path, "todd_widths", []))
    new_path.active_params = deepcopy(path.active_params)
    new_path.x0s = deepcopy(path.x0s)
    return new_path


def _copy_path_headers(paths: Sequence[Path]) -> List[Path]:
    return [_copy_path_header(path) for path in paths]


def find_rank(path, rank):
    for i, mat in enumerate(path):
        if mat.rows < rank:
            return path[max(i-1, 0)]
    return path[-1]
        
def get_matrix(name: Optional[str] = None) -> Matrix:
    if name is None:
        return Matrix.from_numpy(load_matrix_array(DEFAULT_MATRIX_PATH))
    return Matrix.from_numpy(load_matrix_array(f"npy/{name}.npy"))


#--------------------------- RANK ----------------------------
def _is_rank_list(x: Any) -> bool:
    if not isinstance(x, Iterable): 
        return False
    x = list(x) 
    return len(x) > 0 and isinstance(x[0], (list, tuple)) and len(x[0]) == 2

def _to_rank_schedule(x: Any) -> "RankSchedule":
    """Accept RankSchedule | [(rank, value), ...] | scalar and return RankSchedule."""
    if isinstance(x, RankSchedule):
        return x
    if isinstance(x, zip):
        x = [obj for obj in x]
    if _is_rank_list(x):
        return RankSchedule.from_any(list(x))
    return RankSchedule.constant(x)

def _rank_args(x: Any = None, vals=None, *, ranks=None, values=None):
    """Accept old `(x, vals)` and newer `ranks=..., values=...` setter forms."""
    def paired(rank_values, scheduled_values):
        rank_values = list(rank_values)
        scheduled_values = list(scheduled_values)
        if len(rank_values) == len(scheduled_values) - 1:
            # Common mistake: N-1 thresholds given as "boundaries" for N
            # values, forgetting the leading value's own threshold. Prepend
            # the sentinel rank so values[0] pairs with ranks[0] and stays
            # active from the start — RankSchedule.__post_init__ would force
            # the highest-rank point to this same sentinel anyway.
            rank_values = [RANK_SCHEDULE_SENTINEL] + rank_values
        if len(rank_values) != len(scheduled_values):
            raise ValueError(
                "rank schedule requires the same number of ranks and values "
                f"(got {len(rank_values)} ranks and {len(scheduled_values)} values)"
            )
        if not rank_values:
            raise ValueError("rank schedule requires at least one rank/value pair")
        return list(zip(rank_values, scheduled_values))

    if ranks is not None or values is not None:
        if x is not None or vals is not None:
            raise ValueError("use either positional `(x, vals)` or keyword `ranks=..., values=...`, not both")
        if ranks is None or values is None:
            raise ValueError("both `ranks` and `values` are required")
        return paired(ranks, values)
    if vals is not None:
        return paired(x, vals)
    if x is None:
        raise TypeError("missing schedule value")
    return x

def _to_sample_caps(x: Any) -> List[int]:
    if isinstance(x, bool) or isinstance(x, (int, float)):
        return [max(0, int(x)), 0, 0]
    values = list(x)
    if len(values) != 3:
        raise ValueError(f"sample caps must have exactly 3 values: [one_hot, sparse, dense], got {x!r}")
    return [max(0, int(v)) for v in values]

def _to_caps_rank_schedule(x: Any) -> "RankSchedule":
    """Accept RankSchedule | [(rank, [one_hot, sparse, dense]), ...] | caps."""
    if isinstance(x, RankSchedule):
        return x
    if isinstance(x, zip):
        x = [obj for obj in x]
    if _is_rank_list(x):
        return RankSchedule.from_any([(rank, _to_sample_caps(value)) for (rank, value) in x])
    return RankSchedule.constant(_to_sample_caps(x))

def _converted_rank_schedule(x: Any, convert):
    if isinstance(x, RankSchedule):
        return RankSchedule.from_any([(rank, convert(value)) for rank, value in x.points])
    if isinstance(x, zip):
        x = [obj for obj in x]
    if _is_rank_list(x):
        return RankSchedule.from_any([(rank, convert(value)) for rank, value in x])
    return RankSchedule.constant(convert(x))

def _to_erank_schedule(x: Any) -> "RankSchedule":
    """Accept RankSchedule | [(rank, value), ...] | scalar and return RankSchedule."""
    if isinstance(x, RankSchedule):
        return x
    if isinstance(x, zip):
        x = [obj for obj in x]
    if _is_rank_list(x):
        return RankSchedule.from_any([(rank, _to_exploration_score(el)) for (rank, el) in x])
    return RankSchedule.constant(_to_exploration_score(x))

def _to_frank_schedule(x: Any) -> "RankSchedule":
    """Accept RankSchedule | [(rank, value), ...] | scalar and return RankSchedule."""
    if isinstance(x, RankSchedule):
        return x
    if isinstance(x, zip):
        x = [obj for obj in x]
    if _is_rank_list(x):
        return RankSchedule.from_any([(rank, _to_finalization_score(el)) for (rank, el) in x])
    return RankSchedule.constant(_to_finalization_score(x))

def _to_exploration_score(x: Any) -> "ExplorationScore":
    """Accept ExplorationScore | (wred, wdim, wpossible_red) and return ExplorationScore."""
    if isinstance(x, ExplorationScore):
        return x

    x = list(x)
    xs = np.asarray(x)
    x = np.asarray(xs)/np.sqrt(np.sum(xs*xs)) if np.any(xs) else xs
    weights = [float(v) for v in x]
    return ExplorationScore(weights, [0.0] * len(weights))

def _to_finalization_score(x: Any) -> "FinalizationScore":
    """Accept FinalizationScore | (wred, wdim, wpossible_red, wtohpe_dim) and return FinalizationScore."""
    if isinstance(x, FinalizationScore):
        return x
    x = list(x)
    xs = np.asarray(x)
    x = np.asarray(xs)/np.sqrt(np.sum(xs*xs)) if np.any(xs) else xs
    weights = [float(v) for v in x]
    return FinalizationScore(weights, [0.0] * len(weights))

def _to_policy_scores(x: Any) -> "PolicyScores":
    if isinstance(x, PolicyScores):
        return x
    if isinstance(x, Mapping):
        return PolicyScores(
            exploration=_to_exploration_score(x.get("exploration", [1.0, 0.0, 0.0, 0.0, 0.0])),
            final=_to_finalization_score(x.get("final", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        )
    values = list(x)
    if len(values) != 2:
        raise ValueError("PolicyScores expects PolicyScores, mapping, or (exploration, final)")
    return PolicyScores(exploration=_to_exploration_score(values[0]), final=_to_finalization_score(values[1]))

def _to_policy_scores_schedule(x: Any) -> "RankSchedule":
    return _converted_rank_schedule(x, _to_policy_scores)

def _to_action_selection(x: Any) -> "ActionSelection":
    if isinstance(x, ActionSelection):
        return x
    if isinstance(x, Mapping):
        return ActionSelection(
            count=int(x.get("count", 1)),
            mode=str(x.get("mode", "best")),
            temperature=float(x.get("temperature", 0.0)),
        )
    raise TypeError("ActionSelection expects ActionSelection or mapping")

def _to_action_selection_schedule(x: Any) -> "RankSchedule":
    return _converted_rank_schedule(x, _to_action_selection)

def _to_action_pool(x: Any) -> "ActionPool":
    if isinstance(x, ActionPool):
        return x
    if isinstance(x, Mapping):
        return ActionPool(final_size=int(x.get("final_size", 16)))
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return ActionPool(final_size=int(x))
    raise TypeError("ActionPool expects ActionPool, mapping, or integer final_size")

def _to_action_pool_schedule(x: Any) -> "RankSchedule":
    return _converted_rank_schedule(x, _to_action_pool)

def _to_sampling_budget(x: Any) -> "SamplingBudget":
    if isinstance(x, SamplingBudget):
        return x
    if isinstance(x, Mapping):
        return SamplingBudget(
            one_hot=x.get("one_hot", "all"),
            sparse=int(x.get("sparse", 0)),
            dense=int(x.get("dense", 32)),
            sparse_max_weight=int(x.get("sparse_max_weight", 2)),
        )
    values = list(x)
    if len(values) == 3:
        return SamplingBudget(one_hot=values[0], sparse=int(values[1]), dense=int(values[2]), sparse_max_weight=2)
    if len(values) == 4:
        return SamplingBudget(
            one_hot=values[0],
            sparse=int(values[1]),
            dense=int(values[2]),
            sparse_max_weight=int(values[3]),
        )
    raise ValueError("SamplingBudget expects object, mapping, or [one_hot, sparse, dense, sparse_max_weight?]")

def _to_source_pool(x: Any, *, default_keep: int) -> "SourcePool":
    if isinstance(x, SourcePool):
        return x
    if isinstance(x, Mapping):
        return SourcePool(keep=int(x.get("keep", default_keep)), reserve=int(x.get("reserve", 0)))
    raise TypeError("SourcePool expects SourcePool or mapping")

def _to_z_bucket_search(x: Any) -> "ZBucketSearch":
    if isinstance(x, ZBucketSearch):
        return x
    if isinstance(x, Mapping):
        return ZBucketSearch(
            min_buckets=int(x.get("min_buckets", 32)),
            max_buckets=int(x.get("max_buckets", 0)),
            temperature=float(x.get("temperature", 0.0)),
            random_fraction=float(x.get("random_fraction", 0.0)),
            limit_bucket=int(x.get("limit_bucket", -1))
        )
    raise TypeError("ZBucketSearch expects ZBucketSearch or mapping")

def _to_tohpe_search(x: Any) -> "TohpeSearch":
    if isinstance(x, TohpeSearch):
        return x
    if isinstance(x, Mapping):
        return TohpeSearch(
            sampling=_to_sampling_budget(x.get("sampling", SamplingBudget())),
            pool=_to_source_pool(x.get("pool", SourcePool(keep=2, reserve=0)), default_keep=2),
            z_choices=int(x.get("z_choices", 8))
        )
    raise TypeError("TohpeSearch expects TohpeSearch or mapping")

def _to_tohpe_search_schedule(x: Any) -> "RankSchedule":
    return _converted_rank_schedule(x, _to_tohpe_search)

def _to_todd_search(x: Any) -> "ToddSearch":
    if isinstance(x, ToddSearch):
        return x
    if isinstance(x, Mapping):
        return ToddSearch(
            sampling=_to_sampling_budget(x.get("sampling", SamplingBudget())),
            pool=_to_source_pool(x.get("pool", SourcePool(keep=16, reserve=0)), default_keep=16),
            actions_per_bucket=int(x.get("actions_per_bucket", 4)),
            buckets=_to_z_bucket_search(x.get("buckets", ZBucketSearch()))
        )
    raise TypeError("ToddSearch expects ToddSearch or mapping")

def _to_todd_search_schedule(x: Any) -> "RankSchedule":
    return _converted_rank_schedule(x, _to_todd_search)

def float_rank_shedule_to_str(dss: List[RankSchedule], ranks: List[int]):
    output_r = []
    output_v = []
    for i, ds in enumerate(dss):
        down = ranks[i]
        up = ranks[i-1] if i > 0 else 1000000
        for r, v in ds.points:
            if r < up and r >= down:
                output_r.append(r)
                output_v.append(float(f"{float(v):.3f}"))
            elif r < down:
                output_r.append(down)
                output_v.append(float(f"{float(v):.3f}"))
                break
    return [tuple(output_r), tuple(output_v)]

def int_rank_shedule_to_str(dss: List[RankSchedule], ranks: List[int]):
    output_r = []
    output_v = []
    for i, ds in enumerate(dss):
        down = ranks[i]
        up = ranks[i-1] if i > 0 else 1000000
        for r, v in ds.points:
            if r < up and r >= down:
                output_r.append(r)
                output_v.append(int(v))
            elif r < down:
                output_r.append(down)
                output_v.append(int(v))
                break
    return [tuple(output_r), tuple(output_v)]

def caps_rank_shedule_to_str(dss: List[RankSchedule], ranks: List[int]):
    output_r = []
    output_v = []
    for i, ds in enumerate(dss):
        down = ranks[i]
        up = ranks[i-1] if i > 0 else 1000000
        for r, v in ds.points:
            if r < up and r >= down:
                output_r.append(r)
                output_v.append(tuple(_to_sample_caps(v)))
            elif r < down:
                output_r.append(down)
                output_v.append(tuple(_to_sample_caps(v)))
                break
    return [tuple(output_r), tuple(output_v)]

def score_rank_shedule_to_str(dss: List[RankSchedule], ranks: List[int]):
    output_r = []
    output_v = []
    for i, ds in enumerate(dss):
        down = ranks[i]
        up = ranks[i-1] if i > 0 else 1000000
        for r, v in ds.points:
            if r < up and r >= down:
                output_r.append(r)
                weights = [float(f"{v[i]:.3f}") for i in range(len(v))]
                centers = [float(f"{v[i+len(v)]:.3f}") for i in range(len(v))]
                power = float(f"{v.pow():.3f}")
                output = "("
                if any(weights):
                    output = output + f"{weights=},"
                if any(centers):
                    output = output + f"{centers=},"
                output = output + f"{power=})"
                output_v.append(output)
                # output_v.append(f"{weights=}, {centers=}, {pow=}")
            elif r < down:
                output_r.append(down)
                weights = [float(f"{v[i]:.3f}") for i in range(len(v))]
                centers = [float(f"{v[i+len(v)]:.3f}") for i in range(len(v))]
                power = float(f"{v.pow():.3f}")
                output = "("
                if any(weights):
                    output = output + f"{weights=},"
                if any(centers):
                    output = output + f"{centers=},"
                output = output + f"{power=})"
                output_v.append(output)
                break
    return [tuple(output_r), tuple(output_v)]

def object_rank_shedule_to_str(dss: List[RankSchedule], ranks: List[int], convert):
    output_r = []
    output_v = []
    for i, ds in enumerate(dss):
        down = ranks[i]
        up = ranks[i-1] if i > 0 else 1000000
        for r, v in ds.points:
            if r < up and r >= down:
                output_r.append(r)
                output_v.append(convert(v))
            elif r < down:
                output_r.append(down)
                output_v.append(convert(v))
                break
    return [tuple(output_r), tuple(output_v)]
    
def dao_rank_to_str(daos: List[Dao], ranks: List[int]):
    out = {}
    out["action_selection"] = object_rank_shedule_to_str(
        [dao.mode.selection for dao in daos],
        ranks,
        lambda s: f"beamwidth:{int(s.count)}/mode:{s.mode}/temp:{float(s.temperature):.3f}",
    )
    out["action_pool"] = object_rank_shedule_to_str(
        [dao.mode.pool for dao in daos],
        ranks,
        lambda p: f"final:{int(p.final_size)}",
    )
    out["tohpe"] = object_rank_shedule_to_str(
        [dao.mode.tohpe for dao in daos],
        ranks,
        lambda t: f"pool:{int(t.pool.keep)}/{int(t.pool.reserve)} z:{int(t.z_choices)} sampling:{t.sampling}",
    )
    out["todd"] = object_rank_shedule_to_str(
        [dao.mode.todd for dao in daos],
        ranks,
        lambda t: (
            f"pool:{int(t.pool.keep)}/{int(t.pool.reserve)} "
            f"per_bucket:{int(t.actions_per_bucket)} "
            f"z:min_buckets:{int(t.buckets.min_buckets)}/max_buckets:{int(t.buckets.max_buckets)}"
            f"/limit_bucket:{int(t.buckets.limit_bucket)} sampling:{t.sampling}"
        ),
    )
    out["scores"] = object_rank_shedule_to_str(
        [dao.mode.scores for dao in daos],
        ranks,
        lambda s: f"pool:{s.exploration} final:{s.final}",
    )
    return out

def _selected_rank_steps(best_ranks, max_lines=10):
    """
    Return entries with uniformly spaced rank values, always including the last entry.
    """
    n = len(best_ranks)
    if n <= max_lines:
        return list(range(n))
    
    # Get unique ranks and their first occurrence
    rank_to_step = {}
    for i, rank in enumerate(best_ranks):
        if rank not in rank_to_step:  # Keep first occurrence
            rank_to_step[rank] = i
    
    # Sort ranks in ascending order (better ranks first if lower is better)
    unique_ranks = sorted(rank_to_step.keys())
    
    if len(unique_ranks) <= max_lines:
        # If we have few unique ranks, print them all plus last
        selected_steps = set()
        for rank in unique_ranks:
            selected_steps.add(rank_to_step[rank])
        selected_steps.add(n - 1)  # Always include last step
        selected_steps = sorted(selected_steps)
    else:
        selected_steps = []
        
        first_rank = unique_ranks[0]
        selected_steps.append(rank_to_step[first_rank])
        step_size = (len(unique_ranks) - 1) / (max_lines - 2)  # -2 for first and last
        
        for i in range(1, max_lines - 1):
            rank_idx = int(i * step_size)
            if rank_idx < len(unique_ranks):
                rank = unique_ranks[rank_idx]
                selected_steps.append(rank_to_step[rank])
        
        last_step = n - 1
        if last_step not in selected_steps:
            selected_steps.append(last_step)
        
        selected_steps = sorted(set(selected_steps))
    
    return selected_steps


def _format_eval_rank(eval_step, rank) -> str:
    return f"{int(eval_step)}/{int(rank)}"


def _format_segment_improvements(segment_improvements, selected_steps: set[int]) -> str:
    if not segment_improvements:
        return "no improvements"

    tokens = []
    omitted = False
    emitted_any = False
    for local_idx, (step_idx, eval_step, rank) in enumerate(segment_improvements):
        if step_idx not in selected_steps:
            omitted = True
            continue
        if omitted or (not emitted_any and local_idx > 0):
            tokens.append("...")
            omitted = False
        tokens.append(_format_eval_rank(eval_step, rank))
        emitted_any = True
    if omitted:
        tokens.append("...")
    if not emitted_any:
        return "..."
    return " -> ".join(tokens)


def print_uniform_by_rank(best_ranks, best_evals, max_lines=10, init_events=None):
    """
    Print rank improvements grouped by init-rank timeline segments.
    """
    improvements = [
        (idx, int(eval_step), int(rank))
        for idx, (rank, eval_step) in enumerate(zip(best_ranks, best_evals))
    ]
    init_events = sorted(
        {
            (max(1, int(eval_step)), int(init_rank))
            for eval_step, init_rank in (init_events or [])
        }
    )
    if not init_events and improvements:
        init_events = [(1, improvements[0][2])]

    selected_steps = set(_selected_rank_steps(best_ranks, max_lines=max_lines))
    for seg_idx, (init_eval, _) in enumerate(init_events):
        next_eval = init_events[seg_idx + 1][0] if seg_idx + 1 < len(init_events) else None
        segment_improvements = [
            item
            for item in improvements
            if item[1] >= init_eval and (next_eval is None or item[1] < next_eval)
        ]
        if segment_improvements:
            selected_steps.add(segment_improvements[0][0])
            selected_steps.add(segment_improvements[-1][0])

    lines = []
    for seg_idx, (init_eval, init_rank) in enumerate(init_events):
        next_eval = init_events[seg_idx + 1][0] if seg_idx + 1 < len(init_events) else None
        segment_improvements = [
            item
            for item in improvements
            if item[1] >= init_eval and (next_eval is None or item[1] < next_eval)
        ]
        body = _format_segment_improvements(segment_improvements, selected_steps)
        display_init_eval = 0 if seg_idx == 0 and init_eval == 1 else init_eval
        lines.append(f"init={_format_eval_rank(display_init_eval, init_rank)}: {body}")
    if not lines and improvements:
        body = _format_segment_improvements(improvements, selected_steps)
        lines.append(f"improvements: {body}")
    return "".join(f"{line}\n" for line in lines)


def optimizer_landscape_summary(best_ranks, best_evals, total_evals, best_seen) -> str:
    if len(best_ranks) and len(best_evals):
        last = _format_eval_rank(best_evals[-1], best_ranks[-1])
    else:
        last = "none"
    return (
        f"last_improvement={last} total_evals={int(total_evals)} "
        f"best_seen_times={int(best_seen)}\n"
    )


def summarize_path_backups(root_dir: str = "data/path_backups", top_k: int = 10, init_word: str = "init") -> str:
    return PathStore(root_dir=root_dir).summarize(top_k=top_k)

class BaseEvaluator:
    todd: Todd
    _best_rank: int
    best_matrix: np.ndarray
    best_paths: List[Path]
    best_report: str
    best_pcfg: str
    best_eval: int
    total_eval: int
    best_seen: int
    shedule: str
    current_path: Path
    best_ranks: List[int]
    best_evals: List[int]
    init_events: List[Tuple[int, int]]

    @property
    def buckets_space(self) -> int:
        return self.loaded_rank ** 2 + self.loaded_rank

    @staticmethod
    def _path_rank_range(path: Path) -> Tuple[Optional[int], Optional[int]]:
        if path.final_node is None:
            return None, None
        ranks = [int(node.state.rows) for node in path.final_node.path_from_root()]
        if not ranks:
            return None, None
        return min(ranks), max(ranks)

    @staticmethod
    def _default_saved_path_rank_thr(
        path_name: str,
        *,
        margin: int = MIN_SAVED_PATH_MARGIN,
        root_dir: str = "data/path_backups",
        dao_fallback: Optional[Dao] = None,
    ) -> int:
        store = PathStore(root_dir=root_dir)
        paths = store.load(path_name, dao_fallback=dao_fallback or Dao())
        if not paths:
            raise RuntimeError(f"Empty paths for path_name={path_name}")
        final_rank = int(paths[0].final_node.state.rows)
        min_rank, max_rank = BaseEvaluator._path_rank_range(paths[0])
        if min_rank is None or max_rank is None:
            return final_rank + max(MIN_SAVED_PATH_MARGIN, int(margin))
        requested = final_rank + max(MIN_SAVED_PATH_MARGIN, int(margin))
        return max(min_rank - 1, min(requested, max_rank - 1))

    @staticmethod
    def _clamp_saved_path_rank_thr(path: Path, rank_thr: int) -> int:
        final_rank = int(path.final_node.state.rows)
        requested = max(int(rank_thr), final_rank + MIN_SAVED_PATH_MARGIN)
        min_rank, max_rank = BaseEvaluator._path_rank_range(path)
        if min_rank is None or max_rank is None:
            return requested
        if int(rank_thr) >= max_rank:
            return int(rank_thr)
        return max(min_rank - 1, min(requested, max_rank - 1))

    @staticmethod
    def _validate_saved_path_threshold(path: Path, path_name: str, rank_thr: int) -> None:
        if path.branch_path_at(rank_thr=rank_thr) is not None:
            return
        min_rank, max_rank = BaseEvaluator._path_rank_range(path)
        raise ValueError(
            f"cannot extract init matrix at rank_thr={rank_thr} from {path_name} "
            f"(path rank range {min_rank}-{max_rank}). "
            f"Use Evaluator(path_name={path_name!r}, margin=<chosen_margin>, fin_rank=...) "
            f"or choose init_rank_thr below {max_rank}."
        )

    @staticmethod
    def _path_root_rank(path: Path, fallback: Optional[int] = None) -> int:
        if path.final_node is not None:
            nodes = path.final_node.path_from_root()
            if nodes:
                return int(nodes[0].state.rows)
        if fallback is not None:
            return int(fallback)
        raise ValueError("cannot determine path root rank")

    def __init__(
        self,
        path_name: str = "init",
        init_rank_thr: Optional[int] = None,
        mat: Optional[Matrix] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        fin_rank: int = 1200,
        schedule: str = "rank",
        fill_tcounts: bool = False,
        path_root_dir: str = "data/path_backups",
        margin: int = MIN_SAVED_PATH_MARGIN,
    ):
        _ = fill_tcounts
        self.with_report = False
        self.current_path = Path()
        self.is_init = True
        self.initial_rank = None
        self.loaded_rank = None
        self.loaded_path_rank = None
        self.loaded_path_name = None
        self.loaded_path_limit_bucket = None
        self.loaded_path_root_dir = path_root_dir
        self.init_rank_thr = init_rank_thr
        self.margin = int(margin)
        self.best_paths = []
        self.best_ranks = []
        self.best_evals = []
        self._best_rank = 10000
        self.best_eval = 0
        self.total_eval = 0
        self.best_seen = 0
        self._search_started_at = time.perf_counter()
        self.time_to_final_rank_seconds: Optional[float] = None
        self.best_seed = None
        self.dao: Dao = Dao()
        self.max_depth = DEFAULT_MAX_DEPTH if max_depth is None else int(max_depth)
        if mat is None and path_name == "init":
            mat = get_matrix()
            self.initial_rank = int(mat.rows)
            self.init_rank_thr = mat.rows
        elif path_name != "init":
            if init_rank_thr is None:
                init_rank_thr = self._default_saved_path_rank_thr(
                    path_name, margin=self.margin, root_dir=path_root_dir
                )
                self.init_rank_thr = init_rank_thr
            else:
                loaded_paths = PathStore(root_dir=path_root_dir).load(path_name, dao_fallback=self.dao)
                if loaded_paths:
                    init_rank_thr = self._clamp_saved_path_rank_thr(loaded_paths[0], int(init_rank_thr))
                    self.init_rank_thr = init_rank_thr
            mat = self._load_saved_path_as_init(path_name, rank_thr=init_rank_thr, root_dir=path_root_dir)
        if self.current_path.final_node is None:
            self.current_path.final_node = Node(mat)
        if self.initial_rank is None:
            self.initial_rank = self._path_root_rank(self.current_path, fallback=mat.rows)
        self.loaded_rank = int(self.current_path.final_node.state.rows)
        self.init_rank = self.loaded_rank
        self.init_events = [(1, int(self.loaded_rank))]
        self.fin_rank = fin_rank
        self.schedule = str(schedule)
        self.shedule = self.schedule
        self.todd = Todd(self.dao, self.max_depth)
        self.tcount = []
        self.active_params = []
        self.best_params = []
        self._executor = None
        self._executor_key = None
        _register_active_evaluator(self)
        self.x0 = [0 for i in range(X0_LENGTH)]
        if not self.is_init and self.current_path.x0s:
            self.x0 = self.current_path.x0s[-1]

            
        self.reinit()    

    def _record_init_event(self) -> None:
        eval_step = max(1, int(self.total_eval))
        init_rank = int(self.current_path.final_node.state.rows)
        event = (eval_step, init_rank)
        if not self.init_events or self.init_events[-1] != event:
            self.init_events.append(event)

    def _load_saved_path_as_init(
        self,
        path_name: str,
        *,
        rank_thr: int,
        root_dir: str = "data/path_backups",
        xopt=None,
    ) -> Matrix:
        self.loaded_path_name = path_name
        self.loaded_path_root_dir = root_dir
        self.load_path(path_name, root_dir=root_dir)
        loaded_path = self.best_paths[0]
        self.loaded_path_rank = int(loaded_path.final_node.state.rows)
        self.loaded_path_limit_bucket = PathStore._path_limit_buckets({}, loaded_path.daos)
        self.init_rank_thr = int(rank_thr)
        self._validate_saved_path_threshold(loaded_path, path_name, int(rank_thr))
        PathStore(root_dir=root_dir).record_load(path_name, init_rank_thr=int(rank_thr))

        init_path = loaded_path.branch_path_at(rank_thr=int(rank_thr))
        self.current_path = init_path
        self.is_init = False
        self.initial_rank = self._path_root_rank(init_path, fallback=self.loaded_path_rank)
        self.loaded_rank = int(init_path.final_node.state.rows)
        self.init_rank = self.loaded_rank
        if self.current_path.daos:
            self.dao = deepcopy(self.current_path.daos[-1])
        if hasattr(self, "todd"):
            self.todd = Todd(self.dao, self.max_depth)
        if xopt is not None and hasattr(self, "active_params"):
            self.insert(xopt)
        elif init_path.x0s and hasattr(self, "x0"):
            self.x0 = init_path.x0s[-1]

        self.best_paths = []
        self.best_ranks = []
        self.best_evals = []
        self._best_rank = 100000
        self.best_eval = 0
        self.best_seen = 0
        self.best_seed = None
        return init_path.final_node.state

    def set_up_new_init(self, path_num:int, rank_thr:int, xopt=None):

        if path_num >= len(self.best_paths):
            return None
        new_path = self.best_paths[path_num].branch_path_at(rank_thr=rank_thr)
        if new_path is None:
            return None
        self.current_path = new_path
        self.loaded_rank = int(self.current_path.final_node.state.rows)
        self.init_rank = self.loaded_rank
        self.dao = deepcopy(self.current_path.daos[-1])
        self.todd = Todd(self.dao, self.max_depth)
        self._record_init_event()
        if xopt is not None:
            self.insert(xopt)
        else:
            self.x0 = new_path.x0s[-1]
        self.reinit()
        return self.extract_active()
    
    @property
    def init(self):
        return self.current_path.final_node.state

    @property
    def target_final_rank(self) -> int:
        return int(TARGET_FINAL_RANK)

    @property
    def path_num(self):
        return len(self.best_paths)
        
    @property
    def best_rank(self):
        return self._best_rank

    def map_par(self, mapping: callable, *unused_args, **kwargs):
        self.active_params.append(self.idx)
        self.idx += 1
        return mapping(self.x0[self.idx - 1], **kwargs)
        
    def insert(self, x):
        for i, a in enumerate(self.active_params):
            self.x0[a] = x[i]
        return self.x0
    
    def reinit(self):
        self.active_params = []
        self.idx = 0
        self.policy_mapping()
        return self.extract_active()
        
    def extract_active(self):
        return np.asarray([self.x0[a] for a in self.active_params], dtype=float)
        
    def __call__(self, params: Iterable):
        raise NotImplementedError

    def policy_mapping(self):
        raise NotImplementedError

    def _raise_if_soft_deadline(self) -> None:
        if _soft_deadline_reached():
            raise GracefulEvaluationTimeout("executor soft deadline reached")

    def _get_executor(self, executor_cls, executor_kind: str, worker_count: int):
        key = (executor_kind, worker_count)
        if self._executor_key != key:
            self.close_workers()
            self._executor = executor_cls(worker_count)
            self._executor_key = key
        return self._executor

    def close_workers(self, wait: bool = True, cancel_futures: bool = False):
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None
            self._executor_key = None

    def __del__(self):
        try:
            self.close_workers()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_executor"] = None
        state["_executor_key"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._executor = None
        self._executor_key = None

    def merge_run_state_from(self, other: "BaseEvaluator"):
        eval_offset = self.total_eval
        self.total_eval += other.total_eval
        self.tcount.extend(other.tcount)
        other_loaded_rank = getattr(other, "loaded_rank", getattr(other, "init_rank", 1))
        other_init_events = getattr(other, "init_events", [(1, int(other_loaded_rank))])

        if other._best_rank < self._best_rank:
            self.best_seen = other.best_seen
            self.best_ranks = list(other.best_ranks)
            self.best_evals = [eval_offset + e for e in other.best_evals]
            self.init_events = [
                (max(1, eval_offset + eval_step), init_rank)
                for eval_step, init_rank in other_init_events
            ]
            self.best_eval = eval_offset + other.best_eval
            self._best_rank = other._best_rank
            self.best_paths = _copy_path_headers(other.best_paths)
            self.best_seed = other.best_seed
            self.time_to_final_rank_seconds = other.time_to_final_rank_seconds
        elif other._best_rank == self._best_rank:
            self.best_paths.extend(_copy_path_headers(other.best_paths))
            self.best_seen += other.best_seen
            other_time = other.time_to_final_rank_seconds
            if other_time is not None and (
                self.time_to_final_rank_seconds is None
                or other_time < self.time_to_final_rank_seconds
            ):
                self.time_to_final_rank_seconds = other_time

    def _record_run_result(self, seed, node, counters, mats_ranks):
        rank = node.state.rows
        mats_ranks.append(rank)
        self.total_eval += counters[0]
        self.tcount.append(rank)
        if rank < self._best_rank:
            self.time_to_final_rank_seconds = time.perf_counter() - self._search_started_at
            self.best_seen = 0
            self.best_ranks.append(rank)
            self.best_evals.append(self.total_eval)
            self.best_eval = self.total_eval
            self._best_rank = rank
            self.best_paths = [
                self.current_path.branch_path(
                    node,
                    self.dao,
                    self.x0,
                )
            ]
        if rank == self._best_rank:
            self.best_paths.append(
                self.current_path.branch_path(
                    node,
                    self.dao,
                    self.x0,
                )
            )
            self.best_seen += counters[1]
            self.best_seed = seed

    def run(self, params, seeds, max_workers=None):
        if len(params) != len(self.active_params):
            raise RuntimeError(f"Num of params {len(params)} is not equal to the num of active params {len(self.active_params)}")
        self._raise_if_soft_deadline()
        self.insert(params)
        self.reinit()
        if max_workers is None:
            max_workers = DEFAULT_SEED_WORKERS
        worker_count = max(1, min(int(max_workers), len(seeds)))
        results = []
        timed_out = False
        if worker_count == 1:
            for seed in seeds:
                if _soft_deadline_reached():
                    timed_out = True
                    break
                results.append(
                    _worker_run_one_from_template(
                        seed,
                        self.current_path,
                        self.todd,
                    )
                )
        else:
            executor_kind = EXECUTOR_KIND.strip().lower()
            if executor_kind in {"process", "processes", "proc"}:
                executor_cls = ProcessPoolExecutor
            elif executor_kind in {"thread", "threads", "threading"}:
                executor_cls = ThreadPoolExecutor
            else:
                raise ValueError("EXECUTOR_KIND must be 'thread' or 'process'")

            ex = self._get_executor(executor_cls, executor_kind, worker_count)
            futures = [
                ex.submit(_worker_run_one_from_template, seed, self.current_path, self.todd)
                for seed in seeds
            ]
            pending = set(futures)
            while pending:
                if _soft_deadline_reached():
                    timed_out = True
                    for future in pending:
                        future.cancel()
                    break
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    results.append(future.result())

        # process deterministically in seed order
        seed_to_idx = {s:i for i,s in enumerate(seeds)}
        results.sort(key=lambda x: seed_to_idx[x[0]])

        mats_ranks = []
        for seed, node, counters in results:
            self._record_run_result(seed, node, counters, mats_ranks)
        if timed_out:
            raise GracefulEvaluationTimeout("executor soft deadline reached")
        return mats_ranks

    def get_best(self, timeout_salvage: bool = False):
        self.close_workers(wait=not timeout_salvage, cancel_futures=timeout_salvage)
        if not self.best_paths:
            raise RuntimeError("no completed best path available")
        best_path = self._pick_path_for_save()
        stats_start_rank = self._path_stats_start_rank()
        reported_loaded_rank = (
            int(stats_start_rank)
            if stats_start_rank is not None
            else (int(self.loaded_rank) if self.loaded_rank is not None else None)
        )
        reuse_note = ""
        load_note = ""
        if self.initial_rank is not None:
            load_note += f"\ninitial_rank: {self.initial_rank}"
        if self.target_final_rank > 0:
            load_note += f"\ntarget_final_rank: {self.target_final_rank}"
        if reported_loaded_rank is not None:
            load_note += f"\nloaded_rank: {reported_loaded_rank}"
        if self.loaded_path_rank is not None:
            load_note += f"\nloaded_path_rank: {self.loaded_path_rank}"
        if self.loaded_path_name is not None:
            load_note += f"\nloaded_path_name: {self.loaded_path_name}"
            if self.loaded_path_limit_bucket is not None:
                load_note += (
                    f" limit_bucket={PathStore._format_limit_buckets(self.loaded_path_limit_bucket)}"
                )
        if self.loaded_path_name is not None and self.init_rank_thr is not None:
            load_note += f"\ninit_rank_thr: {self.init_rank_thr}"
        if not self.is_init:
            loaded_path_rank = int(self.loaded_path_rank)
            child_improved_loaded = self.best_rank < loaded_path_rank
            parent_info = {
                "parent_path_name": self.loaded_path_name,
                "loaded_rank": int(reported_loaded_rank),
                "loaded_path_rank": loaded_path_rank,
                "init_rank_thr": int(self.init_rank_thr) if self.init_rank_thr is not None else None,
                "child_rank": int(self.best_rank),
                "child_improved_loaded": bool(child_improved_loaded),
            }
            self.path_name = self.save_path(
                "",
                root_dir=self.loaded_path_root_dir,
                parent_info=parent_info,
                path=best_path,
            )
            if child_improved_loaded:
                reuse_note = f"\nloaded_path_improved: 1"
            else:
                reuse_note = (
                    f"\nNOTE: no improvement over loaded path "
                    f"(loaded_path_rank={loaded_path_rank}, best_rank={self.best_rank}); "
                    f"saved_child_path={self.path_name}"
                )
            if self.loaded_path_name is not None and self.loaded_path_rank is not None:
                PathStore(root_dir=self.loaded_path_root_dir).record_result(
                    self.loaded_path_name,
                    loaded_rank=int(self.loaded_path_rank),
                    child_rank=int(self.best_rank),
                    init_rank_thr=self.init_rank_thr,
                )
        else:
            self.path_name = self.save_path("", path=best_path)
        return (
            best_path.final_node.state.to_numpy(), 
            best_path.format_path_stats(start_rank=stats_start_rank),
            "\nsearch_stat:\n" +
            f"rank 0.9q={np.quantile(self.tcount, 0.9) if self.tcount else 'n/a'} \n" +
            f"rank 0.1q={np.quantile(self.tcount, 0.1) if self.tcount else 'n/a'} \n" +
            print_uniform_by_rank(self.best_ranks, self.best_evals, 8, self.init_events) +
            optimizer_landscape_summary(self.best_ranks, self.best_evals, self.total_eval, self.best_seen) +
            f"total_evals: {self.total_eval}" +
            f"\nbest_seen_times: {self.best_seen}" + 
            f"\ntime_to_final_rank_seconds: {self.time_to_final_rank_seconds if self.time_to_final_rank_seconds is not None else 'n/a'}" +
            (f"\ntimeout_salvaged: 1" if timeout_salvage else "") +
            reuse_note +
            load_note +
            f"\nthis path name: {self.path_name}"
            )

    def _path_stats_start_rank(self) -> Optional[int]:
        if not getattr(self, "init_events", None):
            return None
        if self.best_eval:
            ranks = [
                int(init_rank)
                for eval_step, init_rank in self.init_events
                if int(eval_step) <= int(self.best_eval)
            ]
        else:
            ranks = [int(init_rank) for _eval_step, init_rank in self.init_events]
        if not ranks:
            return None
        return max(ranks)

    def _path_name_mid_rank(self, path: Path) -> int:
        rank = int(path.final_node.state.rows)
        init_rank = self._path_root_rank(path, fallback=getattr(self, "initial_rank", rank))
        stats_start_rank = self._path_stats_start_rank()
        if stats_start_rank is not None:
            return int(stats_start_rank)
        if self.init_rank_thr is not None:
            return int(self.init_rank_thr)
        return init_rank

    def _path_depth(self, path: Path) -> int:
        depth = 0
        node = path.final_node
        while node is not None:
            depth += 1
            node = node.parent
        return depth

    def _pick_path_for_save(self) -> Path:
        if not self.best_paths:
            raise ValueError("no best paths available to save")
        best_rank = min(p.final_node.state.rows for p in self.best_paths if p.final_node is not None)
        candidates = [p for p in self.best_paths if p.final_node is not None and p.final_node.state.rows == best_rank]
        # Prefer a shorter solution when ranks tie.
        return min(candidates, key=self._path_depth)

    def _auto_hashed_name(self, name: str, path: Path) -> str:
        rank = int(path.final_node.state.rows)
        init_rank = self._path_root_rank(path, fallback=getattr(self, "initial_rank", rank))
        mat = path.final_node.state.to_numpy()
        mid_rank = self._path_name_mid_rank(path)
        program_id = os.getenv("GIGAEVO_PROGRAM_ID") or PROGRAM_ID
        if program_id:
            return f"{name}i{init_rank}_m{mid_rank}_f{rank}_{program_id[:8]}"
        h = hashlib.blake2b(digest_size=6)
        h.update(str(init_rank).encode("utf-8"))
        h.update(str(mid_rank).encode("utf-8"))
        h.update(str(rank).encode("utf-8"))
        h.update(mat.tobytes())
        return f"{name}i{init_rank}_m{mid_rank}_f{rank}_{h.hexdigest()}"

    def save_path(
        self,
        name: str,
        root_dir: str = "data/path_backups",
        store_daos: bool = True,
        auto_hash: bool = True,
        parent_info: Optional[dict[str, Any]] = None,
        path: Optional[Path] = None,
    ) -> str:
        store = PathStore(root_dir=root_dir)
        best_path = path if path is not None else self._pick_path_for_save()
        save_name = self._auto_hashed_name(name, best_path) if auto_hash else name
        store.save(
            save_name,
            [best_path],
            store_daos=store_daos,
            parent_info=parent_info,
        )
        return save_name

    def load_path(self, name: str, root_dir: str = "data/path_backups", dao_fallback: Optional[Dao] = None):
        store = PathStore(root_dir=root_dir)
        fallback = self.dao if dao_fallback is None else dao_fallback
        self.best_paths = store.load(name, dao_fallback=fallback)
        if len(self.best_paths) == 0:
            raise RuntimeError(f"Empty paths for path_name={name}")
        return self.best_paths
    
    def insert_path(self, name: str, root_dir: str = "data/path_backups", dao_fallback: Optional[Dao] = None):
        store = PathStore(root_dir=root_dir)
        fallback = self.dao if dao_fallback is None else dao_fallback
        self.best_paths = store.load(name, dao_fallback=fallback)
        self._best_rank = self.best_paths[0].final_node.state.rows
        self.best_seen = 1
        self.total_eval = 1
        return self.best_paths
    
    def set_scores(
        self,
        x: Any = None,
        vals=None,
        *,
        ranks=None,
        values=None,
        exploration=None,
        final=None,
    ):
        if exploration is not None or final is not None:
            if x is not None or vals is not None or ranks is not None or values is not None:
                raise ValueError("use either a PolicyScores value/schedule or exploration=.../final=..., not both")
            x = PolicyScores(
                exploration=_to_exploration_score(
                    exploration if exploration is not None else [1.0, 0.0, 0.0, 0.0, 0.0]
                ),
                final=_to_finalization_score(final if final is not None else [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
        else:
            x = _rank_args(x, vals, ranks=ranks, values=values)
        self.dao.mode.scores = _to_policy_scores_schedule(x)

    def set_action_selection(self, x: Any = None, vals=None, *, ranks=None, values=None):
        x = _rank_args(x, vals, ranks=ranks, values=values)
        schedule = _to_action_selection_schedule(x)
        self.dao.mode.selection = schedule

    def set_action_pool(self, x: Any = None, vals=None, *, ranks=None, values=None):
        x = _rank_args(x, vals, ranks=ranks, values=values)
        self.dao.mode.pool = _to_action_pool_schedule(x)

    def set_tohpe_search(self, x: Any = None, vals=None, *, ranks=None, values=None):
        x = _rank_args(x, vals, ranks=ranks, values=values)
        self.dao.mode.tohpe = _to_tohpe_search_schedule(x)

    def set_todd_search(self, x: Any = None, vals=None, *, ranks=None, values=None):
        x = _rank_args(x, vals, ranks=ranks, values=values)
        self.dao.mode.todd = _to_todd_search_schedule(x)
