import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mcts_dao import Dao, Path, RankSchedule
from node import ExplorationScore, FinalizationScore, Matrix, Node, Tensor3D
from path_store import PathStore, X0_LENGTH
from todd import Todd

MIN_SAVED_PATH_MARGIN = 10
DEFAULT_MATRIX_PATH = "npy/gf2^9.npy"

DEFAULT_MATRIX_PATH = "data/init_npy/other/ham15_high.npy"
# DEFAULT_MATRIX_PATH = "data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^64_644310.npy"
# DEFAULT_MATRIX_PATH = "data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^32_3228310.npy"

DEFAULT_SEED_WORKERS = min(4, os.cpu_count() or 1)
EXECUTOR_KIND = "process"
PROGRAM_ID: Optional[str] = os.getenv("GIGAEVO_PROGRAM_ID")


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
    arr = trim_trailing_zero_cols(arr)
    arr = np.asarray(arr != 0, dtype=np.bool_)

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return arr[:0, :0]

    arr = arr[np.any(arr, axis=1)]
    if arr.shape[0] == 0:
        return arr[:, :0]

    rows, first_indices, counts = np.unique(arr, axis=0, return_index=True, return_counts=True)
    keep = (counts & 1) == 1
    order = np.argsort(first_indices[keep])
    rows = rows[keep][order]
    return trim_trailing_zero_cols(np.ascontiguousarray(rows))


def load_matrix_array(path: str | os.PathLike[str]) -> np.ndarray:
    return normalize_matrix_array(np.load(path))


def _worker_run_one_from_template(
    seed: int,
    path: Path,
    todd: Todd,
    bs_width: RankSchedule = RankSchedule.constant(1),
    todd_width: RankSchedule = RankSchedule.constant(1),
):
    # Todd.run only reads the incoming path and creates new child nodes.
    # Deep-copying a GF(2^64) path can recurse through ~1000 Node.parent links.
    node, counters = todd.run(path, bs_width, todd_width, True, seed)
    return seed, node, counters


def _copy_path_header(path: Path) -> Path:
    new_path = Path()
    new_path.final_node = path.final_node
    new_path.ranks_thr = list(path.ranks_thr)
    new_path.daos = deepcopy(path.daos)
    new_path.bs_widths = deepcopy(path.bs_widths)
    new_path.todd_widths = deepcopy(path.todd_widths)
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
    return ExplorationScore(*x)

def _to_finalization_score(x: Any) -> "FinalizationScore":
    """Accept FinalizationScore | (wred, wdim, wpossible_red, wtohpe_dim) and return FinalizationScore."""
    if isinstance(x, FinalizationScore):
        return x
    x = list(x)
    xs = np.asarray(x)
    x = np.asarray(xs)/np.sqrt(np.sum(xs*xs)) if np.any(xs) else xs
    return FinalizationScore(*x)

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
    
def dao_rank_to_str(daos: List[Dao], ranks: List[int]):
    out = {}
    out["tohpe_vector_samples"] = caps_rank_shedule_to_str([dao.mode.tohpe_vector_samples for dao in daos], ranks)
    out["todd_vector_samples"] = caps_rank_shedule_to_str([dao.mode.todd_vector_samples for dao in daos], ranks)
    out["top_pool"] = int_rank_shedule_to_str([dao.mode.top_pool for dao in daos], ranks)
    out["tohpe_pool_size"] = int_rank_shedule_to_str([dao.mode.tohpe_pool_size for dao in daos], ranks)
    out["todd_pool_size"] = int_rank_shedule_to_str([dao.mode.todd_pool_size for dao in daos], ranks)
    out["pool_scores"] = score_rank_shedule_to_str([dao.mode.pool_scores for dao in daos], ranks)
    out["final_scores"] = score_rank_shedule_to_str([dao.mode.final_scores for dao in daos], ranks)
    out["min_z_to_research"] = int_rank_shedule_to_str([dao.mode.min_z_to_research for dao in daos], ranks)
    # dict["temperature"] = float_rank_shedule_to_str([dao.mode.temperature for dao in daos], ranks)
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


def print_uniform_by_rank(best_ranks, best_evals, max_lines=10, init_events=None):
    """
    Print rank improvements with init-rank timeline events.
    """
    selected_steps = _selected_rank_steps(best_ranks, max_lines=max_lines)
    init_events = sorted(
        {
            (max(1, int(eval_step)), int(init_rank))
            for eval_step, init_rank in (init_events or [])
        }
    )

    s = ""
    next_init = 0
    for step in selected_steps:
        rank = best_ranks[step]
        eval_step = best_evals[step]
        while next_init < len(init_events) and init_events[next_init][0] <= eval_step:
            init_eval, init_rank = init_events[next_init]
            s += f"init={init_rank} since eval={init_eval}\n"
            next_init += 1
        s += f"Rank={rank} at eval={eval_step}\n"
    while next_init < len(init_events):
        init_eval, init_rank = init_events[next_init]
        s += f"init={init_rank} since eval={init_eval}\n"
        next_init += 1
    return s


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
    bs_width: RankSchedule
    todd_width: RankSchedule
    current_path: Path
    best_ranks: List[int]
    best_evals: List[int]
    init_events: List[Tuple[int, int]]

    # Backward-compatible alias for typo "best_pathes"
    @property
    def best_pathes(self) -> List[Path]:
        return self.best_paths

    @best_pathes.setter
    def best_pathes(self, value: List[Path]) -> None:
        self.best_paths = value

    @classmethod
    def from_saved_path(
        cls,
        path_name: str,
        rank_thr: Optional[int] = None,
        margin: int = MIN_SAVED_PATH_MARGIN,
        root_dir: str = "data/path_backups",
        **kwargs: Any,
    ) -> "BaseEvaluator":
        if rank_thr is None:
            rank_thr = cls._default_saved_path_rank_thr(path_name, margin=margin, root_dir=root_dir)
        kwargs["path_name"] = path_name
        kwargs["init_rank_thr"] = rank_thr
        kwargs["path_root_dir"] = root_dir
        return cls(**kwargs)

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
            f"Use Evaluator.from_saved_path({path_name!r}, margin=<chosen_margin>, max_depth=...) "
            f"or choose rank_thr below {max_rank}."
        )

    def __init__(
        self,
        path_name: str = "init",
        init_rank_thr: Optional[int] = None,
        mat: Optional[Matrix] = None,
        max_depth: int = 300,
        fin_rank: int = 161,
        shedule: str = "rank",
        fill_tcounts: bool = False,
        path_root_dir: str = "data/path_backups",
    ):
        self.with_report = False
        self.current_path = Path()
        self.is_init = True
        self.loaded_rank = None
        self.loaded_path_name = None
        self.loaded_path_root_dir = path_root_dir
        self.init_rank_thr = init_rank_thr
        self.best_paths = []
        self.best_ranks = []
        self.best_evals = []
        self._best_rank = 10000
        self.best_eval = 0
        self.total_eval = 0
        self.best_seen = 0
        self.best_seed = None
        self.dao: Dao = Dao()
        self.max_depth = max_depth
        if mat is None and path_name == "init":
            mat = get_matrix()
            self.init_rank_thr = mat.rows
        elif path_name != "init":
            if init_rank_thr is None:
                init_rank_thr = self._default_saved_path_rank_thr(path_name, root_dir=path_root_dir)
                self.init_rank_thr = init_rank_thr
            else:
                loaded_paths = PathStore(root_dir=path_root_dir).load(path_name, dao_fallback=self.dao)
                if loaded_paths:
                    init_rank_thr = self._clamp_saved_path_rank_thr(loaded_paths[0], int(init_rank_thr))
                    self.init_rank_thr = init_rank_thr
            mat = self._load_saved_path_as_init(path_name, rank_thr=init_rank_thr, root_dir=path_root_dir)
        if self.current_path.final_node is None:
            self.current_path.final_node = Node(mat)
        self.init_rank = mat.rows
        self.init_events = [(1, int(self.init_rank))]
        self.fin_rank = fin_rank
        self.shedule = shedule
        self.todd = Todd(self.dao, max_depth)
        self.tcount = []
        self.active_params = []
        self.best_params = []
        self._executor = None
        self._executor_key = None
        # self
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
        self.loaded_rank = int(loaded_path.final_node.state.rows)
        self.init_rank_thr = int(rank_thr)
        self._validate_saved_path_threshold(loaded_path, path_name, int(rank_thr))
        PathStore(root_dir=root_dir).record_load(path_name, init_rank_thr=int(rank_thr))

        init_path = loaded_path.branch_path_at(rank_thr=int(rank_thr))
        self.current_path = init_path
        self.is_init = False
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
    def path_num(self):
        return len(self.best_paths)
        
    @property
    def best_rank(self):
        return self._best_rank

    def map_par(self, mapping: callable, thr: int = 0, **kwargs):
        if self.init.rows > thr:
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
        x = []
        for i, a in enumerate(self.active_params):
            x.append(self.x0[a])
        return x
        
    def __call__(self, params: Iterable):
        pass

    def policy_mapping(self):
        pass

    def _get_executor(self, executor_cls, executor_kind: str, max_workers: int):
        key = (executor_kind, max_workers)
        if self._executor_key != key:
            self.close_workers()
            self._executor = executor_cls(max_workers=max_workers)
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
        other_init_events = getattr(other, "init_events", [(1, int(other.init_rank))])

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
        elif other._best_rank == self._best_rank:
            self.best_paths.extend(_copy_path_headers(other.best_paths))
            self.best_seen += other.best_seen

    def _record_run_result(self, seed, node, counters, mats_ranks):
        rank = node.state.rows
        mats_ranks.append(rank)
        self.total_eval += counters[0]
        self.tcount.append(rank)
        if rank < self._best_rank:
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
                    self.bs_width,
                    self.todd_width,
                )
            ]
        if rank == self._best_rank:
            self.best_paths.append(
                self.current_path.branch_path(
                    node,
                    self.dao,
                    self.x0,
                    self.bs_width,
                    self.todd_width,
                )
            )
            self.best_seen += counters[1]
            self.best_seed = seed

    def run(self, params, seeds, max_workers=1):
        if len(params) != len(self.active_params):
            raise RuntimeError(f"Num of params {len(params)} is not equal to the num of active params {len(self.active_params)}")
        self.insert(params)
        self.reinit()
        # self.policy_setup(params)
        if max_workers is None:
            max_workers = DEFAULT_SEED_WORKERS
        max_workers = max(1, min(int(max_workers), len(seeds)))
        if max_workers == 1:
            results = [
                _worker_run_one_from_template(seed, self.current_path, self.todd, self.bs_width, self.todd_width)
                for seed in seeds
            ]
        else:
            executor_kind = EXECUTOR_KIND.strip().lower()
            if executor_kind in {"process", "processes", "proc"}:
                executor_cls = ProcessPoolExecutor
            elif executor_kind in {"thread", "threads", "threading"}:
                executor_cls = ThreadPoolExecutor
            else:
                raise ValueError("EXECUTOR_KIND must be 'thread' or 'process'")

            ex = self._get_executor(executor_cls, executor_kind, max_workers)
            futures = [
                ex.submit(_worker_run_one_from_template, seed, self.current_path, self.todd, self.bs_width, self.todd_width)
                for seed in seeds
            ]
            results = [f.result() for f in futures]

        # process deterministically in seed order
        seed_to_idx = {s:i for i,s in enumerate(seeds)}
        results.sort(key=lambda x: seed_to_idx[x[0]])

        mats_ranks = []
        for seed, node, counters in results:
            self._record_run_result(seed, node, counters, mats_ranks)
        return mats_ranks

    def get_best(self, timeout_salvage: bool = False):
        self.close_workers(wait=not timeout_salvage, cancel_futures=timeout_salvage)
        if not self.best_paths:
            raise RuntimeError("no completed best path available")
        best_path = self._pick_path_for_save()
        reuse_note = ""
        load_note = ""
        if self.loaded_rank is not None:
            load_note = f"\nloaded_rank: {self.loaded_rank}"
            if self.loaded_path_name is not None:
                load_note += f"\nloaded_path_name: {self.loaded_path_name}"
            if self.init_rank_thr is not None:
                load_note += f"\ninit_rank_thr: {self.init_rank_thr}"
        if not self.is_init:
            child_improved_loaded = self.best_rank < self.loaded_rank
            parent_info = {
                "parent_path_name": self.loaded_path_name,
                "loaded_rank": int(self.loaded_rank),
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
                reuse_note = f"\nloaded_path_rank: {self.loaded_rank}\nloaded_path_improved: 1"
            else:
                reuse_note = (
                    f"\nNOTE: no improvement over loaded path "
                    f"(loaded_rank={self.loaded_rank}, best_rank={self.best_rank}); "
                    f"saved_child_path={self.path_name}"
                )
            if self.loaded_path_name is not None and self.loaded_rank is not None:
                PathStore(root_dir=self.loaded_path_root_dir).record_result(
                    self.loaded_path_name,
                    loaded_rank=int(self.loaded_rank),
                    child_rank=int(self.best_rank),
                    init_rank_thr=self.init_rank_thr,
                )
        else:
            self.path_name = self.save_path("", path=best_path)
        stats_start_rank = self._path_stats_start_rank()
        return (
            best_path.final_node.state.to_numpy(), 
            best_path.format_path_stats(start_rank=stats_start_rank),
            "\nbest_policy:\n" +
            str(dao_rank_to_str(best_path.daos, best_path.ranks_thr + [0])) + "\nsearch_stat:\n" +
            f"rank 0.9q={np.quantile(self.tcount, 0.9) if self.tcount else 'n/a'} \n" +
            f"rank 0.1q={np.quantile(self.tcount, 0.1) if self.tcount else 'n/a'} \n" +
            print_uniform_by_rank(self.best_ranks, self.best_evals, 8, self.init_events) +
            f"total_evals: {self.total_eval}" +
            f"\nbest_seen_times: {self.best_seen}" + 
            (f"\ntimeout_salvaged: 1" if timeout_salvage else "") +
            reuse_note +
            load_note +
            f"\nevo path statistics:\n{summarize_path_backups(top_k=11)}" +
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
        path_root = path.final_node.path_from_root()
        init_rank = int(path_root[0].state.rows) if path_root else rank
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
        path_root = path.final_node.path_from_root()
        init_rank = int(path_root[0].state.rows) if path_root else rank
        mat = path.final_node.state.to_numpy()
        mid_rank = self._path_name_mid_rank(path)
        if PROGRAM_ID:
            return f"{name}i{init_rank}_m{mid_rank}_f{rank}_{PROGRAM_ID[:8]}"
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
        # self.dao = 
        return self.best_paths
    
    def set_final_scores(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.final_scores = _to_frank_schedule(x)
        
    def set_pool_scores(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.pool_scores = _to_erank_schedule(x)

    def set_tohpe_vector_samples(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.tohpe_vector_samples = _to_caps_rank_schedule(x)

    def set_todd_vector_samples(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.todd_vector_samples = _to_caps_rank_schedule(x)

    def set_sparse_max_weight(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.sparse_max_weight = _to_rank_schedule(x)

    def set_beamsearch_width(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.bs_width = _to_rank_schedule(x)
        
    def set_todd_width(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.todd_width = _to_rank_schedule(x)

    def set_min_z_to_research(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.min_z_to_research = _to_rank_schedule(x)

    def set_max_z_to_research(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.max_z_to_research = _to_rank_schedule(x)

    def set_min_pool_size(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.min_pool_size = _to_rank_schedule(x)

    def set_temperature(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.temperature = _to_rank_schedule(x)

    def set_max_pool_size(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.top_pool = _to_rank_schedule(x)

    def set_tohpe_pool_size(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.tohpe_pool_size = _to_rank_schedule(x)

    def set_todd_pool_size(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.todd_pool_size = _to_rank_schedule(x)

    def set_min_tohpe_actions(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.min_tohpe_actions = _to_rank_schedule(x)

    def set_min_todd_actions(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.min_todd_actions = _to_rank_schedule(x)

    def set_max_from_single_ns(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.max_from_single_ns = _to_rank_schedule(x)

    def set_tohpe_sample(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.tohpe_sample = _to_rank_schedule(x)

    def set_bucket_temperature(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.bucket_temperature = _to_rank_schedule(x)

    def set_bucket_random_fraction(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.bucket_random_fraction = _to_rank_schedule(x)

    def set_max_per_signature(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.max_per_signature = _to_rank_schedule(x)

    def set_min_reduction(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.min_reduction = _to_rank_schedule(x)

    def set_max_reduction(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.max_reduction  = _to_rank_schedule(x)
