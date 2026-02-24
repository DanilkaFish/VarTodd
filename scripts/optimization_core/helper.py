import numpy as np
from node import Matrix, FinalizationScore, ExplorationScore, Tensor3D
from mcts_dao import Dao, DepthSchedule, RankSchedule
from todd import Todd
from typing import Iterable, Sequence, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

def _worker_run_one_from_template(seed, init, todd: Todd, bs_width: RankSchedule = RankSchedule.constant(1), todd_width: RankSchedule = RankSchedule.constant(1)):
    init = Matrix.from_numpy(init)
    todd = deepcopy(todd)
    # self.todd = Todd(self.dao, max_depth)
    mat, report, counters = todd.run(init, bs_width, todd_width, True, seed, )
    report, path = report
    return seed, mat.shape[0], tuple(path), report, counters


def find_rank(path, rank):
    for i, mat in enumerate(path):
        if mat.rows < rank:
            return path[max(i-1, 0)]
    return None
        
def get_matrix(name:str=None) -> Matrix:
    if name is None:
        # return Matrix.from_numpy(np.load("gf5.npy") )
        # return Matrix.from_numpy(np.load("npy/gf2^3_mult_fr_310.matrix.npy") )
        # return Matrix.from_numpy(np.load("npy/gf2^4_mult_fr_43210.matrix.npy") )
        # return Matrix.from_numpy(np.load("npy/gf2^5_mult_fr_53210.matrix.npy") )
        # return Matrix.from_numpy(np.load("npy/gf2^6_mult_fr_630.matrix.npy") )
        # return Matrix.from_numpy(np.load("npy/gf2^7_mult_fr_730.matrix.npy") )
        # return Matrix.from_numpy(np.load("npy/gf2^8_mult_fr_84310.matrix.npy") )
        # return Matrix.from_numpy(np.load("npy/1733init173-00231.npy") )
        # return Matrix.from_numpy(np.load("npy/1731init129-00168.npy") )
        # return Matrix.from_numpy(np.load("npy/gf2^9_mult_fr_940.matrix.npy") )
        return Matrix.from_numpy(np.load("data/init_npy/gf2^10_mult_fr_1030.matrix.npy") )
        # return Matrix.from_numpy(np.load("npy/gf2^10_mult_1030.qc.matrix.npy").T )
    return Matrix.from_numpy(np.load(f"npy/{name}.npy") )

DepthKV = Sequence[Tuple[int, Any]]

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

def float_rank_shedule_to_str(ds: RankSchedule):
    return [(x, float(f"{y:.2f}")) for x, y in ds.points]

def int_rank_shedule_to_str(ds: RankSchedule):
    return ds.points

def score_rank_shedule_to_str(ds: RankSchedule):
    return [(x, [float(f"{t[i]:.2f}") for i in range(len(t))]) for x, t in ds.points]
    
def dao_rank_to_str(dao: Dao):
    dict = {}
    dict["num_samples"] = int_rank_shedule_to_str(dao.mode.num_samples)
    dict["top_pool"] = int_rank_shedule_to_str(dao.mode.top_pool)
    dict["try_only_tohpe"] = int_rank_shedule_to_str(dao.mode.try_only_tohpe)
    dict["max_tohpe"] = int_rank_shedule_to_str(dao.mode.max_tohpe)
    dict["num_tohpe_sample"] = int_rank_shedule_to_str(dao.mode.num_tohpe_sample)
    dict["max_from_single_ns"] = int_rank_shedule_to_str(dao.mode.max_from_single_ns)
    dict["min_reduction"] = int_rank_shedule_to_str(dao.mode.min_reduction)
    dict["max_reduction"] = int_rank_shedule_to_str(dao.mode.max_reduction)
    dict["pool_weights"] = score_rank_shedule_to_str(dao.mode.pool_weights)
    dict["final_weights"] = score_rank_shedule_to_str(dao.mode.final_weights)
    dict["max_z_to_research"] = float_rank_shedule_to_str(dao.mode.max_z_to_research)
    dict["gen_part"] = float_rank_shedule_to_str(dao.mode.gen_part)
    dict["temperature"] = float_rank_shedule_to_str(dao.mode.temperature)
    # dict["non_improving_prob"] = float_rank_shedule_to_str(dao.mode.non_improving_prob)
    return dict

class BaseEvaluator:
    todd: Todd
    init: Matrix
    best_rank: int=100000
    best_matrix: np.ndarray
    best_report: str
    best_pcfg: str
    best_eval: int = 0
    total_eval: int = 0
    best_seen: int = 0
    shedule: str = "rank"
    bs_width: RankSchedule = RankSchedule.constant(1)
    todd_width: RankSchedule = RankSchedule.constant(1)
    def __init__(self, mat: Matrix,  max_depth: int, fin_rank: int = 161, shedule: str = "rank"):
        self.with_report = False
        self.fin_rank = fin_rank
        self.shedule = shedule
        self.dao: Dao = Dao()
        self.dao.threads = 4
        self.todd = Todd(self.dao, max_depth)
        self.init = mat
        self.root = mat
        self.tcount = []
        self.best_pathes = []
        self.active_params = []
        self.best_params = []
        self.x0 = [0 for i in range(200)]
        self.reinit()    

    def set_up_new_init(self, mat: Matrix, xopt=None):
        self.init = mat
        if xopt is not None:
            self.insert(xopt)
        self.reinit()
        return self.extract_active()
        
    def set_up_activation(self):
        self.policy_setup(self.extract_active())

    def map_par(self, mapping: callable, thr: int, **kwargs):
        if self.init.rows > thr:
            self.active_params.append(self.idx)
        self.idx += 1
        return mapping(self.x0[self.idx - 1], **kwargs)
        
    def insert(self, x):
        for i, a in enumerate(self.active_params):
            self.x0[a] = x[i]
        return self.x0
    
    def reinit(self):
        # for i, a in enumerate(self.active_params):
            # self.x0[a] = x[i]
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

    def run(self, params, seeds, max_workers=4):
        if len(params) != len(self.active_params):
            raise RuntimeError(f"Num of params {len(params)} is not equal to the num of active params {len(self.active_params)}")
        self.insert(params)
        self.reinit()
        # self.policy_setup(params)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            np_mat = self.init.to_numpy()
            futures = [
                ex.submit(_worker_run_one_from_template, seed, np_mat, self.todd, self.bs_width, self.todd_width)
                for seed in seeds
            ]
            results = [(*f.result(), s) for f, s in zip(futures, seeds)]

        # process deterministically in seed order
        seed_to_idx = {s:i for i,s in enumerate(seeds)}
        results.sort(key=lambda x: seed_to_idx[x[0]])

        mats_ranks = []
        for _, rank, path, report, counters, seed in results:
            mats_ranks.append(rank)
            self.total_eval += counters[0]
            self.tcount.append(rank)
            if rank < self.best_rank:
                self.best_seen = 0
                self.best_eval = self.total_eval
                self.best_matrix = path[-1].to_numpy()
                self.best_rank = rank
                self.best_report = f"{self.root.rows}:root\n...\n" + report
                self.best_pcfg = dao_rank_to_str(self.dao)
                self.best_pathes = [path]
                self.best_params = [self.x0]
            if rank == self.best_rank:
                self.best_params.append(self.x0)
                self.best_pathes.append(path)
                self.best_seen += counters[1]
                self.best_seed = seed
        return mats_ranks

    def get_best(self):
        return (
            self.best_matrix, 
            self.best_report, 
            f"\nbest_count/best_first_at/total fun evals: {self.best_seen}/{self.best_eval}/{self.total_eval}\n" + 
                f"rank median={np.median(self.tcount)} " +
                f"rank 0.1q={np.quantile(self.tcount, 0.1)} \n" +
                str(self.best_pcfg) + 
                f"\nbest_seed={self.best_seed}" 
        )
    def set_final_weights(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.final_weights = _to_frank_schedule(x)
        
    def set_pool_weights(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.pool_weights = _to_erank_schedule(x)

    def set_num_samples(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.num_samples = _to_rank_schedule(x)

    def set_gen_part(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.gen_part = _to_rank_schedule(x)

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
        self.dao.mode.max_z_to_research = _to_rank_schedule(x)
        self.dao.threads = 3

    def set_min_pool_size(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.min_pool_size = _to_rank_schedule(x)

    def set_temperature(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.temperature = _to_rank_schedule(x)

    def set_non_improving_prob(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.non_improving_prob = _to_rank_schedule(x)

    def set_top_pool(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.top_pool = _to_rank_schedule(x)

    def set_max_tohpe(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.max_tohpe = _to_rank_schedule(x)

    def set_try_only_tohpe(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.try_only_tohpe = _to_rank_schedule(x)

    def set_max_from_single_ns(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.max_from_single_ns = _to_rank_schedule(x)

    def set_tohpe_num_best(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.num_tohpe_sample = _to_rank_schedule(x)

    def set_min_reduction(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.min_reduction = _to_rank_schedule(x)

    def set_max_reduction(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.max_reduction  = _to_rank_schedule(x)