"""Standalone stand-in for the GigaEvo harness's per-problem variant.py.

Outside that harness, helper.py's `VARIANT = resolve_variant(__file__)` (and
path_store.py's own resolve_variant call) have nothing to resolve against.
This supplies the same interface (a matrix_path, a target_final_rank, and a
data_path for saved search paths) from environment variables instead.
"""

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MATRIX = "data/init_npy/gf_mult_Vandaele_wo_ancilla/gf2^16_1612310.npy"
_DEFAULT_TARGET_FINAL_RANK = 380
_DEFAULT_DATA_PATH = "data/path_backups"


@dataclass(frozen=True)
class Variant:
    matrix_path: str
    target_final_rank: int
    data_path: str


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else _REPO_ROOT / path


def resolve_variant(_file: str) -> Variant:
    matrix_path = _resolve_repo_path(os.getenv("VARTODD_MATRIX_PATH", _DEFAULT_MATRIX))
    if not matrix_path.is_file():
        raise FileNotFoundError(f"VARTODD_MATRIX_PATH does not exist: {matrix_path}")
    data_path = _resolve_repo_path(os.getenv("VARTODD_DATA_PATH", _DEFAULT_DATA_PATH))
    return Variant(
        matrix_path=str(matrix_path),
        target_final_rank=int(os.getenv("VARTODD_TARGET_FINAL_RANK", str(_DEFAULT_TARGET_FINAL_RANK))),
        data_path=str(data_path),
    )
