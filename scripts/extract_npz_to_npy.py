#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def extract_npz_to_npy(
    npz_path: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
    allow_pickle: bool = False,
) -> int:
    if not npz_path.exists():
        raise FileNotFoundError(f"input file not found: {npz_path}")
    if npz_path.suffix.lower() != ".npz":
        raise ValueError(f"expected .npz input, got: {npz_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    with np.load(npz_path, allow_pickle=allow_pickle) as data:
        if not data.files:
            raise ValueError(f"no arrays found in {npz_path}")

        for key in data.files:
            out_path = out_dir / f"{key}.npy"
            if out_path.exists() and not overwrite:
                raise FileExistsError(
                    f"refusing to overwrite existing file: {out_path} "
                    f"(pass --overwrite to replace)"
                )
            np.save(out_path, data[key])
            written += 1
            print(f"saved {key} -> {out_path}")

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all arrays from a .npz archive into separate .npy files."
    )
    parser.add_argument(
        "npz_path",
        nargs="?",
        default="hamming_weight_phase_gradient.npz",
        help="Path to input .npz file (default: hamming_weight_phase_gradient.npz)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default=None,
        help="Output directory. Defaults to '<npz-stem>_npy' next to the input.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files in output directory.",
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow loading object arrays from .npz archives that require pickle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz_path).expanduser().resolve()
    if args.out_dir is None:
        out_dir = npz_path.parent / f"{npz_path.stem}_npy"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()

    count = extract_npz_to_npy(
        npz_path,
        out_dir,
        overwrite=args.overwrite,
        allow_pickle=args.allow_pickle,
    )
    print(f"done: extracted {count} arrays to {out_dir}")


if __name__ == "__main__":
    main()
