# VarTodd

Implementation of the FastTodd algorithm for reducing binary matrices (used in tensor decomposition contexts).

## Build

### Requirements

- CMake 3.20+
- A C++23 compiler (GCC 12+, Clang 15+ recommended)
Probably will be installed by Fetch:
- Boost (program_options)
- Zlib
- Network access during configure (FetchContent pulls `cnpypp`, `unordered_dense`, and optionally `pybind11`)

### Configure and Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Python Bindings (optional)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON
cmake --build build -j
```

The module is emitted into `pyvartodd/` (project root). You can import it by
adding the project root to `PYTHONPATH` or by running Python from the repo.

### Useful CMake Options

- `-DBUILD_DEMONSTRATION=ON|OFF` (default: ON) builds `vartodd_demo`.
- `-DBUILD_PYTHON_BINDINGS=ON|OFF` (default: ON) builds the Python module into `pyvartodd/`.
- `-DBUILD_SHARED_LIBS=ON|OFF` (default: ON) controls library type.

The main artifacts are placed in:

- `build/bin/` for executables
- `build/lib/` for libraries
- `pyvartodd/` for Python bindings

## Demo: `vartodd_demo`

The demo reads a binary matrix from a `.npy` file (2D, `uint8` data) and runs the policy‑iteration reduction.

Build it (if not already built):

```bash
cmake --build build --target vartodd_demo
```

Run:

```bash
./build/bin/vartodd_demo --file data/init_npy/other/adder_8.qc.matrix.npy --output_file out.npy
```

### CLI Options

- `--file, -f` (required): input `.npy` matrix.
- `--output_file, -o`: output `.npy` matrix (optional). If omitted, the demo runs without saving.
- `--num-samples, -ns`: number of samples (default: 40).
- `--max-z, -mz`: maximum Z to research (default: 10000).
- `--max-z-fraction`: fraction for Z exploration (default: 1.0).
- `--escore-wred, -er`: exploration score weight for reduction (default: 1).
- `--fscore-wred, -fr`: finalization score weight for reduction (default: -1).
- `--min-reduction`: minimum reduction (default: 1).
- `--threads, -r`: number of threads (default: 4).
- `--seed, -s`: RNG seed (default: 4).
- `--try-only-tohpe, -ot`: skip the full Todd stage when TOHPE already finds enough actions (default: 1).
- `--enable-tohpe`: run the TOHPE candidate stage (default: 1).
- `--enable-todd`: run the full Todd candidate stage (default: 1).

### Input Format

The demo expects a NumPy `.npy` file storing a 2D `uint8` array. Any non‑zero value is treated as 1.

Example datasets live under `data/init_npy/`.

## GF VarTODD vs FastTODD Comparison

The GF benchmark compares `vartodd_fasttodd_mimic.py` against the Rust FastTODD stage runner on the same GF phase-polynomial matrices. The VarTODD mimic follows FastTODD's policy and action-generation structure while using the VarTODD C++ evaluator. The source data is `benchmark_results/gf_comparison.csv`; benchmark details and regeneration commands are in `scripts/benchmarks/README_gf_comparison.md`.

![TODD stage and TOHPE action time](benchmark_results/gf_todd_stage_time.svg)

![Total runtime and final T-count reduction](benchmark_results/gf_total_time_reduction.svg)

## Project Layout

- `src/core/`: main C++ implementation
- `src/python/`: Python bindings(pybind11)
- `include/`: public headers
- `demos/`: demo programs
- `scripts/`: helper python scripts and generators for gf circuits
- `data/`: sample matrices and datasets
