# GF(2^n) Parity-Matrix Generation (Vandaele, no ancilla)

This folder contains small scripts for generating GF(2^n) multiplication artifacts used in the Vandaele-style construction without ancillae. The main deliverable is a parity matrix (a list of parity vectors over GF(2)) derived from a Toffoli/CCZ decomposition of multiplication.

## Polynomial Convention

All scripts take a list of **exponents** for the irreducible polynomial

```
p(x) = x^m + \sum_i x^{e_i} + 1
```

You provide the exponents as command‑line integers. The leading term `x^m` is handled slightly differently depending on the script:

- `gf2_decomposition.py` expects **all** exponents, including `m` and `0`.
- `gf2_mult.py` and `gf2_mult_compact.py` treat `m = max(indices)` as **implicit** and only use exponents `< m`.

Example for `p(x) = x^4 + x + 1`:

- For `gf2_decomposition.py`: `0 1 4`
- For `gf2_mult.py` / `gf2_mult_compact.py`: `0 1 4` (the `4` is only used to set `m`)

## Scripts

### `gf2_decomposition.py`
Generates a parity matrix for GF(2^n) multiplication by:

1. Building the Q matrix from the polynomial.
2. Synthesizing a CCZ/Toffoli list using the recursive Vandaele construction.
3. Converting CCZs to parity vectors and canceling duplicates mod 2.

**Usage**

```bash
python scripts/gf_vandaele_wo_anc_generation/gf2_decomposition.py <exp_1> <exp_2> ...
```

**Example**

```bash
python scripts/gf_vandaele_wo_anc_generation/gf2_decomposition.py 0 1 4
```

**Output**

- Saves a NumPy array of parity vectors to a file named like:
  - `gf2^<n>_<indices>.npy`
- The array is boolean (`dtype=bool`).

### `gf2_mult.py`
Builds a full GF(2^n) multiplication circuit using Qiskit and writes a `.qc` file (custom format). This is a more verbose version of the circuit generator.

**Usage**

```bash
python scripts/gf_vandaele_wo_anc_generation/gf2_mult.py <exp_1> <exp_2> ...
```

**Output**

- Writes `circuit.qc` in the current working directory.
- Prints circuit size/depth info.

Note: this script uses `QuantumCircuit` but does not import Qiskit explicitly. If you run it, ensure Qiskit is installed and import lines are added (or prefer `gf2_mult_compact.py`).

### `test_gf2.py`
Verifies that the CCZ-based representation (from `gf2_decomposition.py`) matches the algebraic GF(2^n) multiplication polynomial.

**Usage**

```bash
python scripts/gf_vandaele_wo_anc_generation/test_gf2.py
```

It runs a small built‑in example and performs exhaustive testing for small `n` (randomized for larger `n`).

## Requirements

- Python 3
- `numpy` (for `gf2_decomposition.py` and `test_gf2.py`)

## Reference
The decomposition follow recursive algorithms: 

- Vandaele et al., Quantum 9, 1860 (2025). Link: https://quantum-journal.org/papers/q-2025-09-16-1860/
