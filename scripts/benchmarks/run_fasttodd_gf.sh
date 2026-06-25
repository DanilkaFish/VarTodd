#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

CSV_HEADER="algorithm,circuit,initial_t_count,final_t_count,wall_seconds,tohpe_seconds,todd_seconds,stages,tohpe_stages,todd_stages,tohpe_actions,todd_actions,phase_polynomials"
OUT="${OUT:-$ROOT/benchmark_results/fasttodd_gf.csv}"
MATRIX_DIR="${MATRIX_DIR:-$ROOT/data/init_npy/gf_mult_Vandaele_wo_ancilla}"
QCO_DIR="${QCO_DIR:-$ROOT/quantum-circuit-optimization}"
CARGO_BIN="${CARGO_BIN:-cargo}"
FASTTODD_BIN="${FASTTODD_BIN:-$QCO_DIR/target/release/bench_fasttodd_stages}"
TIME_LIMIT="${TIME_LIMIT:-4h}"
HARD_TIME_LIMIT="${HARD_TIME_LIMIT:-}"
DIM_LIST="${DIMS:-4 5 6 7 8 9 10 11 12 13 14 15 16 32 64}"
declare -A GF_MATRIX_FILES=(
    [4]="gf2^4_410.npy"
    [5]="gf2^5_520.npy"
    [6]="gf2^6_610.npy"
    [7]="gf2^7_710.npy"
    [8]="gf2^8_84320.npy"
    [9]="gf2^9_940.npy"
    [10]="gf2^10_1030.npy"
    [11]="gf2^11_1120.npy"
    [12]="gf2^12_126410.npy"
    [13]="gf2^13_134310.npy"
    [14]="gf2^14_148610.npy"
    [15]="gf2^15_1510.npy"
    [16]="gf2^16_1612310.npy"
    [32]="gf2^32_3228320.npy"
    [64]="gf2^64_644310.npy"
)

IFS=' ' read -r -a DIMS_ARR <<< "$DIM_LIST"
mkdir -p -- "$(dirname -- "$OUT")"

FINAL_OUT="$OUT"
if [[ "${APPEND:-0}" == "1" ]]; then
    if [[ ! -s "$OUT" ]]; then
        printf '%s\n' "$CSV_HEADER" > "$OUT"
    fi
else
    OUT="$(mktemp -p "$(dirname -- "$FINAL_OUT")" ".fasttodd_gf.XXXXXX.csv")"
    printf '%s\n' "$CSV_HEADER" > "$OUT"
fi

finish_output() {
    if [[ "$OUT" == "$FINAL_OUT" ]]; then
        return
    fi

    local rows
    rows=$(($(wc -l < "$OUT") - 1))
    if (( rows > 0 )); then
        mv -- "$OUT" "$FINAL_OUT"
        printf 'wrote %s\n' "$FINAL_OUT"
    else
        rm -f -- "$OUT"
        if [[ -e "$FINAL_OUT" ]]; then
            printf 'no completed rows; preserved existing %s\n' "$FINAL_OUT"
        else
            printf '%s\n' "$CSV_HEADER" > "$FINAL_OUT"
            printf 'no completed rows; wrote header to %s\n' "$FINAL_OUT"
        fi
    fi
}

run_with_limit() {
    local label="$1"
    shift

    if [[ -z "$HARD_TIME_LIMIT" || "$HARD_TIME_LIMIT" == "0" ]]; then
        "$@"
        return
    fi

    local status
    if timeout --foreground "$HARD_TIME_LIMIT" "$@"; then
        return
    else
        status=$?
    fi

    if [[ "$status" == "124" || "$status" == "137" ]]; then
        printf '%s hard timed out after %s; preserving completed CSV rows in %s\n' "$label" "$HARD_TIME_LIMIT" "$OUT" >&2
        return
    fi
    return "$status"
}

if [[ "${BUILD_FASTTODD:-1}" == "1" ]]; then
    "$CARGO_BIN" build --release --manifest-path "$QCO_DIR/Cargo.toml" --bin bench_fasttodd_stages
fi

for dim in "${DIMS_ARR[@]}"; do
    if [[ -z "${GF_MATRIX_FILES[$dim]:-}" ]]; then
        printf 'no default GF matrix mapping for dimension: %s\n' "$dim" >&2
        exit 1
    fi
    matrix="$MATRIX_DIR/${GF_MATRIX_FILES[$dim]}"
    if [[ ! -f "$matrix" ]]; then
        printf 'missing FastTODD matrix: %s\n' "$matrix" >&2
        exit 1
    fi
    run_with_limit "FastTODD $dim" "$FASTTODD_BIN" "$matrix" --csv "$OUT" --time-limit "$TIME_LIMIT"
done

finish_output
