#!/usr/bin/env bash
# Bench harness for the CKKS sparse-packing bootstrap example.
#
# Runs example 6 across the supported log_slots range, with one warm-up
# pass per log_slots and multiple measured iterations. Emits CSV on
# stdout: log_slots,iter,bootstrap_ms,min_prec,avg_prec.
#
# Usage:
#   scripts/bench_sparse_bootstrap.sh [iters]
#   iters defaults to 4. log_slots range is [2, 14] inclusive.
#
# The lower bound is 2 because the V-matrix BSGS path degenerates at
# log_slots <= 1 (matrix_count == 1 → empty index vector → null kernel
# pointer). log_slots == 15 is fully packed (use the regular bootstrap
# chain, not this sparse example).

set -euo pipefail

BIN=${BIN:-./build/bin/examples/bootstrapping/6_ckks_sparse_bootstrapping_v2}
ITERS=${1:-4}

if [ ! -x "$BIN" ]; then
    echo "binary not found or not executable: $BIN" >&2
    exit 1
fi

echo "log_slots,iter,bootstrap_ms,min_prec,avg_prec"
for ls in $(seq 2 14); do
    "$BIN" "$ls" >/dev/null 2>&1   # warm-up
    for i in $(seq 1 "$ITERS"); do
        out=$("$BIN" "$ls" 2>&1)
        ms=$(echo "$out"   | awk '/Sparse bootstrapping time:/{print $4}')
        minp=$(echo "$out" | awk '/MIN Prec/{gsub(/│/, ""); print $3}')
        avgp=$(echo "$out" | awk '/AVG Prec/{gsub(/│/, ""); print $3}')
        echo "$ls,$i,$ms,$minp,$avgp"
    done
done
