#!/usr/bin/env bash
# Run the gate tier requested by the exact-SHA pre-push validator.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
TIER="${1:-full}"
case "$TIER" in
    full|heavy) ;;
    *) echo "[push-gates] unknown tier: $TIER" >&2; exit 2 ;;
esac

# A push must not inherit experimental knobs from the caller's shell.
# Preserve the GATE_* paths supplied by pre-push, then establish exact native
# loading after the reset.
source "$ROOT/scripts/gate_common.sh"
gate_reset_test_environment
gate_prepare_paths "$ROOT"

echo "[push-gates] tier=$TIER"
echo "[push-gates] source=$ROOT"
echo "[push-gates] build=$GATE_BUILD_DIR"
echo "[push-gates] logs=$GATE_LOG_DIR"

bash "$ROOT/scripts/verify_gates.sh" || exit 1

if [ "$TIER" = "heavy" ]; then
    echo "== HEAVY production FEM-model build matrix (+FD per model) =="
    FD_MATRIX=1 bash "$ROOT/scripts/model_build_matrix.sh" || exit 1

    echo "== HEAVY FD matrix (gate build) =="
    python3 "$ROOT/scripts/fd_gate.py" || exit 1

    echo "== HEAVY 19-demo sweep =="
    python3 "$ROOT/scripts/demo_verify.py" || exit 1

    echo "== HEAVY three-mode correctness/performance matrix =="
    bench_rc=0
    BENCH_REQUIRE_BASELINE=1 BENCH_REQUIRE_IDLE_GPU=1 \
        python3 "$ROOT/scripts/mode_bench.py" || bench_rc=$?
    if [ "$bench_rc" -eq 2 ]; then
        echo "[push-gates] INFRA-BLOCK: GPU not idle (external compute process)." \
             "Perf numbers under contention are meaningless — free the GPU and re-push." >&2
        exit 2
    elif [ "$bench_rc" -ne 0 ]; then
        exit 1
    fi

    # Release validation is deliberately not caller-configurable: accepting a
    # shortened SANITIZER_TOOLS list would turn an environment variable into a
    # silent tag-gate bypass.
    for tool in memcheck racecheck initcheck synccheck; do
        echo "== HEAVY sanitizer: $tool =="
        bash "$ROOT/scripts/sanitize_anchor.sh" "$tool"
    done
fi

echo "[push-gates] PASS tier=$TIER"
