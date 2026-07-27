#!/usr/bin/env bash
# Hermetic full verification suite.
#
# Usage: scripts/verify_gates.sh [quick]
# "quick" skips the slow towel recipe and is never sufficient for a push.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
QUICK="${1:-}"
GOLD_ANCHOR="f7fb5a786c2d7935"

source "$ROOT/scripts/gate_common.sh"
gate_reset_test_environment
gate_prepare_paths "$ROOT"
cd "$ROOT" || exit 2

# Armed audits are proven bit-transparent and stay enabled for every gate.
export STIFF_MIRROR_AUDIT=1
export STIFF_SLOT_AUDIT=1

declare -a NAMES RESULTS DETAILS
fail_total=0
record()
{
    NAMES+=("$1")
    RESULTS+=("$2")
    DETAILS+=("$3")
    [ "$2" = "PASS" ] || fail_total=$((fail_total + 1))
}

echo "== G0 configure + build exact tree =="
if cmake -S "$ROOT" -B "$GATE_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="${GATE_CMAKE_BUILD_TYPE:-Release}" \
        -DBUILD_PYTHON_BINDINGS=ON -DBUILD_GL_VIEWER=OFF \
        -DSTIFFGIPC_ENABLE_DIAGNOSTICS=ON \
        -DSTIFFGIPC_FEM_MODEL="${GATE_FEM_MODEL:-SNK1}" \
        > "$GATE_LOG_DIR/configure.log" 2>&1 &&
   cmake --build "$GATE_BUILD_DIR" -j"$(nproc)" --target pystiffgipc \
        > "$GATE_LOG_DIR/build.log" 2>&1; then
    echo "BUILD OK"
else
    echo "BUILD FAIL — configure/build tails:"
    tail -25 "$GATE_LOG_DIR/configure.log" 2>/dev/null || true
    tail -25 "$GATE_LOG_DIR/build.log" 2>/dev/null || true
    exit 2
fi

echo "== G0.2 exact native binding =="
native_path="$(
    python3 -c 'import os; from stiff_physics import engine; print(os.path.realpath(engine._C.__file__))' \
        2> "$GATE_LOG_DIR/native-import.err"
)"
case "$native_path" in
    "$GATE_BUILD_DIR"/*)
        echo "NATIVE OK $native_path"
        ;;
    *)
        echo "NATIVE FAIL got=${native_path:-none} expected-under=$GATE_BUILD_DIR"
        cat "$GATE_LOG_DIR/native-import.err" 2>/dev/null || true
        exit 2
        ;;
esac

echo "== G0.3 lifecycle + process-mode lock =="
if timeout 120 python3 scripts/lifecycle_gate.py \
        > "$GATE_LOG_DIR/lifecycle.log" 2>&1 &&
   grep -qF "LIFECYCLE-GATE: PASS" "$GATE_LOG_DIR/lifecycle.log"; then
    echo "LIFECYCLE OK"
else
    cat "$GATE_LOG_DIR/lifecycle.log" 2>/dev/null || true
    exit 2
fi

echo "== G0.5 frozen-zone tripwire =="
FZ_OK=1
fz()
{
    grep -qF "$2" "$1" ||
        { echo "TRIPWIRE: frozen anchor missing in $1 -> $2"; FZ_OK=0; }
}
fz StiffGIPC/mlbvh_modules/03_pair_emission.inl 'bool   smooth = false;'
fz StiffGIPC/mlbvh_modules/00_gates_globals.inl '__device__ int g_ee_nomollify = 0;'
fz StiffGIPC/GIPC.cuh 'void computeSelfCloseVal();'
fz StiffGIPC/GIPC.cuh 'bool checkSelfCloseVal();'
[ "$FZ_OK" = 1 ] && echo "FROZEN OK" || exit 2

echo "== G0.7 exact-SHA pre-push dispatcher =="
if timeout 120 bash scripts/test_pre_push_hook.sh \
        > "$GATE_LOG_DIR/pre-push-hook.log" 2>&1 &&
   grep -qF "PRE-PUSH-HOOK-TEST: PASS" "$GATE_LOG_DIR/pre-push-hook.log"; then
    echo "PRE-PUSH DISPATCH OK"
else
    cat "$GATE_LOG_DIR/pre-push-hook.log" 2>/dev/null || true
    exit 2
fi

STRICT_ENV=(
    STIFF_MULTIENV_MODE=strict
)

echo "== G1 strict bitwise anchor =="
HA="$(
    env "${STRICT_ENV[@]}" SCENE_MODE=strict SCENE_N=2 timeout 900 \
        python3 scripts/anchor_scene.py 2>&1 |
        awk '/VHASH/{print $2}'
)"
if [ "$HA" = "$GOLD_ANCHOR" ]; then
    record anchor PASS "$HA"
else
    record anchor FAIL "got=${HA:-none} want=$GOLD_ANCHOR"
fi

if [ "$QUICK" != "quick" ]; then
    echo "== G2 towel-strict full recipe =="
    env "${STRICT_ENV[@]}" CASE39ME_HEADLESS=1 STIFF_LOG_LEVEL=0 timeout 1200 \
        python3 examples/recipe_towel_scramble.py \
        > "$GATE_LOG_DIR/towel.log" 2>&1
    rc=$?
    marker="$(grep -E '^PASS|^FAIL' "$GATE_LOG_DIR/towel.log" | tail -1)"
    if [ "$rc" -eq 0 ] && [ "$marker" = "PASS" ]; then
        record towel-strict PASS ""
    else
        record towel-strict FAIL "rc=$rc marker=${marker:-none}"
    fi
fi

echo "== G3 midrun quarantine =="
timeout 900 python3 examples/test_env_midrun_quarantine.py \
    > "$GATE_LOG_DIR/midrun-quarantine.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "MIDRUN-QUARANTINE: PASS" "$GATE_LOG_DIR/midrun-quarantine.log"; then
    record midrun-quarantine PASS ""
else
    record midrun-quarantine FAIL "rc=$rc"
fi

echo "== G4 startup quarantine =="
timeout 900 python3 examples/test_env_quarantine.py \
    > "$GATE_LOG_DIR/env-quarantine.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "ENV-QUARANTINE: PASS" "$GATE_LOG_DIR/env-quarantine.log"; then
    record env-quarantine PASS ""
else
    record env-quarantine FAIL "rc=$rc"
fi

echo "== G5 MAS oracle =="
timeout 900 python3 examples/test_mas_oracle.py \
    > "$GATE_LOG_DIR/mas-oracle.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "MAS ORACLE: PASS" "$GATE_LOG_DIR/mas-oracle.log"; then
    record mas-oracle PASS ""
else
    record mas-oracle FAIL "rc=$rc"
fi

echo "== G6 ABD kick precondition =="
timeout 600 python3 examples/test_kick_abd_precond.py \
    > "$GATE_LOG_DIR/kick.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -qxF "PASS" "$GATE_LOG_DIR/kick.log"; then
    record kick PASS ""
else
    record kick FAIL "rc=$rc"
fi

echo "== G7 passive revolute =="
timeout 600 python3 examples/test_passive_revolute.py \
    > "$GATE_LOG_DIR/passive-revolute.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qxF "PASS" "$GATE_LOG_DIR/passive-revolute.log"; then
    record passive-revolute PASS ""
else
    record passive-revolute FAIL "rc=$rc"
fi

echo "== G9 mode envelope + equivalence =="
timeout 1800 python3 scripts/mode_gates.py \
    > "$GATE_LOG_DIR/mode-gates.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "MODE-GATES: PASS" "$GATE_LOG_DIR/mode-gates.log"; then
    record mode-gates PASS ""
else
    record mode-gates FAIL "rc=$rc"
fi

echo "== G10 bad-mesh ABD kinetic =="
timeout 600 python3 examples/test_abd_badmesh_kinetic.py \
    > "$GATE_LOG_DIR/abd-badmesh.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "ABD-BADMESH-KINETIC: PASS" "$GATE_LOG_DIR/abd-badmesh.log" &&
   ! grep -qiE "budget exhausted.*nan|abd-kinetic-nan" "$GATE_LOG_DIR/abd-badmesh.log"; then
    record abd-badmesh PASS ""
else
    record abd-badmesh FAIL "rc=$rc"
fi

echo "== G11 checkpoint format + restart =="
timeout 900 python3 scripts/checkpoint_gate.py \
    > "$GATE_LOG_DIR/checkpoint.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "CHECKPOINT-GATE: PASS" "$GATE_LOG_DIR/checkpoint.log"; then
    record checkpoint PASS ""
else
    record checkpoint FAIL "rc=$rc"
fi

echo "== G12 constitutive + physics invariants =="
MODEL_EXPECT="${GATE_FEM_MODEL:-SNK1}" timeout 900 \
    python3 scripts/model_validation.py \
    > "$GATE_LOG_DIR/model-validation.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "MODEL-VALIDATION: PASS" "$GATE_LOG_DIR/model-validation.log"; then
    record model-validation PASS ""
else
    record model-validation FAIL "rc=$rc"
fi

echo "== G8 foldshirt 30f smoke =="
env CASE39ME_HEADLESS=1 CASE39ME_NUM_ENVS=4 CASE39_FRICTION=0.8 \
    CASE39_FRAME_END=30 STIFF_MULTIENV_MODE=merged STIFF_LOG_LEVEL=0 \
    timeout 600 python3 examples/replay_foldshirt_multienv.py \
    > "$GATE_LOG_DIR/foldshirt-smoke.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   ! grep -qiE "budget exhausted.*nan|abd-kinetic-nan|Traceback|CUDA error" \
       "$GATE_LOG_DIR/foldshirt-smoke.log"; then
    record foldshirt-smoke PASS ""
else
    record foldshirt-smoke FAIL "rc=$rc"
fi

echo
echo "================ GATE SUITE RESULT ================"
for i in "${!NAMES[@]}"; do
    printf "  %-20s %-5s %s\n" "${NAMES[$i]}" "${RESULTS[$i]}" "${DETAILS[$i]}"
done
echo "==================================================="
echo "logs: $GATE_LOG_DIR"
if [ "$fail_total" -eq 0 ]; then
    echo "ALL GATES GREEN"
    exit 0
fi
echo "$fail_total GATE(S) FAILED"
exit 1
