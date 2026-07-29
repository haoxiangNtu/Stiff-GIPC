#!/usr/bin/env bash
# Hermetic full verification suite.
#
# Usage: scripts/verify_gates.sh [quick]
# "quick" skips the slow towel recipe and is never sufficient for a push.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
QUICK="${1:-}"
GOLD_ANCHOR="0544461bd82123ae"

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
   cmake --build "$GATE_BUILD_DIR" -j"$GATE_BUILD_JOBS" --target pystiffgipc \
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
if [ "${STIFFGIPC_IN_PREPUSH:-0}" = "1" ]; then
    # Running INSIDE the pre-push hook: the self-test would invoke the hook
    # script again and block on the flock the enclosing hook already holds
    # (re-entrancy deadlock, caught live on the first gated push). The
    # dispatcher being self-tested IS the enclosing run — skipping here loses
    # nothing; local/CI suite runs still exercise the full self-test.
    echo "PRE-PUSH DISPATCH SKIPPED (inside the hook: re-entrant flock)"
elif timeout 120 bash scripts/test_pre_push_hook.sh \
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

echo "== G13 geometry init-feasibility taxonomy =="
timeout 600 python3 scripts/geometry_gate.py \
    > "$GATE_LOG_DIR/geometry.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "GEOMETRY-GATE: PASS" "$GATE_LOG_DIR/geometry.log"; then
    record geometry PASS ""
else
    record geometry FAIL "rc=$rc"
fi

echo "== G15 RL episode-reset (teleport/health/revival) =="
timeout 900 python3 scripts/rl_reset_gate.py \
    > "$GATE_LOG_DIR/rl_reset.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "RL-RESET-GATE: PASS" "$GATE_LOG_DIR/rl_reset.log"; then
    record rl-reset PASS ""
else
    record rl-reset FAIL "rc=$rc"
fi

echo "== G16 Phase-C frame CUDA Graph transaction =="
timeout 300 python3 scripts/frame_graph_gate.py \
    > "$GATE_LOG_DIR/frame-graph.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "FRAME-GRAPH-GATE: PASS" \
       "$GATE_LOG_DIR/frame-graph.log"; then
    record frame-graph PASS ""
else
    record frame-graph FAIL "rc=$rc"
fi

echo "== G17a Phase-D episode graph (FEM/reuse/bitwise) =="
timeout 300 python3 scripts/episode_graph_gate.py \
    > "$GATE_LOG_DIR/episode-graph.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "EPISODE-GRAPH-GATE: PASS" \
       "$GATE_LOG_DIR/episode-graph.log"; then
    record episode-graph PASS ""
else
    record episode-graph FAIL "rc=$rc"
fi

echo "== G17b Phase-D articulated RL episode graph =="
timeout 300 python3 scripts/episode_graph_rl_gate.py \
    > "$GATE_LOG_DIR/episode-rl-graph.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "EPISODE-RL-GRAPH-GATE: PASS" \
       "$GATE_LOG_DIR/episode-rl-graph.log"; then
    record episode-rl-graph PASS ""
else
    record episode-rl-graph FAIL "rc=$rc"
fi

echo "== G18 C4 collision inside the whole-frame graph =="
timeout 600 python3 scripts/collision_graph_gate.py \
    > "$GATE_LOG_DIR/collision-graph.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "COLLISION-GRAPH-GATE: PASS" \
       "$GATE_LOG_DIR/collision-graph.log"; then
    record collision-graph PASS ""
else
    record collision-graph FAIL "rc=$rc"
fi

echo "== G19 C5 isolated mode inside the whole-frame graph =="
timeout 900 python3 scripts/isolated_graph_gate.py \
    > "$GATE_LOG_DIR/isolated-graph.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "ISOLATED-GRAPH-GATE: PASS" \
       "$GATE_LOG_DIR/isolated-graph.log"; then
    record isolated-graph PASS ""
else
    record isolated-graph FAIL "rc=$rc"
fi

echo "== G17c GPU-native RL device ABI (zero-transfer graph) =="
timeout 300 python3 scripts/gpu_rl_gate.py \
    > "$GATE_LOG_DIR/gpu-rl-graph.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "GPU-RL-GATE: PASS" \
       "$GATE_LOG_DIR/gpu-rl-graph.log"; then
    record gpu-rl-graph PASS ""
else
    record gpu-rl-graph FAIL "rc=$rc"
fi

echo "== G17d GPU-native RL D2D action publication =="
timeout 300 python3 scripts/gpu_native_rl_gate.py \
    > "$GATE_LOG_DIR/gpu-native-rl.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "GPU-NATIVE-RL-GATE: PASS" \
       "$GATE_LOG_DIR/gpu-native-rl.log"; then
    record gpu-native-rl PASS ""
else
    record gpu-native-rl FAIL "rc=$rc"
fi

echo "== G14 STIFF_* knob registry consistency (static) =="
timeout 120 python3 scripts/knob_gate.py \
    > "$GATE_LOG_DIR/knob.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] &&
   grep -qF "KNOB-GATE: PASS" "$GATE_LOG_DIR/knob.log"; then
    record knob-registry PASS ""
else
    record knob-registry FAIL "rc=$rc"
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
