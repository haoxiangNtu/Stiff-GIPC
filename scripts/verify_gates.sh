#!/bin/bash
# ============================================================================
# verify_gates.sh — the push-button verification suite (v0.8.6 refactor net).
#
# EVERY refactor phase must pass this suite before its commit. The suite is
# the codified form of the release-validation protocol this project already
# lives by:
#   G1  strict bitwise anchor        — the determinism contract, byte-exact
#   G2  towel-strict full recipe     — the folded-cloth strict stress case
#   G3  midrun env quarantine        — iron law: mid-run pathology isolates
#   G4  startup env quarantine       — iron law: timeout/NaN freeze
#   G5  MAS oracle                   — preconditioner bit-level CPU oracle
#   G6  ABD kick precondition        — the historical kick root-cause guard
#   G7  passive revolute             — joint limits / passive dynamics
#   G8  foldshirt 30f smoke (merged) — scene bring-up + itermon sanity
#
# Usage:  scripts/verify_gates.sh [quick]
#   quick = skip G2 (the slowest gate) for inner-loop iteration; a phase
#   commit ALWAYS requires the full suite.
# Exit code 0 = all green.
# ============================================================================
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
GOLD_ANCHOR="f7fb5a786c2d7935"
QUICK="${1:-}"

# G0: the suite MUST test the tree it is invoked on. Without this step a
# stale build/ silently validates old binaries (exactly what happened for
# the 2026-07-25 P5..phase4 runs — the armed-audit "validations" tested a
# lib that contained none of the audited code). Build failure = hard stop.
echo "== G0 build =="
if cmake --build build -j"$(nproc)" > /tmp/vg_build.log 2>&1; then
  echo "BUILD OK"
else
  echo "BUILD FAIL — tail of /tmp/vg_build.log:"
  tail -25 /tmp/vg_build.log
  exit 2
fi
declare -a NAMES RESULTS DETAILS
fail_total=0

record() { # name ok detail
  NAMES+=("$1"); RESULTS+=("$2"); DETAILS+=("$3")
  [ "$2" = "PASS" ] || fail_total=$((fail_total + 1))
}

STRICT_ENV=(STIFF_MULTIENV_MODE=isolated STIFF_EE_CANON=1 STIFF_EE_DETGATE=1
            STIFF_CCD_CANON=1 STIFF_SPMV_DET=1)

echo "== G1 strict bitwise anchor =="
HA=$(env "${STRICT_ENV[@]}" SCENE_MODE=strict SCENE_N=2 timeout 900 \
     python3 scripts/anchor_scene.py 2>&1 | awk '/VHASH/{print $2}')
if [ "$HA" = "$GOLD_ANCHOR" ]; then record anchor PASS "$HA"; else record anchor FAIL "got=${HA:-none} want=$GOLD_ANCHOR"; fi

if [ "$QUICK" != "quick" ]; then
  echo "== G2 towel-strict full recipe =="
  env "${STRICT_ENV[@]}" CASE39ME_HEADLESS=1 STIFF_LOG_LEVEL=0 timeout 1200 \
      python3 examples/recipe_towel_scramble.py > /tmp/vg_towel.log 2>&1
  rc=$?; v=$(grep -E '^PASS|^FAIL' /tmp/vg_towel.log | tail -1)
  if [ $rc -eq 0 ] && [ "$v" = "PASS" ]; then record towel-strict PASS ""; else record towel-strict FAIL "rc=$rc $v"; fi
fi

echo "== G3 midrun quarantine =="
timeout 900 python3 examples/test_env_midrun_quarantine.py > /tmp/vg_mq.log 2>&1
rc=$?; v=$(grep -oE 'MIDRUN-QUARANTINE: (PASS|FAIL)' /tmp/vg_mq.log | tail -1)
if [ $rc -eq 0 ]; then record midrun-quarantine PASS ""; else record midrun-quarantine FAIL "rc=$rc $v"; fi

echo "== G4 startup quarantine =="
timeout 900 python3 examples/test_env_quarantine.py > /tmp/vg_q.log 2>&1
rc=$?
if [ $rc -eq 0 ]; then record env-quarantine PASS ""; else record env-quarantine FAIL "rc=$rc"; fi

echo "== G5 MAS oracle =="
timeout 900 python3 examples/test_mas_oracle.py > /tmp/vg_maso.log 2>&1
rc=$?
if [ $rc -eq 0 ]; then record mas-oracle PASS ""; else record mas-oracle FAIL "rc=$rc"; fi

echo "== G6 ABD kick precond =="
timeout 600 python3 examples/test_kick_abd_precond.py > /tmp/vg_kick.log 2>&1
rc=$?
if [ $rc -eq 0 ]; then record kick PASS ""; else record kick FAIL "rc=$rc"; fi

echo "== G7 passive revolute =="
timeout 600 python3 examples/test_passive_revolute.py > /tmp/vg_rev.log 2>&1
rc=$?
if [ $rc -eq 0 ]; then record passive-revolute PASS ""; else record passive-revolute FAIL "rc=$rc"; fi

echo "== G8 foldshirt 30f smoke =="
env CASE39ME_HEADLESS=1 CASE39ME_NUM_ENVS=4 CASE39_FRICTION=0.8 CASE39_FRAME_END=30 \
    STIFF_MULTIENV_MODE=merged STIFF_LOG_LEVEL=0 timeout 600 \
    python3 examples/replay_foldshirt_multienv.py > /tmp/vg_fs.log 2>&1
rc=$?
if [ $rc -eq 0 ]; then record foldshirt-smoke PASS ""; else record foldshirt-smoke FAIL "rc=$rc"; fi

echo
echo "================ GATE SUITE RESULT ================"
for i in "${!NAMES[@]}"; do
  printf "  %-20s %-5s %s\n" "${NAMES[$i]}" "${RESULTS[$i]}" "${DETAILS[$i]}"
done
echo "==================================================="
if [ $fail_total -eq 0 ]; then echo "ALL GATES GREEN"; exit 0; else echo "$fail_total GATE(S) FAILED"; exit 1; fi
