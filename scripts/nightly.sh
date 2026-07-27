#!/bin/bash
# ============================================================================
# nightly.sh — optional legacy/manual heavy-pack runner. It is not installed or
# scheduled by this repository; the versioned pre-push tag path is authoritative.
# Runs FD, demos, the three-mode matrix and compute-sanitizer in a CI worktree.
# ============================================================================
set -u
CI_WT=/home/ps/Downloads/Stiff-GIPC-ci
OUT=$HOME/stiffgipc-ci
LOCK=$OUT/.nightly-lock
mkdir -p "$OUT"; exec 9>"$LOCK"; flock -n 9 || exit 0
LOG=$OUT/nightly-$(date +%Y%m%d).log
cd /home/ps/Downloads/Stiff-GIPC && git fetch origin --quiet
[ -d "$CI_WT" ] || git worktree add "$CI_WT" --detach --quiet
cd "$CI_WT" && git checkout -q --detach "$(git rev-parse origin/internal/v0.8.6-rc1)"
[ -d build ] || cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > "$LOG" 2>&1
cmake --build build -j"$(nproc)" >> "$LOG" 2>&1 || { echo "$(date -Is)  FAIL  nightly-build" >> "$OUT/ci-history.log"; exit 2; }
declare -A R
python3 scripts/fd_gate.py           >> "$LOG" 2>&1; R[fd]=$?
python3 scripts/demo_verify.py       >> "$LOG" 2>&1; R[demos]=$?
python3 scripts/mode_bench.py        >> "$LOG" 2>&1; R[matrix]=$?
bash scripts/sanitize_anchor.sh      >> "$LOG" 2>&1; R[sanitize]=$?
line="$(date -Is)  NIGHTLY  fd=${R[fd]} demos=${R[demos]} matrix=${R[matrix]} sanitize=${R[sanitize]}"
echo "$line" >> "$OUT/ci-history.log"
command -v notify-send >/dev/null && notify-send "StiffGIPC nightly" "$line"
