#!/bin/bash
# ============================================================================
# ci_watch.sh — local CI poller (owner-approved CI item, 2026-07-27).
# Cron-invoked every 10 min (lock-guarded). For each watched branch: if origin
# moved, check out the new sha in the DEDICATED CI worktree (never the dev
# tree), build + run the full armed gate suite, log verdict + notify.
# Usage: ci_watch.sh [--once branch]      (smoke/manual mode)
# ============================================================================
set -u
REPO=/home/ps/Downloads/Stiff-GIPC
CI_WT=/home/ps/Downloads/Stiff-GIPC-ci
OUT=$HOME/stiffgipc-ci
BRANCHES=${CI_BRANCHES:-"internal/v0.8.6-rc1 refactor/v086-energy-modular"}
LOCK=$OUT/.lock
mkdir -p "$OUT"
exec 9>"$LOCK"; flock -n 9 || exit 0        # previous run still going

cd "$REPO" || exit 2
git fetch origin --quiet 2>/dev/null

[ -d "$CI_WT" ] || git worktree add "$CI_WT" --detach --quiet

run_gates() { # sha branch
  local sha=$1 br=$2 tag; tag=$(echo "$br" | tr '/' '_')_${sha:0:9}
  local log=$OUT/${tag}.log
  (
    cd "$CI_WT" || exit 2
    git checkout -q --detach "$sha"
    [ -d build ] || cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > "$log" 2>&1
    bash scripts/verify_gates.sh >> "$log" 2>&1
  )
  local rc=$?
  local verdict=$([ $rc -eq 0 ] && echo PASS || echo FAIL)
  printf '%s  %s  %s  %s\n' "$(date -Is)" "$verdict" "$br" "$sha" >> "$OUT/ci-history.log"
  { echo "# CI status"; echo; tail -20 "$OUT/ci-history.log" | tac; } > "$OUT/ci-status.md"
  command -v notify-send >/dev/null && notify-send "StiffGIPC CI: $verdict" "$br @ ${sha:0:9}"
  return $rc
}

if [ "${1:-}" = "--once" ]; then
  sha=$(git rev-parse "origin/${2:?branch}")
  run_gates "$sha" "$2"; exit $?
fi

for br in $BRANCHES; do
  sha=$(git rev-parse "origin/$br" 2>/dev/null) || continue
  state=$OUT/.last_${br//\//_}
  [ -f "$state" ] && [ "$(cat "$state")" = "$sha" ] && continue
  run_gates "$sha" "$br"
  echo "$sha" > "$state"
done
