#!/usr/bin/env bash
# Full pipeline: physics run -> video -> USD.  Needs ~3 GB free GPU memory.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SOLVER_PY="${SOLVER_PY:-python}"
PXR_PY="${PXR_PY:-python}"
OUT="${OUT:-$REPO/out/lab_tactile}"
mkdir -p "$OUT"
cd "$REPO"
echo "== physics run -> $OUT/lab_demo.npz"
PYTHONPATH=. "$SOLVER_PY" -u examples/lab_tactile/run_lab_demo.py --out "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
echo "== video"
PYTHONPATH=. "$SOLVER_PY" -u examples/lab_tactile/render_lab_video.py "$OUT/lab_demo.npz" --out "$OUT/lab_demo.mp4"
echo "== usd"
"$PXR_PY" examples/lab_tactile/export_lab_usd.py "$OUT/lab_demo.npz" --out "$OUT/lab_demo.usd"
ls -la "$OUT"
