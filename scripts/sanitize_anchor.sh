#!/bin/bash
# [A3] compute-sanitizer passes over the strict anchor scene. NOT part of the
# merge suite (10-40x slowdown) — run nightly or on demand:
#   scripts/sanitize_anchor.sh              # memcheck (default), 10 frames
#   scripts/sanitize_anchor.sh racecheck    # slower
#   SCENE_FRAMES=50 scripts/sanitize_anchor.sh   # full anchor length
# Catches the class behavioral gates cannot: out-of-bounds device access and
# races that happen to produce plausible numbers (the 2b latent-OOB class).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
TOOL="${1:-memcheck}"
SAN=$(command -v compute-sanitizer || echo /usr/local/cuda-12.8/bin/compute-sanitizer)
env STIFF_MULTIENV_MODE=isolated STIFF_EE_CANON=1 STIFF_EE_DETGATE=1 \
    STIFF_CCD_CANON=1 STIFF_SPMV_DET=1 STIFF_LOG_LEVEL=0 \
    SCENE_MODE=strict SCENE_N=2 SCENE_FRAMES="${SCENE_FRAMES:-10}" \
    "$SAN" --tool "$TOOL" --error-exitcode 9 python3 scripts/anchor_scene.py
rc=$?
echo "sanitize($TOOL) exit=$rc"
exit $rc
