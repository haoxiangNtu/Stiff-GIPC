#!/usr/bin/env bash
# [A3] compute-sanitizer passes over the strict anchor scene. These run for
# release tags (not normal branch pushes) or on demand:
#   scripts/sanitize_anchor.sh              # memcheck (default), 10 frames
#   scripts/sanitize_anchor.sh racecheck    # slower
#   SCENE_FRAMES=50 scripts/sanitize_anchor.sh   # full anchor length
# Catches the class behavioral gates cannot: out-of-bounds device access and
# races that happen to produce plausible numbers (the 2b latent-OOB class).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
TOOL="${1:-memcheck}"
case "$TOOL" in
    memcheck|racecheck|initcheck|synccheck) ;;
    *) echo "unsupported sanitizer tool: $TOOL" >&2; exit 2 ;;
esac

cd "$ROOT" || exit 2
SAN="$(command -v compute-sanitizer || true)"
if [ -z "$SAN" ]; then
    SAN=/usr/local/cuda-12.8/bin/compute-sanitizer
fi
if [ ! -x "$SAN" ]; then
    echo "compute-sanitizer not found: $SAN" >&2
    exit 2
fi

SAN_ENV=(
    STIFF_MULTIENV_MODE=strict
    STIFF_LOG_LEVEL=0
    SCENE_MODE=strict
    SCENE_N=2
    SCENE_FRAMES="${SCENE_FRAMES:-10}"
)

# NVIDIA documents racecheck/synccheck/initcheck as unsupported for dynamic
# device launches.  StiffGIPC's default PCG path uses a device-side self-tail
# CUDA Graph launch; on CUDA 12.8 synccheck consequently reports every thread
# at the first otherwise-unconditional block barrier as divergent.  Preserve
# the same K=8 captured kernel sequence, but relaunch it from the host for
# tools that cannot instrument the device launcher.  Memcheck stays on the
# default self-tail path, so that path is still exercised by a sanitizer.
if [ "$TOOL" != "memcheck" ]; then
    SAN_ENV+=(STIFF_PCG_DEVICE_LOOP=0)
    echo "sanitize($TOOL): device self-tail disabled; checking equivalent host graph"
fi

env "${SAN_ENV[@]}" \
    "$SAN" --tool "$TOOL" --error-exitcode 9 \
    python3 scripts/anchor_scene.py
rc=$?
echo "sanitize($TOOL) exit=$rc"
exit $rc
