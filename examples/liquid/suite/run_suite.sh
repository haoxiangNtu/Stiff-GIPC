#!/bin/bash
# Liquid realism regression suite: runs the m-tests for ONE configuration on ONE GPU, one after the other, then prints
# a one-line metric per test (suite/summarize.py). The configuration is whatever LIQ_* environment the caller exports
# (e.g. `source examples/liquid/physical_v2.env`); this script only adds the engine build and the output folder.
#
#   GPU=3 BUILD=build_a800_sti17 OUT=out/suite/v21 FIXTURE=fixtures/flask_fixture.npz \
#       bash examples/liquid/suite/run_suite.sh m6 m14 m15 m16 m9 m12 m13 m4 m10 m5
#
# Test list (default: all). m6 runs three times: its damping ratio has run-to-run noise (sd ~0.008 at dx 5).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
: "${GPU:?GPU}" "${BUILD:?BUILD}" "${OUT:?OUT}"
FIXTURE=${FIXTURE:-}
export STIFFGIPC_NATIVE_DIR=$PWD/$BUILD LD_LIBRARY_PATH=$PWD/$BUILD:${LD_LIBRARY_PATH:-} PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU
PY=${PY:-python}
mkdir -p "$OUT"
env | grep '^LIQ_' | sort > "$OUT/environment.txt"
tests=("$@"); [ ${#tests[@]} -eq 0 ] && tests=(m6 m14 m15 m16 m9 m12 m13 m4 m10 m5)
run() {   # run TAG TIMEOUT script args...
    local tag=$1 to=$2; shift 2
    timeout "$to" $PY -u "$@" > "$OUT/$tag.log" 2>&1
    echo "[suite] $tag exit $? $(date +%H:%M)" | tee -a "$OUT/suite.log"
}
for t in "${tests[@]}"; do
    case $t in
        m6)  for r in 1 2 3; do run m6_r$r 3600 examples/liquid/m6_slosh.py --out "$OUT/m6_r$r"; done ;;
        m14) if [ -n "$FIXTURE" ]; then run m14 7200 examples/liquid/m14_still_water.py "$FIXTURE" --out "$OUT/m14"; else echo "[suite] m14 skipped (no FIXTURE)" | tee -a "$OUT/suite.log"; fi ;;
        m15) run m15y30 7200 examples/liquid/m15_box_rest.py --yaw 30 --shift 0.37 --duration 6 --out "$OUT/m15y30"
             run m15y0 7200 examples/liquid/m15_box_rest.py --duration 6 --out "$OUT/m15y0" ;;
        m16) run m16 7200 examples/liquid/m16_sliding_tank.py --out "$OUT/m16" ;;
        m9)  run m9 7200 examples/liquid/m9_divider.py --out "$OUT/m9" ;;
        m12) run m12 7200 examples/liquid/m12_dambreak_front.py --H 0.114 --ground-mu 0 --energy-rel-tol 1e-10 --out "$OUT/m12" ;;
        m13) run m13 10800 examples/liquid/m13_efflux.py --out "$OUT/m13" ;;
        m4)  run m4 14400 examples/liquid/m4_beaker_tilt.py --out "$OUT/m4" ;;
        m10) run m10 10800 examples/liquid/m10_spill.py --tag suite --duration 6 --out "$OUT/m10" ;;
        m5)  run m5 7200 examples/liquid/m5_dropper.py --out "$OUT/m5" ;;
        *)   echo "[suite] unknown test $t" | tee -a "$OUT/suite.log" ;;
    esac
done
$PY examples/liquid/suite/summarize.py "$OUT" | tee "$OUT/summary.txt"
