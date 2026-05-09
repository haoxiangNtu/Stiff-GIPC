#!/bin/bash
# Method B: hard pin substitution (USE_HARD_PIN=1, M3.5 fixed).
#
# GUI usage: drag joint slider freely. Stable ~250-500ms/step.
#
# Tight-pinning defaults applied:
#   STITCH_THRESH=0.001         only pin FEM verts within 1mm of ABD
#                               (prunes the "loose" tip pins that fight
#                               the root pins under rotation)
#   STITCH_TOP_FRACTION=0.5     keep only the top half — concentrate pins
#                               near the rigid backbone, tip elastically free
#   MAX_REVOLUTE_DEG=0.5        per-step revolute angle clamp
# Override any of them on the command line, e.g.:
#   STITCH_THRESH=0.003 STITCH_TOP_FRACTION=1.0 bash run_hardpin.sh
cd /home/ps/Downloads/Stiff-GIPC-dailyv2
export PYTHONPATH=.
export STITCH_THRESH="${STITCH_THRESH:-0.001}"
export STITCH_TOP_FRACTION="${STITCH_TOP_FRACTION:-0.5}"
export MAX_REVOLUTE_DEG="${MAX_REVOLUTE_DEG:-0.5}"
exec /home/ps/Downloads/Stiff-GIPC/.venv/bin/python3 examples/case_27_mobile_s1_softgripper_cup.py USE_HARD_PIN=1 FEM_BLOBAL=1 "$@"
