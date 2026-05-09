#!/bin/bash
# Method A: stitch spring (USE_HARD_PIN=0). Most stable, recommended.
# GUI: drag any joint slider freely. Stable ~250-500ms/step.
#
# Tight-pinning defaults applied:
#   STITCH_THRESH=0.001         only pin FEM verts within 1mm of ABD
#   STITCH_TOP_FRACTION=0.5     keep only top half (pin near rigid backbone,
#                               tip free to deform elastically)
#   MAX_REVOLUTE_DEG=0.5        per-step revolute angle clamp
# Override any of them on the command line, e.g.:
#   STITCH_THRESH=0.003 bash run_stitch.sh
cd /home/ps/Downloads/Stiff-GIPC-dailyv2
export PYTHONPATH=.
export STITCH_THRESH="${STITCH_THRESH:-0.001}"
export STITCH_TOP_FRACTION="${STITCH_TOP_FRACTION:-0.5}"
export MAX_REVOLUTE_DEG="${MAX_REVOLUTE_DEG:-0.5}"
exec /home/ps/Downloads/Stiff-GIPC/.venv/bin/python3 examples/case_27_mobile_s1_softgripper_cup.py USE_HARD_PIN=0 FEM_BLOBAL=1 "$@"
