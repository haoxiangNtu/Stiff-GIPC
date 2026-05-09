#!/bin/bash
# Method B: hard pin substitution (USE_HARD_PIN=1, M3.5 fixed).
#
# GUI usage: drag joint slider freely.  The engine clamps per-step revolute
# angle change to HARD_PIN_MAX_DEG deg (default 0.5).  Fast drag will spread
# the catch-up over many frames instead of stalling.  Override with
# HARD_PIN_MAX_DEG=N before invoking, e.g.:
#   HARD_PIN_MAX_DEG=0.2 bash run_hardpin.sh
cd /home/ps/Downloads/Stiff-GIPC-dailyv2
export PYTHONPATH=.
exec /home/ps/Downloads/Stiff-GIPC/.venv/bin/python3 examples/case_27_mobile_s1_softgripper_cup.py USE_HARD_PIN=1 FEM_BLOBAL=1 "$@"
