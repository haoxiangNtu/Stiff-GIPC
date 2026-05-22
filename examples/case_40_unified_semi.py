"""case_40_unified_semi.py — semi-implicit variant.

Same physics + scene as case_40_unified.py but with semi_implicit_enabled=True
by default (per case_42 2x2 factorial study: +88% fps in motion+contact pain
case).  Set CASE40_SEMI=0 to override back to the stable reference solver.

All other env vars are forwarded unchanged to case_40_unified.py.
"""
import os, sys, runpy

os.environ.setdefault("CASE40_SEMI", "1")
# Semi runs much faster wall-clock, so the arm moves faster in real time and can
# fling the cloth (+ early-exit slips the grasp).  Halve the per-frame revolute
# step so real-time arm speed matches the stable variant.  Override with
# CASE40_MAX_REV_STEP=... if desired.
os.environ.setdefault("CASE40_MAX_REV_STEP", "0.02")

HERE = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(HERE, "case_40_unified.py"), run_name="__main__")
