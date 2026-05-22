"""case_39_full_scale_semi.py — semi-implicit variant.

Same physics + scene as case_39_full_scale.py but with semi_implicit_enabled=True
by default for the +88% fps gain in motion+contact phases (per case_42 2x2
factorial study: 4.0 → 14.2 fps in the closed-gripper + dragging-arm pain case).

Tradeoff: Newton may early-exit before full convergence; cloth may slip
slightly under aggressive joint motion.  Set CASE39_SEMI=0 to override back
to the stable reference solver.

All other env vars are forwarded unchanged to case_39_full_scale.py.
"""
import os, sys, runpy

os.environ.setdefault("CASE39_SEMI", "1")
# Semi runs ~2.4× faster wall-clock (≈14 vs ≈6 fps), so at the same per-frame
# joint step the arm moves ~2.4× faster in real time and flings the cloth
# (worsened by semi's early-exit slipping the grasp).  Halve the per-frame
# revolute step so the arm's real-time speed matches the stable variant and the
# cloth keeps up.  Override with CASE36_MAX_RAD_PER_FRAME=... if you want.
os.environ.setdefault("CASE36_MAX_RAD_PER_FRAME", "0.02")

HERE = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(HERE, "case_39_full_scale.py"), run_name="__main__")
