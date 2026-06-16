#!/usr/bin/env python3
"""case_umi_finray_ui_obb.py — OBB-arm variant of case_umi_finray_ui.py.

Identical UI-driven finray demo (drag the joint / gripper sliders to move the
arm — NO replay), but the arm uses OBB-box collision geometry
(ridgeback_dual_panda2_OBB_handKEEP.urdf: far arm links are 8-vertex oriented
boxes; the hand + fingers stay detailed) instead of the fully-detailed arm.

This runs cleanly only because of the engine's collision-buffer overflow fixes
(dynamic reduction scratch + guarded/growable pair-emission buffers). Before
those, OBB + finray would balloon/hang. With the detailed arm use
case_umi_finray_ui.py.

Run (GUI):
    CASE39_PRECOND=0 STIFF_SKIP_CCD_SANITY=1 python examples/case_umi_finray_ui_obb.py

Override the OBB URDF with CASE39_OBB_URDF=... (e.g. ridgeback_dual_panda2_OBB_handOBB.urdf
to reproduce the old hand-OBB blow-up geometry).
"""
import os
import runpy

os.environ.setdefault("CASE39_OBB_URDF", "ridgeback_dual_panda2_OBB_handKEEP.urdf")

_HERE = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_HERE, "case_umi_finray_ui.py"), run_name="__main__")
