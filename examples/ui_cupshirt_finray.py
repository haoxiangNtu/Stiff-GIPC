#!/usr/bin/env python3
"""Cup+shirt — UMI finray soft-gripper INTERACTIVE UI (single env).

The arm can replay the recorded trajectory (checkbox); the gripper open/close is a
live slider and the control mode is a live dropdown (pos / stitch / force), so you
can drive the grasp by hand and compare the three gripper modes on this scene.
See examples/umi_finray_lib.py for the shared implementation.

Usage:
    PYTHONPATH=. python examples/ui_cupshirt_finray.py
"""
import umi_finray_lib as L
if __name__ == "__main__":
    L.run_ui("cupshirt")
