#!/usr/bin/env python3
"""Fold-shirt — UMI finray soft-gripper replay (shirt cloth, single env).

Gripper mode via GRIP_MODE = pos (pure position) / stitch (spring-deformation
gauged) / force (real contact-force closed loop). Grip mapping auto-detected.
See examples/umi_finray_lib.py for the shared implementation.

Usage:
    PYTHONPATH=. GRIP_MODE=pos python examples/replay_foldshirt_finray.py
    PYTHONPATH=. CASE39ME_HEADLESS=1 GRIP_MODE=stitch python examples/replay_foldshirt_finray.py
"""
import umi_finray_lib as L
if __name__ == "__main__":
    L.run_replay("foldshirt", default_envs=1)
