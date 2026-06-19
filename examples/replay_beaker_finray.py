#!/usr/bin/env python3
"""Beaker-grasp — UMI finray soft-gripper replay (rigid beaker, single env).

Gripper mode via GRIP_MODE = pos (pure position) / stitch (spring-deformation
gauged) / force (real contact-force closed loop). Grip mapping auto-detected.
See examples/umi_finray_lib.py for the shared implementation.

Usage:
    PYTHONPATH=. GRIP_MODE=pos python examples/replay_beaker_finray.py
    PYTHONPATH=. CASE39ME_HEADLESS=1 GRIP_MODE=stitch python examples/replay_beaker_finray.py
"""
import umi_finray_lib as L
if __name__ == "__main__":
    L.run_replay("beaker", default_envs=1)
