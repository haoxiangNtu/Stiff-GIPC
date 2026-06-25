#!/usr/bin/env python3
"""replay_case39_UMI_beaker.py — beaker-grasp variant of replay_case39_UMI_sf.py.

Same STRATEGY_F hybrid finray gripper, but replays the recorded BEAKER-grasp
trajectory (the gripper picks up a rigid 100 ml beaker) instead of the
fold-shirt trajectory.  This is a thin wrapper that points replay_case39_UMI_sf.py
at the bundled beaker trajectory + rigid-beaker collision mesh.

The beaker scene is contact-light (a rigid ABD object, no cloth), so it defaults
to preconditioner_type=0 (no MAS) — the configuration it was recorded with and
the optimal one for arm/contact-light scenes.  It also runs correctly under MAS
(preconditioner_type=1) now that the metis sort cache is validated by vertex
count (see the fix(metis) commit); pass CASE39_PRECOND=1 to use MAS.

Run (GUI):
    STIFF_SKIP_CCD_SANITY=1 python examples/replay_case39_UMI_beaker.py

    # headless timing:
    CASE39_HEADLESS=1 STIFF_SKIP_CCD_SANITY=1 \
        python examples/replay_case39_UMI_beaker.py --quiet
"""
import os
import sys
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
# Bundled assets live under the repo-root "Assets/" dir (capital A); match the
# pattern used by replay_case39_UMI_sf.py and the other examples.
_ASSETS_DIR = os.path.join(os.path.dirname(_HERE), "Assets")

# Contact-light rigid-object scene -> no MAS by default (also how it was recorded).
os.environ.setdefault("CASE39_PRECOND", "0")

# Default to the bundled beaker-grasp trajectory unless the caller passed --replay.
if "--replay" not in sys.argv:
    sys.argv += ["--replay",
                 os.path.join(_ASSETS_DIR, "trajectories",
                              "episode_grasp_beaker_umi.hdf5")]

runpy.run_path(os.path.join(_HERE, "replay_case39_UMI_sf.py"), run_name="__main__")
