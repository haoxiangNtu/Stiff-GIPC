#!/usr/bin/env python3
"""Demo of the two new engine APIs: engine.set_log_level() and engine.reset().

Run it:
    cd /home/ps/Downloads/Stiff-GIPC-hybrid-mesh
    ./run examples/demo_loglevel_reset.py

What you'll see:
  - WORLD #1 stepped with log_level=1  -> the usual per-frame solver spam
    (solve_subIP >>>, Kappa:, average time cost, TIMING BREAKDOWN) is printed.
  - set_log_level(0) -> the next steps are SILENT (no solver spam).
  - engine.reset()   -> the whole world is torn down (abd_body_count back to 0);
                        a brand-new scene is built + stepped in the SAME process.
"""
import os, sys, time
from pathlib import Path
import numpy as np
from stiff_physics.engine import Engine, Config

ASSETS = str(Path(__file__).resolve().parent.parent / "assets") + "/"


def build_world(eng, fem_y=1e6):
    """1 ABD cube on top of 1 FEM cube — minimal scene."""
    T1 = np.eye(4); T1[:3, :3] *= 0.1; T1[1, 3] = 0.5
    eng.load_mesh("sim_data/tetmesh/cube.msh", dimensions=3, body_type="ABD",
                  transform=T1, young_modulus=1e8)
    T2 = np.eye(4); T2[:3, :3] *= 0.1; T2[1, 3] = 0.0
    eng.load_mesh("sim_data/tetmesh/cube.msh", dimensions=3, body_type="FEM",
                  transform=T2, young_modulus=fem_y)
    eng.finalize()


cfg = Config(dt=0.02, ground_offset=-0.5, assets_dir=ASSETS, preconditioner_type=0)
eng = Engine(cfg)

print("\n========== WORLD #1 : log_level = 1 (verbose, default) ==========")
build_world(eng)
print(f"[demo] abd_body_count = {eng.abd_body_count}")
for i in range(2):
    eng.step()                      # <-- you WILL see solver spam here

print("\n========== set_log_level(0) : next steps are SILENT ==========")
eng.set_log_level(0)
for i in range(2):
    eng.step()                      # <-- NO solver spam
print("[demo] (stepped 2 frames silently — notice no solve_subIP / Kappa above)")

print("\n========== engine.reset() : tear down the whole world ==========")
eng.reset()
print(f"[demo] after reset: abd_body_count = {eng.abd_body_count}  (back to 0)")

print("\n========== WORLD #2 : fresh scene in the SAME process ==========")
eng.set_log_level(0)                # keep it quiet for the rebuild
build_world(eng, fem_y=5e5)         # different params, brand-new world
print(f"[demo] rebuilt: abd_body_count = {eng.abd_body_count}, verts = {len(eng.get_vertices())}")
for i in range(3):
    eng.step()
print("[demo] WORLD #2 stepped fine — reset + rebuild works, no Engine recreation needed.")
print("\nDONE.")
