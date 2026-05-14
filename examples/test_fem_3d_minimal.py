#!/usr/bin/env python3
"""Minimal: 1 ABD cube + 1 FEM cube (dim=3, .msh tet) to verify FEM API works."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from stiff_physics.engine import Engine, Config

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"

cfg = Config(dt=0.020, ground_offset=-0.5, assets_dir=ASSETS_DIR,
             preconditioner_type=0)
eng = Engine(cfg)

# Load 1 ABD cube
T1 = np.eye(4); T1[:3,:3] *= 0.1; T1[1,3] = 0.5
eng.load_mesh("sim_data/tetmesh/cube.msh", dimensions=3, body_type="ABD",
              transform=T1, young_modulus=1e8)
print(f"[test] ABD count: {eng.abd_body_count}", flush=True)

# Load 1 FEM cube
T2 = np.eye(4); T2[:3,:3] *= 0.1; T2[1,3] = 0.0
eng.load_mesh("sim_data/tetmesh/cube.msh", dimensions=3, body_type="FEM",
              transform=T2, young_modulus=1e6)
print(f"[test] After FEM, ABD count: {eng.abd_body_count}", flush=True)

eng.finalize()
print(f"[test] finalized: verts={len(eng.get_vertices())}", flush=True)

for i in range(3):
    t0 = time.perf_counter()
    eng.step()
    print(f"[test] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
