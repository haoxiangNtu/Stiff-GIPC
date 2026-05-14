#!/usr/bin/env python3
"""Test part3.msh as FEM body alone (no arm) — diagnose if mesh is broken."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from stiff_physics.engine import Engine, Config

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"
cfg = Config(dt=0.020, ground_offset=-0.5, assets_dir=ASSETS_DIR,
             preconditioner_type=0,
             skip_all_collision=os.environ.get("SKIP_COL")=="1")
buff = float(os.environ.get("BUFF", "6"))
cfg._cfg.collision_detection_buff_scale = buff
print(f"[test] buff_scale={buff}", flush=True)
eng = Engine(cfg)

# Just 1 ABD ground + 1 FEM softpad (part3.msh)
T1 = np.eye(4); T1[:3,:3] *= 0.5; T1[1,3] = -0.4
eng.load_mesh("sim_data/tetmesh/cube.msh", dimensions=3, body_type="ABD",
              transform=T1, young_modulus=1e8, boundary_type="Fixed")
print(f"[test] ABD count: {eng.abd_body_count}", flush=True)

T2 = np.eye(4); T2[:3,:3] *= 0.3; T2[1,3] = 0.0
YM = float(os.environ.get("YM", "1e6"))
MESH = os.environ.get("MESH", "sim_data/tetmesh/softgriper_part3.msh")
eng.load_mesh(MESH, dimensions=3,
              body_type="FEM", transform=T2, young_modulus=YM)
print(f"[test] FEM mesh={MESH} YM={YM}", flush=True)

# Exclude ABD-vs-FEM (FEM body is the only FEM = global slot 1)
if os.environ.get("EXCLUDE_FEM") == "1":
    eng.add_collision_exclusion(0, 1)  # 0=ABD ground, 1=FEM (last slot)
    print(f"[test] EXCLUDE_FEM=1 added exclusion ABD<->FEM", flush=True)
print(f"[test] After FEM (part3): ABD count: {eng.abd_body_count}", flush=True)

eng.finalize()
print(f"[test] finalized: verts={len(eng.get_vertices())}", flush=True)

for i in range(3):
    t0 = time.perf_counter()
    eng.step()
    print(f"[test] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
