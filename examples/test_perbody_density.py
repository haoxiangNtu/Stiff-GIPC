#!/usr/bin/env python3
"""[per-body density] regression: two identical FEM cubes, one 20x denser.

Both rest on the ground after a short drop. The barrier force balancing a
heavier body needs a smaller gap, so the dense cube's equilibrium ground
clearance must be strictly smaller than the light cube's. Also asserts the
per-body setter path runs (load_mesh(density=...)) and the scene stays finite.

Run:  python3 examples/test_perbody_density.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics.engine import Engine, Config

FRAMES = 120
RHO_LIGHT = 1e3       # = global default (loaded WITHOUT override: exercises fallback)
RHO_HEAVY = 2e4       # 20x, via load_mesh(density=...)

cfg = Config(
    dt=0.01,
    density=RHO_LIGHT,
    young_modulus=1e5,
    poisson_rate=0.49,
    friction_rate=0.4,
    relative_dhat=1e-3,
    ground_offset=0.0,
    assets_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Assets") + "/",
)
eng = Engine(cfg)

tf_light = np.eye(4); tf_light[0, 3] = -0.25; tf_light[1, 3] = 0.15
tf_heavy = np.eye(4); tf_heavy[0, 3] = +0.25; tf_heavy[1, 3] = 0.15
eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="FEM", transform=tf_light)
eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="FEM", transform=tf_heavy,
              density=RHO_HEAVY)
eng.finalize()

recs = eng.get_load_records()
n0 = recs[0].vertex_count if hasattr(recs[0], "vertex_count") else recs[0]["vertex_count"]

for fr in range(FRAMES):
    eng.step()

V = np.asarray(eng.get_vertices())
if not np.isfinite(V).all():
    print("FAIL: non-finite vertices")
    sys.exit(1)
clear_light = V[:n0, 1].min()
clear_heavy = V[n0:, 1].min()
print(f"light cube (rho={RHO_LIGHT:g})  ground clearance = {clear_light:.6e} m")
print(f"heavy cube (rho={RHO_HEAVY:g})  ground clearance = {clear_heavy:.6e} m")

if not (clear_heavy < clear_light):
    print("FAIL: heavy cube does not sit deeper than light cube — "
          "per-body density likely not applied")
    sys.exit(1)
print(f"PASS: heavy/light clearance ratio = {clear_heavy/clear_light:.3f} (<1)")
