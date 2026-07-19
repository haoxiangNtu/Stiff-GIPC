#!/usr/bin/env python3
"""[d-floor fail-fast] regression: a plain FEM bunny drop onto the ground.

The impact frame can legitimately dip a vertex below the 1e-9 m floor for a few
detections (the barrier then pushes it back out). The persistence gate (800
consecutive detections) must therefore NOT throw on this scene. A throw here is
a false positive and fails the test.

Run:  python3 examples/test_dfloor_bunny_drop.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics.engine import Engine, Config

FRAMES = 200

cfg = Config(
    dt=0.01,
    density=1e3,
    young_modulus=1e5,
    poisson_rate=0.49,
    friction_rate=0.4,
    relative_dhat=1e-3,
    ground_offset=0.0,
    assets_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Assets") + "/",
)
eng = Engine(cfg)

tf = np.eye(4)
tf[1, 3] = 0.35   # drop height: free fall then impact
eng.load_mesh("tetMesh/bunny10.msh", dimensions=3, body_type="FEM", transform=tf)
eng.finalize()

min_y_ever = float("inf")
try:
    for fr in range(FRAMES):
        eng.step()
        y = np.asarray(eng.get_vertices())[:, 1].min()
        min_y_ever = min(min_y_ever, y)
        if fr % 25 == 0:
            print(f"  fr={fr:3d} min_vertex_y={y:.6e}")
except RuntimeError as e:
    print(f"FAIL: d-floor threw on a plain bunny drop (false positive):\n{e}")
    sys.exit(1)

print(f"PASS: {FRAMES} frames, no d-floor throw. min vertex y ever = {min_y_ever:.3e} m")
