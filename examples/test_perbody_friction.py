#!/usr/bin/env python3
"""[per-body friction] regression: ground-friction override changes slide distance.

A FEM cube is dropped on the ground with an initial horizontal velocity, twice
in separate engines:
  A) global friction everywhere (mu_gd = 0.4)
  B) same scene, but the cube's ground friction overridden to ~0 via
     set_body_friction(..., ground_mu=0.01)
The slick cube must slide measurably farther. A third run (C) sets the
override EQUAL to the global value and must match A closely (consistency:
override path == scalar path numerically).

Run:  python3 examples/test_perbody_friction.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics.engine import Engine, Config

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Assets") + "/"
FRAMES = 100
V0 = 1.5  # initial horizontal speed, m/s


def run(ground_mu_override):
    cfg = Config(dt=0.01, density=1e3, young_modulus=1e6, poisson_rate=0.49,
                 friction_rate=0.4, relative_dhat=1e-3, ground_offset=0.0,
                 assets_dir=ASSETS)
    eng = Engine(cfg)
    # cube.msh spans y=[0.10, 0.50]; shift so the base sits ~5 mm above ground
    tf = np.eye(4); tf[1, 3] = -0.095
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="FEM", transform=tf)
    if ground_mu_override is not None:
        eng.set_body_friction(0, 0.4, ground_mu=ground_mu_override)
    eng.finalize()
    P = np.asarray(eng.get_vertices())
    v = np.zeros_like(P); v[:, 0] = V0
    # NOTE: set_vertex_velocities_gpu alone does NOT rebuild xTilta, so a bare
    # velocity write never moves the body — teleport (same positions) does.
    eng.native.teleport_fem_vertices(P, v)
    for _ in range(FRAMES):
        eng.step()
    V = np.asarray(eng.get_vertices())
    assert np.isfinite(V).all(), "non-finite vertices"
    return float(V[:, 0].mean())


xA = run(None)     # global mu everywhere (feature off: no per-body tables)
xB = run(0.01)     # slick ground for the cube
xC = run(0.4)      # override == global (consistency check)
print(f"A global mu=0.4      : slide x = {xA:.4f} m")
print(f"B override mu_gd=0.01: slide x = {xB:.4f} m")
print(f"C override mu_gd=0.4 : slide x = {xC:.4f} m")

ok = True
if not (xB > xA + 0.05):
    print("FAIL: slick override did not slide farther — per-body ground mu not applied")
    ok = False
if abs(xC - xA) > 1e-6:
    # not bit-identical by construction (different kernel path) but must be
    # numerically indistinguishable
    print(f"WARN/FAIL: override==global differs from scalar path by {abs(xC-xA):.3e} m")
    ok = abs(xC - xA) < 1e-3 and ok
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
