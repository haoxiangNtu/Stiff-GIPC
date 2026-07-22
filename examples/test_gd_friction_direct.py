#!/usr/bin/env python3
"""[P3/A2] direct assertion: Config.gd_friction_rate reaches the ground solver.

The A2 audit fix made gd_friction_rate a first-class Config arg (previously
only reachable by poking cfg._cfg after construction). This test asserts the
parameter actually changes ground friction physics, with a tangential
force-balance control:

  A) gd_friction_rate=None  -> follows friction_rate (historic behavior)
  B) gd_friction_rate=0.01  -> near-zero ground friction, slides much farther
  C) gd_friction_rate=0.4   -> explicitly equal to global, must match A
  D) gd_friction_rate=1.2   -> high friction: stops early AND stays stopped
                               (tangential equilibrium: last-20-frame drift ~0)

Run:  python3 examples/test_gd_friction_direct.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics.engine import Engine, Config

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Assets") + "/"
FRAMES = 100
V0 = 1.5  # initial horizontal speed, m/s


def run(gd):
    cfg = Config(dt=0.01, density=1e3, young_modulus=1e6, poisson_rate=0.49,
                 friction_rate=0.4, gd_friction_rate=gd, relative_dhat=1e-3,
                 ground_offset=0.0, assets_dir=ASSETS)
    eng = Engine(cfg)
    tf = np.eye(4); tf[1, 3] = -0.095  # cube base ~5 mm above ground
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="FEM", transform=tf)
    eng.finalize()
    P = np.asarray(eng.get_vertices())
    v = np.zeros_like(P); v[:, 0] = V0
    eng.native.teleport_fem_vertices(P, v)  # rebuilds xTilta (bare velocity write would not)
    xs = []
    for _ in range(FRAMES):
        eng.step()
        xs.append(float(np.asarray(eng.get_vertices())[:, 0].mean()))
    V = np.asarray(eng.get_vertices())
    assert np.isfinite(V).all(), "non-finite vertices"
    return np.array(xs)


xa = run(None)   # default: ground mu follows friction_rate
xb = run(0.01)   # slick ground via the Config arg
xc = run(0.4)    # explicit == global
xd = run(1.2)    # high-grip ground

dA, dB, dC, dD = xa[-1], xb[-1], xc[-1], xd[-1]
tail_drift_D = abs(xd[-1] - xd[-20])
print(f"A gd=None : slide x = {dA:.4f} m")
print(f"B gd=0.01 : slide x = {dB:.4f} m")
print(f"C gd=0.4  : slide x = {dC:.4f} m")
print(f"D gd=1.2  : slide x = {dD:.4f} m, last-20-frame drift = {tail_drift_D:.2e} m")

ok = True
if not (dB > dA + 0.05):
    print("FAIL: gd_friction_rate=0.01 did not slide farther — Config arg not reaching solver")
    ok = False
if abs(dC - dA) > 1e-6:
    print(f"FAIL: gd=0.4 differs from default-follow path by {abs(dC-dA):.3e} m "
          "(explicit value must take the identical code path)")
    ok = False
if not (dD <= dA + 1e-9):
    print("FAIL: high-grip ground slid farther than default")
    ok = False
if not (tail_drift_D < 1e-3):
    print(f"FAIL: tangential equilibrium violated on gd=1.2 (tail drift {tail_drift_D:.2e} m)")
    ok = False

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
