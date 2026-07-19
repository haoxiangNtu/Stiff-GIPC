#!/usr/bin/env python3
"""[contact-force distribution + FEM stress] regression on a resting cube.

A FEM cube rests on the ground. Checks:
  1) get_vertex_contact_forces: bottom vertices carry ground contact force,
     top vertices carry ~none; net vertical reaction is within 2x of the
     cube's weight (rho * V * g).
  2) get_fem_von_mises_stress: finite, positive somewhere (self-weight
     compression), zero nowhere-touched check skipped (8-vert cube: all
     vertices belong to tets).

Run:  python3 examples/test_contact_force_stress.py
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stiff_physics.engine import Engine, Config

RHO = 1e3
cfg = Config(dt=0.01, density=RHO, young_modulus=1e6, poisson_rate=0.49,
             friction_rate=0.4, relative_dhat=1e-3, ground_offset=0.0,
             assets_dir=os.path.join(ROOT, "Assets") + "/")
eng = Engine(cfg)
tf = np.eye(4); tf[1, 3] = -0.09   # base ~10 mm above ground
eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="FEM", transform=tf)
eng.finalize()

for _ in range(120):
    eng.step()

V = np.asarray(eng.get_vertices())
F = np.asarray(eng.native.get_vertex_contact_forces(include_ground=True))
S = np.asarray(eng.native.get_fem_von_mises_stress())
print(f"verts={V.shape[0]}  |F| per vertex: {np.linalg.norm(F, axis=1)}")
print(f"von Mises (Pa): min={S.min():.4g} max={S.max():.4g}")

ok = True
if not (np.isfinite(F).all() and np.isfinite(S).all()):
    print("FAIL: non-finite outputs"); ok = False

y = V[:, 1]
bottom = y < y.mean(); top = ~bottom
fb = np.linalg.norm(F[bottom], axis=1).sum()
ft = np.linalg.norm(F[top], axis=1).sum()
print(f"bottom |F| sum = {fb:.3f}, top |F| sum = {ft:.3f}")
if not (fb > 10 * max(ft, 1e-12)):
    print("FAIL: contact force not concentrated on the ground-facing side"); ok = False

# net vertical reaction vs weight (side length 0.4 -> V=0.064 m^3)
weight = RHO * 0.064 * 9.8
net_y = abs(F[:, 1].sum())
print(f"net |Fy| = {net_y:.1f} N  vs weight = {weight:.1f} N")
if not (0.5 * weight < net_y < 2.0 * weight):
    print("FAIL: vertical reaction not within 2x of weight"); ok = False

if not (S.max() > 0.0):
    print("FAIL: von Mises stress identically zero under self-weight"); ok = False

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
