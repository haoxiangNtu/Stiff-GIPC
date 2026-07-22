#!/usr/bin/env python3
"""[M0 sentinel -> v0.8.4.2 regression] A1 axis-anisotropy sentinel: PASS = Z-up/Y-up iteration ratio < 2x, states match.

M0-4 A1 repro: STRICTLY equivalent Y-up vs Z-up (full rigid rotation of the
entire scene: mesh transforms, gravity, ground normal). Compare per-frame
Newton iters + final state mapped back. Verdict: iteration ratio."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from stiff_physics.engine import Engine, Config

Q = np.array([[1.0,0,0],[0,0,-1.0],[0,1.0,0]])   # rot X +90deg: Y-up -> Z-up (y->z)

def run(zup):
    kw = {}
    g = np.array([0.0,-9.8,0.0]); n = np.array([0.0,1.0,0.0])
    if zup: g = Q @ g; n = Q @ n
    cfg = Config(dt=0.01, density=1e3, young_modulus=1e5, poisson_rate=0.45,
                 friction_rate=0.4, relative_dhat=1e-3, ground_offset=0.0,
                 gravity=g.tolist(), ground_normal=n.tolist(),
                 assets_dir=os.path.join(ROOT, "Assets") + "/")
    eng = Engine(cfg)
    # FEM block dropped from 5cm + ABD cube next to it (ground + inter-body contact)
    t1 = np.eye(4); t1[:3,:3] *= 0.5; t1[1,3] = 0.0
    t2 = np.eye(4); t2[:3,:3] *= 0.4; t2[0,3] = 0.28; t2[1,3] = 0.05
    if zup:
        R4 = np.eye(4); R4[:3,:3] = Q
        t1 = R4 @ t1; t2 = R4 @ t2
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD", transform=t2)
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="FEM", transform=t1)
    eng.finalize()
    nat = eng.native
    gi = (lambda: nat.get_total_newton_iters()) if hasattr(nat, "get_total_newton_iters") else (lambda: 0)
    i0 = gi(); iters = []
    for fr in range(120):
        eng.step(); i1 = gi(); iters.append(i1-i0); i0 = i1
    V = np.asarray(eng.get_vertices())
    if zup: V = V @ Q   # map back to Y-up frame (x' = Q^T applied to rows -> V Q)
    return np.array(iters), V

itY, VY = run(False)
itZ, VZ = run(True)
import statistics as st
print(f"[a1] Y-up: total={itY.sum()} median={st.median(itY):.0f} peak={itY.max()}", flush=True)
print(f"[a1] Z-up: total={itZ.sum()} median={st.median(itZ):.0f} peak={itZ.max()}", flush=True)
ratio = itZ.sum()/max(itY.sum(),1)
err = float(np.abs(VZ-VY).max())
print(f"[a1] VERDICT: iteration ratio Z/Y = {ratio:.2f}x ; final-state diff (mapped back) = {err:.3e} m", flush=True)
print(f"[a1] {'ANISOTROPY REAL (>=2x)' if ratio >= 2.0 else 'NO 10x ANISOTROPY in this minimal scene (historical claim needs the original scene or was measurement artifact)'}", flush=True)
ok = ratio < 2.0 and err < 5e-3
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
