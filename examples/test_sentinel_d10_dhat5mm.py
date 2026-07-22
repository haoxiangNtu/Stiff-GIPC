#!/usr/bin/env python3
"""[M0 sentinel -> v0.8.4.2 regression] D10 large-dhat sentinel: PASS = 250 frames complete, no CUDA crash.

M0-3 D10 repro: angular boxes with absolute_dhat=5mm, 250 frames.
Historical: cudaErrorIllegalAddress. Verdict: completes vs crashes."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from stiff_physics.engine import Engine, Config
cfg = Config(dt=0.01, density=1e3, young_modulus=1e7, poisson_rate=0.49,
             friction_rate=0.4, relative_dhat=1e-3, absolute_dhat=0.005,
             ground_offset=0.0, assets_dir=os.path.join(ROOT, "Assets") + "/")
eng = Engine(cfg)
rng = np.random.default_rng(7)
for i in range(4):           # 4 stacked/offset angular boxes
    t = np.eye(4); t[:3,:3] *= 0.4
    t[0,3] = 0.03*i*((-1)**i); t[1,3] = -0.03 + 0.17*i; t[2,3] = 0.02*i
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD", transform=t)
eng.finalize()
try:
    for fr in range(250):
        eng.step()
        if fr % 25 == 0:
            V = np.asarray(eng.get_vertices())
            print(f"[d10] fr={fr:3d} miny={V[:,1].min():+.4f} finite={np.isfinite(V).all()}", flush=True)
    print("[d10] VERDICT: 250 frames COMPLETED with dhat=5mm (no CUDA crash)", flush=True)
    print("PASS")
    sys.exit(0)
except Exception as e:
    print(f"[d10] VERDICT: FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)
    print("FAIL")
    sys.exit(1)
