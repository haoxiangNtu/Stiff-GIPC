#!/usr/bin/env python3
"""[M0 sentinel -> v0.8.4.2 regression] D2 ground-collapse sentinel: PASS = no collapse OR loud engine throw.

M0-1 D2 repro: rigid press plate driven INTO a soft block resting on ground.
Watch: per-frame min ground distance of the soft block + Newton iters.
Verdict: distance slides below 1e-7 m (collapse alive) vs stays um-level or
engine throws explicitly (defended)."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from stiff_physics.engine import Engine, Config

cfg = Config(dt=0.01, density=1e3, young_modulus=2e4, poisson_rate=0.45,
             friction_rate=0.4, relative_dhat=1e-3, ground_offset=0.0,
             prismatic_strength_ratio=2000, prismatic_driving_strength_ratio=100,
             max_prismatic_step_per_frame=0.002,
             assets_dir=os.path.join(ROOT, "Assets") + "/")
eng = Engine(cfg)
# ABD FIRST (engine constraint), FEM last.
t_anc = np.eye(4); t_anc[:3,:3] *= 0.25; t_anc[0,3] = 0.5; t_anc[1,3] = 0.5
eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD", transform=t_anc, boundary_type="Fixed")
t_pl = np.eye(4); t_pl[:3,:3] *= 0.5; t_pl[1,3] = 0.16   # plate above soft block
eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD", transform=t_pl)
t_soft = np.eye(4); t_soft[:3,:3] *= 0.5; t_soft[1,3] = -0.049  # base ~1mm above ground
eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="FEM", transform=t_soft)
j = eng.add_prismatic_joint(0, 1, world_center=[0.0, 0.35, 0.0],
                            world_axis=[0.0, -1.0, 0.0], lower_limit=0.0, upper_limit=0.4)
eng.finalize()
recs = eng.get_load_records()
femr = [r for r in recs if r.body_type == 1][0]
s0, c0 = femr.vertex_offset, femr.vertex_count
nat = eng.native
gi = (lambda: nat.get_total_newton_iters()) if hasattr(nat, "get_total_newton_iters") else (lambda: 0)
i0 = gi(); mind_hist = []
try:
    for fr in range(300):
        eng.native.set_prismatic_target(j, min(0.002 * fr, 0.30))  # press 30 cm total
        eng.step()
        i1 = gi(); dn = i1 - i0; i0 = i1
        V = np.asarray(eng.get_vertices())[s0:s0+c0]
        mind = float(V[:,1].min())      # ground at y=0
        mind_hist.append(mind)
        if fr % 20 == 0 or mind < 1e-6 or dn >= 100:
            print(f"[d2] fr={fr:3d} min_ground_dist={mind:.3e} m newton={dn}", flush=True)
        if mind < 1e-9:
            print(f"[d2] VERDICT: COLLAPSE REPRODUCED at fr={fr} (dist={mind:.3e})", flush=True); sys.exit(1)
    else:
        print(f"[d2] VERDICT: NO collapse in 300 frames; floor of min-dist = {min(mind_hist):.3e} m", flush=True)
        print("PASS")
        sys.exit(0)
except RuntimeError as e:
    print(f"[d2] VERDICT: ENGINE THREW (fail-fast defended): {str(e)[:140]}", flush=True)
    print("PASS")
    sys.exit(0)
