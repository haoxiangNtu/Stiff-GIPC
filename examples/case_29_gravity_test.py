#!/usr/bin/env python3
"""Pure gravity test: Free ABD body + pinned FEM. Let gravity pull
everything down. If FEM verts move → engine setup OK, drag-test bug
was Fixed-ABD-related. If FEM verts still don't move → fundamental
issue in pin propagation."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
from stiff_physics import Engine, Config

UNIFIED_NPZ = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"
ABD_TET_MSH = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/tiny_cube.msh"

cfg = Config(dt=0.020, soft_motion_rate=1e4, poisson_rate=0.49,
             friction_rate=0.4, relative_dhat=1e-4,
             semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
             newton_tol=5e-2, preconditioner_type=0, ground_offset=-0.5,
             assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/")
cfg._cfg.collision_detection_buff_scale = 8.0
eng = Engine(cfg)

# ABD body — FREE this time (gravity drives it)
abd_T = np.eye(4); abd_T[:3, 3] = [0.011, 0.18, 0.017]
eng.load_mesh(ABD_TET_MSH, dimensions=3, body_type="ABD",
              transform=abd_T, young_modulus=1e8)  # default Free
abd_id = eng.get_load_records()[-1].body_offset
print(f"ABD Free body={abd_id}", flush=True)

data = np.load(UNIFIED_NPZ)
verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
vertex_region = data['vertex_region']
fem_young = float(os.environ.get("HYBRID_D_YOUNG", "1e8"))
print(f"Young={fem_young:.1e}", flush=True)
eng.native.load_mesh_from_data(verts, tets, 4, 3, 1, np.eye(4), fem_young, 0)
fem_v_offset = eng.get_load_records()[-1].vertex_offset

rigid_local = np.nonzero(vertex_region == 1)[0]
for i in rigid_local:
    eng.native.add_fem_pin_to_abd(fem_v_offset + int(i), fem_v_offset + int(i),
                                  abd_id, rest_offset_world=(0.0, 0.0, 0.0))
eng.native.add_collision_exclusion(abd_id, eng.get_load_records()[-1].body_offset)
eng.native.add_ground_collision_skip(abd_id)
eng.finalize()
print(f"finalized, {len(rigid_local)} pins", flush=True)

verts0 = eng.get_vertices().copy()
rigid_g = fem_v_offset + rigid_local
fem_g = fem_v_offset + np.nonzero(vertex_region == 0)[0]

print(f"\n--- GRAVITY TEST: Free ABD + pinned FEM, no drag ---", flush=True)
print(f"{'step':>4} {'rigid Δy':>10} {'FEM Δy':>10} {'FEM y range':>20} {'step_ms':>8}", flush=True)

for step in range(15):
    t0 = time.perf_counter()
    eng.step()
    step_ms = (time.perf_counter() - t0) * 1000
    cur = eng.get_vertices()
    rigid_dy = (cur[rigid_g] - verts0[rigid_g])[:, 1].mean()
    fem_dy = (cur[fem_g] - verts0[fem_g])[:, 1]
    print(f"{step:>4} {rigid_dy*1000:>+10.3f} {fem_dy.mean()*1000:>+10.3f} "
          f"[{fem_dy.min()*1000:>+6.2f}, {fem_dy.max()*1000:>+6.2f}] {step_ms:>8.1f}",
          flush=True)
