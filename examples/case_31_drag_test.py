#!/usr/bin/env python3
"""Headless drag test for case_31: ramp Animated target, verify FEM verts
follow ABD via chain-rule.  This is the analog of case_29_drag_test.py
but using Animated ABD body instead of Fixed-vert hack.

If chain-rule works:
  - ABD q.t tracks target (with PD lag depending on stiffness)
  - FEM near-interface verts follow ABD displacement closely
  - FEM far-from-interface verts follow with elastic delay (Young modulus)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
from stiff_physics import Engine, Config

UNIFIED_NPZ = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"
RIGID_MSH   = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid.msh"
RIGID_REMAP = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"

cfg = Config(
    dt=0.020,
    soft_motion_rate=1e4, poisson_rate=0.49, friction_rate=0.4,
    relative_dhat=1e-4,
    semi_implicit_enabled=False,           # force Newton to fully iterate
    newton_tol=5e-2,
    preconditioner_type=0, ground_offset=-0.5,
    assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
)
cfg._cfg.collision_detection_buff_scale = 8.0

eng = Engine(cfg)

# 1. Animated ABD body
eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
              transform=np.eye(4), young_modulus=1e8,
              boundary_type="Animated")
abd_id = eng.get_load_records()[-1].body_offset
abd_v_offset = eng.get_load_records()[-1].vertex_offset

# 2. FEM body (full unified mesh)
data = np.load(UNIFIED_NPZ)
verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
vertex_region = data['vertex_region']
fem_young = float(os.environ.get("HYBRID_D_YOUNG", "1e8"))
print(f"Young={fem_young:.1e}", flush=True)
eng.native.load_mesh_from_data(verts, tets, 4, 3, 1, np.eye(4), fem_young, 0)
fem_v_offset = eng.get_load_records()[-1].vertex_offset
n_abd = sum(1 for r in eng.get_load_records() if r.body_type == 0)
fem_body_global = n_abd + eng.get_load_records()[-1].body_offset

# 3. Chain-rule pins
remap_data = np.load(RIGID_REMAP, allow_pickle=True)
rigid_v_idx = remap_data['rigid_v_idx']
n_rigid = len(rigid_v_idx)
for i in range(n_rigid):
    fem_global = fem_v_offset + int(rigid_v_idx[i])
    abd_global = abd_v_offset + i
    eng.native.add_fem_pin_to_abd(fem_global, abd_global, abd_id, (0.0,0.0,0.0))

# 4. Collision exclusion
eng.native.add_collision_exclusion(abd_id, fem_body_global)

eng.finalize()
print(f"finalized; chain-rule pins: {n_rigid}", flush=True)

# Initial state
abd_xform0 = eng.native.get_abd_body_transforms(np.array([abd_id], dtype=np.int32))
abd_initial_t = np.array(abd_xform0[0, :3, 3], dtype=np.float64).copy()
print(f"ABD initial q.t = {abd_initial_t}", flush=True)

# Settle for a few steps with target = initial (so PD doesn't kick in)
eng.native.set_body_animated_target(abd_id, *abd_initial_t, strength=1e7)
for _ in range(3):
    eng.step()

verts_at_rest = eng.get_vertices().copy()
rigid_global = fem_v_offset + rigid_v_idx
fem_only_local = np.nonzero(vertex_region == 0)[0]
fem_only_global = fem_v_offset + fem_only_local

# Bin FEM verts by Y-distance from rigid centroid
rigid_centroid_Y = verts_at_rest[rigid_global][:, 1].mean()
fem_Y_dist = np.abs(verts_at_rest[fem_only_global][:, 1] - rigid_centroid_Y)
print(f"rigid centroid Y={rigid_centroid_Y:.4f}, "
      f"FEM Y dist range: {fem_Y_dist.min():.4f} to {fem_Y_dist.max():.4f}",
      flush=True)

print("\n--- DRAG TEST: ramp Animated target Y by -2cm over 10 steps ---",
      flush=True)
print(f"{'step':>4} {'tgt Δy':>8} {'abd Δy':>10} {'rigid_FEM Δy':>14} "
      f"{'FEM near Δy':>12} {'FEM mid Δy':>12} {'FEM far Δy':>12} {'ms':>6}",
      flush=True)

for step in range(10):
    delta_y = -0.002 * (step + 1)  # ramp Y target by 2mm per step
    target = abd_initial_t.copy()
    target[1] += delta_y
    eng.native.set_body_animated_target(abd_id, *target, strength=1e7)

    t0 = time.perf_counter()
    eng.step()
    step_ms = (time.perf_counter() - t0) * 1000

    cur = eng.get_vertices()
    cur_xf = eng.native.get_abd_body_transforms(np.array([abd_id], dtype=np.int32))
    abd_dy = cur_xf[0, 1, 3] - abd_initial_t[1]

    rigid_fem_dy = (cur[rigid_global] - verts_at_rest[rigid_global])[:, 1].mean()

    near_mask = fem_Y_dist < 0.02
    mid_mask = (fem_Y_dist >= 0.02) & (fem_Y_dist < 0.05)
    far_mask = fem_Y_dist >= 0.05
    fem_dy_all = (cur[fem_only_global] - verts_at_rest[fem_only_global])[:, 1]
    near_dy = fem_dy_all[near_mask].mean() if near_mask.sum() else 0
    mid_dy = fem_dy_all[mid_mask].mean() if mid_mask.sum() else 0
    far_dy = fem_dy_all[far_mask].mean() if far_mask.sum() else 0

    print(f"{step:>4} {delta_y*1000:>+8.2f} {abd_dy*1000:>+10.2f} "
          f"{rigid_fem_dy*1000:>+14.2f} "
          f"{near_dy*1000:>+12.2f} {mid_dy*1000:>+12.2f} {far_dy*1000:>+12.2f} "
          f"{step_ms:>6.1f}", flush=True)

print("\nUnits: mm.  tgt = Animated target offset, abd = actual q.t offset (PD lag),", flush=True)
print("rigid_FEM = FEM verts pinned to ABD via chain-rule (should match abd).", flush=True)
print("FEM near/mid/far = pure-FEM verts at increasing distance from rigid interface.", flush=True)
print("If chain-rule works: abd ≈ rigid_FEM ≈ FEM_near, decaying for FEM_mid/far.", flush=True)
