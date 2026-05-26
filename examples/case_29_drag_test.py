#!/usr/bin/env python3
"""Headless test of case_29: programmatically drag ABD body, measure how
much FEM verts follow at different distances from interface."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
from stiff_physics import Engine, Config

UNIFIED_NPZ = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"
ABD_TET_MSH = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/tiny_cube.msh"

cfg = Config(
    dt=0.020,
    soft_motion_rate=1e4, poisson_rate=0.49, friction_rate=0.4,
    relative_dhat=1e-4,
    # Disable semi-implicit early-exit; force Newton to actually iterate so
    # elastic propagation reaches free FEM verts past the interface.
    semi_implicit_enabled=False,
    newton_tol=5e-2,
    preconditioner_type=0, ground_offset=-0.5,
    assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
)
cfg._cfg.collision_detection_buff_scale = 8.0

eng = Engine(cfg)

# v16-style: NO ABD body. Mark rigid verts as Fixed (BoundaryType=1) and
# drive them directly via set_vertex_positions_gpu.  Free-pin Hessian
# cross-coupling stays in H_ff → elastic propagation works.
data = np.load(UNIFIED_NPZ)
verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
vertex_region = data['vertex_region']
fem_young = float(os.environ.get("HYBRID_D_YOUNG", "1e8"))
print(f"Young={fem_young:.1e}", flush=True)
eng.native.load_mesh_from_data(verts, tets, 4, 3, 1, np.eye(4), fem_young, 0)
fem_v_offset = eng.get_load_records()[-1].vertex_offset

rigid_local_idx = np.nonzero(vertex_region == 1)[0]
for i in rigid_local_idx:
    eng.native.set_vertex_boundary(int(fem_v_offset + i), 1)  # Fixed
print(f"marked {len(rigid_local_idx)} rigid verts as Fixed", flush=True)
eng.finalize()
print(f"finalized", flush=True)

initial_T = np.eye(4)  # not used in v16-style
verts_at_rest = eng.get_vertices().copy()
fem_v_count = (data['vertices']).shape[0]

# Compute Y-distance of each FEM-region vert from interface (rigid-region centroid)
rigid_global = fem_v_offset + rigid_local_idx
fem_global = fem_v_offset + np.nonzero(vertex_region == 0)[0]
rigid_centroid_Y = verts_at_rest[rigid_global][:, 1].mean()
fem_Y_dist = np.abs(verts_at_rest[fem_global][:, 1] - rigid_centroid_Y)
print(f"rigid centroid Y={rigid_centroid_Y:.4f}, FEM Y dist range: {fem_Y_dist.min():.4f} to {fem_Y_dist.max():.4f}", flush=True)

# Programmatic drag: move ABD by Δ each step, measure FEM displacement
print("\n--- DRAG TEST: move ABD Y by -2cm over 10 steps ---", flush=True)
print(f"{'step':>4} {'abd_Δy':>8} {'rigid_Δy':>10} {'FEM near Δy':>12} {'FEM mid Δy':>12} {'FEM far Δy':>12} {'step_ms':>8}", flush=True)

for step in range(10):
    delta_y = -0.002 * (step + 1)  # ramp Y by 2mm per step
    # Move rigid verts directly via set_vertex_positions_gpu
    cur_all = eng.get_vertices().copy()
    cur_all[rigid_global, 1] = verts_at_rest[rigid_global, 1] + delta_y
    eng.native.set_vertex_positions_gpu(np.ascontiguousarray(cur_all))

    t0 = time.perf_counter()
    eng.step()
    step_ms = (time.perf_counter() - t0) * 1000

    cur = eng.get_vertices()
    rigid_dy = (cur[rigid_global] - verts_at_rest[rigid_global])[:, 1].mean()

    # Bin FEM verts by distance from interface
    near_mask = fem_Y_dist < 0.02   # within 2cm
    mid_mask = (fem_Y_dist >= 0.02) & (fem_Y_dist < 0.05)  # 2-5cm
    far_mask = fem_Y_dist >= 0.05    # 5cm+
    fem_dy_all = (cur[fem_global] - verts_at_rest[fem_global])[:, 1]
    near_dy = fem_dy_all[near_mask].mean() if near_mask.sum() else 0
    mid_dy = fem_dy_all[mid_mask].mean() if mid_mask.sum() else 0
    far_dy = fem_dy_all[far_mask].mean() if far_mask.sum() else 0

    print(f"{step:>4} {delta_y*1000:>+8.2f} {rigid_dy*1000:>+10.2f} {near_dy*1000:>+12.2f} {mid_dy*1000:>+12.2f} {far_dy*1000:>+12.2f} {step_ms:>8.1f}", flush=True)

print("\nUnits: mm.  Compare rigid_Δy (target) vs FEM bands at increasing distance from interface.", flush=True)
print("If FEM Δy ≈ rigid Δy at all distances → mesh follows tightly.", flush=True)
print("If FEM Δy decays with distance → elastic propagation lag (Young too low or step too big).", flush=True)
