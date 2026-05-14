#!/usr/bin/env python3
"""case_29_hybrid_d_unified.py — Strategy D demo, working version.

Loads ONE unified tet mesh (finger + softpad merged via convex hull +
fTetWild) as a single FEM body.  Rigid-region verts are marked
BoundaryType=Fixed and driven externally each frame from sliders — this
is mathematically equivalent to an Animated ABD body whose 12 DOF q
controls the rigid region's vertices via x = q.t + R(q)*lo_p, EXCEPT
that we drive vertex positions directly (avoiding M3.5 chain-rule
which loses free-pin Hessian coupling when ABD body is Fixed/Animated).

Drag-test verified (case_29_drag_test.py): FEM near/mid/far all follow
ABD movement exactly (rigid translation).

Run:
    cd /home/ps/Downloads/Stiff-GIPC-hybrid-mesh
    ./run examples/case_29_hybrid_d_unified.py
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from stiff_physics import Engine, Config


UNIFIED_NPZ = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"


def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("tile_reflection")

    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1e4,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-4,
        joint_strength_ratio=100.0, revolute_driving_strength_ratio=100.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.5,
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 8.0

    eng = Engine(cfg)
    print(f"\n[case29] === Strategy D unified-mesh demo ===", flush=True)

    # Load UNIFIED mesh as single FEM body
    data = np.load(UNIFIED_NPZ)
    verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
    tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
    vertex_region = data['vertex_region']
    n_rigid_v = int((vertex_region == 1).sum())
    n_fem_v = int((vertex_region == 0).sum())

    fem_young = float(os.environ.get("HYBRID_D_YOUNG", "1e8"))
    print(f"[case29] mesh V={len(verts)} T={len(tets)}, rigid_v={n_rigid_v}, "
          f"FEM_v={n_fem_v}, Young={fem_young:.1e}", flush=True)

    eng.native.load_mesh_from_data(verts, tets, 4, 3, 1, np.eye(4), fem_young, 0)
    fem_rec = eng.get_load_records()[-1]
    fem_v_offset = fem_rec.vertex_offset

    # Mark rigid-region verts as Fixed (BoundaryType=1).
    # This is the engine's standard Dirichlet-BC handling — free-pin Hessian
    # coupling stays intact in H_ff, Newton properly propagates rigid
    # displacement to free FEM verts.  M3.5 chain-rule (add_fem_pin_to_abd)
    # is BYPASSED because it routes free-pin H to a non-existent ABD column
    # when ABD is Fixed/Animated, breaking propagation.
    rigid_local_idx = np.nonzero(vertex_region == 1)[0]
    for i in rigid_local_idx:
        eng.native.set_vertex_boundary(int(fem_v_offset + i), 1)
    print(f"[case29] {len(rigid_local_idx)} rigid verts marked Fixed (Dirichlet BC)",
          flush=True)

    eng.finalize()
    print(f"[case29] finalized", flush=True)

    rest_world = eng.get_vertices().copy()
    rigid_global_idx = fem_v_offset + rigid_local_idx
    rest_rigid = rest_world[rigid_global_idx].copy()

    auto_n = int(os.environ.get("AUTO_STEP", "0"))
    if auto_n > 0:
        for i in range(auto_n):
            t0 = time.perf_counter()
            eng.step()
            print(f"[case29] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms",
                  flush=True)
        return

    verts_world = eng.get_vertices()
    faces_surf = eng.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts_world, faces_surf,
                                    smooth_shade=True)
    mesh.set_color((0.7, 0.75, 0.85))
    region_marker = np.zeros(len(verts_world), dtype=np.float32)
    region_marker[rigid_global_idx] = 1.0
    mesh.add_scalar_quantity("region (1=rigid red ABD-driven, 0=FEM)",
                             region_marker, enabled=True, cmap='reds')

    # Slider state (target) vs driven state (rate-limited actual position).
    # Big slider jumps would otherwise make rigid verts teleport too far
    # per step → interface tets distort severely → IPC barrier triggers
    # near-infinite Newton iterations (engine "locks up").
    state = dict(running=False, step_count=0,
                 x=0.0, y=0.0, z=0.0,         # slider target (cm)
                 dx=0.0, dy=0.0, dz=0.0,      # actually-driven offset (cm)
                 last_step_ms=0.0)

    # Rate limit per Newton step.  STRATEGY_F mesh has avg tet ~3mm; pushing
    # rigid more than ~10% of tet size per step risks interface tet inversion.
    # 0.05cm/step (0.5mm) = ~17% of tet size — moderate.  At 60 fps GUI: 3cm/s.
    MAX_DRIVE_PER_STEP_CM = float(os.environ.get("HYBRID_D_MAX_DRIVE_CM", "0.05"))
    # Max slider range — mesh bbox ~3cm, so ±2cm offset is already large
    # deformation (66% of mesh extent).  Larger risks tet quality issues.
    SLIDER_MAX_CM = float(os.environ.get("HYBRID_D_SLIDER_MAX_CM", "2.0"))

    def drive_rigid():
        """Move rigid region toward slider target, rate-limited."""
        # Rate limit: dx moves toward x by at most MAX_DRIVE_PER_STEP_CM
        for axis in ('x', 'y', 'z'):
            target = state[axis]
            current = state['d' + axis]
            delta = target - current
            if abs(delta) > MAX_DRIVE_PER_STEP_CM:
                current += MAX_DRIVE_PER_STEP_CM * np.sign(delta)
            else:
                current = target
            state['d' + axis] = current

        cur_all = eng.get_vertices().copy()
        offset = np.array([state['dx']*0.01, state['dy']*0.01, state['dz']*0.01])
        cur_all[rigid_global_idx] = rest_rigid + offset
        eng.native.set_vertex_positions_gpu(np.ascontiguousarray(cur_all))

    def do_step():
        drive_rigid()
        t0 = time.perf_counter()
        eng.step()
        state['last_step_ms'] = (time.perf_counter() - t0) * 1000.0
        state['step_count'] += 1
        mesh.update_vertex_positions(eng.get_vertices())

    def callback():
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((400, 0), psim.ImGuiCond_Once)
        psim.Begin("case_29 Strategy D — drag rigid region (red), FEM follows")
        psim.Text(f"Mesh: V={len(verts)} T={len(tets)}, "
                  f"rigid_v={n_rigid_v}, FEM_v={n_fem_v}")
        psim.Text(f"Young={fem_young:.1e}, step #{state['step_count']}: "
                  f"{state['last_step_ms']:.1f} ms")
        psim.Separator()

        if state['running']:
            if psim.Button("Pause"): state['running'] = False
        else:
            if psim.Button("Run"): state['running'] = True
        psim.SameLine()
        if psim.Button("Step Once"): do_step()
        psim.SameLine()
        if psim.Button("Reset"):
            state['x'] = state['y'] = state['z'] = 0.0
            state['dx'] = state['dy'] = state['dz'] = 0.0
            drive_rigid()
            mesh.update_vertex_positions(eng.get_vertices())

        psim.Separator()
        psim.Text(f"Drag rigid (cm) — rate {MAX_DRIVE_PER_STEP_CM}cm/step, "
                  f"max ±{SLIDER_MAX_CM}cm:")
        chg, val = psim.SliderFloat("X##cx", state['x'],
                                    v_min=-SLIDER_MAX_CM, v_max=SLIDER_MAX_CM)
        if chg: state['x'] = val
        chg, val = psim.SliderFloat("Y##cy", state['y'],
                                    v_min=-SLIDER_MAX_CM, v_max=SLIDER_MAX_CM)
        if chg: state['y'] = val
        chg, val = psim.SliderFloat("Z##cz", state['z'],
                                    v_min=-SLIDER_MAX_CM, v_max=SLIDER_MAX_CM)
        if chg: state['z'] = val
        psim.Text(f"  driven (current actual): "
                  f"x={state['dx']:+.2f} y={state['dy']:+.2f} z={state['dz']:+.2f}")
        # Auto-step toward target when slider moved (so user doesn't need to
        # press Run). If at target, no action.
        if (abs(state['x']-state['dx']) > 1e-4 or
            abs(state['y']-state['dy']) > 1e-4 or
            abs(state['z']-state['dz']) > 1e-4):
            do_step()

        psim.End()

        if state['running']:
            do_step()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
