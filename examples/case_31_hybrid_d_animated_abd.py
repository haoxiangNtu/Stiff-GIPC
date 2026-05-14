#!/usr/bin/env python3
"""case_31_hybrid_d_animated_abd.py — Strategy D with REAL ABD body.

Architecture (the *root* fix replacing case_29's Fixed-vert hack):

    rigid_only.msh                    full unified mesh
        |                                    |
        v                                    v
    ABD body (Animated)              FEM body (Free)
        - 150 verts                       - 450 verts (incl. 150 in rigid region)
        - 370 tets                        - 1263 tets
        - real mass / inertia from        - 109 interface tets cross-couple
          rigid tets                        FEM ↔ ABD via M3.5 chain-rule
        - 12 DOF q in PCG                 - 370 rigid-internal tets in this
        - PD-driven by                      mesh are skipped via Phase 4
          set_body_animated_target          (tet_to_abd auto-populated at
        - can be jointed to other           finalize for tet[v] all in ABD
          ABD bodies                        body — same mechanism as M3.5)

Why this is the right architecture (vs case_29's Fixed-vert hack):
  - Rigid region IS an ABD body → has q, has mass → can attach joints
  - chain-rule routes interface H to ABD's live PCG column → FEM follows
  - GUI drag = update Animated target (not direct vertex writes)
  - Future: add_revolute_joint(other_abd_id, gripper_abd_id, ...) "just works"

Run:
    cd /home/ps/Downloads/Stiff-GIPC-hybrid-mesh
    ./run examples/case_31_hybrid_d_animated_abd.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from stiff_physics import Engine, Config


UNIFIED_NPZ = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"
RIGID_MSH   = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid.msh"
RIGID_REMAP = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"


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
    print(f"\n[case31] === Strategy D with REAL Animated ABD body ===", flush=True)

    # ---- 1. Load extracted rigid sub-mesh as ABD body (Animated) ----
    eng.load_mesh(RIGID_MSH,
                  dimensions=3,
                  body_type="ABD",
                  transform=np.eye(4),
                  young_modulus=1e8,
                  boundary_type="Animated")     # ← KEY: PD-driven, 12 DOF stay in PCG
    abd_rec = eng.get_load_records()[-1]
    abd_id = abd_rec.body_offset
    abd_v_offset = abd_rec.vertex_offset
    print(f"[case31] ABD body={abd_id} verts={abd_rec.vertex_count} "
          f"v_offset={abd_v_offset}", flush=True)

    # ---- 2. Load full unified mesh as FEM body (Free) ----
    data = np.load(UNIFIED_NPZ)
    verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
    tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
    vertex_region = data['vertex_region']

    fem_young = float(os.environ.get("HYBRID_D_YOUNG", "1e8"))
    eng.native.load_mesh_from_data(verts, tets, 4, 3, 1, np.eye(4), fem_young, 0)
    fem_rec = eng.get_load_records()[-1]
    fem_v_offset = fem_rec.vertex_offset
    # IMPORTANT: BodyLoadRecord.body_offset is per-TYPE local id (ABD-local or
    # FEM-local).  Engine-internal global body id (for collision_exclusion,
    # add_fem_pin_to_abd's abd_body_id, etc.) is:
    #   global_id = (body_type==ABD) ? abd_local_id : abd_body_num + fem_local_id
    n_abd = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    fem_body_global = n_abd + fem_rec.body_offset
    print(f"[case31] FEM body=local{fem_rec.body_offset}/global{fem_body_global} "
          f"verts={fem_rec.vertex_count} v_offset={fem_v_offset}, "
          f"Young={fem_young:.1e}", flush=True)

    # ---- 3. Pin overlapping verts via M3.5 chain-rule ----
    # The extracted rigid_only.msh's vert indices [0..149] correspond 1:1 to
    # the unified mesh's rigid_v_idx[0..149].  So:
    #   ABD vert at local i  <-->  unified vert rigid_v_idx[i]
    # Both at the same world position initially (no transform).
    remap_data = np.load(RIGID_REMAP, allow_pickle=True)
    rigid_v_idx = remap_data['rigid_v_idx']        # (150,) int32, unified-indexing
    n_rigid = len(rigid_v_idx)

    for i in range(n_rigid):
        fem_global = fem_v_offset + int(rigid_v_idx[i])
        abd_global = abd_v_offset + i
        eng.native.add_fem_pin_to_abd(
            fem_global,
            abd_global,
            abd_id,
            rest_offset_world=(0.0, 0.0, 0.0),  # FEM and ABD vert coincide
        )
    print(f"[case31] M3.5 chain-rule: pinned {n_rigid} FEM verts → ABD body {abd_id}",
          flush=True)

    # ---- 4. Exclude collision between ABD and FEM bodies (they overlap) ----
    # Use GLOBAL body ids — abd_id is already global (ABD bodies come first),
    # fem_body_global = n_abd + fem_local.
    eng.native.add_collision_exclusion(abd_id, fem_body_global)
    print(f"[case31] collision exclusion: ABD global={abd_id} <-> FEM global={fem_body_global}",
          flush=True)

    # ---- 5. Finalize ----
    eng.finalize()
    print(f"[case31] finalized", flush=True)

    # ---- Establish target reference: ABD body's initial centroid ----
    # When boundary_type=Animated, the engine pulls q.t toward
    # body_motor_params[0:2].  We set it equal to current q.t so the body
    # holds still until the user moves a slider.
    abd_xform0 = eng.native.get_abd_body_transforms(np.array([abd_id], dtype=np.int32))
    abd_initial_t = np.array(abd_xform0[0, :3, 3], dtype=np.float64).copy()
    print(f"[case31] ABD initial q.t = ({abd_initial_t[0]:.4f}, "
          f"{abd_initial_t[1]:.4f}, {abd_initial_t[2]:.4f})", flush=True)

    # Initialize the Animated target so body holds still at t=0.
    eng.native.set_body_animated_target(abd_id,
                                         abd_initial_t[0],
                                         abd_initial_t[1],
                                         abd_initial_t[2],
                                         strength=1e6)

    auto_n = int(os.environ.get("AUTO_STEP", "0"))
    if auto_n > 0:
        for i in range(auto_n):
            t0 = time.perf_counter()
            eng.step()
            cur_xf = eng.native.get_abd_body_transforms(np.array([abd_id], dtype=np.int32))
            print(f"[case31] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms  "
                  f"abd q.t=({cur_xf[0,0,3]:+.4f}, {cur_xf[0,1,3]:+.4f}, "
                  f"{cur_xf[0,2,3]:+.4f})", flush=True)
        return

    # ---- GUI ----
    verts_world = eng.get_vertices()
    faces_surf = eng.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts_world, faces_surf,
                                    smooth_shade=True)
    mesh.set_color((0.7, 0.75, 0.85))

    # Mark FEM rigid-region verts (visualize where chain-rule pins are)
    region_marker = np.zeros(len(verts_world), dtype=np.float32)
    for i in range(n_rigid):
        region_marker[fem_v_offset + int(rigid_v_idx[i])] = 1.0
    # ABD verts (separate body, also at same positions)
    for i in range(n_rigid):
        region_marker[abd_v_offset + i] = 1.0
    mesh.add_scalar_quantity(
        "rigid (red) — Animated ABD target driven via slider",
        region_marker, enabled=True, cmap='reds')

    state = dict(running=False, step_count=0,
                 dx=0.0, dy=0.0, dz=0.0,            # slider target offsets (cm)
                 strength=1e6, last_step_ms=0.0)

    SLIDER_MAX_CM = float(os.environ.get("HYBRID_D_SLIDER_MAX_CM", "10.0"))

    def update_target():
        # Animated target = initial q.t + slider offset (m)
        tx = abd_initial_t[0] + state['dx'] * 0.01
        ty = abd_initial_t[1] + state['dy'] * 0.01
        tz = abd_initial_t[2] + state['dz'] * 0.01
        eng.native.set_body_animated_target(abd_id, tx, ty, tz,
                                             strength=state['strength'])

    def do_step():
        update_target()
        t0 = time.perf_counter()
        eng.step()
        state['last_step_ms'] = (time.perf_counter() - t0) * 1000.0
        state['step_count'] += 1
        mesh.update_vertex_positions(eng.get_vertices())

    def callback():
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((420, 0), psim.ImGuiCond_Once)
        psim.Begin("case_31 — Strategy D with Animated ABD body")
        psim.Text(f"ABD body={abd_id} (Animated, 12 DOF, mass from {abd_rec.vertex_count} verts)")
        psim.Text(f"FEM body=global{fem_body_global} ({fem_rec.vertex_count} verts, "
                  f"chain-rule {n_rigid} pins)")
        psim.Text(f"step #{state['step_count']}: {state['last_step_ms']:.1f} ms")
        psim.Separator()

        if state['running']:
            if psim.Button("Pause"): state['running'] = False
        else:
            if psim.Button("Run"): state['running'] = True
        psim.SameLine()
        if psim.Button("Step Once"): do_step()
        psim.SameLine()
        if psim.Button("Reset"):
            state['dx'] = state['dy'] = state['dz'] = 0.0

        psim.Separator()
        psim.Text(f"Animated target offset (cm) — soft PD, max ±{SLIDER_MAX_CM}cm:")
        chg, val = psim.SliderFloat("X##cx", state['dx'],
                                    v_min=-SLIDER_MAX_CM, v_max=SLIDER_MAX_CM)
        if chg: state['dx'] = val
        chg, val = psim.SliderFloat("Y##cy", state['dy'],
                                    v_min=-SLIDER_MAX_CM, v_max=SLIDER_MAX_CM)
        if chg: state['dy'] = val
        chg, val = psim.SliderFloat("Z##cz", state['dz'],
                                    v_min=-SLIDER_MAX_CM, v_max=SLIDER_MAX_CM)
        if chg: state['dz'] = val

        chg, val = psim.SliderFloat(
            "Stiffness (log10)##stiff",
            float(np.log10(max(state['strength'], 1.0))),
            v_min=3.0, v_max=10.0)
        if chg: state['strength'] = 10.0 ** val

        # Show actual ABD q.t (so user sees PD lag vs target)
        cur_xf = eng.native.get_abd_body_transforms(np.array([abd_id], dtype=np.int32))
        actual = np.array(cur_xf[0, :3, 3], dtype=np.float64) - abd_initial_t
        psim.Text(f"  target Δ:  ({state['dx']:+.2f}, {state['dy']:+.2f}, {state['dz']:+.2f}) cm")
        psim.Text(f"  actual Δ:  ({actual[0]*100:+.2f}, {actual[1]*100:+.2f}, "
                  f"{actual[2]*100:+.2f}) cm")
        psim.End()

        # Always step (gravity + PD must be active even if no slider change,
        # so the Animated body settles)
        if state['running']:
            do_step()
        else:
            # Step once when slider changes (so user sees movement)
            slider_diff = (abs(state['dx']) > 1e-6 or abs(state['dy']) > 1e-6
                           or abs(state['dz']) > 1e-6 or chg)
            actual_diff = (abs(actual[0]*100 - state['dx']) > 0.05 or
                           abs(actual[1]*100 - state['dy']) > 0.05 or
                           abs(actual[2]*100 - state['dz']) > 0.05)
            if slider_diff or actual_diff:
                do_step()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
