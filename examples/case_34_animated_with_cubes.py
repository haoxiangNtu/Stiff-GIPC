#!/usr/bin/env python3
"""case_34_animated_with_cubes.py — case_30 redone on case_31 architecture.

Goal: validate that the *real* hybrid mesh framework (case_31 — Animated
ABD body + M3.5 chain-rule, NOT the case_29 Fixed-vert hack) works
correctly when the gripper contacts other objects.

Scene:
    gripper (case_31 hybrid):
        rigid_only.msh  → Animated ABD body  (12 DOF, PD-driven)
        unified mesh   → FEM body
        chain-rule pin: rigid-region FEM verts ↔ gripper ABD
    + external ABD cube (Free, 3cm, falls under gravity)
    + external FEM cube (Free, 3cm, soft, falls and deforms)
    + ground plane

Contact scenarios validated:
    A.  rigid ABD ↔ external ABD cube   →  classic ABD-ABD contact
    B.  rigid ABD ↔ external FEM cube   →  ABD-FEM contact
    C.  softpad FEM ↔ external ABD cube →  FEM-ABD contact (chain-rule pinned)
    D.  softpad FEM ↔ external FEM cube →  FEM-FEM contact (most complex —
                                           softpad is chain-rule pinned,
                                           target cube is free)

GUI:
    XYZ sliders move gripper ABD's Animated target. PD spring pulls
    gripper to slider position. Drag toward each cube to test contacts.
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from stiff_physics import Engine, Config


UNIFIED_NPZ = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"
RIGID_MSH   = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid.msh"
RIGID_REMAP = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"
CUBE_MSH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/tetmesh/cube.msh"


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
        preconditioner_type=0, ground_offset=-0.040,
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 8.0

    eng = Engine(cfg)
    print(f"\n[case34] === case_31 hybrid + 2 external cubes ===", flush=True)

    # ---- 1. External ABD cube — Free, 3cm just below gripper (left) ----
    # Gripper centroid is at (0.012, 0.083, 0.016); cube center at Y=0.025
    # → cube top Y=0.04, ~4cm below gripper centroid.  Drag gripper Y<0
    # to press cubes down; X<0 to push ABD cube left/sideways.
    abd_T = np.eye(4); abd_T[:3, :3] *= 0.03; abd_T[3, 3] = 1.0
    abd_T[:3, 3] = [-0.05, -0.020, 0.016]
    eng.load_mesh(CUBE_MSH, dimensions=3, body_type="ABD",
                  transform=abd_T, young_modulus=1e8, boundary_type="Free")
    ext_abd_rec = eng.get_load_records()[-1]
    ext_abd_id = ext_abd_rec.body_offset                   # global = 0
    print(f"[case34] external ABD cube body={ext_abd_id} verts={ext_abd_rec.vertex_count}",
          flush=True)

    # ---- 2. Hybrid gripper: rigid sub-mesh as Animated ABD ----
    # Will be PD-driven via set_body_animated_target each step.
    eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
                  transform=np.eye(4), young_modulus=1e8,
                  boundary_type="Animated")
    gripper_rec = eng.get_load_records()[-1]
    gripper_id = gripper_rec.body_offset                   # global = 1
    gripper_v_offset = gripper_rec.vertex_offset
    print(f"[case34] gripper ABD (Animated) body={gripper_id} verts={gripper_rec.vertex_count}",
          flush=True)

    # ---- 3. External FEM cube — Free, soft, 3cm just below gripper (right) ----
    fem_T = np.eye(4); fem_T[:3, :3] *= 0.03; fem_T[3, 3] = 1.0
    fem_T[:3, 3] = [0.08, -0.020, 0.016]
    eng.load_mesh(CUBE_MSH, dimensions=3, body_type="FEM",
                  transform=fem_T, young_modulus=5e6, boundary_type="Free")
    ext_fem_rec = eng.get_load_records()[-1]
    ext_fem_v_offset = ext_fem_rec.vertex_offset
    print(f"[case34] external FEM cube body=local{ext_fem_rec.body_offset} "
          f"verts={ext_fem_rec.vertex_count}", flush=True)

    # ---- 4. Hybrid gripper: full unified mesh as FEM body ----
    data = np.load(UNIFIED_NPZ)
    verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
    tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
    vertex_region = data['vertex_region']
    fem_young = float(os.environ.get("HYBRID_D_YOUNG", "1e8"))
    eng.native.load_mesh_from_data(verts, tets, 4, 3, 1, np.eye(4), fem_young, 0)
    gripper_fem_rec = eng.get_load_records()[-1]
    gripper_fem_v_offset = gripper_fem_rec.vertex_offset
    print(f"[case34] gripper FEM body=local{gripper_fem_rec.body_offset} "
          f"verts={gripper_fem_rec.vertex_count} Young={fem_young:.1e}",
          flush=True)

    n_abd = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    ext_fem_global = n_abd + ext_fem_rec.body_offset       # global = 2
    gripper_fem_global = n_abd + gripper_fem_rec.body_offset  # global = 3

    # ---- 5. M3.5 chain-rule pin: rigid-region FEM verts ↔ gripper ABD ----
    remap_data = np.load(RIGID_REMAP, allow_pickle=True)
    rigid_v_idx = remap_data['rigid_v_idx']
    n_rigid = len(rigid_v_idx)
    for i in range(n_rigid):
        fem_global = gripper_fem_v_offset + int(rigid_v_idx[i])
        abd_global = gripper_v_offset + i
        eng.native.add_fem_pin_to_abd(fem_global, abd_global, gripper_id, (0,0,0))
    print(f"[case34] M3.5 chain-rule: {n_rigid} pins (gripper FEM ↔ ABD {gripper_id})",
          flush=True)

    # ---- 6. Collision exclusions ----
    # gripper rigid ABD and gripper FEM overlap geometrically — exclude.
    eng.native.add_collision_exclusion(gripper_id, gripper_fem_global)
    print(f"[case34] excl ({gripper_id},{gripper_fem_global}) gripper rigid↔FEM (overlap)",
          flush=True)

    eng.finalize()
    print(f"[case34] finalized\n", flush=True)

    # ---- 7. Disable gravity on gripper ABD (case_32 lesson) ----
    # Gripper is fully PD-driven via Animated target — gravity would just
    # cause drift and force the PD to constantly fight it.
    eng.native.set_body_apply_gravity(gripper_id, False)

    # Initial Animated target = current ABD pose
    abd_xform0 = eng.native.get_abd_body_transforms(np.array([gripper_id], dtype=np.int32))
    abd_initial_t = np.array(abd_xform0[0, :3, 3], dtype=np.float64).copy()
    print(f"[case34] gripper ABD initial q.t = {abd_initial_t}", flush=True)
    eng.native.set_body_animated_target(gripper_id, *abd_initial_t, strength=1e6)

    auto_n = int(os.environ.get("AUTO_STEP", "0"))
    if auto_n > 0:
        for i in range(auto_n):
            t0 = time.perf_counter()
            eng.step()
            print(f"[case34] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms",
                  flush=True)
        return

    # ---- GUI ----
    # Render each body separately (ABD gripper surface overlaps FEM gripper —
    # show only FEM gripper, separate render for cubes)
    verts_world = eng.get_vertices()
    all_faces = eng.get_surface_faces()
    recs = eng.get_load_records()

    # Helper to register a single-body mesh
    def reg_body(name, rec, color):
        v_off = rec.vertex_offset
        v_end = v_off + rec.vertex_count
        face_mask = np.all((all_faces >= v_off) & (all_faces < v_end), axis=1)
        faces_local = all_faces[face_mask] - v_off
        m = ps.register_surface_mesh(name, verts_world[v_off:v_end],
                                      faces_local, smooth_shade=True)
        m.set_color(color)
        return m, v_off, v_end

    ext_abd_mesh,    ext_abd_v0,    ext_abd_v1    = reg_body("ext_abd_cube",  ext_abd_rec,     (0.4, 0.7, 0.9))
    ext_fem_mesh,    ext_fem_v0,    ext_fem_v1    = reg_body("ext_fem_cube",  ext_fem_rec,     (0.9, 0.6, 0.4))
    gripper_fem_mesh, gripper_v0,   gripper_v1   = reg_body("gripper",        gripper_fem_rec, (0.85, 0.85, 0.92))

    # Color rigid region on the gripper FEM body
    fem_region_marker = np.zeros(gripper_fem_rec.vertex_count, dtype=np.float32)
    fem_region_marker[rigid_v_idx] = 1.0
    gripper_fem_mesh.add_scalar_quantity("region (red=rigid, ABD-driven)",
                                          fem_region_marker, enabled=True, cmap='reds')

    state = dict(
        running=False, step_count=0,
        dx=0.0, dy=0.0, dz=0.0,
        last_step_ms=0.0, strength=1e6,
    )

    SLIDER_MAX = float(os.environ.get("CASE34_SLIDER_MAX", "15.0"))

    def update_target():
        tx = abd_initial_t[0] + state['dx'] * 0.01
        ty = abd_initial_t[1] + state['dy'] * 0.01
        tz = abd_initial_t[2] + state['dz'] * 0.01
        eng.native.set_body_animated_target(gripper_id, tx, ty, tz,
                                             strength=state['strength'])

    def do_step():
        update_target()
        t0 = time.perf_counter()
        eng.step()
        state['last_step_ms'] = (time.perf_counter() - t0) * 1000.0
        state['step_count'] += 1
        cur = eng.get_vertices()
        ext_abd_mesh.update_vertex_positions(cur[ext_abd_v0:ext_abd_v1])
        ext_fem_mesh.update_vertex_positions(cur[ext_fem_v0:ext_fem_v1])
        gripper_fem_mesh.update_vertex_positions(cur[gripper_v0:gripper_v1])

    def callback():
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((440, 0), psim.ImGuiCond_Once)
        psim.Begin("case_34 — case_31 hybrid + 2 external cubes")
        psim.Text(f"gripper ABD={gripper_id}(Animated) FEM=g{gripper_fem_global}")
        psim.Text(f"ext_ABD={ext_abd_id}  ext_FEM=g{ext_fem_global}")
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
        psim.Text(f"Animated target offset (cm) — max ±{SLIDER_MAX}cm:")
        chg, val = psim.SliderFloat("X##cx", state['dx'], v_min=-SLIDER_MAX, v_max=SLIDER_MAX)
        if chg: state['dx'] = val
        chg, val = psim.SliderFloat("Y##cy", state['dy'], v_min=-SLIDER_MAX, v_max=SLIDER_MAX)
        if chg: state['dy'] = val
        chg, val = psim.SliderFloat("Z##cz", state['dz'], v_min=-SLIDER_MAX, v_max=SLIDER_MAX)
        if chg: state['dz'] = val

        chg, val = psim.SliderFloat(
            "Stiffness log10##s",
            float(np.log10(max(state['strength'], 1.0))),
            v_min=3.0, v_max=10.0)
        if chg: state['strength'] = 10.0 ** val

        cur_xf = eng.native.get_abd_body_transforms(np.array([gripper_id], dtype=np.int32))
        actual = np.array(cur_xf[0, :3, 3], dtype=np.float64) - abd_initial_t
        psim.Text(f"  target Δ: ({state['dx']:+.2f}, {state['dy']:+.2f}, {state['dz']:+.2f}) cm")
        psim.Text(f"  actual Δ: ({actual[0]*100:+.2f}, {actual[1]*100:+.2f}, "
                  f"{actual[2]*100:+.2f}) cm")
        psim.Separator()
        psim.Text("Tips:")
        psim.Text("- Drag X<0 → push ext_ABD cube (left)")
        psim.Text("- Drag X>0 → push ext_FEM cube (right, deforms)")
        psim.Text("- Drag Y<0 → press down on cubes (squeeze test)")
        psim.End()

        slider_changed = (abs(state['dx']) > 1e-6 or abs(state['dy']) > 1e-6
                          or abs(state['dz']) > 1e-6)
        actual_diff = (abs(actual[0]*100 - state['dx']) > 0.05 or
                       abs(actual[1]*100 - state['dy']) > 0.05 or
                       abs(actual[2]*100 - state['dz']) > 0.05)
        if state['running'] or slider_changed or actual_diff:
            do_step()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
