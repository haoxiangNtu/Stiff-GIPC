#!/usr/bin/env python3
"""case_32_hybrid_d_with_joint.py — case_31 hybrid mesh + revolute joint
to a Fixed external "arm link" ABD body.

This proves the *production* path: a hybrid ABD+FEM mesh CAN be jointed
to other ABD bodies (URDF link, parent gripper base, etc.) via the
standard joint API — exactly because the rigid region IS a real ABD body
with 12 DOF q in PCG.

Scene:

           arm_link (Fixed ABD cube)
                  |
           [revolute joint Y-axis]
                  |
        gripper rigid (Free ABD)
        + softpad (FEM, chain-rule pinned)

GUI:  drag the joint-angle slider → gripper swings → FEM softpad follows.

Run:
    cd /home/ps/Downloads/Stiff-GIPC-hybrid-mesh
    ./run examples/case_32_hybrid_d_with_joint.py
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
        joint_strength_ratio=200.0, revolute_driving_strength_ratio=200.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.5,
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 8.0

    eng = Engine(cfg)
    print(f"\n[case32] === hybrid mesh + revolute joint to external ABD ===",
          flush=True)

    # ---- 1. arm_link: Fixed ABD cube above gripper ----
    # The gripper rigid region's centroid is around (0.012, 0.083, 0.016).
    # Place arm_link 8cm above it.
    arm_T = np.eye(4)
    arm_T[:3, :3] *= 0.04                       # 4cm cube
    arm_T[3, 3] = 1.0
    arm_T[:3, 3] = [0.012, 0.18, 0.016]         # above gripper, ground=-0.5
    eng.load_mesh(CUBE_MSH, dimensions=3, body_type="ABD",
                  transform=arm_T, young_modulus=1e8,
                  boundary_type="Fixed")        # ← static base
    arm_rec = eng.get_load_records()[-1]
    arm_id = arm_rec.body_offset                # global = 0 (first ABD)
    print(f"[case32] arm_link Fixed ABD body={arm_id} verts={arm_rec.vertex_count}",
          flush=True)

    # ---- 2. gripper rigid sub-mesh: Free ABD (constrained by joint) ----
    eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
                  transform=np.eye(4), young_modulus=1e8,
                  boundary_type="Free")         # ← Free, not Animated
    gripper_rec = eng.get_load_records()[-1]
    gripper_id = gripper_rec.body_offset        # global = 1 (second ABD)
    gripper_v_offset = gripper_rec.vertex_offset
    print(f"[case32] gripper Free ABD body={gripper_id} verts={gripper_rec.vertex_count} "
          f"v_offset={gripper_v_offset}", flush=True)

    # ---- 3. Full unified mesh as FEM body ----
    data = np.load(UNIFIED_NPZ)
    verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
    tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
    vertex_region = data['vertex_region']
    fem_young = float(os.environ.get("HYBRID_D_YOUNG", "1e8"))
    eng.native.load_mesh_from_data(verts, tets, 4, 3, 1, np.eye(4), fem_young, 0)
    fem_rec = eng.get_load_records()[-1]
    fem_v_offset = fem_rec.vertex_offset
    n_abd = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    fem_global = n_abd + fem_rec.body_offset    # global = 2
    print(f"[case32] FEM body=local{fem_rec.body_offset}/global{fem_global} "
          f"verts={fem_rec.vertex_count}, Young={fem_young:.1e}", flush=True)

    # ---- 4. M3.5 chain-rule pins (gripper rigid FEM verts → gripper ABD) ----
    remap_data = np.load(RIGID_REMAP, allow_pickle=True)
    rigid_v_idx = remap_data['rigid_v_idx']
    n_rigid = len(rigid_v_idx)
    for i in range(n_rigid):
        fem_pin = fem_v_offset + int(rigid_v_idx[i])
        abd_anchor = gripper_v_offset + i
        eng.native.add_fem_pin_to_abd(fem_pin, abd_anchor, gripper_id, (0.0,0.0,0.0))
    print(f"[case32] M3.5 chain-rule: {n_rigid} FEM verts → gripper ABD {gripper_id}",
          flush=True)

    # ---- 5. Revolute joint: arm_link --[Y axis]--> gripper ABD ----
    # Joint anchor: midway between arm_link bottom and gripper top
    joint_anchor = np.array([0.012, 0.14, 0.016])
    joint_axis   = np.array([0.0, 0.0, 1.0])    # Z axis — gripper swings in XY plane
    joint_idx = eng.native.add_revolute_joint(
        parent_body=arm_id,
        child_body=gripper_id,
        world_axis=joint_axis,
        joint_pos=joint_anchor,
        lower_limit=-math.radians(90),
        upper_limit= math.radians(90),
        initial_angle=0.0,
        name="arm_to_gripper",
    )
    print(f"[case32] revolute joint #{joint_idx}: arm({arm_id}) → gripper({gripper_id}) "
          f"axis={joint_axis} anchor={joint_anchor}", flush=True)

    # ---- 6. Collision exclusions ----
    # arm_link is far away geometrically, but exclusion is cheap insurance.
    # gripper ABD ↔ FEM body overlap by design (chain-rule mesh).
    eng.native.add_collision_exclusion(arm_id, gripper_id)
    eng.native.add_collision_exclusion(arm_id, fem_global)
    eng.native.add_collision_exclusion(gripper_id, fem_global)
    print(f"[case32] collision exclusions: ({arm_id},{gripper_id}), "
          f"({arm_id},{fem_global}), ({gripper_id},{fem_global})", flush=True)

    eng.finalize()
    print(f"[case32] finalized\n", flush=True)

    # Disable gravity on gripper ABD body — it's anchored to Fixed arm_link via
    # revolute joint, shouldn't fall independently.  Without this:
    #   - joint penalty must constantly cancel mass*g, leaving residual that
    #     accumulates as position drift after each rotation cycle
    #   - Newton can't find self-consistent state when chain-rule + joint +
    #     gravity all pull on ABD's q → barrier 卡死, lineSearchCount=9 spam
    eng.native.set_body_apply_gravity(gripper_id, False)
    print(f"[case32] disabled gravity on gripper ABD body {gripper_id}", flush=True)

    # Set initial joint target = 0 (gripper hangs straight down from arm)
    eng.native.set_revolute_target(joint_idx, 0.0)

    auto_n = int(os.environ.get("AUTO_STEP", "0"))
    if auto_n > 0:
        for i in range(auto_n):
            t0 = time.perf_counter()
            eng.step()
            xf = eng.native.get_abd_body_transforms(np.array([gripper_id], dtype=np.int32))
            print(f"[case32] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms  "
                  f"gripper q.t=({xf[0,0,3]:+.4f}, {xf[0,1,3]:+.4f}, {xf[0,2,3]:+.4f})",
                  flush=True)
        return

    # ---- GUI ----
    # IMPORTANT: ABD gripper body and FEM body BOTH have surface triangles
    # in the rigid region (same geometry, different vertex sets).  Rendering
    # both via eng.get_surface_faces() causes z-fighting + appears as
    # "ABD/FEM rotating in opposite directions" during transient mismatch.
    # Solution: render ONLY the FEM body (which carries the full unified
    # geometry including rigid region — chain-rule keeps those verts in sync
    # with ABD), plus arm_link separately for visualization.
    verts_world = eng.get_vertices()
    all_faces = eng.get_surface_faces()

    # Filter surface faces to FEM body's vertex range only
    fem_v_end = fem_v_offset + fem_rec.vertex_count
    fem_face_mask = np.all((all_faces >= fem_v_offset) & (all_faces < fem_v_end), axis=1)
    fem_faces = all_faces[fem_face_mask]
    fem_verts = verts_world[fem_v_offset:fem_v_end]
    fem_faces_local = fem_faces - fem_v_offset
    print(f"[case32] FEM surface faces: {len(fem_faces)} (filtered from {len(all_faces)})",
          flush=True)

    fem_mesh = ps.register_surface_mesh("gripper_unified", fem_verts, fem_faces_local,
                                        smooth_shade=True)
    fem_mesh.set_color((0.85, 0.85, 0.92))

    # Color rigid region red on the FEM body
    fem_region_marker = np.zeros(fem_rec.vertex_count, dtype=np.float32)
    fem_region_marker[rigid_v_idx] = 1.0
    fem_mesh.add_scalar_quantity("region (red=rigid driven by ABD)",
                                  fem_region_marker, enabled=True, cmap='reds')

    # arm_link as separate mesh (Fixed, doesn't move)
    arm_v_end = arm_rec.vertex_offset + arm_rec.vertex_count
    arm_face_mask = np.all((all_faces >= arm_rec.vertex_offset) &
                           (all_faces < arm_v_end), axis=1)
    arm_faces = all_faces[arm_face_mask] - arm_rec.vertex_offset
    arm_verts = verts_world[arm_rec.vertex_offset:arm_v_end]
    arm_mesh = ps.register_surface_mesh("arm_link", arm_verts, arm_faces,
                                        smooth_shade=False)
    arm_mesh.set_color((0.4, 0.6, 0.8))
    print(f"[case32] arm_link surface faces: {len(arm_faces)}", flush=True)

    state = dict(running=False, step_count=0,
                 angle_deg=0.0,        # slider target (deg)
                 driven_deg=0.0,       # actually-applied target (rate-limited)
                 last_step_ms=0.0,
                 strength=1.0)

    # Rate limit: max angle change per step (deg).  Avoids softpad FEM
    # self-intersection when slider jumps — the chain-rule kernel hard-pins
    # rigid FEM verts to the new ABD pose each step, and free FEM verts
    # (softpad) only have ~1 Newton-iter to follow.  Angular velocity > tet
    # size / step ⇒ softpad gets "scrunched" into itself → INTERSECT spam,
    # lineSearchCount=9, eventual hang.  0.3°/step ≈ 18°/sec at 60fps.
    MAX_ANGLE_PER_STEP_DEG = float(os.environ.get("HYBRID_D_MAX_ANGLE_DEG", "0.3"))
    # Slider range cap.  At ±10° the gripper-tip displacement ≈ 1cm — within
    # softpad's elastic recovery range.  Bigger angles (±30°+) put the
    # softpad past geometric stability regardless of rate limit (this is a
    # mesh-quality limit, not engine bug — would need finer tet refinement
    # in the softpad to support larger excursion).
    SLIDER_ANGLE_MAX = float(os.environ.get("HYBRID_D_SLIDER_ANGLE_MAX", "10.0"))

    def do_step():
        # Rate-limit driven angle toward slider target.
        delta = state['angle_deg'] - state['driven_deg']
        if abs(delta) > MAX_ANGLE_PER_STEP_DEG:
            state['driven_deg'] += MAX_ANGLE_PER_STEP_DEG * np.sign(delta)
        else:
            state['driven_deg'] = state['angle_deg']

        eng.native.set_revolute_target(joint_idx, math.radians(state['driven_deg']))
        eng.native.set_revolute_strength(joint_idx, state['strength'])
        t0 = time.perf_counter()
        eng.step()
        state['last_step_ms'] = (time.perf_counter() - t0) * 1000.0
        state['step_count'] += 1
        cur_verts = eng.get_vertices()
        fem_mesh.update_vertex_positions(cur_verts[fem_v_offset:fem_v_end])
        arm_mesh.update_vertex_positions(cur_verts[arm_rec.vertex_offset:arm_v_end])

    def callback():
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((420, 0), psim.ImGuiCond_Once)
        psim.Begin("case_32 — hybrid gripper jointed to arm_link")
        psim.Text(f"arm_link Fixed ABD={arm_id} | gripper Free ABD={gripper_id}")
        psim.Text(f"FEM body=global{fem_global} | revolute joint #{joint_idx}")
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
            state['angle_deg'] = 0.0

        psim.Separator()
        psim.Text(f"Joint angle target (deg) — rate {MAX_ANGLE_PER_STEP_DEG}°/step, "
                  f"max ±{SLIDER_ANGLE_MAX}°:")
        chg_a, val = psim.SliderFloat("angle##j", state['angle_deg'],
                                      v_min=-SLIDER_ANGLE_MAX, v_max=SLIDER_ANGLE_MAX)
        if chg_a: state['angle_deg'] = val

        chg_s, val = psim.SliderFloat("joint stiffness##s", state['strength'],
                                      v_min=0.1, v_max=10.0)
        if chg_s: state['strength'] = val

        psim.Text(f"  slider target: {state['angle_deg']:+.1f}°  "
                  f"driven: {state['driven_deg']:+.1f}°")

        # Show actual gripper transform (joint may have lag if stiffness low)
        xf = eng.native.get_abd_body_transforms(np.array([gripper_id], dtype=np.int32))
        psim.Text(f"  gripper q.t = ({xf[0,0,3]:+.3f}, {xf[0,1,3]:+.3f}, "
                  f"{xf[0,2,3]:+.3f})")
        psim.Text(f"  gripper A_xx = {xf[0,0,0]:+.3f}, A_yx = {xf[0,1,0]:+.3f}")
        psim.End()

        # Auto-step when slider differs from driven OR Run is on
        slider_chasing = abs(state['angle_deg'] - state['driven_deg']) > 1e-3
        if state['running'] or slider_chasing:
            do_step()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
