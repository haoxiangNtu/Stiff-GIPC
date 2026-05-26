#!/usr/bin/env python3
"""case_35_hard_pin.py — case_35 with M3.5 chain-rule HARD PIN coupling.

Sibling of case_35_stitch_spring.py.  Both share the same URDF + hybrid
gripper setup; only the FEM↔ABD coupling differs:

  HARD PIN (this file): apply_fem_pins kinematically projects pin verts to
    ABD-derived position each Newton iter.  Chain-rule routing transfers
    FEM Hessian/gradient onto ABD's q DOFs — Newton-consistent under joint
    motion, but ABD body strongly feels FEM reactive force.  At high softpad
    young modulus (~1e8) the joint chain can be effectively locked.

Use case: when you need precise ABD↔FEM tracking and don't mind the joint
feeling the softpad load.  Sibling stitch_spring file is softer.

GUI: sliders for all 16 revolute joints.
"""
import sys, os, math, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
from scipy.spatial.transform import Rotation
import polyscope as ps
import polyscope.imgui as psim

from stiff_physics import Engine, Config
from stiff_physics.robot import Robot


URDF_PATH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_softgripper_no_finger.urdf"
# Original URDF (with soft_material links) — parsed for FK to find where the
# softpad SHOULD live in world space, given hand's current FK pose.  This is
# the same approach case_27_softgripper_cup.py uses for its hybrid_d path.
ORIGINAL_URDF = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_full.urdf"
RIGID_MSH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid.msh"
RIGID_REMAP  = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"
UNIFIED_NPZ  = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"

ARM_SCALE = 0.3


def make_arm_tf(scale: float) -> np.ndarray:
    """case_27's transform: scale + rotate URDF z-up to scene y-up."""
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix()
    tf[0, 3] = 0.0
    tf[1, 3] = -0.9
    tf[2, 3] = 0.0
    return tf


def _parse_xyz_rpy(s):
    return np.array([float(x) for x in s.split()], dtype=float)


def parse_link_world_tf(urdf_path: str, link_name: str,
                         scale: float, base_tf: np.ndarray) -> np.ndarray:
    """Direct port of case_27.parse_soft_material_world_tfs logic for ONE
    link.  Walks URDF joint tree from link → root, accumulating local
    joint origin transforms.  Then composes link's collision/visual mesh
    origin (from <collision>/<visual> origin tag inside the link)."""
    src = open(urdf_path).read()
    joints = {}
    for m in re.finditer(r'<joint\s+name="([^"]+)"[^>]*>(.*?)</joint>', src, re.DOTALL):
        body = m.group(2)
        pm = re.search(r'<parent\s+link="([^"]+)"', body)
        cm = re.search(r'<child\s+link="([^"]+)"', body)
        if not (pm and cm): continue
        om_xyz = re.search(r'<origin[^/>]*xyz="([^"]+)"', body)
        om_rpy = re.search(r'<origin[^/>]*rpy="([^"]+)"', body)
        joints[cm.group(1)] = dict(
            parent=pm.group(1),
            xyz=_parse_xyz_rpy(om_xyz.group(1)) if om_xyz else np.zeros(3),
            rpy=_parse_xyz_rpy(om_rpy.group(1)) if om_rpy else np.zeros(3),
        )

    def world_tf(link):
        T = np.eye(4); cur = link
        while cur in joints:
            j = joints[cur]
            R = Rotation.from_euler('xyz', j['rpy']).as_matrix()
            local = np.eye(4); local[:3, :3] = R; local[:3, 3] = j['xyz']
            T = local @ T
            cur = j['parent']
        return T

    def visual_origin_in_link(name):
        m = re.search(r'<link\s+name="' + re.escape(name) + r'"[^>]*>(.*?)</link>',
                      src, re.DOTALL)
        if not m: return np.eye(4)
        body = m.group(1)
        section = (re.search(r'<collision[^>]*>(.*?)</collision>', body, re.DOTALL)
                   or re.search(r'<visual[^>]*>(.*?)</visual>', body, re.DOTALL))
        if not section: return np.eye(4)
        sbody = section.group(1)
        om_xyz = re.search(r'<origin[^/>]*xyz="([^"]+)"', sbody)
        om_rpy = re.search(r'<origin[^/>]*rpy="([^"]+)"', sbody)
        if not (om_xyz or om_rpy): return np.eye(4)
        xyz = _parse_xyz_rpy(om_xyz.group(1)) if om_xyz else np.zeros(3)
        rpy = _parse_xyz_rpy(om_rpy.group(1)) if om_rpy else np.zeros(3)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
        T[:3, 3] = xyz
        return T

    T_link_in_world = world_tf(link_name)
    T_mesh_in_link = visual_origin_in_link(link_name)
    return base_tf @ T_link_in_world @ T_mesh_in_link


def main():
    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1.0,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
        joint_strength_ratio=100.0,             # case_27 default
        # case_27 default — sufficient now that fem_young is reasonable (1e6
        # not 1e8).  Higher K only needed when softpad elasticity overpowers
        # joint motion, which happens at fem_young >> 1e6.
        revolute_driving_strength_ratio=float(os.environ.get("CASE35_PD_K", "100")),
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.5,
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 64.0   # case_27 setting

    eng = Engine(cfg)
    print(f"\n[case35] === ridgeback + hybrid gripper ===", flush=True)

    # --- 1. Load URDF (no mesh override; OBB collision is light enough) ---
    arm_tf = make_arm_tf(ARM_SCALE)
    # revolute_as_motor=False (case_27 standard).
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})
    n_urdf = eng.abd_body_count
    urdf_recs = list(eng.get_load_records())
    hand_rec = next((r for r in urdf_recs if r.label == "left_arm_hand"), None)
    if hand_rec is None:
        raise RuntimeError("left_arm_hand not loaded — check URDF mesh paths")
    print(f"[case35] URDF: {n_urdf} ABD bodies, hand body_offset={hand_rec.body_offset}",
          flush=True)

    # Skip ground collision for all arm bodies (case_27 pattern)
    for b in range(n_urdf):
        eng.add_ground_collision_skip(b)

    # OPTION A: Disable gravity on URDF arm ABD bodies. URDF is loaded with
    # kinematic FK pose; without gravity, arm holds exactly at that pose,
    # making "slider angle == actual joint angle" true.  With gravity on,
    # finite PD stiffness lets arm sag to dynamic equilibrium (~30 mm hand
    # drift), and UI's reported joint angle no longer matches reality.
    # NOTE: must be called after finalize, deferred below.

    # --- 2. Compute gripper transform.  Two-step approach to handle joint
    # clamping correctly:
    #   (a) From ORIGINAL_URDF (has soft_material link), parse soft_material
    #       and hand link transforms at "zero arm pose" but in robot-base
    #       frame (no arm_tf, no scale).  Compute soft_in_hand = relative tf.
    #       All joints between hand and soft_material are fixed or prismatic
    #       at 0 (valid), so this relative offset is independent of arm pose.
    #   (b) Get hand's ACTUAL world transform from importer (which now uses
    #       clamped joint angles for FK — joint6 starts at lower limit 31°
    #       instead of invalid 0).
    #   (c) Compose: gripper_world = hand_world × soft_in_hand
    SOFT_LINK = "left_arm_leftfinger_soft_material"
    T_soft_root = parse_link_world_tf(ORIGINAL_URDF, SOFT_LINK, 1.0, np.eye(4))
    T_hand_root = parse_link_world_tf(ORIGINAL_URDF, "left_arm_hand", 1.0, np.eye(4))
    T_soft_in_hand = np.linalg.inv(T_hand_root) @ T_soft_root

    hand_T_world = eng.native.get_urdf_link_transform("left_arm_hand")
    hand_center = hand_T_world[:3, 3]
    gripper_T = hand_T_world @ T_soft_in_hand
    print(f"[case35] gripper transform: hand_world × soft_in_hand\n{gripper_T}",
          flush=True)

    eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
                  transform=gripper_T, young_modulus=1e8,
                  boundary_type="Free")
    gripper_rec = eng.get_load_records()[-1]
    gripper_id = gripper_rec.body_offset
    gripper_v_offset = gripper_rec.vertex_offset
    print(f"[case35] gripper ABD body={gripper_id} verts={gripper_rec.vertex_count}",
          flush=True)

    # --- 4. Hybrid gripper unified mesh as FEM body (same transform) ---
    data = np.load(UNIFIED_NPZ)
    verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
    tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
    # case_27_softgripper_cup uses 1e6 for softpad — 1e8 makes softpad
    # so stiff that chain-rule pin forces dominate joint dynamics, and the
    # arm becomes effectively "frozen" by the gripper resistance.
    fem_young = float(os.environ.get("HYBRID_D_YOUNG", "1e6"))
    # Use load_mesh_from_data with the same scale+translate transform
    eng.native.load_mesh_from_data(verts, tets, 4, 3, 1, gripper_T, fem_young, 0)
    gripper_fem_rec = eng.get_load_records()[-1]
    gripper_fem_v_offset = gripper_fem_rec.vertex_offset
    n_abd_total = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    gripper_fem_global = n_abd_total + gripper_fem_rec.body_offset
    print(f"[case35] gripper FEM global={gripper_fem_global} verts={gripper_fem_rec.vertex_count}",
          flush=True)

    # --- 5. FEM↔ABD coupling: M3.5 chain-rule HARD PIN ---
    # apply_fem_pins kernel kinematically writes pinned FEM vert position =
    # q.t + R(q) * lo_p each Newton iter.  Chain-rule routing transfers
    # FEM Hessian + gradient at pin verts onto ABD body's q DOFs — fully
    # bidirectional, Newton-consistent under joint motion, ABD feels full
    # FEM reactive force.
    remap = np.load(RIGID_REMAP, allow_pickle=True)
    rigid_v_idx = remap['rigid_v_idx']
    n_rigid = len(rigid_v_idx)
    for i in range(n_rigid):
        eng.native.add_fem_pin_to_abd(
            gripper_fem_v_offset + int(rigid_v_idx[i]),
            gripper_v_offset + i,
            gripper_id, (0, 0, 0))
    print(f"[case35-hardpin] M3.5 chain-rule hard pin: {n_rigid} pins",
          flush=True)

    # --- 6. Fixed joint: left_arm_hand ↔ gripper ABD ---
    hand_id = hand_rec.body_offset
    fj_idx = eng.native.add_fixed_joint(
        parent_body=hand_id, child_body=gripper_id,
        world_anchor=hand_center,
        world_normal=np.array([1.0, 0.0, 0.0]),
        world_bitangent=np.array([0.0, 0.0, 1.0]),
    )
    print(f"[case35] fixed_joint #{fj_idx}: hand({hand_id}) → gripper({gripper_id})",
          flush=True)

    # --- 7. Collision exclusions ---
    # gripper ABD ↔ gripper FEM (overlap)
    eng.native.add_collision_exclusion(gripper_id, gripper_fem_global)
    # arm bodies ↔ gripper FEM (don't want arm pushing softpad)
    # gripper rigid ↔ hand (geometric overlap)
    eng.native.add_collision_exclusion(hand_id, gripper_id)
    eng.native.add_collision_exclusion(hand_id, gripper_fem_global)
    for r in urdf_recs:
        if r.body_type == 0 and r.body_offset != hand_id:
            eng.native.add_collision_exclusion(r.body_offset, gripper_id)
            eng.native.add_collision_exclusion(r.body_offset, gripper_fem_global)

    eng.finalize()

    # --- 8. Disable gravity on gripper ABD + all URDF arm ABD bodies ---
    # (Option A from the analysis: without gravity, UI joint angle ≡ actual
    # joint angle since PD spring isn't fighting gravity sag.)
    eng.native.set_body_apply_gravity(gripper_id, False)
    n_urdf_grav_off = 0
    if int(os.environ.get("CASE35_DISABLE_URDF_GRAVITY", "1")):
        for r in urdf_recs:
            if r.body_type == 0:
                eng.native.set_body_apply_gravity(r.body_offset, False)
                n_urdf_grav_off += 1
        print(f"[case35] disabled gravity on {n_urdf_grav_off} URDF ABD bodies",
              flush=True)

    # --- Lift the engine's hardcoded per-frame revolute target rate-limit ---
    # ENGINE BUG/QUIRK: update_revolute_driving_targets caps target_angle
    # change to max_revolute_step_per_frame (default 0.1 rad ≈ 5.7°/frame).
    # This means GUI slider jumps are throttled — slider says 100° but actual
    # engine target only moves 5.7° per frame regardless of PD K.  Caller
    # must explicitly raise this cap or accept "joint always lags slider".
    # 0.5 rad/frame (~30°/frame) responsive without violating self-collision.
    eng.native.set_max_revolute_step_per_frame(
        float(os.environ.get("CASE35_MAX_RAD_PER_FRAME", "0.5")))

    # --- 9. Crank up fixed_joint stiffness on JUST our joint ---
    # Default kappa ~20 (joint_strength_ratio × mass) too weak — gripper
    # ABD lags 8mm+ per cm of hand motion.  1e3 = sweet spot: hand-ABD lag
    # ~1mm, no self-intersect.  Higher (1e4+) is too tight — gripper jumps
    # too fast each step, softpad's free verts can't follow chain-rule pin
    # motion → mesh self-intersects (same pattern as case_32 rate limit).
    # Doesn't affect URDF internal joints (per-joint kappa entry).
    eng.native.set_fixed_joint_strength(fj_idx, 1e3)

    # --- Wrap engine in Robot helper (case_27 pattern) ---
    # URDF importer now clamps joint target to limit at load time
    # (urdf_scene_importer.cpp), so app-level clamping no longer needed.
    robot = Robot(eng)
    print(f"[case35] {len(robot.revolute_joints)} revolute joints, "
          f"{len(robot.prismatic_joints)} prismatic joints", flush=True)
    print(f"[case35] finalized\n", flush=True)

    auto_n = int(os.environ.get("AUTO_STEP", "0"))
    if auto_n > 0:
        for i in range(auto_n):
            t0 = time.perf_counter()
            eng.step()
            print(f"[case35] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms",
                  flush=True)
        return

    # --- GUI ---
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    verts_world = eng.get_vertices()
    all_faces = eng.get_surface_faces()
    recs = eng.get_load_records()

    body_meshes = []
    for r in recs:
        v_off = r.vertex_offset
        v_end = v_off + r.vertex_count
        face_mask = np.all((all_faces >= v_off) & (all_faces < v_end), axis=1)
        if not face_mask.any():
            continue
        faces_local = all_faces[face_mask] - v_off
        name = (r.label or f"body{r.body_offset}").replace(" ", "_")[:32]
        # DEBUG-mode coloring to distinguish 3 hybrid components:
        #   1. URDF left_arm_hand (cyan) — the hand the gripper is welded to
        #   2. gripper ABD body (BRIGHT GREEN) — should track hand via fixed_joint
        #   3. gripper FEM body (red region scalar) — softpad pinned to ABD
        #
        # If hand moves but green ABD doesn't follow → fixed_joint failure
        # If green ABD moves but FEM red region drifts → chain-rule failure
        # If green ABD coincides with FEM red region always → both work
        if r.body_type == 1 and r.body_offset == gripper_fem_rec.body_offset:
            color = (0.85, 0.85, 0.92)
        elif r.body_offset == gripper_id and r.body_type == 0:
            color = (0.2, 0.95, 0.3)   # BRIGHT GREEN — gripper ABD body
        elif r.body_offset == hand_id:
            color = (0.3, 0.5, 0.85)   # blue — hand
        else:
            color = (0.55, 0.55, 0.6)  # arm grey
        m = ps.register_surface_mesh(name, verts_world[v_off:v_end], faces_local,
                                      smooth_shade=True)
        m.set_color(color)
        body_meshes.append((m, v_off, v_end))
        if r.body_type == 1 and r.body_offset == gripper_fem_rec.body_offset:
            region = np.zeros(r.vertex_count, dtype=np.float32)
            region[rigid_v_idx] = 1.0
            m.add_scalar_quantity("rigid (red)", region, enabled=True, cmap='reds')

    # case_27 pattern: Robot.set_revolute_position writes target + clamps to
    # joint limit internally.  PD spring tracks target (lag depends on K).
    # No explicit rate-limit — case_27 doesn't use one either.
    state = dict(running=False, step_count=0, last_step_ms=0.0)

    def do_step():
        t0 = time.perf_counter()
        eng.step()
        state['last_step_ms'] = (time.perf_counter() - t0) * 1000.0
        state['step_count'] += 1
        cur_verts = eng.get_vertices()
        for m, v0, v1 in body_meshes:
            m.update_vertex_positions(cur_verts[v0:v1])

    def callback():
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((480, 0), psim.ImGuiCond_Once)
        psim.Begin("case_35 — ridgeback + hybrid gripper")
        psim.Text(f"step #{state['step_count']}: {state['last_step_ms']:.1f} ms")
        psim.Text(f"URDF bodies: {n_urdf}, hybrid: 1 ABD + 1 FEM")
        psim.Separator()

        if state['running']:
            if psim.Button("Pause"): state['running'] = False
        else:
            if psim.Button("Run"): state['running'] = True
        psim.SameLine()
        if psim.Button("Step"): do_step()
        psim.SameLine()
        if psim.Button("Reset"): robot.reset_all()

        psim.Separator()
        if robot.revolute_joints:
            psim.Text(f"Revolute Joints ({len(robot.revolute_joints)})")
            psim.Separator()
            for i, ji in enumerate(robot.revolute_joints):
                cur = robot.get_revolute_target_deg(i)
                chg, new_val = psim.SliderFloat(
                    ji.name, cur, ji.lower_limit_deg, ji.upper_limit_deg)
                if chg:
                    robot.set_revolute_position(i, new_val, degree=True)

        if robot.prismatic_joints:
            psim.Spacing()
            psim.Text(f"Prismatic Joints ({len(robot.prismatic_joints)})")
            psim.Separator()
            for i, ji in enumerate(robot.prismatic_joints):
                lo_mm = ji.lower_limit * 1000.0
                hi_mm = ji.upper_limit * 1000.0
                cur_mm = robot.get_prismatic_target_mm(i)
                chg, new_val = psim.SliderFloat(
                    f"{ji.name} (mm)", cur_mm, lo_mm, hi_mm)
                if chg:
                    robot.set_prismatic_position(i, new_val, millimeters=True)
        psim.End()

        if state['running']:
            do_step()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
