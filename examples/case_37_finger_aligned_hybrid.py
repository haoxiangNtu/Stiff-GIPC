#!/usr/bin/env python3
"""case_37 — hybrid grippers built FROM finger.stl + softpad merge.

Sibling of case_36_quad_hybrid_grippers.py.  Same architecture (4 fingers,
each with hybrid_rigid ABD + hybrid FEM, fixed_joint to URDF finger,
stitch_spring inside hybrid).  Difference:

  case_36 uses the pre-existing STRATEGY_F mesh (built in some unknown
  frame).  Required scaled-Procrustes alignment to fit hybrid rigid
  verts onto URDF finger comp 0 verts; ~1mm fit residual.

  case_37 (this file) uses CASE37_unified.npz, BUILT FRESH by
  tools/build_case37_finger_softpad_hybrid.py from:
    - finger.stl lower segment (URDF finger collision mesh, 337 verts)
    - softgriper_part3 softpad (in same mesh-local frame, no transform
      needed because both URDF visual_origin tags are identical)
    - convex hull simplification + tetgen tetrahedralization
    - vertex_region label: inside finger.stl convex hull → rigid

  Result: hybrid rigid sub-mesh shares EXACTLY the finger.stl mesh-local
  frame.  Loading hybrid with the URDF finger ABD's engine-FK transform
  produces ZERO alignment error — green ABD literally overlays the
  blue finger ABD body.

GUI: 4 prismatic finger_joints + 16 revolute, same as case_36.
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


URDF_PATH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_softgripper.urdf"
RIGID_MSH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/CASE37_rigid.msh"
RIGID_REMAP  = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/CASE37_rigid_remap.npz"
UNIFIED_NPZ  = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/CASE37_unified.npz"

ARM_SCALE = 0.3
FINGER_LABELS = [
    'left_arm_leftfinger',  'left_arm_rightfinger',
    'right_arm_leftfinger', 'right_arm_rightfinger',
]


def make_arm_tf(scale: float) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix()
    tf[1, 3] = -0.9
    return tf


def main():
    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100,
        soft_motion_rate=float(os.environ.get("CASE37_SOFT_RATE", "1e4")),
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
        joint_strength_ratio=100.0,
        revolute_driving_strength_ratio=float(os.environ.get("CASE37_PD_K", "100")),
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.5,
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 64.0
    eng = Engine(cfg)
    print("\n[case37] === ridgeback + 4 finger-aligned hybrid grippers ===",
          flush=True)

    # 1. URDF (37 ABD bodies including 4 fingers + 4 prismatic joints)
    arm_tf = make_arm_tf(ARM_SCALE)
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})
    n_urdf = eng.abd_body_count
    urdf_recs = list(eng.get_load_records())
    finger_recs = {r.label: r for r in urdf_recs if r.body_type == 0
                   and r.label in FINGER_LABELS}
    print(f"[case37] URDF: {n_urdf} ABD bodies", flush=True)
    for b in range(n_urdf):
        eng.add_ground_collision_skip(b)

    # 2. Per-finger transform = finger ABD body's full transform.
    #
    # URDF importer's finger ABD body uses:
    #   full_T = link_global × collision_origin × scale
    # but `get_urdf_link_transform()` returns ONLY link_global.  We
    # must explicitly compose collision_origin to match what finger
    # ABD body sees — otherwise hybrid lands in link frame while
    # finger ABD body lives in (link × collision_origin) frame, and
    # they differ by ~24mm (verified: collision_origin's xyz is
    # 0.128m × 0.3 scale ≈ 38mm offset).
    #
    # finger collision_origin in URDF (same for all 4 fingers):
    #   rpy=(-1.57079632679, 0.20245819348, -1.57079632679)
    #   xyz=(-0.0165, 0.0165, 0.12773331296)
    _co_rpy = np.array([-1.57079632679, 0.20245819348, -1.57079632679])
    _co_xyz = np.array([-0.0165, 0.0165, 0.12773331296])
    collision_origin = np.eye(4)
    collision_origin[:3, :3] = Rotation.from_euler('xyz', _co_rpy).as_matrix()
    collision_origin[:3, 3] = _co_xyz

    finger_T = {}
    for label in FINGER_LABELS:
        link_T = eng.native.get_urdf_link_transform(label)
        finger_T[label] = link_T @ collision_origin
        print(f"[case37] {label}: link_T translation = {link_T[:3, 3]}, "
              f"full hybrid load translation = {finger_T[label][:3, 3]}",
              flush=True)

    fem_young = float(os.environ.get("CASE37_FEM_YOUNG", "1e6"))
    fj_kappa = float(os.environ.get("CASE37_FJ_KAPPA", "1e3"))

    rigid_remap = np.load(RIGID_REMAP, allow_pickle=True)
    rigid_v_idx = rigid_remap['rigid_v_idx']  # indices into FEM unified
    n_rigid_v = len(rigid_v_idx)
    hybrid_data = np.load(UNIFIED_NPZ)
    hybrid_verts = np.ascontiguousarray(hybrid_data['vertices'], dtype=np.float64)
    hybrid_tets  = np.ascontiguousarray(hybrid_data['tets'], dtype=np.int32)

    # Engine constraint: load all ABD before any FEM.
    grippers = []  # populated across passes

    # Pass 1: 4 hybrid rigid sub-meshes (ABD)
    for label in FINGER_LABELS:
        T = finger_T[label]
        eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
                      transform=T, young_modulus=1e8, boundary_type="Free")
        rigid_rec = eng.get_load_records()[-1]
        grippers.append(dict(
            label=label, finger_id=finger_recs[label].body_offset,
            T=T, abd_id=rigid_rec.body_offset, abd_v_off=rigid_rec.vertex_offset,
        ))

    # Pass 2: 4 hybrid FEM unified meshes
    for g in grippers:
        eng.native.load_mesh_from_data(
            hybrid_verts, hybrid_tets, 4, 3, 1, g['T'], fem_young, 0)
        fem_rec = eng.get_load_records()[-1]
        g['fem_rec'] = fem_rec
        g['fem_v_off'] = fem_rec.vertex_offset

    n_abd_total = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    for g in grippers:
        g['fem_global_id'] = n_abd_total + g['fem_rec'].body_offset

    # Pass 3: stitch springs + fixed joints
    # rigid_v_idx[i] → unified FEM vert index that corresponds to rigid
    # ABD vert i.  Both share the SAME mesh-local frame (rigid mesh is
    # extracted as a subset of unified).  rest_offset=(0,0,0) is exact.
    for g in grippers:
        for i in range(n_rigid_v):
            eng.add_stitch_spring(
                g['fem_v_off'] + int(rigid_v_idx[i]),
                g['abd_v_off'] + i,
                g['abd_id'],
                rest_offset_world=(0.0, 0.0, 0.0))
        g['fj_idx'] = eng.native.add_fixed_joint(
            parent_body=g['finger_id'], child_body=g['abd_id'],
            world_anchor=g['T'][:3, 3],
            world_normal=np.array([1.0, 0.0, 0.0]),
            world_bitangent=np.array([0.0, 0.0, 1.0]),
        )
        print(f"[case37] {g['label']}: finger={g['finger_id']}, "
              f"hybrid_abd={g['abd_id']}, fem={g['fem_rec'].body_offset}, "
              f"fj={g['fj_idx']}, n_stitch={n_rigid_v}", flush=True)

    # Collision exclusions
    arm_ids = [r.body_offset for r in urdf_recs if r.body_type == 0]
    for g in grippers:
        eng.native.add_collision_exclusion(g['abd_id'], g['fem_global_id'])
        eng.native.add_collision_exclusion(g['abd_id'], g['finger_id'])
        eng.native.add_collision_exclusion(g['fem_global_id'], g['finger_id'])
        for arm_id in arm_ids:
            if arm_id == g['finger_id']: continue
            eng.native.add_collision_exclusion(g['abd_id'], arm_id)
            eng.native.add_collision_exclusion(g['fem_global_id'], arm_id)
    # Cross-pair exclusion (left/right finger of same arm overlap at prismatic=0)
    def _arm_prefix(label):
        return 'left' if label.startswith('left_') else 'right'
    for i, gi in enumerate(grippers):
        for gj in grippers[i+1:]:
            if _arm_prefix(gi['label']) != _arm_prefix(gj['label']): continue
            eng.native.add_collision_exclusion(gi['abd_id'], gj['abd_id'])
            eng.native.add_collision_exclusion(gi['abd_id'], gj['fem_global_id'])
            eng.native.add_collision_exclusion(gi['fem_global_id'], gj['abd_id'])
            eng.native.add_collision_exclusion(gi['fem_global_id'], gj['fem_global_id'])

    eng.finalize()

    if int(os.environ.get("CASE37_DISABLE_GRAVITY", "1")):
        for arm_id in arm_ids:
            eng.native.set_body_apply_gravity(arm_id, False)
        for g in grippers:
            eng.native.set_body_apply_gravity(g['abd_id'], False)
    for g in grippers:
        eng.native.set_fixed_joint_strength(g['fj_idx'], fj_kappa)
    eng.native.set_max_revolute_step_per_frame(
        float(os.environ.get("CASE37_MAX_RAD_PER_FRAME", "0.5")))

    robot = Robot(eng)
    print(f"[case37] {len(robot.revolute_joints)} revolute, "
          f"{len(robot.prismatic_joints)} prismatic", flush=True)
    print(f"[case37] finalized\n", flush=True)

    auto_n = int(os.environ.get("AUTO_STEP", "0"))
    if auto_n > 0:
        for i in range(auto_n):
            t0 = time.perf_counter()
            eng.step()
            print(f"[case37] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms",
                  flush=True)
        return

    # ---- GUI ----
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    verts_world = eng.get_vertices()
    all_faces = eng.get_surface_faces()
    recs = eng.get_load_records()
    finger_id_set = {g['finger_id'] for g in grippers}
    abd_id_set = {g['abd_id'] for g in grippers}
    fem_local_id_set = {g['fem_rec'].body_offset for g in grippers}

    body_meshes = []
    for r in recs:
        v_off, v_end = r.vertex_offset, r.vertex_offset + r.vertex_count
        face_mask = np.all((all_faces >= v_off) & (all_faces < v_end), axis=1)
        if not face_mask.any(): continue
        faces_local = all_faces[face_mask] - v_off
        base = (r.label or f"body{r.body_offset}").replace(" ", "_")[:24]
        name = f"{base}_b{r.body_offset}"
        if r.body_type == 1 and r.body_offset in fem_local_id_set:
            color = (0.85, 0.85, 0.92)
        elif r.body_type == 0 and r.body_offset in abd_id_set:
            color = (0.2, 0.95, 0.3)
        elif r.body_type == 0 and r.body_offset in finger_id_set:
            color = (0.3, 0.5, 0.85)
        else:
            color = (0.55, 0.55, 0.6)
        m = ps.register_surface_mesh(name, verts_world[v_off:v_end], faces_local,
                                      smooth_shade=True)
        m.set_color(color)
        body_meshes.append((m, v_off, v_end))
        if r.body_type == 1 and r.body_offset in fem_local_id_set:
            region = np.zeros(r.vertex_count, dtype=np.float32)
            region[rigid_v_idx] = 1.0
            m.add_scalar_quantity("rigid (red)", region, enabled=True, cmap='reds')

    auto_run = bool(int(os.environ.get("GUI_AUTO_RUN", "0")))
    auto_quit_after = int(os.environ.get("GUI_QUIT_AFTER_STEPS", "0"))
    state = dict(running=auto_run, step_count=0, last_step_ms=0.0)

    def do_step():
        t0 = time.perf_counter()
        eng.step()
        state['last_step_ms'] = (time.perf_counter() - t0) * 1000.0
        state['step_count'] += 1
        cur_verts = eng.get_vertices()
        for m, v0, v1 in body_meshes:
            m.update_vertex_positions(cur_verts[v0:v1])
        if auto_quit_after and state['step_count'] >= auto_quit_after:
            print(f"[case37] auto-quit after {auto_quit_after} GUI steps", flush=True)
            os._exit(0)

    def callback():
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((520, 0), psim.ImGuiCond_Once)
        psim.Begin("case_37 — finger-aligned hybrid grippers")
        psim.Text(f"step #{state['step_count']}: {state['last_step_ms']:.1f} ms")
        psim.Text(f"URDF: {n_urdf} ABD, hybrid: {len(grippers)} ABD + {len(grippers)} FEM")
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
        if robot.prismatic_joints:
            psim.Text(f"Prismatic Joints (gripper open/close, {len(robot.prismatic_joints)})")
            psim.Separator()
            for i, ji in enumerate(robot.prismatic_joints):
                lo_mm = ji.lower_limit * 1000.0
                hi_mm = ji.upper_limit * 1000.0
                cur_mm = robot.get_prismatic_target_mm(i)
                chg, new_val = psim.SliderFloat(f"{ji.name} (mm)", cur_mm, lo_mm, hi_mm)
                if chg: robot.set_prismatic_position(i, new_val, millimeters=True)
        if robot.revolute_joints:
            psim.Spacing()
            psim.Text(f"Revolute Joints ({len(robot.revolute_joints)})")
            psim.Separator()
            for i, ji in enumerate(robot.revolute_joints):
                cur = robot.get_revolute_target_deg(i)
                chg, new_val = psim.SliderFloat(ji.name, cur, ji.lower_limit_deg, ji.upper_limit_deg)
                if chg: robot.set_revolute_position(i, new_val, degree=True)
        psim.End()
        if state['running']:
            do_step()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
