#!/usr/bin/env python3
"""case_36 — D variant: hybrid FEM directly stitched to URDF finger ABDs.

Sibling of case_36_quad_hybrid_grippers.py (the A variant).  Both attach
4 hybrid grippers to the URDF finger backbones.  The architecture differs:

  A: URDF finger ABD → fixed_joint → hybrid rigid sub-mesh ABD → stitch →
     hybrid FEM unified mesh.  Three-layer; FEM never feels the URDF
     finger directly, only the intermediate rigid ABD.

  D (this file): URDF finger ABD → stitch (NN-matched) → hybrid FEM
     unified mesh DIRECTLY.  Two-layer; the hybrid mesh's rigid sub-set
     is dropped (only its FEM part is loaded).  Stitches use case_27's
     nearest-neighbor matching (FEM rigid-subset verts → finger ABD verts)
     with rest_offset = current spatial offset to avoid yanking all FEM
     verts onto a single ABD anchor point.

Trade-off vs A:
  + Simpler: no extra ABD bodies, no fixed_joints, fewer Hessian DOFs.
  + Closer to case_27 baseline (which is well-tuned for this URDF).
  - URDF finger ABD feels FEM reactive force directly through stitch
    (no intermediate buffer body) — at high softpad young the joint
    chain is more directly affected.

GUI: 4 prismatic finger_joints + 16 revolute joints (same as A).
"""
import sys, os, math, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import polyscope as ps
import polyscope.imgui as psim

from stiff_physics import Engine, Config
from stiff_physics.robot import Robot


URDF_PATH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_softgripper.urdf"
ORIGINAL_URDF = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_full.urdf"
RIGID_REMAP  = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"
UNIFIED_NPZ  = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"

ARM_SCALE = 0.3
FINGER_LABELS = [
    'left_arm_leftfinger',  'left_arm_rightfinger',
    'right_arm_leftfinger', 'right_arm_rightfinger',
]
SOFT_LABELS = [
    'left_arm_leftfinger_soft_material',
    'left_arm_rightfinger_soft_material',
    'right_arm_leftfinger_soft_material',
    'right_arm_rightfinger_soft_material',
]


def make_arm_tf(scale: float) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix()
    tf[1, 3] = -0.9
    return tf


def _parse_xyz_rpy(s):
    return np.array([float(x) for x in s.split()], dtype=float)


def parse_link_world_tf(urdf_path: str, link_name: str, base_tf: np.ndarray) -> np.ndarray:
    """case_27 / case_35-style URDF tree FK at startup (movable joints=0),
    composed with the link's <visual>/<collision> mesh origin.  Used to
    place the hybrid mesh at the same world location as case_27's softpad."""
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
            T = local @ T; cur = j['parent']
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
        T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix(); T[:3, 3] = xyz
        return T
    return base_tf @ world_tf(link_name) @ visual_origin_in_link(link_name)


def main():
    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100,
        # case_27's calibrated value — stitch K = soft_motion_rate.  D variant
        # uses the same rate so spring strength is comparable.
        soft_motion_rate=float(os.environ.get("CASE36D_SOFT_RATE", "1e4")),
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
        joint_strength_ratio=100.0,
        revolute_driving_strength_ratio=float(os.environ.get("CASE36D_PD_K", "100")),
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.5,
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 64.0
    eng = Engine(cfg)
    print("\n[case36-D] === ridgeback + 4 hybrid FEM (direct stitch) ===",
          flush=True)

    # --- 1. Load URDF (37 ABD bodies including 4 fingers) ---
    arm_tf = make_arm_tf(ARM_SCALE)
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})
    n_urdf = eng.abd_body_count
    urdf_recs = list(eng.get_load_records())
    finger_recs = {r.label: r for r in urdf_recs if r.body_type == 0
                   and r.label in FINGER_LABELS}
    if len(finger_recs) != 4:
        raise RuntimeError(f"expected 4 finger ABDs, got {len(finger_recs)}: "
                           f"{list(finger_recs.keys())}")
    print(f"[case36-D] URDF: {n_urdf} ABD bodies", flush=True)

    for b in range(n_urdf):
        eng.add_ground_collision_skip(b)

    # --- 2. Per-finger hybrid transform ---
    # Default "finger_full" (NEW — same fix as case_37): T = engine_link_T
    # × collision_origin.  Matches URDF importer's full_T applied to
    # finger ABD body, no missing collision_origin offset.
    # Other modes preserved for comparison; "procrustes" still works
    # around STRATEGY_F's frame mismatch.
    tf_mode = os.environ.get("CASE36D_TF_MODE", "finger_full").lower()
    _co_rpy = np.array([-1.57079632679, 0.20245819348, -1.57079632679])
    _co_xyz = np.array([-0.0165, 0.0165, 0.12773331296])
    collision_origin = np.eye(4)
    collision_origin[:3, :3] = Rotation.from_euler('xyz', _co_rpy).as_matrix()
    collision_origin[:3, 3] = _co_xyz
    rigid_remap_data = np.load(RIGID_REMAP, allow_pickle=True)
    rigid_v_idx_glb = rigid_remap_data['rigid_v_idx']
    hybrid_verts_local = np.load(UNIFIED_NPZ)['vertices']
    rigid_centroid_local = hybrid_verts_local[rigid_v_idx_glb].mean(axis=0)
    rigid_local_verts = hybrid_verts_local[rigid_v_idx_glb]

    def _scaled_procrustes(A_local, B_world):
        cA = A_local.mean(0); cB = B_world.mean(0)
        Ac = A_local - cA; Bc = B_world - cB
        H = Ac.T @ Bc
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        s = (np.array([S[0], S[1], S[2]*d]).sum()) / (Ac**2).sum()
        t = cB - s * R @ cA
        T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
        return T

    import trimesh as _tm
    _finger_obj_path = ("/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/"
                        "ridgeback_dual_panda_soft/meshes/plate/visual/"
                        "soft_hard_segmenation/finger_clean.obj")
    _finger_mesh = _tm.load(_finger_obj_path, process=False)
    _comps = _finger_mesh.split(only_watertight=False)
    _largest_comp = max(_comps, key=lambda c: len(c.vertices))
    _orig_v = np.asarray(_finger_mesh.vertices)
    _orig_lookup = {tuple(np.round(p, 8)): i for i, p in enumerate(_orig_v)}
    _lower_seg_idx = np.array([
        _orig_lookup[tuple(np.round(p, 8))]
        for p in np.asarray(_largest_comp.vertices)
        if tuple(np.round(p, 8)) in _orig_lookup
    ], dtype=int)
    print(f"[case36-D] finger lower seg: {len(_lower_seg_idx)}/{len(_orig_v)} verts",
          flush=True)

    all_v_pre = eng.native.get_vertices_host()
    finger_lower_world_center = {}
    finger_lower_world_verts = {}
    for label in FINGER_LABELS:
        rec = finger_recs[label]
        v = all_v_pre[rec.vertex_offset:rec.vertex_offset + rec.vertex_count]
        if len(v) == len(_orig_v) and len(_lower_seg_idx) > 0:
            lv = v[_lower_seg_idx]
            finger_lower_world_center[label] = (lv.min(0) + lv.max(0)) * 0.5
            finger_lower_world_verts[label] = lv
        else:
            finger_lower_world_center[label] = (v.min(0) + v.max(0)) * 0.5
            finger_lower_world_verts[label] = v

    print(f"[case36-D] CASE36D_TF_MODE={tf_mode}", flush=True)
    from scipy.spatial import cKDTree as _KDTree
    soft_T = {}
    for finger_label, soft_label in zip(FINGER_LABELS, SOFT_LABELS):
        finger_T_engine = eng.native.get_urdf_link_transform(finger_label)
        soft_link_T = parse_link_world_tf(ORIGINAL_URDF, soft_label, arm_tf)
        lower_ctr = finger_lower_world_center[finger_label]
        if tf_mode == "finger_full":
            T = finger_T_engine @ collision_origin
        elif tf_mode == "soft_link":
            T = soft_link_T
        elif tf_mode == "centroid":
            T = soft_link_T.copy()
            T[:3, 3] = (finger_T_engine[:3, 3]
                        - soft_link_T[:3, :3] @ rigid_centroid_local)
        elif tf_mode == "finger":
            T = finger_T_engine
        elif tf_mode == "lower_seg":
            T = soft_link_T.copy()
            T[:3, 3] = lower_ctr - soft_link_T[:3, :3] @ rigid_centroid_local
        else:  # "procrustes"
            T_init = soft_link_T.copy()
            T_init[:3, 3] = lower_ctr - soft_link_T[:3, :3] @ rigid_centroid_local
            rigid_world_init = (T_init[:3,:3] @ rigid_local_verts.T).T + T_init[:3,3]
            tree = _KDTree(finger_lower_world_verts[finger_label])
            _, nn_idx = tree.query(rigid_world_init)
            target_world = finger_lower_world_verts[finger_label][nn_idx]
            T = _scaled_procrustes(rigid_local_verts, target_world)
            fit_world = (T[:3,:3] @ rigid_local_verts.T).T + T[:3,3]
            residual = np.linalg.norm(fit_world - target_world, axis=1)
            print(f"[case36-D/procrustes] {finger_label}: scale={np.linalg.norm(T[:3,0]):.4f}  "
                  f"fit_residual mean={residual.mean()*1000:.2f}mm max={residual.max()*1000:.2f}mm",
                  flush=True)
        soft_T[finger_label] = T
        print(f"[case36-D/diag] {finger_label}: load_t={T[:3,3]}",
              flush=True)

    # --- 3. Load 4 hybrid FEM unified meshes (one per finger) ---
    # Hybrid mesh's rigid sub-mesh is intentionally NOT loaded — only its
    # FEM part participates.  The rigid_v_idx subset (150 verts) marks the
    # vertices that originally were "rigid" — these are the ones we stitch
    # to the finger ABD; the rest stay free as the soft outer envelope.
    fem_young = float(os.environ.get("CASE36D_FEM_YOUNG", "1e6"))
    rigid_remap = np.load(RIGID_REMAP, allow_pickle=True)
    rigid_v_idx = rigid_remap['rigid_v_idx']

    hybrid_data = np.load(UNIFIED_NPZ)
    hybrid_verts = np.ascontiguousarray(hybrid_data['vertices'], dtype=np.float64)
    hybrid_tets  = np.ascontiguousarray(hybrid_data['tets'], dtype=np.int32)

    grippers = []  # {label, finger_id, gripper_T, fem_rec, fem_v_off}
    for label in FINGER_LABELS:
        finger_rec = finger_recs[label]
        T = soft_T[label]
        eng.native.load_mesh_from_data(
            hybrid_verts, hybrid_tets, 4, 3, 1, T, fem_young, 0)
        fem_rec = eng.get_load_records()[-1]
        grippers.append(dict(
            label=label, finger_id=finger_rec.body_offset,
            gripper_T=T,
            fem_rec=fem_rec, fem_v_off=fem_rec.vertex_offset,
        ))

    # FEM global ids (n_abd_total stable since no new ABDs were loaded)
    n_abd_total = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    for g in grippers:
        g['fem_global_id'] = n_abd_total + g['fem_rec'].body_offset

    # --- 4. Direct stitch: FEM rigid-subset verts → NN finger ABD verts ---
    # case_27 NN logic: for each rigid-subset FEM vertex, find nearest
    # finger ABD vertex; rest_offset = current (fem_pos - abd_pos).
    # Threshold prunes outliers (FEM vert with no finger geometry nearby).
    all_verts = eng.native.get_vertices_host()
    # Hybrid mesh sits at soft_material location (top of finger), but
    # finger ABD vertices span the whole ~3cm finger body — most are
    # below the softpad zone.  100mm = generous; case_27 uses 20mm only
    # because its softpad mesh and finger backbone are nearly coincident.
    nn_thresh = float(os.environ.get("CASE36D_NN_THRESH_MM", "100.0")) / 1000.0
    total_pairs = 0
    for g in grippers:
        f_rec = finger_recs[g['label']]
        finger_verts = all_verts[f_rec.vertex_offset:
                                 f_rec.vertex_offset + f_rec.vertex_count]
        fem_verts = all_verts[g['fem_v_off']:
                              g['fem_v_off'] + g['fem_rec'].vertex_count]
        # Only the rigid-subset FEM verts get stitched (these are the
        # "interface" verts that originally were rigid in unified mesh).
        rigid_fem_pos = fem_verts[rigid_v_idx]
        f_tree = cKDTree(finger_verts)
        d, idx = f_tree.query(rigid_fem_pos)
        n_pairs = 0
        for k in range(len(rigid_v_idx)):
            if d[k] >= nn_thresh:
                continue
            fem_global = g['fem_v_off'] + int(rigid_v_idx[k])
            abd_global = f_rec.vertex_offset + int(idx[k])
            rest_off = (rigid_fem_pos[k] - finger_verts[idx[k]]).tolist()
            eng.add_stitch_spring(
                fem_global, abd_global, g['finger_id'],
                rest_offset_world=rest_off)
            n_pairs += 1
        g['n_stitch'] = n_pairs
        total_pairs += n_pairs
        print(f"[case36-D] {g['label']}: {n_pairs}/{len(rigid_v_idx)} stitch "
              f"(d_min={d.min()*1000:.2f}mm, d_used_mean={1000*d[d<nn_thresh].mean():.2f}mm "
              f"if matched else N/A)", flush=True)
    print(f"[case36-D] total stitch pairs: {total_pairs}", flush=True)

    # --- 5. Collision exclusions ---
    # Each FEM ↔ ALL arm ABD bodies (overlap with own finger; don't push
    # other arm bodies).  Plus: cross-gripper FEM ↔ FEM within same arm
    # pair (left/right finger overlap at prismatic=0).
    arm_ids = [r.body_offset for r in urdf_recs if r.body_type == 0]
    for g in grippers:
        for arm_id in arm_ids:
            eng.native.add_collision_exclusion(g['fem_global_id'], arm_id)

    def _arm_prefix(label):
        return 'left' if label.startswith('left_') else 'right'
    for i, gi in enumerate(grippers):
        for gj in grippers[i+1:]:
            if _arm_prefix(gi['label']) == _arm_prefix(gj['label']):
                eng.native.add_collision_exclusion(
                    gi['fem_global_id'], gj['fem_global_id'])

    eng.finalize()

    # --- 6. Disable gravity for static-pose UI testing ---
    if int(os.environ.get("CASE36D_DISABLE_GRAVITY", "1")):
        for arm_id in arm_ids:
            eng.native.set_body_apply_gravity(arm_id, False)

    eng.native.set_max_revolute_step_per_frame(
        float(os.environ.get("CASE36D_MAX_RAD_PER_FRAME", "0.5")))

    robot = Robot(eng)
    prismatic_strength = float(os.environ.get("CASE36D_PRISMATIC_K", "10.0"))
    for i in range(len(robot.prismatic_joints)):
        eng.native.set_prismatic_strength(i, prismatic_strength)
    print(f"[case36-D] {len(robot.revolute_joints)} revolute, "
          f"{len(robot.prismatic_joints)} prismatic "
          f"(prismatic strength × {prismatic_strength})", flush=True)
    print(f"[case36-D] finalized\n", flush=True)

    auto_n = int(os.environ.get("AUTO_STEP", "0"))
    if auto_n > 0:
        for i in range(auto_n):
            t0 = time.perf_counter()
            eng.step()
            print(f"[case36-D] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms",
                  flush=True)
        return

    # --- GUI ---
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    verts_world = eng.get_vertices()
    all_faces = eng.get_surface_faces()
    recs = eng.get_load_records()

    finger_id_set = {g['finger_id'] for g in grippers}
    fem_local_id_set = {g['fem_rec'].body_offset for g in grippers}

    body_meshes = []
    for r in recs:
        v_off = r.vertex_offset
        v_end = v_off + r.vertex_count
        face_mask = np.all((all_faces >= v_off) & (all_faces < v_end), axis=1)
        if not face_mask.any():
            continue
        faces_local = all_faces[face_mask] - v_off
        # 4 hybrid FEM bodies share the same mesh file (label collides);
        # append body_offset so polyscope's registry stays unique.
        base = (r.label or f"body{r.body_offset}").replace(" ", "_")[:24]
        name = f"{base}_b{r.body_offset}"
        if r.body_type == 1 and r.body_offset in fem_local_id_set:
            color = (0.85, 0.85, 0.92)
        elif r.body_type == 0 and r.body_offset in finger_id_set:
            color = (0.3, 0.5, 0.85)   # blue — finger
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
            print(f"[case36-D] auto-quit after {auto_quit_after} GUI steps",
                  flush=True)
            os._exit(0)

    def callback():
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((520, 0), psim.ImGuiCond_Once)
        psim.Begin("case_36_D — direct stitch (FEM ↔ finger)")
        psim.Text(f"step #{state['step_count']}: {state['last_step_ms']:.1f} ms")
        psim.Text(f"URDF bodies: {n_urdf}, hybrid: 0 ABD + {len(grippers)} FEM")
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
            left_idxs  = [i for i, ji in enumerate(robot.prismatic_joints)
                          if ji.name.startswith('left_arm')]
            right_idxs = [i for i, ji in enumerate(robot.prismatic_joints)
                          if ji.name.startswith('right_arm')]
            psim.Text("Gripper open/close (1 slider per arm — both fingers together)")
            psim.Separator()
            for label, idxs in [("left arm gripper", left_idxs),
                                ("right arm gripper", right_idxs)]:
                if not idxs: continue
                ji0 = robot.prismatic_joints[idxs[0]]
                lo_mm = ji0.lower_limit * 1000.0
                hi_mm = ji0.upper_limit * 1000.0
                cur_mm = robot.get_prismatic_target_mm(idxs[0])
                chg, new_val = psim.SliderFloat(f"{label} (mm)", cur_mm, lo_mm, hi_mm)
                if chg:
                    for i in idxs:
                        robot.set_prismatic_position(i, new_val, millimeters=True)

        if robot.revolute_joints:
            psim.Spacing()
            psim.Text(f"Revolute Joints ({len(robot.revolute_joints)})")
            psim.Separator()
            for i, ji in enumerate(robot.revolute_joints):
                cur = robot.get_revolute_target_deg(i)
                chg, new_val = psim.SliderFloat(
                    ji.name, cur, ji.lower_limit_deg, ji.upper_limit_deg)
                if chg:
                    robot.set_revolute_position(i, new_val, degree=True)
        psim.End()

        if state['running']:
            do_step()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
