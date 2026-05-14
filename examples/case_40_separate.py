#!/usr/bin/env python3
"""case_40_separate — case_27 style at scale 1.0: COARSE softpad
(softgriper_part2.msh) loaded AS-IS as 4 separate FEM bodies, each
NN-stitched to its corresponding URDF finger ABD body.  No hybrid
unified mesh, no convex hull simplification — preserves the
original case_27 softpad mesh structure exactly.

Architecture (per finger, ×4):
    URDF chain → finger ABD (α, blue) ← prismatic open/close
                  ↓ stitch_spring (NN match, sub-sampled)
              softpad FEM body (γ, white)        ← case_27 part3.msh

NO intermediate hybrid_rigid ABD body, NO fixed_joint, NO convex hull.

Companion to case_40_unified.py (Strategy D PLC merge of finger.stl +
softpad outer surface, retetrahedralized).

Same scene as case_39: cup + shirt + ground.  ARM_SCALE=1.0.
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


# URDF with 4 finger ABDs + prismatic finger_joints (case_27 uses same one)
URDF_PATH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_softgripper.urdf"
# _full URDF has the *_soft_material child links with mesh-in-link origin —
# case_27 uses this to compute where each softpad SHOULD live in world
# (prismatic=0 baseline).  We do the same so the hybrid sits at the same
# location as case_27's softpad.
ORIGINAL_URDF = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_full.urdf"
# case_40 separate: softpad mesh AS-IS (case_27 part2 (12k tet/finger, mid density, full bottom))
SOFTPAD_MSH  = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/tetmesh/softgriper_part2.msh"
# Finger geometry (used as NN-stitch target via URDF importer's finger ABD)
FINGER_OBJ   = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/meshes/plate/visual/soft_hard_segmenation/finger_clean.obj"
CUP_MSH      = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/tetmesh/softgriper_cup.msh"
# Shirt: 2D FEM triangle mesh as in case_27_mobile_s1_hybrid.py
SHIRT_OBJ    = "triMesh/shirt_6436v.obj"

ARM_SCALE = 1.0   # case_39: full scale (was 0.3 in case_38)
# parallel arrays: finger ABD label vs the soft_material child link whose
# world transform we use to place the hybrid mesh.
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
    tf[1, 3] = -3.0   # case_39: was -0.9 at scale 0.3, scaled 3.33×
    return tf




def _parse_xyz_rpy(s):
    return np.array([float(x) for x in s.split()], dtype=float)


def parse_link_world_tf(urdf_path: str, link_name: str, base_tf: np.ndarray) -> np.ndarray:
    """Walk URDF joint tree from link → root, accumulating local joint
    origins (assumes all movable joints at 0 — true for prismatic at
    startup).  Then compose the link's <visual>/<collision> mesh origin.
    Result: world transform of the link's mesh at startup.

    Direct port of case_27.parse_soft_material_world_tfs's per-link logic.
    """
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

    return base_tf @ world_tf(link_name) @ visual_origin_in_link(link_name)


def main():
    # Joint stiffness:
    #   CASE38_ALL_JOINT_K=K  → set BOTH revolute_driving_strength_ratio
    #     AND joint_strength_ratio (used by prismatic/fixed) to K, AND
    #     reset prismatic per-joint multiplier to 1.0 → effective K_revolute
    #     == effective K_prismatic == K * mass.  Use this to test "all
    #     joints same strength" scenarios (e.g. CASE38_ALL_JOINT_K=2000).
    #   Otherwise the per-channel CASE36_PD_K / CASE38_JOINT_K /
    #   CASE36_PRISMATIC_K env vars apply independently.
    all_K = float(os.environ.get("CASE38_ALL_JOINT_K", "0"))
    if all_K > 0:
        joint_K        = all_K
        revolute_K     = all_K
        prismatic_mult = 1.0
        print(f"[case38] CASE38_ALL_JOINT_K={all_K} — all joints same K", flush=True)
    else:
        joint_K        = float(os.environ.get("CASE38_JOINT_K", "100"))
        revolute_K     = float(os.environ.get("CASE36_PD_K", "100"))
        # Default prismatic effective K = 100 × 15 = 1500·mass — strong
        # enough that the slider doesn't visibly lag.
        prismatic_mult = float(os.environ.get("CASE36_PRISMATIC_K", "15"))

    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100,
        soft_motion_rate=float(os.environ.get("CASE36_SOFT_RATE", "1e4")),
        # case_40 uses original softpad mesh (sub-mm edges) → need
        # case_27's relative_dhat=1e-4 to avoid CCD overflow.
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-4,
        joint_strength_ratio=joint_K,
        revolute_driving_strength_ratio=revolute_K,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-1.67,   # case_39 full-scale
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 64.0
    eng = Engine(cfg)
    print("\n[case36] === ridgeback + 4 hybrid grippers ===", flush=True)

    # --- 1. Load URDF (37 ABD bodies including 4 fingers) ---
    arm_tf = make_arm_tf(ARM_SCALE)
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})
    n_urdf = eng.abd_body_count
    urdf_recs = list(eng.get_load_records())
    finger_recs = {r.label: r for r in urdf_recs if r.body_type == 0
                   and r.label in FINGER_LABELS}
    if len(finger_recs) != 4:
        raise RuntimeError(
            f"expected 4 finger ABDs, got {len(finger_recs)}: "
            f"{list(finger_recs.keys())}")
    print(f"[case36] URDF: {n_urdf} ABD bodies, finger offsets: "
          f"{[(k, v.body_offset) for k, v in finger_recs.items()]}", flush=True)

    for b in range(n_urdf):
        eng.add_ground_collision_skip(b)

    # --- 2. Per-finger world transforms — case_27 style ---
    # The hybrid mesh's footprint already matches a single softpad
    # (4cm × 10cm × 2.3cm at unit scale; with ARM_SCALE 0.3 → ~1.2cm ×
    # 3cm × 0.7cm).  We DON'T add an extra gripper_scale — case_27's
    # softpad mesh has the same property and uses arm_tf directly.
    #
    # Place each hybrid where case_27's softpad lives: at the corresponding
    # *_soft_material link's mesh origin (in finger's local frame, with
    # rpy=(-pi/2, 0.20, -pi/2), xyz=(-0.0165, 0.0165, 0.128)).  Use
    # _full URDF (the only one that has soft_material links) for parsing,
    # then base_tf = arm_tf so SCALE is already baked in.
    # Compute per-finger hybrid transform.
    # case_40 separate: simple finger_full transform (engine_link_T ×
    # collision_origin) — matches what URDF importer applied to finger
    # ABD body, so softpad mesh lands at the case_27 soft_material
    # world position.  No procrustes / centroid hacks needed since
    # softpad has no rigid sub-mesh to align.
    _co_rpy = np.array([-1.57079632679, 0.20245819348, -1.57079632679])
    _co_xyz = np.array([-0.0165, 0.0165, 0.12773331296])
    collision_origin = np.eye(4)
    collision_origin[:3, :3] = Rotation.from_euler('xyz', _co_rpy).as_matrix()
    collision_origin[:3, 3] = _co_xyz
    soft_T = {}
    for finger_label in FINGER_LABELS:
        finger_T_engine = eng.native.get_urdf_link_transform(finger_label)
        soft_T[finger_label] = finger_T_engine @ collision_origin
        print(f"[case40] {finger_label}: softpad load t = {soft_T[finger_label][:3, 3]}",
              flush=True)

    # --- 3. case_40 separate: load softpad as 4 FEM bodies (no hybrid_rigid) ---
    fem_young = float(os.environ.get("CASE40_FEM_YOUNG", "1e6"))

    # Track per-finger info; no hybrid_rigid ABD needed.
    grippers = []
    for label in FINGER_LABELS:
        finger_rec = finger_recs[label]
        grippers.append(dict(
            label=label, finger_id=finger_rec.body_offset,
            gripper_T=soft_T[label],
        ))

    # Pass 1.5: load cup (ABD) — must be BEFORE any FEM.  Bigger cup
    # (case_27 default scale 0.2) sitting on the ground half-plane.
    # Ground is at y = ground_offset = -0.5.
    # Cup placed well OFFSET from the shirt (z=-0.30) so the shirt
    # falls clear of the cup — they sit side-by-side with comfortable
    # clearance, still within the gripper work area.
    # case_39: full-scale cup (×2.67 vs case_38)
    cup_scale = float(os.environ.get("CASE39_CUP_SCALE", "0.8"))
    cup_xyz   = np.array([float(s) for s in
        os.environ.get("CASE39_CUP_XYZ", "0.67,-1.00,-1.00").split(",")])
    cup_T = np.eye(4)
    cup_T[:3, :3] *= cup_scale
    cup_T[:3, 3] = cup_xyz
    eng.load_mesh(CUP_MSH, dimensions=3, body_type="ABD",
                  transform=cup_T, young_modulus=1e8, boundary_type="Free")
    cup_rec = eng.get_load_records()[-1]
    cup_id = cup_rec.body_offset
    print(f"[case38] cup body_id={cup_id} verts={cup_rec.vertex_count} "
          f"scale={cup_scale} at {cup_xyz}", flush=True)

    # Pass 2: load 4 softpad FEM bodies (case_27 part3.msh AS-IS)
    for g in grippers:
        eng.load_mesh(SOFTPAD_MSH, dimensions=3, body_type="FEM",
                      transform=g['gripper_T'], young_modulus=fem_young)
        fem_rec = eng.get_load_records()[-1]
        g['fem_rec'] = fem_rec
        g['fem_v_off'] = fem_rec.vertex_offset

    # Pass 2.5: load shirt (case_27 style — 2D FEM triangle mesh)
    # case_39: full-scale shirt (×2 vs case_38's 0.5; ×3.33 position)
    shirt_scale = float(os.environ.get("CASE39_SHIRT_SCALE", "1.0"))
    shirt_xyz   = np.array([float(s) for s in
        os.environ.get("CASE39_SHIRT_XYZ", "0.67,0.00,0.00").split(",")])
    shirt_T = np.eye(4)
    shirt_T[:3, :3] *= shirt_scale
    shirt_T[:3, 3] = shirt_xyz
    eng.load_mesh(SHIRT_OBJ, dimensions=2, body_type="FEM",
                  transform=shirt_T,
                  young_modulus=float(os.environ.get("CASE38_SHIRT_YOUNG", "1e2")))
    shirt_rec = eng.get_load_records()[-1]
    print(f"[case38] shirt fem_local_id={shirt_rec.body_offset} "
          f"verts={shirt_rec.vertex_count} scale={shirt_scale} at {shirt_xyz}",
          flush=True)

    # Compute FEM global ids now (n_abd_total stable after all ABD loaded)
    n_abd_total = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    for g in grippers:
        g['fem_global_id'] = n_abd_total + g['fem_rec'].body_offset

    # Pass 3: NN-stitch each softpad FEM vert to nearest finger ABD vert.
    # case_27 pattern: subsample softpad verts (every Nth) + NN-match
    # within threshold; rest_offset = current spatial offset so spring
    # rest length = current geometry (avoids yanking all FEM verts onto
    # one ABD anchor point).
    sub_n     = int(os.environ.get("CASE40_STITCH_SUB", "8"))
    nn_thresh = float(os.environ.get("CASE40_NN_THRESH_MM", "30.0")) / 1000.0
    all_v_after = eng.native.get_vertices_host()
    total_pairs = 0
    for g in grippers:
        f_rec = finger_recs[g['label']]
        finger_verts = all_v_after[f_rec.vertex_offset:
                                   f_rec.vertex_offset + f_rec.vertex_count]
        fem_verts = all_v_after[g['fem_v_off']:
                                g['fem_v_off'] + g['fem_rec'].vertex_count]
        # Sub-sample softpad verts to bound stitch count
        e_sub_idx = np.arange(0, len(fem_verts), sub_n)
        e_sub = fem_verts[e_sub_idx]
        f_tree = cKDTree(finger_verts)
        d, idx = f_tree.query(e_sub)
        n_pairs = 0
        for k in range(len(e_sub)):
            if d[k] >= nn_thresh: continue
            i_e = int(e_sub_idx[k])
            j_f = int(idx[k])
            fem_global = g['fem_v_off'] + i_e
            abd_global = f_rec.vertex_offset + j_f
            rest_off = (fem_verts[i_e] - finger_verts[j_f]).tolist()
            eng.add_stitch_spring(
                fem_global, abd_global, g['finger_id'],
                rest_offset_world=rest_off)
            n_pairs += 1
        g['n_stitch'] = n_pairs
        total_pairs += n_pairs
        print(f"[case40] {g['label']}: finger={g['finger_id']}, "
              f"fem={g['fem_rec'].body_offset}, "
              f"{n_pairs}/{len(e_sub)} stitch (sub={sub_n}, thresh={nn_thresh*1000:.0f}mm)",
              flush=True)
    print(f"[case40] total stitch pairs: {total_pairs}", flush=True)

    # --- 4. Collision exclusions ---
    # For each gripper:
    #   (a) hybrid_abd ↔ own FEM (overlap by construction)
    #   (b) hybrid_abd ↔ own finger (overlap by construction)
    #   (c) hybrid_abd ↔ ALL other arm ABD bodies (don't push arm around)
    #   (d) FEM        ↔ ALL other arm ABD bodies (same)
    # Plus: hybrid pair within the same arm (left/right finger of one
    # hand) starts geometrically overlapped at prismatic=0 — exclude
    # them mutually so IPC doesn't reject the initial config.
    # case_40 separate: NO hybrid_rigid ABD; just exclude FEM softpad
    # vs all arm ABD bodies (overlap with own finger; don't push other
    # arm bodies).
    arm_ids = [r.body_offset for r in urdf_recs if r.body_type == 0]
    for g in grippers:
        for arm_id in arm_ids:
            eng.native.add_collision_exclusion(g['fem_global_id'], arm_id)

    # Exclude non-finger arm bodies from cup (don't bash the cup with arm
    # link/hand collision OBBs — only the finger gripper should touch it).
    finger_offsets = {g['finger_id'] for g in grippers}
    for arm_id in arm_ids:
        if arm_id in finger_offsets:
            continue
        eng.native.add_collision_exclusion(arm_id, cup_id)

    # SHIRT collision policy (case_27 fast-path style): exclude EVERY
    # URDF arm body (links + finger ABDs) from shirt collision detection.
    # Only the hybrid gripper components (rigid ABD + FEM softpad) collide
    # with the shirt.  This avoids costly contact processing against the
    # blocky OBB arm geometry — shirt only "sees" the soft gripper.
    n_abd_total_for_shirt = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    shirt_global_id = n_abd_total_for_shirt + shirt_rec.body_offset
    for arm_id in arm_ids:
        eng.native.add_collision_exclusion(arm_id, shirt_global_id)
    # Hybrid rigid ABD ↔ shirt KEPT (gripper closes on the shirt).
    # Hybrid FEM ↔ shirt KEPT (softpad presses the shirt).
    # Cup ↔ shirt: shirt may settle on cup — keep collision (default).
    print(f"[case38] shirt global_id={shirt_global_id}: excluded vs all "
          f"{len(arm_ids)} arm ABD bodies; collides only with hybrid grippers + cup",
          flush=True)

    # Hybrid FEM ↔ ground half-plane: default SKIP (env=0).  Set
    # CASE38_FEM_GROUND_COLLISION=1 to enable hybrid softpad ↔ ground
    # contact.  Useful so softpad doesn't grab the ground when arm is
    # near rest pose.  (URDF arm bodies already ground_collision_skip'd
    # at Pass 1.)
    fem_ground_collide = int(os.environ.get("CASE38_FEM_GROUND_COLLISION", "0"))
    if not fem_ground_collide:
        for g in grippers:
            eng.add_ground_collision_skip(g['fem_global_id'])
        print(f"[case38] hybrid FEM × ground collision: SKIPPED (default; "
              f"set CASE38_FEM_GROUND_COLLISION=1 to enable)", flush=True)
    else:
        print(f"[case38] hybrid FEM × ground collision: ENABLED", flush=True)

    # Cross-gripper FEM exclusion within same arm (overlap at prismatic=0)
    def _arm_prefix(label):
        return 'left' if label.startswith('left_') else 'right'
    for i, gi in enumerate(grippers):
        for gj in grippers[i+1:]:
            if _arm_prefix(gi['label']) != _arm_prefix(gj['label']):
                continue
            eng.native.add_collision_exclusion(
                gi['fem_global_id'], gj['fem_global_id'])

    eng.finalize()

    # --- 5. Disable gravity on URDF arm (no hybrid_rigid in separate variant) ---
    if int(os.environ.get("CASE36_DISABLE_GRAVITY", "1")):
        for arm_id in arm_ids:
            eng.native.set_body_apply_gravity(arm_id, False)

    # (separate variant has no fixed_joint to override.)

    eng.native.set_max_revolute_step_per_frame(
        float(os.environ.get("CASE36_MAX_RAD_PER_FRAME", "0.5")))

    robot = Robot(eng)
    # --- 7. Bump prismatic strength so finger keeps up with slider ---
    # Default per-joint multiplier is 1.0; effective K = soft_motion_rate
    # × strength × (m_p+m_c).  With finger masses small relative to
    # arm/hybrid attached payload, the default lags noticeably.  10× is
    # a reasonable starting bump matching revolute responsiveness.
    # Use the prismatic multiplier resolved earlier (= 1.0 if
    # CASE38_ALL_JOINT_K is set, otherwise CASE36_PRISMATIC_K).
    for i, ji in enumerate(robot.prismatic_joints):
        eng.native.set_prismatic_strength(i, prismatic_mult)
    print(f"[case38] {len(robot.revolute_joints)} revolute (global K={revolute_K}), "
          f"{len(robot.prismatic_joints)} prismatic "
          f"(global K={joint_K} × per-joint {prismatic_mult} = effective {joint_K*prismatic_mult})",
          flush=True)
    print(f"[case38] finalized\n", flush=True)

    auto_n = int(os.environ.get("AUTO_STEP", "0"))
    if auto_n > 0:
        for i in range(auto_n):
            t0 = time.perf_counter()
            eng.step()
            print(f"[case36] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms",
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
    # case_40 separate: no hybrid_rigid → no green ABD set
    fem_local_id_set = {g['fem_rec'].body_offset for g in grippers}
    shirt_local_id = shirt_rec.body_offset

    body_meshes = []
    for r in recs:
        v_off = r.vertex_offset
        v_end = v_off + r.vertex_count
        face_mask = np.all((all_faces >= v_off) & (all_faces < v_end), axis=1)
        if not face_mask.any():
            continue
        faces_local = all_faces[face_mask] - v_off
        # NOTE: 4 hybrid bodies all come from the same mesh file (label
        # collides), and 4 hybrid FEM bodies likewise.  Append body_offset
        # so polyscope's name-keyed registry stays unique — otherwise later
        # loads silently overwrite earlier ones, leading to update_vertex
        # _positions(150) on a "size 0" stub mesh and a polyscope crash.
        base = (r.label or f"body{r.body_offset}").replace(" ", "_")[:24]
        name = f"{base}_b{r.body_offset}"
        if r.body_type == 1 and r.body_offset in fem_local_id_set:
            color = (0.85, 0.85, 0.92)
        elif r.body_type == 1 and r.body_offset == shirt_local_id:
            color = (0.95, 0.85, 0.30)   # shirt — yellow
        elif r.body_type == 0 and r.body_offset in finger_id_set:
            color = (0.3, 0.5, 0.85)     # finger blue
        elif r.body_type == 0 and r.body_offset == cup_id:
            color = (0.85, 0.30, 0.30)   # cup — red
        else:
            color = (0.55, 0.55, 0.6)    # other arm grey
        m = ps.register_surface_mesh(name, verts_world[v_off:v_end], faces_local,
                                      smooth_shade=True)
        m.set_color(color)
        body_meshes.append((m, v_off, v_end))
        # case_40 separate: pure FEM softpad (no rigid sub-mesh) → no overlay

    # Optional: auto-run + auto-quit after N GUI steps for smoke tests
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
            print(f"[case36] auto-quit after {auto_quit_after} GUI steps",
                  flush=True)
            os._exit(0)

    def callback():
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((520, 0), psim.ImGuiCond_Once)
        psim.Begin("case_40 separate — coarse softpad FEM + finger ABD (case_27 style)")
        psim.Text(f"step #{state['step_count']}: {state['last_step_ms']:.1f} ms")
        psim.Text(f"URDF bodies: {n_urdf}, hybrid: {len(grippers)} ABD + {len(grippers)} FEM")
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
        # Prismatic — group by arm (left/right), one slider per arm drives
        # both fingers of that arm together (mirror open/close).
        if robot.prismatic_joints:
            left_idxs  = [i for i, ji in enumerate(robot.prismatic_joints)
                          if ji.name.startswith('left_arm')]
            right_idxs = [i for i, ji in enumerate(robot.prismatic_joints)
                          if ji.name.startswith('right_arm')]
            psim.Text(f"Gripper open/close (1 slider per arm — drives both fingers together)")
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
