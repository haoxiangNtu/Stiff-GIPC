#!/usr/bin/env python3
"""case_42 — case_39 with simplified shirt (831v instead of 6436v).

Goal: isolate whether GUI lag comes from arm+gripper compute or from shirt FEM
(6436v shirt was likely dominating contact + cloth elasticity cost).  If case_42
feels much smoother than case_39 → shirt is the bottleneck.  If similar → cost
is in arm/gripper itself (Newton + contact resolution).

Differences from case_39:
  SHIRT_OBJ              shirt_6436v.obj  →  shirt_831v.obj  (7.7× fewer verts)
  (everything else identical: ARM_SCALE=1.0, full case_39 config)

Same scene/architecture as case_38_gripper_cup_cloth.py but every
mesh loads at its native 1:1 scale: arm at its true URDF size,
cup at original mesh size, shirt at original mesh size, positions
multiplied ~3.3× from case_38 to keep relative layout.

Purpose: benchmark whether step-time scales meaningfully with world
size (BVH coverage / contact pair count / IPC distance metric).
Compare `AUTO_STEP=200 ./run` of case_38 vs case_39 to see the gap.

Differences from case_38:
  ARM_SCALE              0.3   → 1.0
  arm_tf y-translation  -0.9   → -3.0
  cup_scale              0.30  → 1.0
  cup_xyz                       multiplied 3.33×
  shirt_scale            0.5   → 1.0
  shirt_xyz                     multiplied 3.33×
  ground_offset         -0.5   → -1.67
"""
import sys, os, math, time, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
from scipy.spatial.transform import Rotation
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
RIGID_MSH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid.msh"
RIGID_REMAP  = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"
UNIFIED_NPZ  = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"
CUP_MSH      = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/tetmesh/softgriper_cup.msh"
# Shirt: 2D FEM triangle mesh as in case_27_mobile_s1_hybrid.py
SHIRT_OBJ    = "triMesh/shirt_831v.obj"

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


def load_urdf_capture_joint_indices(eng, urdf_path, arm_tf, *load_urdf_args):
    """Wrap eng.native.load_urdf() with fd-level stdout capture so we can
    parse the URDF importer's "Joint constraint 'NAME' (Fixed|Revolute): body P <-> C"
    lines and extract each joint's constraint_index.

    URDF importer skips joints with unresolved bodies (link has no collision),
    so URDF source order ≠ engine's joint_constraints[] index.  This wrapper
    is the only reliable way to get the name → constraint_index mapping.

    Returns:
        (load_urdf_return, {joint_name: constraint_index})
    """
    import os as _os, sys as _sys, tempfile as _tf
    log_path = _tf.mktemp(prefix='urdf_load_log_', suffix='.txt')
    saved_fd1 = _os.dup(1)
    f = _os.open(log_path, _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC)
    _os.dup2(f, 1)
    _os.close(f)
    try:
        ret = eng.native.load_urdf(urdf_path, arm_tf, *load_urdf_args)
        _sys.stdout.flush()
        _os.fsync(1)
    finally:
        _os.dup2(saved_fd1, 1)
        _os.close(saved_fd1)
    log = open(log_path).read()
    _os.unlink(log_path)
    # Re-emit log to actual stdout so user sees it
    print(log, end='', flush=True)
    # Parse: "[UrdfSceneImporter] Joint constraint 'NAME' (TYPE): body P <-> C"
    out = {}
    idx = 0
    for m in re.finditer(
            r"\[UrdfSceneImporter\] Joint constraint '([^']+)' \((Fixed|Revolute)\):",
            log):
        out[m.group(1)] = idx
        idx += 1
    return ret, out


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

    # Prismatic constraint kappa — independent of revolute/fixed joint_strength_ratio.
    # Default 100 makes prismatic finger_joint constraint kappa = 100·(m_finger+m_hand)
    # ≈ 72.  But case_39 manually sets fixed_joint kappa = 1000 (absolute) for the
    # green hybrid gripper attached to blue finger ABD.  Ratio fixed:prismatic ≈
    # 1000:72 = ~14:1 — hybrid pulls finger off the prismatic axis (visible as
    # blue finger separating from gray hand when arm slider is dragged).
    # Bump prismatic constraint kappa to ~1e4 (mass-multiplied = ~7000) to match
    # fixed_joint and keep finger on-axis.
    prismatic_constraint_K = float(os.environ.get("CASE39_PRISMATIC_CONSTRAINT_K", "2000"))

    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100,
        soft_motion_rate=float(os.environ.get("CASE36_SOFT_RATE", "1e4")),
        poisson_rate=0.49, friction_rate=0.4,
        relative_dhat=float(os.environ.get("CASE42_RDHAT", "1e-3")),
        joint_strength_ratio=joint_K,
        revolute_driving_strength_ratio=revolute_K,
        prismatic_strength_ratio=prismatic_constraint_K,
        # Default S2+semi config: replay-trace profiling shows 4.0→14.2 fps in
        # CLOSED+motion pain case, with S2 (max_step=0.04) handling the smoothing
        # job (S1 alpha-smoothing is redundant with S2, see 2x2 factorial study).
        semi_implicit_enabled=bool(int(os.environ.get("CASE42_SEMI", "1"))),
        semi_implicit_beta_tol=float(os.environ.get("CASE42_SEMI_BETA", "5e-2")),
        semi_implicit_min_iter=int(os.environ.get("CASE42_SEMI_MIN_ITER", "1")),
        newton_tol=5e-2,
        newton_iter_cap=int(os.environ.get("CASE39_NEWTON_CAP", "100")),
        preconditioner_type=int(os.environ.get("CASE42_PRECOND", "0")),  # 1=MAS (needs stitch metis-remap fix)
        ground_offset=-1.67,   # case_39 full-scale
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 64.0
    eng = Engine(cfg)
    print("\n[case36] === ridgeback + 4 hybrid grippers ===", flush=True)

    # --- 1. Load URDF (37 ABD bodies including 4 fingers) ---
    # Use wrapper to capture stdout + extract joint_constraint name → idx map.
    # Needed for per-joint set_fixed_joint_strength on *_arm_link8 (joint6 jitter fix).
    arm_tf = make_arm_tf(ARM_SCALE)
    _, joint_constraint_indices = load_urdf_capture_joint_indices(
        eng, URDF_PATH, arm_tf, True, False, 1e7, {})
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
    # CASE36_TF_MODE policy:
    #   "finger_full" (default, NEW — same fix as case_37):
    #       T = engine_link_T × collision_origin.  Matches what URDF
    #       importer applied to finger ABD body, eliminating the 24mm
    #       offset all previous modes had.  NOTE: STRATEGY_F mesh's
    #       local frame is NOT identical to finger.stl's mesh local
    #       frame — residual misfit may remain because STRATEGY_F was
    #       built with an unknown transform.  case_37's fresh-built
    #       mesh avoids this; case_36 mesh predates the fix.
    #   "procrustes": scaled-Kabsch fit hybrid rigid → finger comp 0
    #       (works around STRATEGY_F frame mismatch).
    #   Legacy modes ("lower_seg" / "bbox_center" / "centroid" /
    #   "soft_link" / "finger") all suffer from missing collision_origin.
    tf_mode = os.environ.get("CASE36_TF_MODE", "finger_full").lower()
    # collision_origin from URDF (identical for all 4 finger links):
    _co_rpy = np.array([-1.57079632679, 0.20245819348, -1.57079632679])
    _co_xyz = np.array([-0.0165, 0.0165, 0.12773331296])
    collision_origin = np.eye(4)
    collision_origin[:3, :3] = Rotation.from_euler('xyz', _co_rpy).as_matrix()
    collision_origin[:3, 3] = _co_xyz
    rigid_remap_data = np.load(RIGID_REMAP, allow_pickle=True)
    rigid_v_idx_glb = rigid_remap_data['rigid_v_idx']
    hybrid_verts_local = np.load(UNIFIED_NPZ)['vertices']
    rigid_centroid_local = hybrid_verts_local[rigid_v_idx_glb].mean(axis=0)
    rigid_local_verts = hybrid_verts_local[rigid_v_idx_glb]  # (150, 3) for procrustes

    def _scaled_procrustes(A_local, B_world):
        """Find s, R, t minimizing ||s*R*A + t - B||.  Returns 4x4 transform.
        A: (N,3) source mesh local; B: (N,3) target world.  Includes scale
        because A and B may be at different unit scales (e.g., ARM_SCALE=0.3).
        """
        cA = A_local.mean(0); cB = B_world.mean(0)
        Ac = A_local - cA; Bc = B_world - cB
        H = Ac.T @ Bc
        U, S, Vt = np.linalg.svd(H)
        # Reflection guard
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        # Scale = sum(S * sign) / sum(Ac*Ac)
        s = (np.array([S[0], S[1], S[2]*d]).sum()) / (Ac**2).sum()
        t = cB - s * R @ cA
        T = np.eye(4)
        T[:3, :3] = s * R
        T[:3, 3]  = t
        return T

    import trimesh as _tm
    _finger_obj_path = ("/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/"
                        "ridgeback_dual_panda_soft/meshes/plate/visual/"
                        "soft_hard_segmenation/finger_clean.obj")
    _finger_mesh = _tm.load(_finger_obj_path, process=False)
    _comps = _finger_mesh.split(only_watertight=False)
    # Largest component = lower segment
    _largest_comp = max(_comps, key=lambda c: len(c.vertices))
    # Map comp vertices back to original-mesh vertex indices via position match
    _orig_v = np.asarray(_finger_mesh.vertices)
    _comp_v = np.asarray(_largest_comp.vertices)
    # Build position → orig-index map (within float tolerance)
    _orig_lookup = {tuple(np.round(p, 8)): i for i, p in enumerate(_orig_v)}
    _lower_seg_idx = np.array([
        _orig_lookup[tuple(np.round(p, 8))] for p in _comp_v
        if tuple(np.round(p, 8)) in _orig_lookup
    ], dtype=int)
    print(f"[case36] finger.stl lower segment: {len(_lower_seg_idx)}/{len(_orig_v)} verts",
          flush=True)

    all_v_pre = eng.native.get_vertices_host()
    finger_lower_world_center = {}
    finger_lower_world_verts = {}  # for procrustes NN matching
    for label in FINGER_LABELS:
        rec = finger_recs[label]
        v = all_v_pre[rec.vertex_offset:rec.vertex_offset + rec.vertex_count]
        if len(v) == len(_orig_v) and len(_lower_seg_idx) > 0:
            lower_v = v[_lower_seg_idx]
            finger_lower_world_center[label] = (lower_v.min(0) + lower_v.max(0)) * 0.5
            finger_lower_world_verts[label] = lower_v
        else:
            finger_lower_world_center[label] = (v.min(0) + v.max(0)) * 0.5
            finger_lower_world_verts[label] = v

    print(f"[case36] CASE36_TF_MODE={tf_mode}  "
          f"rigid centroid (mesh local) = {rigid_centroid_local}", flush=True)
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
        elif tf_mode == "bbox_center":
            rec = finger_recs[finger_label]
            v = all_v_pre[rec.vertex_offset:rec.vertex_offset + rec.vertex_count]
            T = soft_link_T.copy()
            T[:3, 3] = (v.min(0) + v.max(0)) * 0.5 - soft_link_T[:3, :3] @ rigid_centroid_local
        elif tf_mode == "lower_seg":
            T = soft_link_T.copy()
            T[:3, 3] = lower_ctr - soft_link_T[:3, :3] @ rigid_centroid_local
        else:  # "procrustes" default
            # NN-match each hybrid rigid local vert to its closest finger
            # comp 0 world vert.  Initialize with lower_seg transform so
            # NN starts from a sensible orientation.
            T_init = soft_link_T.copy()
            T_init[:3, 3] = lower_ctr - soft_link_T[:3, :3] @ rigid_centroid_local
            rigid_world_init = (T_init[:3,:3] @ rigid_local_verts.T).T + T_init[:3,3]
            tree = _KDTree(finger_lower_world_verts[finger_label])
            _, nn_idx = tree.query(rigid_world_init)
            target_world = finger_lower_world_verts[finger_label][nn_idx]
            T = _scaled_procrustes(rigid_local_verts, target_world)
            # Diagnostic: residual fit error
            fit_world = (T[:3,:3] @ rigid_local_verts.T).T + T[:3,3]
            residual = np.linalg.norm(fit_world - target_world, axis=1)
            print(f"[case36/diag-procrustes] {finger_label}: scale={np.linalg.norm(T[:3,0]):.4f}  "
                  f"fit_residual mean={residual.mean()*1000:.2f}mm max={residual.max()*1000:.2f}mm",
                  flush=True)
        soft_T[finger_label] = T
        rcw = T[:3, :3] @ rigid_centroid_local + T[:3, 3]
        print(f"[case36/diag] {finger_label}: rigid_centroid_world={rcw}  "
              f"finger_lower_ctr={lower_ctr}  "
              f"delta={np.linalg.norm(rcw - lower_ctr)*1000:.2f}mm",
              flush=True)

    # --- 3. For each finger, load hybrid rigid + FEM and wire stitch + fj ---
    fem_young = float(os.environ.get("CASE36_FEM_YOUNG", "1e7"))
    fj_kappa = float(os.environ.get("CASE36_FJ_KAPPA", "1e3"))

    rigid_remap = np.load(RIGID_REMAP, allow_pickle=True)
    rigid_v_idx = rigid_remap['rigid_v_idx']
    n_rigid_v = len(rigid_v_idx)

    hybrid_data = np.load(UNIFIED_NPZ)
    hybrid_verts = np.ascontiguousarray(hybrid_data['vertices'], dtype=np.float64)
    hybrid_tets  = np.ascontiguousarray(hybrid_data['tets'], dtype=np.int32)

    # ENGINE CONSTRAINT: all ABD must load before any FEM.  So we do
    # 3 passes — load all hybrid rigid ABDs, then all hybrid FEM bodies,
    # then wire stitch springs + fixed joints.
    grippers = []  # list of dicts populated across passes

    # Pass 1: load 4 hybrid rigid sub-meshes (ABD)
    for label in FINGER_LABELS:
        finger_rec = finger_recs[label]
        gripper_T = soft_T[label]
        eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
                      transform=gripper_T, young_modulus=1e8,
                      boundary_type="Free")
        rigid_rec = eng.get_load_records()[-1]
        grippers.append(dict(
            label=label, finger_id=finger_rec.body_offset,
            gripper_T=gripper_T,
            abd_id=rigid_rec.body_offset, abd_v_off=rigid_rec.vertex_offset,
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

    # Pass 2: load 4 hybrid FEM unified meshes
    for g in grippers:
        eng.native.load_mesh_from_data(
            hybrid_verts, hybrid_tets, 4, 3, 1, g['gripper_T'], fem_young, 0)
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

    # Pass 3: stitch springs + fixed joints per gripper
    for g in grippers:
        for i in range(n_rigid_v):
            eng.add_stitch_spring(
                g['fem_v_off'] + int(rigid_v_idx[i]),
                g['abd_v_off'] + i,
                g['abd_id'],
                rest_offset_world=(0.0, 0.0, 0.0))
        anchor = g['gripper_T'][:3, 3]
        g['fj_idx'] = eng.native.add_fixed_joint(
            parent_body=g['finger_id'], child_body=g['abd_id'],
            world_anchor=anchor,
            world_normal=np.array([1.0, 0.0, 0.0]),
            world_bitangent=np.array([0.0, 0.0, 1.0]),
        )
        print(f"[case36] {g['label']}: finger={g['finger_id']}, "
              f"hybrid_abd={g['abd_id']}, fem={g['fem_rec'].body_offset}, "
              f"fj={g['fj_idx']}", flush=True)

    # --- 4. Collision exclusions ---
    # For each gripper:
    #   (a) hybrid_abd ↔ own FEM (overlap by construction)
    #   (b) hybrid_abd ↔ own finger (overlap by construction)
    #   (c) hybrid_abd ↔ ALL other arm ABD bodies (don't push arm around)
    #   (d) FEM        ↔ ALL other arm ABD bodies (same)
    # Plus: hybrid pair within the same arm (left/right finger of one
    # hand) starts geometrically overlapped at prismatic=0 — exclude
    # them mutually so IPC doesn't reject the initial config.
    arm_ids = [r.body_offset for r in urdf_recs if r.body_type == 0]
    for g in grippers:
        eng.native.add_collision_exclusion(g['abd_id'], g['fem_global_id'])
        eng.native.add_collision_exclusion(g['abd_id'], g['finger_id'])
        eng.native.add_collision_exclusion(g['fem_global_id'], g['finger_id'])
        for arm_id in arm_ids:
            if arm_id == g['finger_id']:
                continue
            eng.native.add_collision_exclusion(g['abd_id'], arm_id)
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

    # Hybrid GREEN rigid sub-mesh ABD × ground collision: default SKIP.
    # Pass 1 loop above only skipped URDF bodies (0..n_urdf-1).  The 4 green
    # hybrid rigid sub-mesh ABDs were loaded AFTER URDF (body ids n_urdf..
    # n_urdf+3), so without this they would collide with the ground when arm
    # lowers — the ground reaction force pulls the green ABD up, which drags
    # the blue finger ABD via fixed_joint, which can break the prismatic
    # constraint to the gray hand → gripper "falls off" visually.
    abd_ground_collide = int(os.environ.get("CASE39_HYBRID_ABD_GROUND_COLLISION", "0"))
    if not abd_ground_collide:
        for g in grippers:
            eng.add_ground_collision_skip(g['abd_id'])
        print(f"[case39] hybrid GREEN ABD × ground collision: SKIPPED (default; "
              f"set CASE39_HYBRID_ABD_GROUND_COLLISION=1 to enable)", flush=True)
    else:
        print(f"[case39] hybrid GREEN ABD × ground collision: ENABLED", flush=True)

    # Cross-gripper exclusion within the same arm pair
    def _arm_prefix(label):
        return 'left' if label.startswith('left_') else 'right'
    for i, gi in enumerate(grippers):
        for gj in grippers[i+1:]:
            if _arm_prefix(gi['label']) != _arm_prefix(gj['label']):
                continue
            eng.native.add_collision_exclusion(gi['abd_id'], gj['abd_id'])
            eng.native.add_collision_exclusion(gi['abd_id'], gj['fem_global_id'])
            eng.native.add_collision_exclusion(gi['fem_global_id'], gj['abd_id'])
            eng.native.add_collision_exclusion(gi['fem_global_id'], gj['fem_global_id'])

    eng.finalize()

    # --- 5. Disable gravity (Option A: keep poses static under PD) ---
    if int(os.environ.get("CASE36_DISABLE_GRAVITY", "1")):
        for arm_id in arm_ids:
            eng.native.set_body_apply_gravity(arm_id, False)
        for g in grippers:
            eng.native.set_body_apply_gravity(g['abd_id'], False)

    # --- 6. Per-fixed-joint kappa override (default ~20 too weak) ---
    for g in grippers:
        eng.native.set_fixed_joint_strength(g['fj_idx'], fj_kappa)

    # [case_42 jitter fix] joint6 wrist roll exposes the *_hand_joint_*_arm_link8
    # FIXED constraint as the weakest link in the kinematic chain: it carries
    # hand + 4 grippers + softpads but URDF importer initializes its kappa to
    # joint_strength_ratio·mass ≈ 140 (with default ratio=100, mass~1.4kg).
    # Bump it to absolute kappa ~1000 like the manually-added gripper fj's.
    # This is targeted — does NOT bump revolute constraints (so joint1 stays
    # fast).
    hand_link8_K = float(os.environ.get("CASE42_HAND_LINK8_K", "1000"))
    bumped = 0
    for jname, jidx in joint_constraint_indices.items():
        if "_arm_link8" in jname and "hand_joint" in jname:
            eng.native.set_fixed_joint_strength(jidx, hand_link8_K)
            print(f"[case42] bumped fixed_joint '{jname}' (idx={jidx}) kappa = {hand_link8_K}",
                  flush=True)
            bumped += 1
    if bumped == 0:
        print(f"[case42] WARN: no *_hand_joint_*_arm_link8 found in joint_constraint_indices "
              f"(got {len(joint_constraint_indices)} joints) — fix may not apply", flush=True)

    eng.native.set_max_revolute_step_per_frame(
        float(os.environ.get("CASE42_MAX_REV_STEP", os.environ.get("CASE36_MAX_RAD_PER_FRAME", "0.04"))))

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
    abd_id_set    = {g['abd_id']    for g in grippers}
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
        elif r.body_type == 0 and r.body_offset in abd_id_set:
            color = (0.2, 0.95, 0.3)     # hybrid rigid green
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
        if r.body_type == 1 and r.body_offset in fem_local_id_set:
            region = np.zeros(r.vertex_count, dtype=np.float32)
            region[rigid_v_idx] = 1.0
            m.add_scalar_quantity("rigid (red)", region, enabled=True, cmap='reds')

    # Optional: auto-run + auto-quit after N GUI steps for smoke tests
    auto_run = bool(int(os.environ.get("GUI_AUTO_RUN", "0")))
    auto_quit_after = int(os.environ.get("GUI_QUIT_AFTER_STEPS", "0"))
    from collections import deque
    state = dict(running=auto_run, step_count=0, last_step_ms=0.0,
                 step_ms_history=deque(maxlen=30),
                 smooth_alpha=float(os.environ.get("CASE42_SMOOTH_ALPHA", "1.0")))

    # --- S1 target smoothing (low-pass on slider input) ---
    # Slider writes to raw_*; engine sees driven_* which lags raw_* with
    # exponential smoothing.  alpha=1.0 = no smoothing (original).  Lower
    # alpha = smoother drag, less impulsive force injection into the ABD-FEM
    # chain (prevents "cloth getting yanked" when slider is dragged fast).
    raw_rev    = [robot.get_revolute_target_deg(i)
                  for i in range(len(robot.revolute_joints))]
    driven_rev = list(raw_rev)
    raw_prism    = [robot.get_prismatic_target_mm(i)
                    for i in range(len(robot.prismatic_joints))]
    driven_prism = list(raw_prism)

    def smooth_and_apply():
        a = state['smooth_alpha']
        for i in range(len(driven_rev)):
            driven_rev[i] += a * (raw_rev[i] - driven_rev[i])
            robot.set_revolute_position(i, driven_rev[i], degree=True)
        for i in range(len(driven_prism)):
            driven_prism[i] += a * (raw_prism[i] - driven_prism[i])
            robot.set_prismatic_position(i, driven_prism[i], millimeters=True)

    # --- trajectory recorder (CASE42_RECORD=<path.jsonl>) ---
    # Writes one JSON line per stepped frame with all joint targets, so the
    # same drag can be replayed headless via CASE42_REPLAY for profiling.
    record_path = os.environ.get("CASE42_RECORD")
    record_file = None
    if record_path:
        record_file = open(record_path, "w")
        record_file.write(json.dumps({
            "meta": True,
            "rev_names":   [ji.name for ji in robot.revolute_joints],
            "prism_names": [ji.name for ji in robot.prismatic_joints],
        }) + "\n")
        record_file.flush()
        print(f"[record] writing trajectory to {record_path}", flush=True)
        import atexit
        atexit.register(lambda: (record_file.flush(), record_file.close(),
                                 print(f"[record] saved {state['step_count']} frames",
                                       flush=True)))

    def do_step():
        smooth_and_apply()
        t0 = time.perf_counter()
        eng.step()
        ms = (time.perf_counter() - t0) * 1000.0
        state['last_step_ms'] = ms
        state['step_ms_history'].append(ms)
        state['step_count'] += 1
        if record_file:
            record_file.write(json.dumps({
                "frame": state['step_count'],
                "ms": ms,
                "rev":   list(driven_rev),
                "prism": list(driven_prism),
            }) + "\n")
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
        psim.Begin("case_39 — full scale (ARM=1.0) A variant")
        psim.Text(f"step #{state['step_count']}: {state['last_step_ms']:.1f} ms")
        # FPS row: instantaneous (last frame) + sliding avg over last 30 frames
        last_ms = state['last_step_ms']
        inst_fps = 1000.0 / last_ms if last_ms > 0 else 0.0
        hist = state['step_ms_history']
        avg_ms = sum(hist) / len(hist) if hist else 0.0
        avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        psim.Text(f"FPS: inst {inst_fps:5.1f}  |  avg(30) {avg_fps:5.1f}  ({avg_ms:.1f} ms/step)")
        psim.Text(f"URDF bodies: {n_urdf}, hybrid: {len(grippers)} ABD + {len(grippers)} FEM")
        psim.Separator()
        if state['running']:
            if psim.Button("Pause"): state['running'] = False
        else:
            if psim.Button("Run"): state['running'] = True
        psim.SameLine()
        if psim.Button("Step"): do_step()
        psim.SameLine()
        if psim.Button("Reset"):
            robot.reset_all()
            for i in range(len(raw_rev)):
                raw_rev[i] = driven_rev[i] = robot.get_revolute_target_deg(i)
            for i in range(len(raw_prism)):
                raw_prism[i] = driven_prism[i] = robot.get_prismatic_target_mm(i)

        psim.Separator()
        # Smoothing alpha (live tunable)
        chg_a, new_a = psim.SliderFloat("target_smooth_alpha (1=off, 0.05=very smooth)",
                                         state['smooth_alpha'], 0.05, 1.0)
        if chg_a:
            state['smooth_alpha'] = new_a
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
                # Slider shows + writes raw target; driven catches up smoothly.
                cur_mm = raw_prism[idxs[0]]
                chg, new_val = psim.SliderFloat(f"{label} (mm)", cur_mm, lo_mm, hi_mm)
                if chg:
                    for i in idxs:
                        raw_prism[i] = new_val

        if robot.revolute_joints:
            psim.Spacing()
            psim.Text(f"Revolute Joints ({len(robot.revolute_joints)})")
            psim.Separator()
            for i, ji in enumerate(robot.revolute_joints):
                # Slider shows + writes raw target; driven catches up smoothly.
                cur = raw_rev[i]
                chg, new_val = psim.SliderFloat(
                    ji.name, cur, ji.lower_limit_deg, ji.upper_limit_deg)
                if chg:
                    raw_rev[i] = new_val
        psim.End()

        if state['running']:
            do_step()

    # ---- BENCH_MODE: skip GUI, sweep (max_rev_step, K_ratio) configs ----
    if os.environ.get("BENCH_MODE"):
        _bench_joint_motion(eng, robot)
        return

    # ---- CASE42_REPLAY or STIFFGIPC_REPLAY: replay recorded trajectory ----
    import _case4x_replay
    if _case4x_replay.maybe_replay(eng, robot, body_meshes, ps, psim,
                                    env_name="CASE42"):
        return

    ps.set_user_callback(callback)
    ps.show()


def _bench_joint_motion(eng, robot):
    """Headless 9-config sweep of (max_rev_step, revolute_K_ratio).
    Push panda_joint4 by +0.5 rad, measure idle/motion step ms + final tracking.
    Triggered by BENCH_MODE=1 env var.  Single-process, ~6-10 min.
    """
    import statistics
    import math
    import time as _time

    # Single-config mode (BENCH_SINGLE=1): just run one A=0.02 B=1.0 measurement
    # and print result in CSV-parseable format.  Used by external K sweep wrappers
    # (which vary CASE39_PRISMATIC_CONSTRAINT_K via subprocess, since prismatic K
    # is set at Config-time and can't be re-set post-finalize).
    if os.environ.get("BENCH_SINGLE"):
        CONFIGS = [(0.02, 1.0, "single")]
    else:
        CONFIGS = [
            # (max_rev_step rad, per-joint K multiplier, label)
            (0.5,  1.0,   "A=0.50 B=1.00 (current case_39)"),
            (0.1,  1.0,   "A=0.10 B=1.00 (engine default A)"),
            (0.05, 1.0,   "A=0.05 B=1.00"),
            (0.02, 1.0,   "A=0.02 B=1.00"),
            (0.01, 1.0,   "A=0.01 B=1.00"),
            (0.05, 0.50,  "A=0.05 B=0.50"),
            (0.05, 0.25,  "A=0.05 B=0.25"),
            (0.05, 0.10,  "A=0.05 B=0.10"),
            (0.02, 0.25,  "A=0.02 B=0.25"),
        ]

    # Pick a joint where 0 is INSIDE [lower, upper] (avoids Y5 clamp issue —
    # if 0 is outside limits, initial_angle_offset gets clamped + driving force
    # is zero at theta=0 regardless of target).
    # Print all joints + limits for debugging:
    print(f"[BENCH] revolute joints + limits:")
    for i, ji in enumerate(robot.revolute_joints):
        in_range = (ji.lower_limit <= 0.0 <= ji.upper_limit)
        marker = " ✓" if in_range else "  "
        print(f"  [{i:2}] {marker} {ji.name:<30} [{math.degrees(ji.lower_limit):>7.2f}°, {math.degrees(ji.upper_limit):>7.2f}°]")

    # Pick first joint where 0 is in valid range AND name contains 'arm'
    # (avoid finger / wheel joints — those have small range or are prismatic-like)
    joint_idx = None
    JOINT_NAME = None
    for i, ji in enumerate(robot.revolute_joints):
        if (ji.lower_limit <= 0.0 <= ji.upper_limit
            and "arm" in ji.name and "finger" not in ji.name):
            range_rad = ji.upper_limit - ji.lower_limit
            if range_rad > 1.0:  # plenty of room for ±0.5 rad push
                joint_idx = i
                JOINT_NAME = ji.name
                break
    if joint_idx is None:
        print(f"[BENCH] WARN: no suitable arm joint, falling back to joint 0")
        joint_idx = 0
        JOINT_NAME = robot.revolute_joints[0].name
    n_joints = len(robot.revolute_joints)

    # Use raw native API for fast/scriptable angle reads
    angles_buf = eng.native.get_revolute_current_angles()
    base_angle = float(angles_buf[joint_idx])

    # Push delta — +0.5 rad ≈ +28.6°
    PUSH_DELTA = 0.5
    target_angle = base_angle + PUSH_DELTA

    # Clamp target to joint limits if needed
    ji = robot.revolute_joints[joint_idx]
    if target_angle > ji.upper_limit:
        target_angle = ji.upper_limit
    elif target_angle < ji.lower_limit:
        target_angle = ji.lower_limit
    actual_delta = target_angle - base_angle

    print(f"\n{'='*90}")
    print(f"[BENCH] case_39 joint motion benchmark (semi_implicit=False)")
    print(f"[BENCH] driving '{JOINT_NAME}' joint #{joint_idx}")
    print(f"[BENCH] base={math.degrees(base_angle):.2f}°  target={math.degrees(target_angle):.2f}°  "
          f"Δ={math.degrees(actual_delta):.2f}° ({actual_delta:.3f} rad)")
    print(f"[BENCH] limits=[{math.degrees(ji.lower_limit):.2f}°, {math.degrees(ji.upper_limit):.2f}°]")
    print(f"{'='*90}\n")

    results = []
    N_SETTLE = 30
    N_IDLE_MEAS = 20
    N_MOTION_MEAS = 50

    for (A, B_ratio, label) in CONFIGS:
        # Apply config
        eng.native.set_max_revolute_step_per_frame(A)
        for i in range(n_joints):
            eng.native.set_revolute_strength(i, B_ratio)

        # Pull joint back to base
        robot.set_revolute_position(joint_idx, base_angle, degree=False)

        # Settle (no measurement)
        for _ in range(N_SETTLE):
            eng.step()

        # Measure idle
        idle_samples = []
        for _ in range(N_IDLE_MEAS):
            t0 = _time.perf_counter()
            eng.step()
            idle_samples.append((_time.perf_counter() - t0) * 1000.0)
        idle_avg = statistics.mean(idle_samples)

        # Push target
        robot.set_revolute_position(joint_idx, target_angle, degree=False)

        # Measure motion
        motion_samples = []
        for _ in range(N_MOTION_MEAS):
            t0 = _time.perf_counter()
            eng.step()
            motion_samples.append((_time.perf_counter() - t0) * 1000.0)
        motion_avg = statistics.mean(motion_samples)
        motion_p95 = sorted(motion_samples)[int(0.95 * len(motion_samples))]
        motion_max = max(motion_samples)

        # Final actual offset
        angles_buf = eng.native.get_revolute_current_angles()
        actual_final = float(angles_buf[joint_idx])
        offset_deg = math.degrees(abs(actual_final - target_angle))

        results.append((label, A, B_ratio, idle_avg, motion_avg, motion_p95, motion_max, offset_deg))
        print(f"  {label:<40} idle={idle_avg:6.1f}  motion_avg={motion_avg:7.1f}  "
              f"p95={motion_p95:7.1f}  max={motion_max:7.1f}  offset={offset_deg:5.2f}°")
        # Machine-parseable CSV row (consumed by external K-sweep wrappers)
        prismatic_K = os.environ.get("CASE39_PRISMATIC_CONSTRAINT_K", "2000")
        print(f"BENCH_CSV,prismatic_K={prismatic_K},A={A},B={B_ratio},"
              f"idle={idle_avg:.2f},mot_avg={motion_avg:.2f},mot_p95={motion_p95:.2f},"
              f"mot_max={motion_max:.2f},offset_deg={offset_deg:.3f}")

    # Summary table
    print(f"\n\n{'='*100}")
    print(f"BENCH RESULTS — case_39 joint motion (semi_implicit=False), drive '{JOINT_NAME}' +{math.degrees(actual_delta):.1f}°")
    print(f"{'='*100}")
    print(f"{'config':<42} {'idle_ms':>8} {'mot_avg':>9} {'mot_p95':>9} {'mot_max':>9} {'off_deg':>9}")
    print(f"{'-'*100}")
    for r in results:
        print(f"  {r[0]:<40} {r[3]:>8.1f} {r[4]:>9.1f} {r[5]:>9.1f} {r[6]:>9.1f} {r[7]:>9.2f}")
    print(f"{'='*100}")
    # Best by motion_avg subject to constraints
    valid = [r for r in results if r[5] < 1000.0]  # M3 < 1s constraint
    if valid:
        best = min(valid, key=lambda r: r[4])
        print(f"\n[BENCH] BEST (by motion_avg, p95<1000ms): {best[0]}")
        print(f"        idle={best[3]:.1f}  motion_avg={best[4]:.1f}  p95={best[5]:.1f}  offset={best[7]:.2f}°")
    else:
        print(f"\n[BENCH] WARN: no config met motion_p95<1000ms constraint")


if __name__ == "__main__":
    main()
