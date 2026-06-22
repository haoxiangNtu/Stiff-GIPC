#!/usr/bin/env python3
"""Replay a pre-recorded qpos trajectory on the case_39 fold-shirt scene with the
UMI finray gripper rebuilt in the STRATEGY_F HYBRID pattern (mirroring
replay_case39_general.py), instead of the direct-stitch route of
replay_case39_UMI.py.

The key difference vs replay_case39_UMI.py (direct stitch):

  * The finray's SOLID MOUNTING ROOT (bottom band, centroid Z < 0.032) is split
    off as a SEPARATE RIGID ABD body (young 1e8), extracted from the SAME
    unified tet mesh by build_umi_finray_strategyF.py.  Only the hollow fin-ray
    truss + solid tip above stays FEM.

  * Per finger: load `UMI_finray_{side}_rigid.msh` as a rigid ABD at the gripper
    transform, AND load the full unified finray mesh as FEM at the same
    transform.  Stitch each unified rigid vert (`rigid_v_idx[i]`) to rigid-ABD
    vert i with rest_offset=(0,0,0) — GAP IS ZERO because the .msh verts are
    coincident with the unified mesh's rigid verts.  This + the engine DEFAULT
    high soft_motion_rate (1e4) gives a wrinkle-free join (no soft_rate=1e2
    workaround needed, unlike the direct-stitch version).

  * An add_fixed_joint pins the finray rigid ABD to the FINGER ABD seat, so the
    rigid root is driven by the prismatic joint (same as case_39's STRATEGY_F).

This fixes the local wrinkling we got from soft-spring-stitching the WHOLE
finray to the finger: now only the truss is deformable, the root is rigid.

Everything else (URDF load, cloth FEM actors, joint mapping with UMI's mirrored
prismatic limits, headless timing loop, polyscope GUI) matches
replay_case39_UMI.py.

Usage (GUI, uses the bundled fold-shirt trajectory by default):
    CASE39_PRECOND=0 STIFF_SKIP_CCD_SANITY=1 \
        python examples/replay_case39_UMI_sf.py

    # headless timing:
    CASE39_HEADLESS=1 CASE39_FRAME_END=30 CASE39_PRECOND=0 \
        STIFF_SKIP_CCD_SANITY=1 python examples/replay_case39_UMI_sf.py --quiet
"""
import sys, os, math, time, re
from pathlib import Path

_ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "Assets") + "/"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from stiff_physics import Engine, Config
from stiff_physics.robot import Robot
import polyscope as ps
import polyscope.imgui as psim

URDF_PATH = _ASSETS_DIR + "sim_data/urdf/ridgeback_dual_panda_UMI/ridgeback_dual_panda2.urdf"
# STRATEGY_F hybrid assets (built by build_umi_finray_strategyF.py): per side,
# a unified tet mesh + a rigid sub-mesh .msh extracted from it + the rigid_v_idx
# remap.  CASE39UMI_FEM_SRC is accepted for CLI compat but the sf assets live in
# a single dir regardless.
_FEM_DIR = "sim_data/" + os.environ.get("CASE39UMI_FINRAY_DIR", "umi_hybrid_sf_v800")


def _sf_paths(side):
    base = _ASSETS_DIR + _FEM_DIR + f"/UMI_finray_{side}"
    return (base + "_unified.npz", base + "_rigid.msh", base + "_rigid_remap.npz")

ARM_SCALE = 1.0
FINGER_LABELS = [
    'left_arm_leftfinger',  'left_arm_rightfinger',
    'right_arm_leftfinger', 'right_arm_rightfinger',
]


def make_arm_tf(pos, scale: float) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix()
    tf[:3, 3] = Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix() @ np.asarray(pos)
    return tf


def _parse_xyz_rpy(s):
    return np.array([float(x) for x in s.split()], dtype=float)


def link_visual_origin(urdf_path: str, link_name: str) -> np.ndarray:
    """Return the 4x4 visual/collision <origin> transform of `link_name`
    (mesh-local -> link frame).  The UMI mount links carry rpy="0 0 -1.5708"
    xyz="0 +/-0.01 0"; the finray FEM is in the same mesh-local frame as the
    mount mesh, so this is the transform that maps FEM-local into the link frame
    (compose with the link world transform to get FEM-local -> world)."""
    src = open(urdf_path).read()
    m = re.search(r'<link\s+name="' + re.escape(link_name) + r'"[^>]*>(.*?)</link>',
                  src, re.DOTALL)
    if not m:
        return np.eye(4)
    body = m.group(1)
    section = (re.search(r'<collision[^>]*>(.*?)</collision>', body, re.DOTALL)
               or re.search(r'<visual[^>]*>(.*?)</visual>', body, re.DOTALL))
    if not section:
        return np.eye(4)
    sbody = section.group(1)
    om_xyz = re.search(r'<origin[^/>]*xyz="([^"]+)"', sbody)
    om_rpy = re.search(r'<origin[^/>]*rpy="([^"]+)"', sbody)
    xyz = _parse_xyz_rpy(om_xyz.group(1)) if om_xyz else np.zeros(3)
    rpy = _parse_xyz_rpy(om_rpy.group(1)) if om_rpy else np.zeros(3)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T


_BOX_EDGES = np.array([  # 12 edges of a box over its 8 corners (see order below)
    [0, 1], [1, 2], [2, 3], [3, 0],   # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],   # top face
    [0, 4], [1, 5], [2, 6], [3, 7],   # verticals
], dtype=np.int64)


def _aabbs_to_box_wireframe(aabbs: np.ndarray):
    """Turn an (N, 6) array of [lx,ly,lz,ux,uy,uz] AABBs into a polyscope
    curve-network (nodes (8N,3), edges (12N,2)) of box wireframes."""
    n = aabbs.shape[0]
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int64)
    lx, ly, lz = aabbs[:, 0], aabbs[:, 1], aabbs[:, 2]
    ux, uy, uz = aabbs[:, 3], aabbs[:, 4], aabbs[:, 5]
    # 8 corners per box, matching the _BOX_EDGES index order.
    corners = np.empty((n, 8, 3), dtype=np.float64)
    corners[:, 0] = np.stack([lx, ly, lz], axis=1)
    corners[:, 1] = np.stack([ux, ly, lz], axis=1)
    corners[:, 2] = np.stack([ux, uy, lz], axis=1)
    corners[:, 3] = np.stack([lx, uy, lz], axis=1)
    corners[:, 4] = np.stack([lx, ly, uz], axis=1)
    corners[:, 5] = np.stack([ux, ly, uz], axis=1)
    corners[:, 6] = np.stack([ux, uy, uz], axis=1)
    corners[:, 7] = np.stack([lx, uy, uz], axis=1)
    nodes = corners.reshape(-1, 3)
    edges = (_BOX_EDGES[None, :, :] + (np.arange(n) * 8)[:, None, None]).reshape(-1, 2)
    return nodes, edges


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=str,
                        default=_ASSETS_DIR + "trajectories/episode_fold_shirt_umi.hdf5")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    import h5py, json
    with h5py.File(args.replay, "r") as f:
        init_info = json.loads(f.attrs["object_init_info"])
        actions = f["actions"][:]
        robot_init_pose = f.attrs["robot_init_pose"]
        env_cfg = json.loads(f.attrs["env_cfg"])

    # --- Physics config (case_39 values) ---
    prismatic_constraint_K = float(os.environ.get("CASE39_PRISMATIC_CONSTRAINT_K", "2000"))
    joint_K        = float(os.environ.get("CASE38_JOINT_K", "100"))
    revolute_K     = float(os.environ.get("CASE36_PD_K", "100"))
    prismatic_mult = float(os.environ.get("CASE36_PRISMATIC_K", "15"))

    default_cfg = dict(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100,
        soft_motion_rate=float(os.environ.get("CASE36_SOFT_RATE", "1e4")),
        poisson_rate=0.49,
        friction_rate=float(os.environ.get("CASE39_FRICTION", "0.8")),
        relative_dhat=1e-3,
        joint_strength_ratio=joint_K,
        revolute_driving_strength_ratio=revolute_K,
        prismatic_strength_ratio=prismatic_constraint_K,
        semi_implicit_enabled=bool(int(os.environ.get("CASE39_SEMI", "0"))),
        semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1,
        newton_tol=float(os.environ.get("CASE39_NEWTON_TOL", "5e-2")),
        newton_iter_cap=int(os.environ.get("CASE39_NEWTON_CAP", "50")),
        preconditioner_type=int(os.environ.get("CASE39_PRECOND", "1")),
        ground_offset=0.75,
        assets_dir=_ASSETS_DIR,
    )

    env_cfg["assets_dir"] = _ASSETS_DIR

    print("default_cfg:\n", default_cfg)
    print("record_cfg:\n", env_cfg)

    cfg = Config(**default_cfg)
    eng = Engine(cfg)
    print("\n[replay_case39_UMI] === building case_39 (UMI gripper) scene ===", flush=True)

    if args.quiet:
        eng.set_log_level(0)

    ################################### robot init

    # --- 1. Load URDF (UMI option X: each finger link IS the rigid mount seat;
    #         no separate *_mount ABD links any more) ---
    arm_tf = make_arm_tf(robot_init_pose[:3], ARM_SCALE)
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})
    n_urdf = eng.abd_body_count
    urdf_recs = list(eng.get_load_records())
    rec_by_label = {r.label: r for r in urdf_recs if r.body_type == 0}
    finger_recs = {lbl: rec_by_label[lbl] for lbl in FINGER_LABELS
                   if lbl in rec_by_label}
    if len(finger_recs) != 4:
        raise RuntimeError(f"expected 4 finger ABDs, got {len(finger_recs)}")
    print(f"[replay_case39_UMI] URDF: {n_urdf} ABD bodies "
          f"(4 finger mount-seats, no separate mount links)", flush=True)

    for b in range(n_urdf):
        eng.add_ground_collision_skip(b)

    # --- 2. Per-finger finray STRATEGY_F hybrid setup ---
    # The finger link carries the mount-seat mesh + its visual origin, so:
    #   FEM-local -> world = finger_link_world_T @ finger_visual_origin
    # Both the rigid sub-mesh (.msh) and the unified FEM mesh are placed at this
    # same gripper_T, so the rigid-ABD verts are coincident with the unified
    # mesh's rigid verts (gap=0 stitch).
    grippers = []
    for label in FINGER_LABELS:
        side = 'L' if label.endswith('leftfinger') else 'R'
        unified_npz, rigid_msh, rigid_remap = _sf_paths(side)
        d = np.load(unified_npz)
        verts = np.ascontiguousarray(d['vertices'], dtype=np.float64)
        tets  = np.ascontiguousarray(d['tets'], dtype=np.int32)
        vertex_region = np.asarray(d['vertex_region'], dtype=np.int32)
        rigid_v_idx = np.asarray(
            np.load(rigid_remap)['rigid_v_idx'], dtype=np.int64)

        finger_link_T = np.asarray(eng.native.get_urdf_link_transform(label))
        vis_T = link_visual_origin(URDF_PATH, label)
        gripper_T = finger_link_T @ vis_T

        grippers.append(dict(
            label=label, side=side,
            finger_id=finger_recs[label].body_offset,
            finger_rec=finger_recs[label],
            gripper_T=gripper_T,
            verts=verts, tets=tets,
            vertex_region=vertex_region, rigid_v_idx=rigid_v_idx,
            rigid_msh=rigid_msh,
        ))

    # Pass 1: per-finger RIGID finray-root ABD (extracted sub-mesh, young 1e8),
    # placed at the gripper transform.  Mirrors case_39 STRATEGY_F pass 1.
    rigid_young = float(os.environ.get("CASE39UMI_RIGID_YOUNG", "1e8"))
    for g in grippers:
        eng.load_mesh(g['rigid_msh'], dimensions=3, body_type="ABD",
                      transform=g['gripper_T'], young_modulus=rigid_young,
                      boundary_type="Free")
        rigid_rec = eng.get_load_records()[-1]
        g['rigid_abd_id'] = rigid_rec.body_offset
        g['rigid_abd_v_off'] = rigid_rec.vertex_offset
        g['rigid_rec'] = rigid_rec

    ################################################################
    # load actors from the recorded init_info. FEM actors (the cloth/shirt) are
    # loaded straight by the engine (eng.load_mesh); only ABD actors (e.g. a
    # rigid cup) need trimesh to read their surface mesh. The fold-shirt replay
    # has no ABD actors, so trimesh is imported lazily below — keeping it off the
    # required-dependency list for this example.
    abd_objs = []
    fem_objs = []
    for key, value in init_info.items():
        init_pose = value['initial_pose']
        # Resolve the recorded collision-mesh path to a file under THIS repo's
        # assets/ dir. The shipped trajectory stores a clean `assets/...` path;
        # older recordings stored an absolute data-generation path — strip any
        # such known prefix so both resolve under _ASSETS_DIR.
        collision_mesh = value["collision_mesh"]
        for _pre in ("/data/stiff-physics/franka_sim/assets_new/",
                     "/data/stiff-physics/franka_sim/assets/",
                     "/data/stiff-physics/assets/",
                     "assets/"):
            if collision_mesh.startswith(_pre):
                collision_mesh = _ASSETS_DIR + collision_mesh[len(_pre):]
                break
        else:
            if not os.path.isabs(collision_mesh):
                collision_mesh = _ASSETS_DIR + collision_mesh
        body_type = value["body_type"]
        if body_type == "ABD":
            abd_objs.append([init_pose, collision_mesh])
        else:
            fem_objs.append([init_pose, collision_mesh])

    abd_records = []
    if abd_objs:
        import trimesh  # only needed when the replay has rigid (ABD) actor meshes
    for init_pose, collision_mesh in abd_objs:
        mesh = trimesh.load(collision_mesh)
        cube_v = np.asarray(mesh.vertices)
        cube_f = np.asarray(mesh.faces)
        eng.load_mesh_from_data(
            vertices=cube_v, faces=cube_f,
            verts_per_face=3,
            dimensions=3,
            body_type="ABD",
            transform=np.asarray(init_pose).reshape(4, 4),
            young_modulus=1e5,
            boundary_type="Free",
        )
        abd_records.append(eng.get_load_records()[-1])

    #################################

    # Pass 2: finray FEM (full unified per-side mesh, same transform as rigid)
    fem_young = float(os.environ.get("CASE36_FEM_YOUNG", "1e7"))
    for g in grippers:
        eng.native.load_mesh_from_data(
            g['verts'], g['tets'], 4, 3, 1, g['gripper_T'], fem_young, 0)
        fem_rec = eng.get_load_records()[-1]
        g['fem_rec'] = fem_rec
        g['fem_v_off'] = fem_rec.vertex_offset

    ##################################
    # cloth FEM actors (shirt) from init_info
    fem_records = []
    _skip_cloth = int(os.environ.get("CASE39UMI_SKIP_CLOTH", "0"))
    for init_pose, collision_mesh in (fem_objs if not _skip_cloth else []):
        eng.load_mesh(collision_mesh, body_type="FEM",
                      transform=np.asarray(init_pose).reshape(4, 4),
                      young_modulus=1e2, dimensions=2)
        fem_records.append(eng.get_load_records()[-1])

    #######################################

    n_abd_total = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    for g in grippers:
        g['fem_global_id'] = n_abd_total + g['fem_rec'].body_offset

    # Pass 3: STRATEGY_F gap=0 stitch (unified FEM rigid verts -> rigid ABD verts)
    # + fixed joint pinning the finray rigid root to the finger seat.
    # rigid_v_idx[i] is the unified-mesh index of the rigid sub-mesh's i-th vert
    # (same order as the .msh), so FEM-vert (fem_v_off + rigid_v_idx[i]) and
    # rigid-ABD-vert (rigid_abd_v_off + i) are COINCIDENT in world: gap = 0.
    all_v = eng.native.get_vertices_host()
    total_pairs = 0
    for g in grippers:
        rvidx = g['rigid_v_idx']
        for i in range(len(rvidx)):
            eng.add_stitch_spring(
                g['fem_v_off'] + int(rvidx[i]),
                g['rigid_abd_v_off'] + i,
                g['rigid_abd_id'],
                rest_offset_world=(0.0, 0.0, 0.0))
        g['n_stitch'] = len(rvidx)
        total_pairs += len(rvidx)
        # Report actual world gap between the stitched pairs (should be ~0).
        fem_verts = all_v[g['fem_v_off']:g['fem_v_off'] + g['fem_rec'].vertex_count]
        rigid_verts = all_v[g['rigid_abd_v_off']:
                            g['rigid_abd_v_off'] + g['rigid_rec'].vertex_count]
        gap = np.linalg.norm(fem_verts[rvidx] - rigid_verts, axis=1)
        print(f"[UMI-sf-stitch] {g['label']} ({g['side']}): "
              f"finger={g['finger_id']}, rigid_abd={g['rigid_abd_id']}, "
              f"fem={g['fem_rec'].body_offset}, "
              f"{len(rvidx)} stitch pairs "
              f"(gap_mean={1000*gap.mean():.4f}mm, gap_max={1000*gap.max():.4f}mm)",
              flush=True)
    print(f"[UMI-sf-stitch] total stitch pairs: {total_pairs}", flush=True)

    # Fixed joint: pin each finray rigid root to its finger ABD seat, so the
    # rigid root (and via gap=0 stitch, the FEM root) is driven by the prismatic.
    for g in grippers:
        anchor = g['gripper_T'][:3, 3]
        g['fj_idx'] = eng.native.add_fixed_joint(
            parent_body=g['finger_id'], child_body=g['rigid_abd_id'],
            world_anchor=anchor,
            world_normal=np.array([1.0, 0.0, 0.0]),
            world_bitangent=np.array([0.0, 0.0, 1.0]),
        )

    # --- 4. Collision exclusions (STRATEGY_F structure, mirrors case_39) ---
    # For each gripper:
    #   rigid-finray-ABD ↔ own finger ABD   (fixed-jointed to it)
    #   rigid-finray-ABD ↔ own FEM          (coincident, stitched)
    #   rigid-finray-ABD ↔ all OTHER arm ABDs
    #   FEM ↔ own finger ABD                (sits on the finger mount seat)
    #   FEM ↔ own rigid-finray-ABD
    #   FEM ↔ all OTHER arm ABDs
    # Plus same-arm finger-pair mutual exclusions (overlap at prismatic=0).
    arm_ids = [r.body_offset for r in urdf_recs if r.body_type == 0]
    for g in grippers:
        eng.native.add_collision_exclusion(g['rigid_abd_id'], g['fem_global_id'])
        eng.native.add_collision_exclusion(g['rigid_abd_id'], g['finger_id'])
        eng.native.add_collision_exclusion(g['fem_global_id'], g['finger_id'])
        for arm_id in arm_ids:
            if arm_id == g['finger_id']:
                continue
            eng.native.add_collision_exclusion(g['rigid_abd_id'], arm_id)
            eng.native.add_collision_exclusion(g['fem_global_id'], arm_id)

    # cup ABDs: exclude from every arm ABD that is not a finger (the arm bodies
    # near the cup get spurious OBB contact otherwise — same as case_39).
    finger_offsets = {g['finger_id'] for g in grippers}
    for arm_id in arm_ids:
        if arm_id in finger_offsets:
            continue
        for abd_rec in abd_records:
            eng.native.add_collision_exclusion(arm_id, abd_rec.body_offset)

    # cloth (shirt) FEM ↔ all arm ABDs
    n_abd_total_for_shirt = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    for fem_rec in fem_records:
        fem_global_id = n_abd_total_for_shirt + fem_rec.body_offset
        for arm_id in arm_ids:
            eng.native.add_collision_exclusion(arm_id, fem_global_id)

    for g in grippers:
        eng.add_ground_collision_skip(g['fem_global_id'])
        eng.add_ground_collision_skip(g['rigid_abd_id'])

    # same-arm finger-pair mutual exclusions (left/right finger of one hand
    # start geometrically overlapped at prismatic=0).  Each "finger" now has 3
    # bodies: finger ABD, rigid-finray ABD, FEM.
    def _arm_prefix(label):
        return 'left' if label.startswith('left_') else 'right'
    for i, gi in enumerate(grippers):
        for gj in grippers[i+1:]:
            if _arm_prefix(gi['label']) != _arm_prefix(gj['label']):
                continue
            gi_bodies = (gi['rigid_abd_id'], gi['fem_global_id'])
            gj_bodies = (gj['rigid_abd_id'], gj['fem_global_id'], gj['finger_id'])
            for a in gi_bodies:
                for b in gj_bodies:
                    eng.native.add_collision_exclusion(a, b)
            # gi's finger vs gj's rigid/fem
            eng.native.add_collision_exclusion(gi['finger_id'], gj['rigid_abd_id'])
            eng.native.add_collision_exclusion(gi['finger_id'], gj['fem_global_id'])

    eng.finalize()

    # --- 5. Post-finalize setup ---
    if int(os.environ.get("CASE36_DISABLE_GRAVITY", "1")):
        for arm_id in arm_ids:
            eng.native.set_body_apply_gravity(arm_id, False)
        for g in grippers:
            eng.native.set_body_apply_gravity(g['rigid_abd_id'], False)

    fj_kappa = float(os.environ.get("CASE36_FJ_KAPPA", "1e3"))
    for g in grippers:
        eng.native.set_fixed_joint_strength(g['fj_idx'], fj_kappa)

    eng.native.set_max_revolute_step_per_frame(
        float(os.environ.get("CASE36_MAX_RAD_PER_FRAME", "0.04")))

    robot = Robot(eng)
    for i in range(len(robot.prismatic_joints)):
        eng.native.set_prismatic_strength(i, prismatic_mult)

    # Map joint indices: qpos = [L×7, R×7, grip_L, grip_R].
    # NOTE: the UMI URDF names the LEFT prismatic joints `leftarm_finger_joint*`
    # (no underscore between "left" and "arm"), unlike the right arm's
    # `right_arm_finger_joint*`.  Match both 'left_arm' and 'leftarm'.
    left_rev_indices  = [i for i, ji in enumerate(robot.revolute_joints)
                         if ji.name.startswith('left_arm_joint')]
    right_rev_indices = [i for i, ji in enumerate(robot.revolute_joints)
                         if ji.name.startswith('right_arm_joint')]
    left_pris_indices  = [i for i, ji in enumerate(robot.prismatic_joints)
                          if ji.name.startswith('left_arm')
                          or ji.name.startswith('leftarm')]
    right_pris_indices = [i for i, ji in enumerate(robot.prismatic_joints)
                          if ji.name.startswith('right_arm')
                          or ji.name.startswith('rightarm')]

    assert len(left_rev_indices) == 7 and len(right_rev_indices) == 7, \
        f"expected 7+7 revolute joints, got {len(left_rev_indices)}+{len(right_rev_indices)}"
    print(f"[replay_case39_UMI] {len(robot.revolute_joints)} revolute, "
          f"{len(robot.prismatic_joints)} prismatic; "
          f"left_pris={left_pris_indices} right_pris={right_pris_indices}",
          flush=True)
    if not left_pris_indices:
        print("[replay_case39_UMI] WARNING: no left prismatic joints matched!", flush=True)

    # ============ HEADLESS timing mode ============
    # tqdm is an optional progress-bar nicety (not in the documented dep set);
    # fall back to a plain range so headless runs work without it installed.
    try:
        import tqdm
        _progress = tqdm.tqdm
    except ImportError:
        def _progress(it):
            return it
    if int(os.environ.get("CASE39_HEADLESS", "0")):
        qpos_all = actions  # [L×7, gripL, R×7, gripR]

        ms_log = []
        nt_log = []

        _close_r = float(os.environ.get("CASE39_CLOSE_RATIO", "0."))
        _frame_start = int(os.environ.get("CASE39_FRAME_START", "0"))
        _frame_end = int(os.environ.get("CASE39_FRAME_END", str(len(qpos_all))))
        _frame_end = min(_frame_end, len(qpos_all))
        for f in _progress(range(_frame_start, _frame_end)):
            raw = qpos_all[f]
            for i, rev_idx in enumerate(left_rev_indices):
                robot.set_revolute_position(rev_idx, float(raw[i]), degree=False)
            for i, rev_idx in enumerate(right_rev_indices):
                robot.set_revolute_position(rev_idx, float(raw[7 + i]), degree=False)
            grip_L = float(raw[14]) if len(raw) > 14 else 0.0
            grip_R = float(raw[15]) if len(raw) > 15 else 0.0
            # UMI's two fingers per arm have OPPOSITE prismatic limits
            # (joint1: [0,+0.041]; joint2: [-0.041,0]) — they open in mirrored
            # directions, with 0 = closed (fingers together) and the limit
            # FARTHEST from 0 = fully open.  So per finger:
            #   open  position = lo if |lo|>|hi| else hi   (farthest-from-0 end)
            #   close position = the near-0 end (≈0)
            # A shared value or a blind "use upper" would clamp finger2 and make
            # the gripper open asymmetrically.  case_39's symmetric limits hid
            # this; UMI exposes it.
            for grip, idxs in ((grip_L, left_pris_indices), (grip_R, right_pris_indices)):
                for pi in idxs:
                    lo = robot.prismatic_joints[pi].lower_limit
                    hi = robot.prismatic_joints[pi].upper_limit
                    op = lo if abs(lo) > abs(hi) else hi    # fully-open end
                    cl = hi if abs(lo) > abs(hi) else lo    # near-0 (closed) end
                    gp = op if grip >= 0 else (op + (1.0 - _close_r) * (cl - op))
                    robot.set_prismatic_position(pi, gp, millimeters=False)
            nt0 = eng.native.get_total_newton_iters()
            t0 = time.perf_counter()
            eng.step()
            ms_log.append((time.perf_counter() - t0) * 1000.0)
            nt_log.append(eng.native.get_total_newton_iters() - nt0)

        _dump = os.environ.get("CASE39_MS_DUMP", "")
        if _dump:
            np.save(_dump, np.array(ms_log, dtype=np.float64))
            np.save(_dump.replace('.npy', '_nt.npy'), np.array(nt_log, dtype=np.int32))
            print(f"[hl] dumped {len(ms_log)} per-frame ms to {_dump}", flush=True)
        mean_ms = float(np.mean(ms_log)) if ms_log else float('nan')
        print(f"\n[hl] ran frames {_frame_start}-{_frame_end}", flush=True)
        print(f"[hl] mean step: {mean_ms:.1f} ms "
              f"({1000.0/mean_ms if mean_ms == mean_ms and mean_ms > 0 else 0.0:.1f} fps)",
              flush=True)
        return

    # --- 6. Polyscope + replay loop ---
    verts = eng.get_vertices()
    faces = eng.get_surface_faces()

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_program_name("replay_case39_UMI (polyscope)")

    state = dict(
        idx=0, running=False, last_ms=0.0,
        mesh=ps.register_surface_mesh("scene", verts, faces, color=(0.6, 0.7, 0.8)),
        verts=verts, faces=faces,
        show_edges=False, show_bvh=False, bvh_net=None,
    )
    state['mesh'].set_edge_width(0.0)
    BVH_MAX_BOXES = int(os.environ.get("CASE39UMI_BVH_MAX_BOXES", "6000"))

    def _refresh_bvh_viz():
        """(Re)build the edge-BVH box wireframe from the engine's edge BVH.
        get_edge_bvh_aabbs() is empty until after finalize + first step."""
        # get_edge_bvh_aabbs() is an optional debug-viz API not present in every
        # released wheel (e.g. v0.6.2). Degrade gracefully: if the binding lacks
        # it, the "show edge BVH" toggle just does nothing.
        try:
            aabbs = eng.native.get_edge_bvh_aabbs()
        except AttributeError:
            return
        if aabbs.shape[0] == 0:
            return  # BVH not built yet (no step taken / no collisions)
        # The BVH array is internal nodes first, leaves last. The top internal
        # nodes are HUGE boxes spanning the whole scene (arm+cloth+ground) — if
        # we draw those the wireframes cover the whole view and read as solid
        # white. Default to leaves only (the tight per-edge boxes); set
        # CASE39UMI_BVH_LEAVES=0 to draw the full tree.
        if int(os.environ.get("CASE39UMI_BVH_LEAVES", "1")):
            n_edges = (aabbs.shape[0] + 1) // 2   # edge_Num = (2N-1+1)/2
            aabbs = aabbs[-n_edges:]              # leaf nodes only
        if BVH_MAX_BOXES > 0 and aabbs.shape[0] > BVH_MAX_BOXES:
            stride = int(np.ceil(aabbs.shape[0] / BVH_MAX_BOXES))
            aabbs = aabbs[::stride]
        nodes, edges = _aabbs_to_box_wireframe(aabbs)
        net = ps.register_curve_network("edge_BVH", nodes, edges, radius=0.0006)
        net.set_color((0.95, 0.55, 0.10))
        state['bvh_net'] = net

    # Color the finray FEM verts by vertex_region (red=rigid root, blue=FEM
    # truss); the rigid ABD bodies keep the base color.
    base_color = np.array([0.6, 0.7, 0.8])
    red  = np.array([0.85, 0.25, 0.25])
    blue = np.array([0.25, 0.45, 0.85])
    colors = np.tile(base_color, (verts.shape[0], 1))
    for g in grippers:
        fo = g['fem_v_off']; fc = g['fem_rec'].vertex_count
        vr = g['vertex_region'][:fc]
        colors[fo:fo + fc] = np.where((vr == 1)[:, None], red, blue)
    state['mesh'].add_color_quantity(
        "region (red=rigid root, blue=FEM truss)", colors, defined_on='vertices',
        enabled=True)

    qpos_all = actions  # [L×7, gripL, R×7, gripR]

    def callback():
        if state['running']:
            if psim.Button("Pause"):
                state['running'] = False
        else:
            if psim.Button("Start" if state['idx'] == 0 else "Resume"):
                state['running'] = True
        psim.SameLine()
        if psim.Button("Reset"):
            state['idx'] = 0
            state['running'] = False
        psim.Text(f"frame {state['idx']:>4d} / {len(qpos_all)}")
        psim.Text(f"step: {state['last_ms']:6.1f} ms   "
                  f"FPS {(1000.0/state['last_ms']) if state['last_ms']>0 else 0.0:5.1f}")

        # --- Display toggles ---
        ch_e, val_e = psim.Checkbox("show mesh edges", state['show_edges'])
        if ch_e:
            state['show_edges'] = val_e
            state['mesh'].set_edge_width(0.5 if val_e else 0.0)
        psim.SameLine()
        ch_b, val_b = psim.Checkbox("show edge BVH", state['show_bvh'])
        if ch_b:
            state['show_bvh'] = val_b
            if val_b:
                _refresh_bvh_viz()
            elif state['bvh_net'] is not None:
                ps.remove_curve_network("edge_BVH")
                state['bvh_net'] = None

        if not state['running'] or state['idx'] >= len(qpos_all):
            return

        raw = qpos_all[state['idx']]
        q_left  = raw[0:7]
        q_right = raw[8:15]
        grip_L  = float(raw[7])
        grip_R  = float(raw[15])

        for i, rev_idx in enumerate(left_rev_indices):
            robot.set_revolute_position(rev_idx, q_left[i], degree=False)
        for i, rev_idx in enumerate(right_rev_indices):
            robot.set_revolute_position(rev_idx, q_right[i], degree=False)
        _close_r = float(os.environ.get("CASE39_CLOSE_RATIO", "0."))
        # per-finger limits (UMI's two fingers open in mirrored directions:
        # open = limit farthest from 0, closed = near-0 end — see headless loop)
        for grip, idxs in ((grip_L, left_pris_indices), (grip_R, right_pris_indices)):
            for pi in idxs:
                lo = robot.prismatic_joints[pi].lower_limit
                hi = robot.prismatic_joints[pi].upper_limit
                op = lo if abs(lo) > abs(hi) else hi
                cl = hi if abs(lo) > abs(hi) else lo
                grip_pos = op if grip >= 0 else (op + (1.0 - _close_r) * (cl - op))
                robot.set_prismatic_position(pi, grip_pos, millimeters=False)

        t0 = time.perf_counter()
        eng.step()
        state['last_ms'] = (time.perf_counter() - t0) * 1000.0

        v = eng.get_vertices()
        f = eng.get_surface_faces()
        if v.shape[0] != state['verts'].shape[0] or f.shape != state['faces'].shape:
            state['mesh'] = ps.register_surface_mesh(
                "scene", v, f, color=(0.6, 0.7, 0.8))
            state['mesh'].set_edge_width(0.5 if state['show_edges'] else 0.0)
            state['verts'], state['faces'] = v, f
        else:
            state['mesh'].update_vertex_positions(v)

        # Keep the edge-BVH wireframe in sync with the stepped geometry.
        if state['show_bvh']:
            _refresh_bvh_viz()

        state['idx'] += 1

    ps.set_user_callback(callback)
    ps.show()
    print(f"Replay finished or window closed at frame {state['idx']}/{len(qpos_all)}")


if __name__ == "__main__":
    main()
