#!/usr/bin/env python3
"""OBB arm + UMI finray gripper, CUP-GRASP trajectory + shirt — FORCE-controlled
gripper variant, with three closing modes (CASE39_GRIP_MODE).

The gripper OPENS by position-hold (no external force). It CLOSES per
CASE39_GRIP_MODE:
  * force      — pure external force (strength=0). Adaptive (finds the opening by
                 force balance) but DRIFTS/creeps under load (no position anchor).
  * impedance  — soft position-drive toward closed (strength=GRIP_K). On a soft
                 finray it still nearly fully closes (the soft cup can't balance
                 the spring before cl). Stable (no drift).
  * forcelimit — force-limited position via the DRIVE force gate. Works for RIGID
                 jaws; on the compliant finray the drive force never reflects the
                 grip (fingers keep moving) so the gate doesn't bite. Kept for
                 comparison / rigid grippers.
  * forcelock  — (DEFAULT) force-grasp (adaptive) for GRIP_HOLD_DELAY frames, then
                 LOCK the achieved opening with position-hold. Adaptive grasp +
                 stiff hold (no drift). Robust, needs no contact-force readout.
  * truegrip   — close until the REAL IPC contact force on the finger reaches
                 CASE39_GRIP_TARGET, then lock. Closes the loop on the actual grip
                 (tactile-sensor style via get_body_contact_force) — the
                 physically-correct force grasp; the lock opening adapts to the
                 object + target force. (NOTE: measure ONE finger — summing both
                 cancels by Newton's 3rd law; the finray grip force is small.)
The arm revolute joints stay position-controlled. Verified (detailed arm):
forcelock & truegrip both grasp the cup and lift it ~7 cm, held stable, no drift.

  Requires the force-control engine build (set_prismatic_force /
  get_prismatic_drive_force) — run from the Stiff-GIPC-forcetest worktree (its
  build_force has the APIs; the released v0.6.3 wheel does NOT).
  Knobs: CASE39_GRIP_MODE, CASE39_GRIP_FORCE (N), CASE39_GRIP_K, CASE39_GRIP_CLOSE_FRAMES.


This is replay_case39_UMI_sf_obb.py (OBB ridgeback arm + STRATEGY_F UMI finray
gripper) but driven by the SOLID-gripper cup-grasp trajectory qpos_case39.h5
(1018 frames, [L x7, R x7, gripL, gripR]) and with the cup + shirt loaded
EXPLICITLY (same placement/scale as replay_case39.py) instead of from a recorded
init_info.  So: OBB arm, finray soft gripper, cup + shirt unchanged, replay the
qpos_case39.h5 trajectory.

  Run (GUI):
    STIFF_SKIP_CCD_SANITY=1 python examples/replay_case39_UMI_obb_cup_shirt.py
    # custom trajectory:  ... replay_case39_UMI_obb_cup_shirt.py /path/to/qpos.h5
    # headless timing:    CASE39_HEADLESS=1 CASE39_FRAME_END=30 STIFF_SKIP_CCD_SANITY=1 ...
  Notes:
    * Requires stiff-physics >= 0.6.3 (the OBB scene tripped two GPU OOB bugs on
      0.6.2 — fixed by fc13ae9 squeue + a105ada pair-emission guards). On 0.6.3
      OBB runs ~64 ms/frame; detailed arm via CASE39_OBB_URDF=ridgeback_dual_panda2.urdf.
    * Higher-res finray: CASE39UMI_FINRAY_DIR=umi_hybrid_sf_v1340 (or _v1690).

The finray gripper is built in the STRATEGY_F HYBRID pattern (rigid mount root +
FEM truss), exactly as in replay_case39_UMI_sf_obb.py:

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
        python examples/replay_case39_UMI_sf_obb.py

    # headless timing:
    CASE39_HEADLESS=1 CASE39_FRAME_END=30 \
        STIFF_SKIP_CCD_SANITY=1 python examples/replay_case39_UMI_sf_obb.py --quiet
"""
import sys, os, math, time, re
from pathlib import Path

_ASSETS_ROOT = Path(__file__).resolve().parent.parent
_ASSETS_DIR = str(_ASSETS_ROOT / ("assets" if (_ASSETS_ROOT / "assets").is_dir() else "Assets")) + "/"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from stiff_physics import Engine, Config
from stiff_physics.robot import Robot
import polyscope as ps
import polyscope.imgui as psim

# Default to the OBB arm (ridgeback_dual_panda2_OBB.urdf == hand-KEEP variant).
# NOTE: this requires stiff-physics >= 0.6.3. On 0.6.2 the OBB scene hit two GPU
# out-of-bounds bugs (squeue + pair-emission overflow) that manifested as a
# ~3.5 s/frame "balloon" and a segfault ~frame 4; both are fixed in 0.6.3, where
# OBB runs cleanly at ~64 ms/frame. Override with CASE39_OBB_URDF=
# ridgeback_dual_panda2.urdf for the detailed arm.
URDF_PATH = _ASSETS_DIR + "sim_data/urdf/ridgeback_dual_panda_UMI/" + \
    os.environ.get("CASE39_OBB_URDF", "ridgeback_dual_panda2_OBB.urdf")
# cup + shirt actors, placed exactly like replay_case39.py (the cup-grasp scene)
CUP_MSH   = _ASSETS_DIR + "sim_data/tetmesh/softgriper_cup.msh"
SHIRT_OBJ = _ASSETS_DIR + "triMesh/shirt_6436v.obj"
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
    parser.add_argument("qpos", nargs="?",
                        default=_ASSETS_DIR + "trajectories/qpos_case39.h5")
    parser.add_argument("--quiet", action="store_true")
    args, _ = parser.parse_known_args()

    # qpos_case39.h5 is the SOLID-gripper cup-grasp trajectory: [N, 16] =
    # [L_arm x7, R_arm x7, gripL, gripR]. No init_info/actors — we load the cup
    # and shirt explicitly below (same placement as replay_case39.py).
    import h5py
    with h5py.File(args.qpos, "r") as f:
        actions = f["qpos"][:]
    init_info = {}
    robot_init_pose = np.zeros(3)
    env_cfg = {}
    print(f"Loaded {len(actions)} qpos frames from {args.qpos}  shape={actions.shape}", flush=True)

    # --- Physics config (case_39 values) ---
    prismatic_constraint_K = float(os.environ.get("CASE39_PRISMATIC_CONSTRAINT_K", "2000"))
    joint_K        = float(os.environ.get("CASE38_JOINT_K", "100"))
    revolute_K     = float(os.environ.get("CASE36_PD_K", "100"))
    prismatic_mult = float(os.environ.get("CASE36_PRISMATIC_K", "15"))

    default_cfg = dict(
        dt=float(os.environ.get("CASE39UMI_DT", "0.020")),
        velocity_damping=float(os.environ.get("CASE39_VEL_DAMP", "0.0")),
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100,
        soft_motion_rate=float(os.environ.get("CASE36_SOFT_RATE", "1e4")),
        poisson_rate=0.49,
        friction_rate=float(os.environ.get("CASE39_FRICTION", "0.8")),
        relative_dhat=float(os.environ.get("CASE39UMI_REL_DHAT", "1e-3")),
        joint_strength_ratio=joint_K,
        revolute_driving_strength_ratio=revolute_K,
        prismatic_strength_ratio=prismatic_constraint_K,
        semi_implicit_enabled=bool(int(os.environ.get("CASE39_SEMI", "0"))),
        semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1,
        newton_tol=float(os.environ.get("CASE39_NEWTON_TOL", "5e-2")),
        newton_iter_cap=int(os.environ.get("CASE39_NEWTON_CAP", "50")),
        preconditioner_type=int(os.environ.get("CASE39_PRECOND", "1")),
        # ground at y=-1.67 like replay_case39.py (the cup-grasp scene): the cup
        # (y~-0.4) and shirt (y~0) must have a floor BELOW them. The fold-shirt
        # scene used ground_offset=0.75 (cloth is grasped, no floor needed) — that
        # height is ABOVE the cup/shirt here, which is why they fell forever.
        ground_offset=float(os.environ.get("CASE39_GROUND_OFFSET", "-1.67")),
        assets_dir=_ASSETS_DIR,
    )

    env_cfg["assets_dir"] = _ASSETS_DIR

    print("default_cfg:\n", default_cfg)
    print("record_cfg:\n", env_cfg)

    cfg = Config(**default_cfg)
    # With the v0.6.3 pair-emission auto-grow fix (a105ada), the collision buffers
    # grow on demand, so we no longer need the old buff_scale=64 workaround (which
    # pre-allocated ~125M CCD pairs and OOMs on a 24 GB GPU). Default scale 6 lets
    # the engine auto-grow to the real pair count (~17M here).
    cfg._cfg.collision_detection_buff_scale = float(
        os.environ.get("CASE39UMI_BUFF_SCALE", "6.0"))
    eng = Engine(cfg)
    print("\n[replay_case39_UMI] === building case_39 (UMI gripper) scene ===", flush=True)

    if args.quiet:
        eng.set_log_level(0)

    ################################### robot init

    # --- 1. Load URDF (UMI option X: each finger link IS the rigid mount seat;
    #         no separate *_mount ABD links any more) ---
    # Arm base placed exactly like replay_case39.py (cup-grasp scene): rotate
    # -90deg about X, translate to y=-3.0. This MUST match where qpos_case39.h5
    # was recorded, otherwise the cup at [0.67,-0.2,-0.4] is unreachable.
    arm_tf = np.eye(4)
    arm_tf[:3, :3] = ARM_SCALE * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    arm_tf[1, 3] = -3.0
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
    # load actors (cup/beaker ABD + cloth FEM) from the recorded init_info.
    # FEM actors (cloth/shirt) are loaded straight by the engine; only ABD
    # actors (e.g. a rigid cup/beaker) need trimesh to read their surface mesh.
    # The fold-shirt replay has no ABD actors, so trimesh is imported lazily
    # below — keeping it off the required-dependency list for that example.
    abd_objs = []
    fem_objs = []
    for key, value in init_info.items():
        init_pose = value['initial_pose']
        collision_mesh = value["collision_mesh"]
        if "/data/stiff-physics/franka_sim/assets_new" in collision_mesh:
            collision_mesh = collision_mesh.replace(
                "/data/stiff-physics/franka_sim/assets_new/", _ASSETS_DIR)
            print(f"remapped collision mesh path to {collision_mesh}")
        if "/data/stiff-physics/franka_sim/assets" in collision_mesh:
            collision_mesh = collision_mesh.replace(
                "/data/stiff-physics/franka_sim/assets/", _ASSETS_DIR)
            print(f"remapped collision mesh path to {collision_mesh}")
        body_type = value["body_type"]
        if body_type == "ABD":
            abd_objs.append([init_pose, collision_mesh])
        else:
            fem_objs.append([init_pose, collision_mesh])

    # --- explicit cup (rigid ABD), placed exactly like replay_case39.py ---
    # (init_info is empty for qpos_case39.h5, so we load the cup directly.)
    abd_records = []
    cup_scale = float(os.environ.get("CASE39_CUP_SCALE", "0.8"))
    cup_xyz = np.array([float(s) for s in
                        os.environ.get("CASE39_CUP_XYZ", "0.67,-0.2,-0.4").split(",")])
    cup_T = np.eye(4); cup_T[:3, :3] *= cup_scale; cup_T[:3, 3] = cup_xyz
    eng.load_mesh(CUP_MSH, dimensions=3, body_type="ABD",
                  transform=cup_T, young_modulus=1e8, boundary_type="Free")
    cup_rec = eng.get_load_records()[-1]
    abd_records.append(cup_rec)
    print(f"[replay_case39_UMI_obb] cup body_id={cup_rec.body_offset} "
          f"scale={cup_scale} at {cup_xyz}", flush=True)

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
    # --- explicit shirt (cloth FEM), placed exactly like replay_case39.py ---
    fem_records = []
    if not int(os.environ.get("CASE39UMI_SKIP_CLOTH", "0")):
        shirt_scale = float(os.environ.get("CASE39_SHIRT_SCALE", "1.0"))
        shirt_xyz = np.array([float(s) for s in
                              os.environ.get("CASE39_SHIRT_XYZ", "0.67,0,0").split(",")])
        shirt_T = np.eye(4); shirt_T[:3, :3] *= shirt_scale; shirt_T[:3, 3] = shirt_xyz
        eng.load_mesh(SHIRT_OBJ, body_type="FEM",
                      transform=shirt_T, young_modulus=1e2, dimensions=2)
        fem_records.append(eng.get_load_records()[-1])
        print(f"[replay_case39_UMI_obb] shirt body_id={fem_records[-1].body_offset} "
              f"at {shirt_xyz}", flush=True)

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
        # stitches are added in call order, so this finger's springs occupy a
        # contiguous range [stitch_pair_start, +n_stitch) — used by the on-GPU
        # get_stitch_max_stretch(start, count) gate (no full-vertex D2H).
        g['stitch_pair_start'] = total_pairs
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

    # cloth (shirt) FEM ↔ all arm ABDs, and ↔ finray rigid roots (the rigid
    # mount root never touches the cloth — only the FEM truss grasps it).
    n_abd_total_for_shirt = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    for fem_rec in fem_records:
        fem_global_id = n_abd_total_for_shirt + fem_rec.body_offset
        for arm_id in arm_ids:
            eng.native.add_collision_exclusion(arm_id, fem_global_id)
        for g in grippers:
            eng.native.add_collision_exclusion(g['rigid_abd_id'], fem_global_id)

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

    # Optional: exclude finray FEM self-collision. DEFAULT OFF (self-collision
    # KEPT — physically complete). Set CASE39_NO_FINRAY_SELF=1 to remove the dense
    # finray-self CCD pairs (~1.2x faster over a full replay, no effect on Newton
    # convergence) at the cost of the truss being able to self-penetrate.
    if int(os.environ.get("CASE39_NO_FINRAY_SELF", "0")):
        for g in grippers:
            eng.native.add_collision_exclusion(g['fem_global_id'], g['fem_global_id'])
        print("[DIAG] finray FEM self-collision DISABLED for bodies "
              + str([g['fem_global_id'] for g in grippers]), flush=True)

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
    # FORCE-CONTROL variant — clean definition (avoids the "gripper flies off"
    # failure of pure strength=0 force control):
    #   * OPEN  (grip >= 0): POSITION control holds the fingers at the open end,
    #     NO external force. (Pure force during open pushes the unconstrained
    #     joint past its limit -> fingers fly off.)
    #   * CLOSE (grip <  0): switch that joint to FORCE control (strength=0) and
    #     apply a constant closing force; it closes until it grips the cup / the
    #     fingers meet (reaction balances the force) — no position target needed.
    # The closing force is the only "guessed" knob (CASE39_GRIP_FORCE).
    # Three gripper-control modes (CASE39_GRIP_MODE), to compare:
    #   force       (A) pure force, strength=0 — droops/creeps under load (bad).
    #   impedance   (1) soft position toward closed, strength=GRIP_K. Grip force
    #               = K*compression (emergent, bounded by the soft spring). Stable.
    #   forcelimit  (2) force-limited position (= a real gripper): position-drive
    #               toward closed with a STIFF spring, but advance the target only
    #               while the drive force < GRIP_FORCE; once it hits the limit,
    #               stop -> equilibrium at exactly GRIP_FORCE, opening = object
    #               width. Stiff (no droop) AND force-capped (won't crush).
    # DEFAULT trackgrip: contact-force-regulated POSITION drive (same logic as
    # case_umi_finray_force_ui.py). Always position-driven (stable, no pure-force
    # overshoot/creep); the commanded target MARCHES toward closed at CLOSE_SPEED
    # but PAUSES once the REAL grip (contact) force on the soft fingertip reaches
    # GRIP_TARGET -> force-limited PARTIAL close (the "half-closed" grasp). It
    # auto-resumes if the object is removed (grip force drops). Unlike the rigid
    # prismatic lag, the contact force actually registers the soft finray grasp.
    # Satisfies stable + adaptive + resume-on-vanish. 4 older modes kept for cmp.
    GRIP_MODE  = os.environ.get("CASE39_GRIP_MODE", "trackgrip")
    GRIP_FORCE = float(os.environ.get("CASE39_GRIP_FORCE", "30.0"))   # F_max (force/forcelimit/forcelock)
    GRIP_K     = float(os.environ.get("CASE39_GRIP_K", "15.0"))       # closing position stiffness
    GRIP_CLOSE_FRAMES = float(os.environ.get("CASE39_GRIP_CLOSE_FRAMES", "40"))  # forcelimit advance rate
    GRIP_HOLD_DELAY = int(os.environ.get("CASE39_GRIP_HOLD_DELAY", "25"))  # forcelock: force frames before locking
    GRIP_LOCK_K = float(os.environ.get("CASE39_GRIP_LOCK_K", "15.0"))      # forcelock/truegrip: hold stiffness
    GRIP_CLOSE_FORCE = float(os.environ.get("CASE39_GRIP_CLOSE_FORCE", "40.0"))  # truegrip: closing actuation force
    # TARGET grip (contact) force in IPC units — the trackgrip/truegrip firmness
    # knob. Bigger = squeeze harder before pausing; smaller = gentler/earlier stop.
    GRIP_TARGET = float(os.environ.get("CASE39_GRIP_TARGET", "0.02"))
    # stitchgrip: pause the march once the MAX per-pair STITCH STRETCH reaches
    # STITCH_THRESH (a length, mm). The stitch ties the soft truss base to the
    # rigid root; as the fingertip grips, the most-loaded stitch spring stretches.
    # That max stretch is the gate signal — CHEAP (vertex distances, no BVH) and
    # cleaner/less noisy than the vector net force. Bigger thresh = firmer grasp.
    #   (force in the single tightest stitch spring = STITCH_K * stretch.)
    STITCH_K     = float(os.environ.get("CASE36_SOFT_RATE", "1e4"))                # stitch stiffness (force readout)
    STITCH_THRESH = float(os.environ.get("CASE39_STITCH_THRESH", "0.005")) * 1e-3  # max stitch stretch (mm->m)
    # HYSTERESIS: once the grip signal reaches STITCH_THRESH the target LATCHES
    # (stops advancing) -> a STABLE hold, no slow creep that squeezes the object
    # out. It only RESUMES closing if the signal drops below STITCH_THRESH*RESUME_FRAC
    # (object loosened / removed). Lower RESUME_FRAC = stickier hold.
    RESUME_FRAC = float(os.environ.get("CASE39_STITCH_RESUME_FRAC", "0.5"))
    # STITCH_GPU=1: compute the gate signal with the on-GPU reduction
    # (get_stitch_max_stretch) — returns a scalar, NO full-vertex D2H. =0 uses the
    # CPU path (get_vertices + numpy). Same signal; this is the perf comparison.
    STITCH_GPU = int(os.environ.get("CASE39_STITCH_GPU", "0"))
    # forcebarrier: PURE FORCE close, but a solver-side IPC barrier armed at the
    # closed limit cl guarantees d never crosses cl (hard no-overshoot). A small
    # force is enough — the barrier sets the stop, not the force magnitude.
    BARRIER_FORCE = float(os.environ.get("CASE39_BARRIER_FORCE", "60.0"))           # closing force (N)
    BARRIER_DHAT  = float(os.environ.get("CASE39_BARRIER_DHAT", "0.002"))          # standoff band from cl (m)
    BARRIER_KAPPA = float(os.environ.get("CASE39_BARRIER_KAPPA", "1e3"))           # barrier stiffness
    # trackgrip march speed (how fast the setpoint advances toward closed)
    CLOSE_SPEED  = float(os.environ.get("CASE39_GRIP_CLOSE_SPEED", "0.0012"))  # target march/frame (m)
    for i in range(len(robot.prismatic_joints)):
        eng.native.set_prismatic_strength(i, prismatic_mult)  # start in position(open) mode
    # forcebarrier: arm the one-sided IPC barrier at each finger's closed limit cl
    # ONCE, so the solver can never let the opening cross cl no matter the force.
    if GRIP_MODE == "forcebarrier":
        for i in range(len(robot.prismatic_joints)):
            lo = robot.prismatic_joints[i].lower_limit
            hi = robot.prismatic_joints[i].upper_limit
            op = lo if abs(lo) > abs(hi) else hi
            cl = hi if abs(lo) > abs(hi) else lo
            bdir = 1.0 if (op - cl) > 0 else -1.0     # gap = (d-cl)*bdir > 0 while open
            eng.native.set_prismatic_limit_barrier(i, cl, bdir, BARRIER_DHAT, BARRIER_KAPPA)
        print(f"[forcebarrier] armed cl-barrier on {len(robot.prismatic_joints)} prismatic "
              f"joints (dhat={BARRIER_DHAT*1000:.1f}mm, kappa={BARRIER_KAPPA:g}, F={BARRIER_FORCE}N)",
              flush=True)
    _fl_tgt = {}   # forcelimit: per-joint advancing target (meters)
    _grip_tgt = {} # trackgrip/stitchgrip: per-joint marched setpoint (meters)
    _grip_latch = {} # stitchgrip hysteresis: per-joint True once gripped (hold, no creep)
    _lock = {}     # forcelock/truegrip: per-joint {'n':.., 'd': locked opening or None}

    def _finger_contact(fem_recs):
        """Max |IPC contact force| over a side's fingers (each measured alone, so
        the two fingers' opposite grip forces don't cancel) = the real grip force.
        EXPENSIVE: rebuilds the BVH + collision pairs every call."""
        gmax = 0.0
        for fr in (fem_recs or []):
            cf = eng.native.get_body_contact_force(fr.vertex_offset, fr.vertex_count)
            gmax = max(gmax, float((cf[0]**2 + cf[1]**2 + cf[2]**2) ** 0.5))
        return gmax

    def _side_stitch(grippers_side):
        """Max per-pair STITCH STRETCH (m) over a side's fingers = how far the
        most-loaded stitch spring is pulled (the rigid root's reaction to the soft
        truss grip). Two paths, SAME signal:
          STITCH_GPU=1: on-GPU reduction get_stitch_max_stretch() -> scalar, no
                        full-vertex D2H (only 8 bytes/finger come back).
          STITCH_GPU=0: get_vertices() (full D2H) + numpy max. NOTE the CPU path
                        MUST use get_vertices() (LIVE GPU); get_vertices_host() is
                        a stale host cache and would report ~0 forever."""
        if STITCH_GPU:
            m = _stitch_cache['maxes']
            if m is None:                      # safety: compute if not refreshed this frame
                m = eng.native.get_stitch_max_stretch_batched(_seg_starts, _seg_counts)
            return float(max((m[g['seg_idx']] for g in (grippers_side or [])), default=0.0))
        allv = np.asarray(eng.native.get_vertices())
        fmax = 0.0
        for g in (grippers_side or []):
            rv = g['rigid_v_idx']
            fe = allv[g['fem_v_off'] + rv]
            ri = allv[g['rigid_abd_v_off']: g['rigid_abd_v_off'] + len(rv)]
            smax = float(np.linalg.norm(fe - ri, axis=1).max())   # max per-pair stretch (m)
            fmax = max(fmax, smax)
        return fmax

    def drive_gripper(grip, idxs, fem_recs=None, grippers_side=None):
        """Drive one arm's gripper fingers. grip>=0 = OPEN (position-hold, no
        force); grip<0 = CLOSE per GRIP_MODE."""
        # trackgrip/truegrip need the side's REAL contact force (measured once per
        # call). For the compliant finray this — not the prismatic lag — is the
        # signal that actually rises when the soft fingertip grips the object (the
        # rigid prismatic root keeps moving as the soft truss deforms, so its lag
        # never reports the grasp). stitchgrip uses the cheaper stitch stretch.
        gforce = _finger_contact(fem_recs) if (GRIP_MODE in ("truegrip", "trackgrip") and grip < 0) else 0.0
        sstretch = _side_stitch(grippers_side) if (GRIP_MODE == "stitchgrip" and grip < 0) else 0.0
        for pi in idxs:
            lo = robot.prismatic_joints[pi].lower_limit
            hi = robot.prismatic_joints[pi].upper_limit
            op = lo if abs(lo) > abs(hi) else hi    # fully-open end
            cl = hi if abs(lo) > abs(hi) else lo    # near-0 (closed) end
            if GRIP_MODE == "position":              # BASELINE: plain direct-set prismatic, NO gate
                # original behaviour — just command closed/open; reads NO grip
                # signal (no stitch, no contact). The benchmark reference.
                eng.native.set_prismatic_force(pi, 0.0)
                eng.native.set_prismatic_strength(pi, prismatic_mult)
                eng.native.set_prismatic_target(pi, cl if grip < 0 else op)
                continue
            if GRIP_MODE == "forcebarrier":          # PURE FORCE + solver cl-barrier (hard no-overshoot)
                cd = 1.0 if (cl - op) > 0 else -1.0
                if grip < 0:                          # CLOSE: constant force; barrier stops at cl
                    eng.native.set_prismatic_strength(pi, 0.0)
                    eng.native.set_prismatic_force(pi, cd * BARRIER_FORCE)
                else:                                 # OPEN: position-hold to open end
                    eng.native.set_prismatic_force(pi, 0.0)
                    eng.native.set_prismatic_strength(pi, prismatic_mult)
                    eng.native.set_prismatic_target(pi, op)
                continue
            if GRIP_MODE == "trackgrip":             # DEFAULT: contact-force-regulated position march
                cdir = 1.0 if (cl - op) > 0 else -1.0
                d = eng.native.get_prismatic_current_distance(pi)
                eng.native.set_prismatic_force(pi, 0.0)            # position drive only
                eng.native.set_prismatic_strength(pi, prismatic_mult)
                tgt = _grip_tgt.get(pi, d)
                if grip < 0:                          # CLOSE: march in while grip force below target
                    if gforce < GRIP_TARGET:         # not gripping hard enough yet -> keep closing
                        tgt += cdir * CLOSE_SPEED
                    # else holding at the target grip force -> pause (force-limited
                    # PARTIAL close); auto-resumes if the object is removed (gforce
                    # drops). NOT locked, so it tracks the object.
                    tgt = min(tgt, cl) if cdir > 0 else max(tgt, cl)
                else:                                 # OPEN: march back to open end
                    tgt += -cdir * CLOSE_SPEED
                    tgt = max(tgt, op) if cdir > 0 else min(tgt, op)
                _grip_tgt[pi] = tgt
                eng.native.set_prismatic_target(pi, tgt)
                continue
            if GRIP_MODE == "stitchgrip":            # contact-free variant: gate on STITCH stretch
                cdir = 1.0 if (cl - op) > 0 else -1.0
                d = eng.native.get_prismatic_current_distance(pi)
                eng.native.set_prismatic_force(pi, 0.0)            # position drive only
                eng.native.set_prismatic_strength(pi, prismatic_mult)
                tgt = _grip_tgt.get(pi, d)
                if grip < 0:                          # CLOSE with HYSTERESIS latch (stable hold)
                    latched = _grip_latch.get(pi, False)
                    if not latched:
                        if sstretch >= STITCH_THRESH: # gripped -> LATCH: freeze, stop creeping in
                            latched = True
                        else:
                            tgt += cdir * CLOSE_SPEED  # not gripping yet -> keep closing
                    elif sstretch < STITCH_THRESH * RESUME_FRAC:  # object loosened/gone -> resume
                        latched = False
                        tgt += cdir * CLOSE_SPEED
                    _grip_latch[pi] = latched
                    tgt = min(tgt, cl) if cdir > 0 else max(tgt, cl)
                else:                                 # OPEN
                    _grip_latch.pop(pi, None)
                    tgt += -cdir * CLOSE_SPEED
                    tgt = max(tgt, op) if cdir > 0 else min(tgt, op)
                _grip_tgt[pi] = tgt
                eng.native.set_prismatic_target(pi, tgt)
                continue
            if grip >= 0:                            # OPEN -> position, no force
                eng.native.set_prismatic_force(pi, 0.0)
                eng.native.set_prismatic_strength(pi, prismatic_mult)
                eng.native.set_prismatic_target(pi, op)
                _fl_tgt.pop(pi, None); _lock.pop(pi, None)
                continue
            cd = 1.0 if (cl - op) > 0 else -1.0
            if GRIP_MODE == "force":                 # (A) pure force (adaptive but drifts)
                eng.native.set_prismatic_strength(pi, 0.0)
                eng.native.set_prismatic_force(pi, cd * GRIP_FORCE)
            elif GRIP_MODE == "impedance":           # (1) soft position to closed
                eng.native.set_prismatic_force(pi, 0.0)
                eng.native.set_prismatic_strength(pi, GRIP_K)
                eng.native.set_prismatic_target(pi, cl)
            elif GRIP_MODE == "forcelimit":          # (2) force-limited position
                eng.native.set_prismatic_force(pi, 0.0)
                eng.native.set_prismatic_strength(pi, GRIP_K)
                if pi not in _fl_tgt:
                    _fl_tgt[pi] = op                 # start advancing from open
                if abs(eng.native.get_prismatic_drive_force(pi)) < GRIP_FORCE:
                    _fl_tgt[pi] += cd * abs(op - cl) / GRIP_CLOSE_FRAMES
                    _fl_tgt[pi] = max(min(_fl_tgt[pi], max(op, cl)), min(op, cl))
                eng.native.set_prismatic_target(pi, _fl_tgt[pi])
            elif GRIP_MODE == "forcelock":           # (3) force-grasp (timed) then lock
                st_pi = _lock.setdefault(pi, {'n': 0, 'd': None})
                if st_pi['d'] is None:               # PHASE 1: pure force grasp (adaptive)
                    eng.native.set_prismatic_strength(pi, 0.0)
                    eng.native.set_prismatic_force(pi, cd * GRIP_FORCE)
                    st_pi['n'] += 1
                    if st_pi['n'] >= GRIP_HOLD_DELAY:    # grasp settled -> lock current opening
                        st_pi['d'] = eng.native.get_prismatic_current_distance(pi)
                else:                                # PHASE 2: position-hold at the grasped opening
                    eng.native.set_prismatic_force(pi, 0.0)
                    eng.native.set_prismatic_strength(pi, GRIP_LOCK_K)
                    eng.native.set_prismatic_target(pi, st_pi['d'])
            else:                                    # (4) truegrip: close until the REAL
                # IPC contact force on the finger reaches GRIP_TARGET, then lock.
                # This closes the loop on the actual grip (tactile-sensor style),
                # so it stops/locks at the true grip force regardless of finray
                # compliance — the physically-correct force grasp.
                st_pi = _lock.setdefault(pi, {'d': None})
                if st_pi['d'] is None:               # PHASE 1: close by force, watch real grip
                    if gforce < GRIP_TARGET:
                        eng.native.set_prismatic_strength(pi, 0.0)
                        eng.native.set_prismatic_force(pi, cd * GRIP_CLOSE_FORCE)
                    else:                            # reached target grip -> lock opening
                        st_pi['d'] = eng.native.get_prismatic_current_distance(pi)
                else:                                # PHASE 2: position-hold at the grasped opening
                    eng.native.set_prismatic_force(pi, 0.0)
                    eng.native.set_prismatic_strength(pi, GRIP_LOCK_K)
                    eng.native.set_prismatic_target(pi, st_pi['d'])

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

    # Per-side FEM finger bodies (for truegrip's real contact-force readout).
    left_fem  = [g['fem_rec'] for g in grippers if g['label'].startswith('left')]
    right_fem = [g['fem_rec'] for g in grippers if g['label'].startswith('right')]
    left_grippers  = [g for g in grippers if g['label'].startswith('left')]
    right_grippers = [g for g in grippers if g['label'].startswith('right')]

    # BATCHED on-GPU stitch gate: one segment per finger (here 4; for multi-env it
    # would be every finger of every env). One kernel launch/frame computes all
    # per-finger maxes; _refresh_stitch_gpu() caches them so both sides reuse the
    # single launch. Per-finger isolation (block-per-segment) => no env crosstalk.
    for _i, _g in enumerate(grippers):
        _g['seg_idx'] = _i
    _seg_starts = np.array([g['stitch_pair_start'] for g in grippers], dtype=np.int32)
    _seg_counts = np.array([g['n_stitch']          for g in grippers], dtype=np.int32)
    _stitch_cache = {'maxes': None}

    def _refresh_stitch_gpu():
        """One batched kernel launch/frame -> per-finger max stretch, cached."""
        _stitch_cache['maxes'] = eng.native.get_stitch_max_stretch_batched(
            _seg_starts, _seg_counts)

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
        # CASE39_BENCH=1: skip the per-frame [cup] diagnostic (it calls the
        # EXPENSIVE get_body_contact_force + get_vertices regardless of mode, which
        # would dominate/pollute a controller-overhead benchmark).
        _BENCH = int(os.environ.get("CASE39_BENCH", "0"))
        for f in _progress(range(_frame_start, _frame_end)):
            raw = qpos_all[f]
            for i, rev_idx in enumerate(left_rev_indices):
                robot.set_revolute_position(rev_idx, float(raw[i]), degree=False)
            for i, rev_idx in enumerate(right_rev_indices):
                robot.set_revolute_position(rev_idx, float(raw[7 + i]), degree=False)
            grip_L = float(raw[14]) if len(raw) > 14 else 0.0
            grip_R = float(raw[15]) if len(raw) > 15 else 0.0
            if STITCH_GPU and GRIP_MODE == "stitchgrip" and (grip_L < 0 or grip_R < 0):
                _refresh_stitch_gpu()        # ONE batched kernel launch for all fingers
            drive_gripper(grip_L, left_pris_indices, left_fem, left_grippers)
            drive_gripper(grip_R, right_pris_indices, right_fem, right_grippers)
            nt0 = eng.native.get_total_newton_iters()
            t0 = time.perf_counter()
            eng.step()
            ms_log.append((time.perf_counter() - t0) * 1000.0)
            nt_log.append(eng.native.get_total_newton_iters() - nt0)
            # diagnostic: cup centroid + LEFT gripper opening d + drive force.
            # (d = actual finger opening along the rail; for a force-limited grasp
            #  it should STOP at the cup width, NOT go to ~0 / fully closed.)
            if (not _BENCH) and (f % 25 == 0 or (grip_L < 0 and f % 5 == 0)):
                cv = np.asarray(eng.get_vertices())[cup_rec.vertex_offset:
                                                    cup_rec.vertex_offset + cup_rec.vertex_count]
                cc = cv.mean(axis=0)
                _lp = left_pris_indices[0] if left_pris_indices else -1
                d_open = eng.native.get_prismatic_current_distance(_lp) if _lp >= 0 else 0.0
                f_drv  = eng.native.get_prismatic_drive_force(_lp) if _lp >= 0 else 0.0
                # REAL grip force = |net IPC contact force| on ONE finger's FEM
                # (summing BOTH fingers cancels: they press the cup from opposite
                #  sides). Print each left finger separately + the cup.
                _lf = [g for g in grippers if g['label'].startswith('left')]
                gper = [float(np.linalg.norm(eng.native.get_body_contact_force(
                            g['fem_rec'].vertex_offset, g['fem_rec'].vertex_count)))
                        for g in _lf]
                _cupf = float(np.linalg.norm(eng.native.get_body_contact_force(
                    cup_rec.vertex_offset, cup_rec.vertex_count)))
                # STITCH stretch on left fingers: ||pos_fem - pos_abd|| over the
                # stitched pairs (= stitch spring load / k). If the soft-truss grip
                # reaction reaches the rigid root, the stitch is stretched and the
                # prismatic drive force f_drv rises; if the soft truss absorbs it,
                # both stay ~0 while the contact force (gper) is what rises.
                _allv = np.asarray(eng.get_vertices())
                _smax = 0.0       # max per-pair stretch (mm)
                _sforce = 0.0     # net stitch reaction force (N) = k*|Σ Δ|
                for g in _lf:
                    rv = g['rigid_v_idx']
                    fe = _allv[g['fem_v_off'] + rv]
                    ri = _allv[g['rigid_abd_v_off']: g['rigid_abd_v_off'] + len(rv)]
                    _smax = max(_smax, float(np.linalg.norm(fe - ri, axis=1).max()))
                    _sforce = max(_sforce, STITCH_K * float(np.linalg.norm((fe - ri).sum(axis=0))))
                print(f"[cup] f={f} gripL={grip_L:+.1f} "
                      f"cup=({cc[0]:+.3f},{cc[1]:+.3f},{cc[2]:+.3f}) "
                      f"L_open={d_open:+.4f}m f_drv={f_drv:+.3f} "
                      f"stitch={_smax*1000:.3f}mm stitchF={_sforce:.3f}N "
                      f"fingerGRIP={['%.2f'%x for x in gper]} CUPf={_cupf:.2f}", flush=True)

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
        aabbs = eng.native.get_edge_bvh_aabbs()
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

        # --- MANUAL control mode: YOU drive the arm (a pose slider over the
        # recorded frames) and the gripper via a BINARY CLOSE/OPEN button. The
        # physics steps continuously so you can hold a pose, hit CLOSE, and watch
        # the force grip settle / hold / lift. (Replay mode = checkbox off.)
        state.setdefault('manual', False)
        state.setdefault('ui_close', False)
        state.setdefault('arm_frame', 0)
        _, state['manual'] = psim.Checkbox("MANUAL control (drive arm + binary gripper)",
                                           state['manual'])
        if state['manual']:
            _, state['arm_frame'] = psim.SliderInt("arm pose (frame)", state['arm_frame'],
                                                   0, len(qpos_all) - 1)
            if psim.Button("CLOSE (force)"):
                state['ui_close'] = True
            psim.SameLine()
            if psim.Button("OPEN"):
                state['ui_close'] = False
            psim.SameLine()
            psim.TextUnformatted("gripper: [CLOSED]" if state['ui_close'] else "gripper: [open]")

        if state['manual']:
            raw = qpos_all[state['arm_frame']]
            grip_L = grip_R = (-1.0 if state['ui_close'] else 1.0)
        else:
            if not state['running'] or state['idx'] >= len(qpos_all):
                return
            raw = qpos_all[state['idx']]
            grip_L = float(raw[14])
            grip_R = float(raw[15])

        q_left  = raw[0:7]                  # qpos_case39.h5 layout: [L x7, R x7, gripL, gripR]
        q_right = raw[7:14]
        for i, rev_idx in enumerate(left_rev_indices):
            robot.set_revolute_position(rev_idx, q_left[i], degree=False)
        for i, rev_idx in enumerate(right_rev_indices):
            robot.set_revolute_position(rev_idx, q_right[i], degree=False)
        if STITCH_GPU and GRIP_MODE == "stitchgrip" and (grip_L < 0 or grip_R < 0):
            _refresh_stitch_gpu()            # ONE batched kernel launch for all fingers
        drive_gripper(grip_L, left_pris_indices, left_fem, left_grippers)
        drive_gripper(grip_R, right_pris_indices, right_fem, right_grippers)

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

        if not state['manual']:        # only the auto-replay advances the frame
            state['idx'] += 1

    ps.set_user_callback(callback)
    ps.show()
    print(f"Replay finished or window closed at frame {state['idx']}/{len(qpos_all)}")


if __name__ == "__main__":
    main()
