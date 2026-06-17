#!/usr/bin/env python3
"""Interactive (UI-driven, NO replay) demo of the UMI finray gripper grasping
the case_39 fold-shirt cloth, with an ABD cup/beaker in the scene.

Same physics scene as replay_case39_UMI_sf.py — non-OBB (detailed) arm
collision + STRATEGY_F hybrid finray gripper (rigid mount root + FEM truss,
gap-0 stitch) — but instead of replaying a recorded qpos trajectory, the arm
joints and gripper open/close are driven LIVE by polyscope/imgui sliders.

Scene:
  * ridgeback dual-panda UMI arm (detailed URDF collision)
  * 4 STRATEGY_F finray grippers (FEM truss + rigid root)
  * case_39 shirt (cloth FEM), placed from the bundled trajectory's init pose
  * case_39 cup/beaker (softgriper_cup.msh) as a rigid ABD on the table
    (pose anchored to the shirt + CASE_UMI_CUP_OFF; set CASE_UMI_CUP=0 to skip)

The bundled .hdf5 is read ONLY for the shirt placement + the arm's initial
pose (actions[0]); no trajectory is played back.

FORCE-GRIPPER variant: instead of the 0..1 position sliders, the gripper is a
BINARY pure-FORCE control — "CLOSE (force)" / "OPEN" buttons. CLOSE drives the
prismatic with a constant external force (strength=0); the fingers find their
opening by force balance against the object, and continue to fully close if the
object is removed. OPEN position-holds the fingers open. The arm 7-DOF joints are
still driven by per-joint sliders. So you manually pose the arm and binary-grip
to verify the pure force control. Requires the force build (set_prismatic_force).

UI: "Run/Pause", "Reset pose", binary gripper "CLOSE (force)"/"OPEN", per-joint
sliders for both 7-DOF arms. Tune the close force with CASE_UMI_GRIP_FORCE=<N>.

Usage (GUI):
    CASE39_PRECOND=0 STIFF_SKIP_CCD_SANITY=1 \
        python examples/case_umi_finray_ui.py
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

URDF_PATH = _ASSETS_DIR + "sim_data/urdf/ridgeback_dual_panda_UMI/" + \
    os.environ.get("CASE39_OBB_URDF", "ridgeback_dual_panda2_OBB_handKEEP.urdf")
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
    parser.add_argument("--replay", type=str, default="",
                        help="optional: an .hdf5 to take the shirt/arm init pose "
                             "from. If omitted, self-contained hardcoded poses "
                             "(case_39 fold-shirt setup) are used — NO trajectory "
                             "is ever played back; the arm is UI-driven.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Self-contained scene poses (baked from the case_39 fold-shirt setup) so the
    # example needs NO replay file. --replay only overrides these if given.
    SHIRT_POSE = np.array([
        -0.147693, 0.989033, 0.0, -0.038994,  -0.0, -0.0, 1.0, 1.025979,
         0.989033, 0.147693, 0.0, -0.059208,   0.0,  0.0, 0.0, 1.0]).reshape(4, 4)
    SHIRT_MESH = _ASSETS_DIR + "objects/m-panda_single/scaled.obj"
    ROBOT_INIT_POSE = np.array([-0.8, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    ARM_INIT_Q = np.array([                       # [L×7, gripL, R×7, gripR]
        -0.423562, 0.076357, 0.267709, -2.317659, -1.071812, 2.201335, 1.420468, 1.0,
         0.966705, 0.349781, -1.353784, -1.946211, 1.223486, 1.982947, 0.045421, 1.0])

    import json
    if args.replay and os.path.exists(args.replay):
        import h5py
        with h5py.File(args.replay, "r") as f:
            init_info = json.loads(f.attrs["object_init_info"])
            actions = f["actions"][:]
            robot_init_pose = f.attrs["robot_init_pose"]
            env_cfg = json.loads(f.attrs["env_cfg"])
        print(f"[UI] loaded init pose from {args.replay}", flush=True)
    else:
        init_info = {"panda_cloth_0_0": {"initial_pose": SHIRT_POSE.tolist(),
                                         "collision_mesh": SHIRT_MESH,
                                         "body_type": "FEM"}}
        actions = ARM_INIT_Q[None, :]
        robot_init_pose = ROBOT_INIT_POSE
        env_cfg = {}
        print("[UI] self-contained scene (no replay file)", flush=True)

    # --- Physics config (case_39 values) ---
    prismatic_constraint_K = float(os.environ.get("CASE39_PRISMATIC_CONSTRAINT_K", "2000"))
    joint_K        = float(os.environ.get("CASE38_JOINT_K", "100"))
    revolute_K     = float(os.environ.get("CASE36_PD_K", "100"))
    prismatic_mult = float(os.environ.get("CASE36_PRISMATIC_K", "15"))

    default_cfg = dict(
        dt=0.020,
        velocity_damping=float(os.environ.get("CASE_UMI_VEL_DAMP", "0.0")),
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
    # Born-posed: load the arm ALREADY at the start pose (actions[0]) by passing
    # initial_joint_angles. The importer syncs target_angle = initial_angle_offset
    # = the given angle, so the joint sits there with ZERO driving force and the
    # UI sliders (which read get_revolute_target_deg) show that pose. Without this
    # the arm loads at the URDF default and the joint controller slews toward the
    # start pose on its own over many steps — which looks exactly like a replay.
    _q0 = (actions[0] if len(actions) else np.zeros(16)).astype(float)
    init_joint_angles = {}
    for _i in range(7):
        init_joint_angles[f"left_arm_joint{_i+1}"]  = float(_q0[_i])
        init_joint_angles[f"right_arm_joint{_i+1}"] = float(_q0[8 + _i])
    arm_tf = make_arm_tf(robot_init_pose[:3], ARM_SCALE)
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, init_joint_angles)
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

    # --- ABD cup/beaker (case_39 asset softgriper_cup.msh), placed beside the
    #     shirt on the table. Loaded as a rigid ABD; the cup-vs-arm exclusion
    #     loop below (which iterates abd_records) keeps it from spuriously
    #     contacting the OBB-free arm links. Pose anchored to the shirt +
    #     offset so it lands near the gripper; tune with CASE_UMI_CUP_OFF. ---
    if int(os.environ.get("CASE_UMI_CUP", "0")):
        CUP_MSH = _ASSETS_DIR + "sim_data/tetmesh/softgriper_cup.msh"
        # Config matches case_39 exactly (rigid ABD, scale 0.8, young 1e8). The
        # cup is NOT soft — earlier it just LOOKED soft because it spawned inside
        # the shirt and the contact solver shoved it around.
        #
        # Placement mirrors case_39's philosophy (cup well OFFSET from the shirt,
        # both resting on the ground) — but in THIS scene's frame, not case_39's.
        # Shirt world footprint here: x~[-0.36,0.27], z~[-0.35,0.23] @ y~[0.89,1.16];
        # ground is at ground_offset (=0.75). case_39's literal xyz would land
        # the cup ~1.7 m below this ground, and the old default (0.15,1.0,-0.06)
        # sat dead-centre INSIDE the shirt's bbox. So we anchor the cup to the
        # shirt's (x,z) and drop it onto the ground, offset in -z to clear the
        # shirt.  Override the offset with CASE_UMI_CUP_OFF="dx,dy,dz" or pin an
        # absolute position with CASE39_CUP_XYZ="x,y,z".
        _cup_scale = float(os.environ.get("CASE39_CUP_SCALE", "0.8"))
        # shirt world translation (first actor in init_info), fallback to bake
        _shirt_t = np.array([-0.046, 1.025, -0.058])
        for _v in init_info.values():
            _shirt_t = np.asarray(_v['initial_pose'], dtype=float).reshape(4, 4)[:3, 3]
            break
        _ground_y = float(default_cfg.get('ground_offset', 0.75))
        # cup mesh local bottom (after scale) ≈ scale*(-0.0016); lift so the cup
        # bottom rests ~1 cm above the ground, then it settles in a step or two.
        _cup_off = np.array([float(s) for s in
                             os.environ.get("CASE_UMI_CUP_OFF", "0.0,0.0,-0.42").split(",")])
        # cup local bottom after scale ≈ 0.8*(-0.0016) = -0.0013, so cup_y =
        # ground + 0.0013 puts the bottom exactly on the ground; +~3 mm gap keeps
        # it clear of the ground contact barrier at spawn.
        _cup_default = np.array([_shirt_t[0] + _cup_off[0],
                                 _ground_y + 0.0043 + _cup_off[1],
                                 _shirt_t[2] + _cup_off[2]])
        if os.environ.get("CASE39_CUP_XYZ"):
            _cup_xyz = np.array([float(s) for s in
                                 os.environ["CASE39_CUP_XYZ"].split(",")])
        else:
            _cup_xyz = _cup_default
        cup_T = np.eye(4)
        cup_T[:3, :3] *= _cup_scale
        cup_T[:3, 3] = _cup_xyz
        eng.load_mesh(CUP_MSH, dimensions=3, body_type="ABD",
                      transform=cup_T, young_modulus=1e8, boundary_type="Free")
        cup_rec = eng.get_load_records()[-1]
        abd_records.append(cup_rec)
        print(f"[UI] loaded ABD cup: v_off=%d nverts=%d at xyz=%s"
              % (cup_rec.vertex_offset, cup_rec.vertex_count, cup_T[:3,3].round(3)),
              flush=True)
    else:
        cup_rec = None

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
        # stitches are added in call order -> each finger's springs are a
        # contiguous range [stitch_pair_start, +n_stitch) for the on-GPU
        # get_stitch_max_stretch_batched() gate.
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
    # MULTI-ANCHOR: a single anchor only fixes translation at that point — the
    # grip reaction torque (force x lever arm) then rotates/relaxes the pin, so
    # the finray visibly slides/tilts off the mount under load. Adding 2-3 anchors
    # offset along the gripper frame fixes rotation too (a rigid frame, no single
    # lever). Tune count/span with CASE_UMI_FJ_ANCHORS / CASE_UMI_FJ_SPAN.
    FJ_ANCHORS = int(os.environ.get("CASE_UMI_FJ_ANCHORS", "1"))
    FJ_SPAN    = float(os.environ.get("CASE_UMI_FJ_SPAN", "0.02"))   # extra-anchor offset (m)
    for g in grippers:
        T = g['gripper_T']
        o = T[:3, 3]
        ax_n = T[:3, :3] @ np.array([1.0, 0.0, 0.0])   # gripper local x (world)
        ax_b = T[:3, :3] @ np.array([0.0, 0.0, 1.0])   # gripper local z (world)
        anchors = [o]
        if FJ_ANCHORS >= 2: anchors.append(o + FJ_SPAN * ax_n)
        if FJ_ANCHORS >= 3: anchors.append(o + FJ_SPAN * ax_b)
        if FJ_ANCHORS >= 4: anchors.append(o - FJ_SPAN * ax_n)
        g['fj_idxs'] = []
        for a in anchors[:max(1, FJ_ANCHORS)]:
            g['fj_idxs'].append(eng.native.add_fixed_joint(
                parent_body=g['finger_id'], child_body=g['rigid_abd_id'],
                world_anchor=a,
                world_normal=np.array([1.0, 0.0, 0.0]),
                world_bitangent=np.array([0.0, 0.0, 1.0]),
            ))
        g['fj_idx'] = g['fj_idxs'][0]   # back-compat

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
    # finray-self CCD pairs (~1.2x faster, no effect on Newton convergence) at the
    # cost of the truss being able to self-penetrate.
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

    # higher default kappa (1e3 -> 1e5) so the pins resist the grip reaction; with
    # the multi-anchor frame this stops the finray sliding/tilting off the mount.
    fj_kappa = float(os.environ.get("CASE36_FJ_KAPPA", "1e5"))
    for g in grippers:
        for _fj in g.get('fj_idxs', [g['fj_idx']]):
            eng.native.set_fixed_joint_strength(_fj, fj_kappa)

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

    # NOTE: the arm was born already at the start pose (load_urdf
    # initial_joint_angles, above) with target == initial pose, so it holds with
    # zero driving force — no pre-pose settle loop is needed and the GUI opens
    # grasp-ready and static.

    # --- 6. Polyscope + UI loop ---
    verts = eng.get_vertices()
    faces = eng.get_surface_faces()

    if cup_rec is not None:
        _co = cup_rec.vertex_offset; _cc = cup_rec.vertex_count
        _cf = int(((faces >= _co) & (faces < _co + _cc)).any(axis=1).sum())
        print(f"[UI-DIAG] verts={verts.shape[0]} faces={faces.shape[0]}  "
              f"cup verts[{_co}:{_co+_cc}]  cup surface-faces={_cf}  "
              f"{'(RENDERS)' if _cf>0 else '(NOT IN SURFACE!)'}", flush=True)

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
    # color the ABD cup green so it's clearly visible vs the gray arm
    if cup_rec is not None:
        co = cup_rec.vertex_offset; cc = cup_rec.vertex_count
        colors[co:co + cc] = np.array([0.20, 0.80, 0.35])
        # how many render faces actually reference cup verts? (0 => not drawn)
        _cupfaces = int(((faces >= co) & (faces < co + cc)).any(axis=1).sum())
        print(f"[UI] render: verts={verts.shape[0]} faces={faces.shape[0]}  "
              f"cup verts[{co}:{co+cc}]  cup faces in surface={_cupfaces}", flush=True)
    state['mesh'].add_color_quantity(
        "region (red=rigid root, blue=FEM truss, green=cup)", colors,
        defined_on='vertices', enabled=True)

    # ============ UI-DRIVEN control (no replay) ============
    # Mirror case_39_full_scale.py's UI-link-control pattern exactly:
    #   * each slider DISPLAYS the joint's current target (get_revolute_target_deg
    #     / get_prismatic_target_mm),
    #   * a target is pushed ONLY when its slider actually changes.
    # The arm was born at the start pose (load_urdf initial_joint_angles) with
    # target == that pose, so every slider opens showing the start pose and the
    # arm holds it with zero driving force. Nothing moves until you drag a slider.
    a0 = (actions[0] if len(actions) else np.zeros(16)).astype(float)
    state['running'] = False      # start PAUSED — step only after the user clicks Run
    # BINARY pure-FORCE gripper, PER ARM: left / right each have their own
    # open(False)/closed(True) state. Start OPEN.
    state['gripL_closed'] = False
    state['gripR_closed'] = False
    # CONTACT-FORCE-REGULATED POSITION control. The prismatic is ALWAYS
    # position-driven (stable, no pure-force overshoot). "Closing" marches the
    # commanded target toward the closed end at CLOSE_SPEED — but PAUSES once the
    # REAL grip (contact) force on the soft fingertip reaches GRIP_TARGET. This is
    # the signal that actually registers a soft-finray grasp (the rigid prismatic
    # lag does NOT — the rigid root keeps moving as the soft truss deforms, so it
    # would never report the grasp and the gripper would crush the object). Remove
    # the object and the contact force drops -> the march resumes to full close.
    CLOSE_SPEED = float(os.environ.get("CASE_UMI_CLOSE_SPEED", "0.0012"))  # target march/frame (m)
    # Which grip-force signal gates the half-close:
    #   contact (default) — REAL IPC contact force on the soft fingertip. Most
    #                       direct, but EXPENSIVE (rebuilds BVH+pairs every frame).
    #   stitch            — stitch-spring stretch at the soft-truss base (the
    #                       reaction the rigid root feels). CHEAP (vertex distances,
    #                       no BVH) and rises monotonically with grip.
    GATE = os.environ.get("CASE_UMI_GATE", "contact").lower()
    # TARGET grip (contact) force in IPC units — the firmness knob (GATE=contact).
    GRIP_TARGET = float(os.environ.get("CASE_UMI_GRIP_TARGET", "0.02"))
    # firmness knob for GATE=stitch: pause once the MAX per-pair stitch STRETCH
    # (a length, mm) reaches this. As the fingertip grips, the most-loaded stitch
    # spring (soft-truss base ↔ rigid root) stretches; that max stretch is the
    # gate signal. Bigger thresh = firmer grasp. (force in that spring = k*stretch.)
    STITCH_K     = float(os.environ.get("CASE36_SOFT_RATE", "1e4"))                  # stitch stiffness (readout)
    STITCH_THRESH = float(os.environ.get("CASE_UMI_STITCH_THRESH", "0.005")) * 1e-3  # max stitch stretch (mm->m)
    # STITCH_GPU=1: compute the stitch gate signal with the on-GPU batched
    # reduction (get_stitch_max_stretch_batched) — one kernel, scalar back, no
    # full-vertex D2H, multi-env-isolated. =0 uses the CPU path (get_vertices+numpy).
    STITCH_GPU = int(os.environ.get("CASE_UMI_STITCH_GPU", "0"))
    # MODE selects the gripper control LAW:
    #   stitch / contact (default) — pure POSITION drive, force-feedback-regulated
    #       march (gate = stitch stretch / contact force). Stable, auto-resumes if
    #       the object is removed. (MODE defaults to GATE for back-compat.)
    #   forcelock — PURE FORCE grasp (set_prismatic_force, strength=0) for
    #       FORCELOCK_DELAY frames, THEN LOCK the achieved opening with position
    #       control. = "纯力控抓取 + 位控锁定" (the replay's forcelock). Adaptive
    #       grasp + stiff hold (no drift); does NOT resume if the object vanishes.
    #   forcebarrier — PURE FORCE the whole time, but an IPC barrier armed at the
    #       closed limit cl makes the SOLVER guarantee d never crosses cl (hard
    #       no-overshoot). Literal force control + provable no-cross; resumes if
    #       the object vanishes (force keeps pushing into the barrier at cl).
    MODE = os.environ.get("CASE_UMI_MODE", "").lower() or GATE
    GRIP_FORCE     = float(os.environ.get("CASE_UMI_GRIP_FORCE", "8.0"))   # forcelock/forcebarrier grasp force (N)
    FORCELOCK_DELAY = int(os.environ.get("CASE_UMI_FORCELOCK_DELAY", "25")) # frames of force before locking
    LOCK_K         = float(os.environ.get("CASE_UMI_LOCK_K", "15.0"))       # forcelock phase-2 hold stiffness
    BARRIER_DHAT   = float(os.environ.get("CASE_UMI_BARRIER_DHAT", "0.002"))  # barrier activation band (m)
    BARRIER_KAPPA  = float(os.environ.get("CASE_UMI_BARRIER_KAPPA", "1e3"))   # barrier stiffness (tune)
    BARRIER_FORCE  = float(os.environ.get("CASE_UMI_BARRIER_FORCE", "3.0"))   # forcebarrier closing force (N; small is enough — barrier sets the stop)
    # HYSTERESIS: once the grip signal hits thresh the target LATCHES (stable hold,
    # no slow creep that squeezes the object out); resumes only if the signal drops
    # below thresh*RESUME_FRAC (object loosened/removed). Lower = stickier hold.
    RESUME_FRAC = float(os.environ.get("CASE_UMI_STITCH_RESUME_FRAC", "0.5"))
    _grip_tgt = {}     # per-joint commanded position target (the marched setpoint)
    _grip_latch = {}   # per-joint True once gripped (hysteresis hold, no creep)
    _lock = {}         # forcelock: per-joint {'n':frames, 'd': locked opening or None}

    # forcebarrier: arm the one-sided IPC barrier at each finger's closed limit cl
    # ONCE, so the solver can never let the opening cross cl no matter the force.
    if MODE == "forcebarrier":
        for pi in range(len(robot.prismatic_joints)):
            lo = robot.prismatic_joints[pi].lower_limit
            hi = robot.prismatic_joints[pi].upper_limit
            op = lo if abs(lo) > abs(hi) else hi      # open end
            cl = hi if abs(lo) > abs(hi) else lo      # closed end
            bdir = 1.0 if (op - cl) > 0 else -1.0     # gap = (d-cl)*bdir > 0 while open
            eng.native.set_prismatic_limit_barrier(pi, cl, bdir, BARRIER_DHAT, BARRIER_KAPPA)
        print(f"[forcebarrier] armed cl-barrier on {len(robot.prismatic_joints)} prismatic "
              f"joints (dhat={BARRIER_DHAT*1000:.1f}mm, kappa={BARRIER_KAPPA:g})", flush=True)

    # per-side soft-finray FEM bodies + gripper dicts (grip-force readout)
    left_fem  = [g['fem_rec'] for g in grippers if g['label'].startswith('left')]
    right_fem = [g['fem_rec'] for g in grippers if g['label'].startswith('right')]
    left_grippers  = [g for g in grippers if g['label'].startswith('left')]
    right_grippers = [g for g in grippers if g['label'].startswith('right')]
    # batched on-GPU stitch gate: one segment per finger (multi-env would add more)
    for _i, _g in enumerate(grippers):
        _g['seg_idx'] = _i
    _seg_starts = np.array([g['stitch_pair_start'] for g in grippers], dtype=np.int32)
    _seg_counts = np.array([g['n_stitch']          for g in grippers], dtype=np.int32)
    _stitch_cache = {'maxes': None}

    def _refresh_stitch_gpu():
        """One batched kernel launch/frame -> per-finger max stretch, cached."""
        _stitch_cache['maxes'] = eng.native.get_stitch_max_stretch_batched(
            _seg_starts, _seg_counts)

    def _finger_contact(fem_recs):
        """Max |IPC contact force| over a side's finger FEM bodies (each measured
        alone so the two fingers' opposite grip forces don't cancel). EXPENSIVE."""
        gmax = 0.0
        for fr in (fem_recs or []):
            cf = eng.native.get_body_contact_force(fr.vertex_offset, fr.vertex_count)
            gmax = max(gmax, float((cf[0]**2 + cf[1]**2 + cf[2]**2) ** 0.5))
        return gmax

    def _side_stitch(grippers_side):
        """Max per-pair STITCH STRETCH (m) over a side's fingers = how far the
        most-loaded stitch spring is pulled (the rigid root's reaction to the soft
        truss grip). Two paths, SAME signal:
          STITCH_GPU=1: on-GPU batched reduction (scalar back, no full-vertex D2H).
          STITCH_GPU=0: get_vertices() (LIVE GPU) + numpy. (NOT get_vertices_host —
                        that is a stale host cache and would report ~0 forever.)"""
        if STITCH_GPU:
            m = _stitch_cache['maxes']
            if m is None:
                m = eng.native.get_stitch_max_stretch_batched(_seg_starts, _seg_counts)
            return float(max((m[g['seg_idx']] for g in (grippers_side or [])), default=0.0))
        allv = np.asarray(eng.native.get_vertices())
        fmax = 0.0
        for g in (grippers_side or []):
            rv = g['rigid_v_idx']
            fe = allv[g['fem_v_off'] + rv]
            ri = allv[g['rigid_abd_v_off']: g['rigid_abd_v_off'] + len(rv)]
            fmax = max(fmax, float(np.linalg.norm(fe - ri, axis=1).max()))   # max per-pair stretch (m)
        return fmax

    def _apply_gripper_force(closed, idxs, fem_recs=None, grippers_side=None):
        """Per-arm binary gripper, applied EVERY frame, per CASE_UMI_MODE:

        forcelock — PURE FORCE grasp then position LOCK. closed=True: phase 1 drives
          the prismatic with a constant external force (strength=0) so the fingers
          find the object by force balance; after FORCELOCK_DELAY frames it captures
          and LOCKS the achieved opening with position control. closed=False:
          release the lock + position-hold open. ("纯力控抓取 + 位控锁定".)

        stitch/contact — force-REGULATED POSITION march (no raw force, no overshoot):
          march the target toward closed at CLOSE_SPEED, PAUSING once the grip
          signal (stitch stretch / contact force) hits its threshold -> partial
          close that auto-resumes if the object is removed."""
        if MODE == "forcebarrier":
            # FORCE drive + RETAINED constraint stiffness (libuipc-style: external
            # prismatic force AND the position-drive constraint coexist). CLOSE:
            # apply the closing force, but KEEP strength>0 with target tracking the
            # CURRENT opening — so the PD adds NO net pull (err~0) yet provides the
            # implicit Hessian stiffness that anchors the seat. Without this (pure
            # strength=0) the seat is un-anchored, the force shoves it, and the soft
            # fixed-joint (finray root <-> seat) stretches -> the finray slides off
            # the mount. The barrier still caps at cl (no overshoot). OPEN: hold open.
            for pi in idxs:
                lo = robot.prismatic_joints[pi].lower_limit
                hi = robot.prismatic_joints[pi].upper_limit
                op = lo if abs(lo) > abs(hi) else hi
                cl = hi if abs(lo) > abs(hi) else lo
                cd = 1.0 if (cl - op) > 0 else -1.0
                if closed:
                    eng.native.set_prismatic_strength(pi, 0.0)             # PURE FORCE -> adaptive half-close
                    eng.native.set_prismatic_force(pi, cd * BARRIER_FORCE) # closing drive; stops at object / barrier
                else:
                    eng.native.set_prismatic_force(pi, 0.0)
                    eng.native.set_prismatic_strength(pi, prismatic_mult)
                    eng.native.set_prismatic_target(pi, op)
            return
        if MODE == "forcelock":
            for pi in idxs:
                lo = robot.prismatic_joints[pi].lower_limit
                hi = robot.prismatic_joints[pi].upper_limit
                op = lo if abs(lo) > abs(hi) else hi
                cl = hi if abs(lo) > abs(hi) else lo
                cd = 1.0 if (cl - op) > 0 else -1.0
                if closed:
                    st = _lock.setdefault(pi, {'n': 0, 'd': None})
                    if st['d'] is None:               # PHASE 1: pure-force grasp
                        d = eng.native.get_prismatic_current_distance(pi)
                        remaining = (d - cl) * cd * -1.0  # >0 while still open
                        # SAFETY: if the finger reaches the closed end (e.g. CLOSE
                        # on nothing — pure force has no end-stop and would run away
                        # / cross), lock at cl immediately. Otherwise force-grasp
                        # until the object balances it, then lock after the delay.
                        if remaining <= 1e-4:
                            st['d'] = cl              # lock at closed end (no cross)
                        else:
                            eng.native.set_prismatic_strength(pi, 0.0)
                            eng.native.set_prismatic_force(pi, cd * GRIP_FORCE)
                            st['n'] += 1
                            if st['n'] >= FORCELOCK_DELAY:   # grasp settled -> lock
                                # clamp the captured opening into [op, cl] so a
                                # one-step force overshoot can't lock past cl.
                                dd = eng.native.get_prismatic_current_distance(pi)
                                st['d'] = (min(max(dd, min(op, cl)), max(op, cl)))
                    else:                             # PHASE 2: position-hold (lock)
                        eng.native.set_prismatic_force(pi, 0.0)
                        eng.native.set_prismatic_strength(pi, LOCK_K)
                        eng.native.set_prismatic_target(pi, st['d'])
                else:                                 # OPEN: release lock, hold open
                    _lock.pop(pi, None)
                    eng.native.set_prismatic_force(pi, 0.0)
                    eng.native.set_prismatic_strength(pi, prismatic_mult)
                    eng.native.set_prismatic_target(pi, op)
            return
        # --- stitch / contact: force-regulated POSITION march ---
        if closed and MODE == "stitch":
            signal, thresh = _side_stitch(grippers_side), STITCH_THRESH
        elif closed:
            signal, thresh = _finger_contact(fem_recs), GRIP_TARGET
        else:
            signal, thresh = 0.0, 1.0
        for pi in idxs:
            lo = robot.prismatic_joints[pi].lower_limit
            hi = robot.prismatic_joints[pi].upper_limit
            op = lo if abs(lo) > abs(hi) else hi      # open end (farthest from 0)
            cl = hi if abs(lo) > abs(hi) else lo      # closed end (near 0)
            cdir = 1.0 if (cl - op) > 0 else -1.0     # sign of closing motion
            d = eng.native.get_prismatic_current_distance(pi)
            # position drive only — feed the marched setpoint, never a raw force
            eng.native.set_prismatic_force(pi, 0.0)
            eng.native.set_prismatic_strength(pi, prismatic_mult)
            tgt = _grip_tgt.get(pi, d)
            if closed:
                # HYSTERESIS latch: once the grip signal reaches thresh, FREEZE the
                # target -> stable hold (no slow creep that squeezes the object out).
                # Resume closing only if the signal drops below thresh*RESUME_FRAC
                # (object loosened / removed).
                latched = _grip_latch.get(pi, False)
                if not latched:
                    if signal >= thresh:
                        latched = True
                    else:
                        tgt += cdir * CLOSE_SPEED
                elif signal < thresh * RESUME_FRAC:
                    latched = False
                    tgt += cdir * CLOSE_SPEED
                _grip_latch[pi] = latched
                tgt = min(tgt, cl) if cdir > 0 else max(tgt, cl)   # never past closed end
            else:
                _grip_latch.pop(pi, None)
                tgt += -cdir * CLOSE_SPEED            # march back toward open
                tgt = max(tgt, op) if cdir > 0 else min(tgt, op)
            _grip_tgt[pi] = tgt
            eng.native.set_prismatic_target(pi, tgt)

    # Headless self-test (CASE_UMI_FORCE_TEST=1): close both grippers on nothing
    # and confirm the opening clamps at the closed limit (does NOT cross negative
    # / interpenetrate).
    if int(os.environ.get("CASE_UMI_FORCE_TEST", "0")):
        def _probe(tag, fr):
            ds = [eng.native.get_prismatic_current_distance(pi) for pi in left_pris_indices]
            try:
                fs = [eng.native.get_prismatic_drive_force(pi) for pi in left_pris_indices]
                fstr = " force=" + str(['%+.3f'%x for x in fs])
            except Exception:
                fstr = ""
            print(f"[force-test] {tag} f={fr} left openings={['%+.4f'%x for x in ds]}{fstr}", flush=True)
        for fr in range(40):                     # OPEN first
            _apply_gripper_force(False, left_pris_indices, left_fem, left_grippers)
            _apply_gripper_force(False, right_pris_indices, right_fem, right_grippers)
            eng.step()
            if fr % 20 == 0: _probe("OPEN", fr)
        _probe("OPEN", 39)
        for fr in range(80):                     # then CLOSE on nothing
            _apply_gripper_force(True, left_pris_indices, left_fem, left_grippers)
            _apply_gripper_force(True, right_pris_indices, right_fem, right_grippers)
            eng.step()
            if fr % 5 == 0: _probe("CLOSE", fr)
        _probe("CLOSE", 79)
        print("[force-test] done — CLOSE openings should clamp near cl=0, NOT go negative", flush=True)
        return

    def callback():
        # run / pause / reset-pose
        if state['running']:
            if psim.Button("Pause"):
                state['running'] = False
        else:
            if psim.Button("Run"):
                state['running'] = True
        psim.SameLine()
        if psim.Button("Reset pose"):
            # NOTE: robot.reset_all() resets targets to ZERO (URDF default), which
            # would swing this born-posed arm away — so re-push the START pose.
            for i, rev_idx in enumerate(left_rev_indices):
                robot.set_revolute_position(rev_idx, float(a0[i]), degree=False)
            for i, rev_idx in enumerate(right_rev_indices):
                robot.set_revolute_position(rev_idx, float(a0[8 + i]), degree=False)
            state['gripL_closed'] = False   # reset both grippers to OPEN
            state['gripR_closed'] = False
        psim.Text(f"step: {state['last_ms']:6.1f} ms   "
                  f"FPS {(1000.0/state['last_ms']) if state['last_ms']>0 else 0.0:5.1f}")

        ch_e, val_e = psim.Checkbox("show mesh edges", state['show_edges'])
        if ch_e:
            state['show_edges'] = val_e
            state['mesh'].set_edge_width(0.5 if val_e else 0.0)

        # --- grippers: BINARY pure-force control, PER ARM (left / right) ---
        psim.Separator()
        if MODE == "forcelock":
            psim.Text(f"Grippers (forcelock: pure-force {GRIP_FORCE:.0f}N grasp "
                      f"-> lock after {FORCELOCK_DELAY}f)")
        else:
            _gate_str = (f"stitch<{STITCH_THRESH*1000:.3f}mm" if MODE == "stitch"
                         else f"contact<{GRIP_TARGET:.3f}")
            psim.Text(f"Grippers (force-regulated position; mode={MODE} [{_gate_str}], "
                      f"speed={CLOSE_SPEED*1000:.2f}mm/f)")
        if psim.Button("L CLOSE"):
            state['gripL_closed'] = True
        psim.SameLine()
        if psim.Button("L OPEN"):
            state['gripL_closed'] = False
        psim.SameLine()
        psim.TextUnformatted("left [CLOSED]" if state['gripL_closed'] else "left [open]")
        if psim.Button("R CLOSE"):
            state['gripR_closed'] = True
        psim.SameLine()
        if psim.Button("R OPEN"):
            state['gripR_closed'] = False
        psim.SameLine()
        psim.TextUnformatted("right [CLOSED]" if state['gripR_closed'] else "right [open]")

        # --- arm joint sliders (read live target, push on change) ---
        psim.Separator()
        psim.Text("Left arm joints (deg)")
        for rev_idx in left_rev_indices:
            ji  = robot.revolute_joints[rev_idx]
            cur = robot.get_revolute_target_deg(rev_idx)
            ch, v = psim.SliderFloat(ji.name, cur, ji.lower_limit_deg, ji.upper_limit_deg)
            if ch:
                robot.set_revolute_position(rev_idx, v, degree=True)
        psim.Separator()
        psim.Text("Right arm joints (deg)")
        for rev_idx in right_rev_indices:
            ji  = robot.revolute_joints[rev_idx]
            cur = robot.get_revolute_target_deg(rev_idx)
            ch, v = psim.SliderFloat(ji.name, cur, ji.lower_limit_deg, ji.upper_limit_deg)
            if ch:
                robot.set_revolute_position(rev_idx, v, degree=True)

        if not state['running']:
            return

        # re-assert the per-arm binary force grippers EVERY frame (force control
        # is continuous, unlike the on-change position sliders)
        if STITCH_GPU and MODE == "stitch" and (state['gripL_closed'] or state['gripR_closed']):
            _refresh_stitch_gpu()        # ONE batched kernel launch for all fingers
        _apply_gripper_force(state['gripL_closed'], left_pris_indices, left_fem, left_grippers)
        _apply_gripper_force(state['gripR_closed'], right_pris_indices, right_fem, right_grippers)

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

    ps.set_user_callback(callback)
    ps.show()
    print("UI session ended.")


if __name__ == "__main__":
    main()
