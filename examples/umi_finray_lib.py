#!/usr/bin/env python3
"""Shared library for the UMI finray soft-gripper example suite.

Backs all the thin entry scripts (single/multi-env x replay/UI x 3 scenes), so the
heavy logic — finray STRATEGY_F gripper build, scene loading, the 3 gripper-control
modes, multi-env tiling, the replay/UI harnesses — lives in ONE place.

Scenes (SCENES):
  foldshirt  episode_fold_shirt_umi.hdf5   shirt (cloth FEM), no rigid object
  beaker     episode_grasp_beaker_umi.hdf5 beaker (rigid ABD), no cloth
  cupshirt   qpos_case39.h5                cup (rigid ABD) + shirt (cloth FEM)

Grip mapping is AUTO-DETECTED per trajectory: if the recorded grip column has <= 2
distinct values it is treated as BINARY (open/closed); otherwise CONTINUOUS (the
grip value maps linearly to finger opening). So foldshirt/beaker -> continuous,
cupshirt(qpos_case39) -> binary, automatically.

Gripper modes (GRIP_MODE, all are *position* actuated — the finray FEM truss gives
the grasp compliance):
  pos     pure position control: drive each prismatic joint straight to the
          grip-mapped opening (no feedback).
  stitch  spring-deformation-gauged: on a close command, march the opening toward
          closed until the finray STITCH-spring stretch (get_stitch_max_stretch)
          exceeds a threshold = "deformed enough, gripping", then latch.
  force   force-combined position: on a close command, march toward closed until
          the REAL IPC contact force on the finger (get_body_contact_force) reaches
          a target, then latch the opening. (tactile-style closed loop)

Arm revolute joints are always position-controlled. Per-env line-search (S1-S4)
is available via STIFF_PERENV_ALPHA=1 (+ STIFF_PERENV_MASK=1) at run time.
"""
import sys, os, math, time, json, re
from pathlib import Path

_ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from stiff_physics import Engine, Config
from stiff_physics.robot import Robot

_URDF_DIR = _ASSETS_DIR + "sim_data/urdf/ridgeback_dual_panda_UMI/"
_FEM_DIR = "sim_data/" + os.environ.get("CASE39UMI_FINRAY_DIR", "umi_hybrid_sf_v800")
ARM_SCALE = 1.0
FINGER_LABELS = ['left_arm_leftfinger', 'left_arm_rightfinger',
                 'right_arm_leftfinger', 'right_arm_rightfinger']


def urdf_path():
    """OBB coarse arm by default; GRIP_URDF=detailed for the fine collision arm."""
    if os.environ.get("GRIP_URDF", "obb") == "detailed":
        return _URDF_DIR + "ridgeback_dual_panda2.urdf"
    return _URDF_DIR + "ridgeback_dual_panda2_OBB.urdf"


def _sf_paths(side):
    base = _ASSETS_DIR + _FEM_DIR + f"/UMI_finray_{side}"
    return (base + "_unified.npz", base + "_rigid.msh", base + "_rigid_remap.npz")


def make_arm_tf(pos, scale=1.0):
    R = Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    tf = np.eye(4); tf[:3, :3] = scale * R; tf[:3, 3] = R @ np.asarray(pos)
    return tf


def make_env_offsets(n, spacing):
    cols = int(math.ceil(math.sqrt(n))); rows = int(math.ceil(n / cols))
    offs = []
    for e in range(n):
        r, c = divmod(e, cols)
        o = np.eye(4)
        o[0, 3] = (c - (cols - 1) / 2.0) * spacing
        o[2, 3] = (r - (rows - 1) / 2.0) * spacing
        offs.append(o)
    return offs


def link_visual_origin(urdf, link_name):
    src = open(urdf).read()
    m = re.search(r'<link\s+name="' + re.escape(link_name) + r'"[^>]*>(.*?)</link>', src, re.DOTALL)
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
    xyz = np.array([float(x) for x in om_xyz.group(1).split()]) if om_xyz else np.zeros(3)
    rpy = np.array([float(x) for x in om_rpy.group(1).split()]) if om_rpy else np.zeros(3)
    T = np.eye(4); T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix(); T[:3, 3] = xyz
    return T


def load_finray_sides():
    up = urdf_path()
    sides = {}
    for side in ('L', 'R'):
        unified_npz, rigid_msh, rigid_remap = _sf_paths(side)
        d = np.load(unified_npz); rr = np.load(rigid_remap)
        sides[side] = dict(
            verts=np.ascontiguousarray(d['vertices'], np.float64),
            tets=np.ascontiguousarray(d['tets'], np.int32),
            rigid_v_idx=np.asarray(rr['rigid_v_idx'], np.int64),
            rigid_msh=rigid_msh,
            vis={lbl: link_visual_origin(up, lbl) for lbl in FINGER_LABELS})
    return sides


# ----------------------------------------------------------------------------
# Scene preparation: trajectory + action layout + objects + physics defaults.
# ----------------------------------------------------------------------------
def _resolve_mesh(p):
    for pre in ("/data/stiff-physics/franka_sim/assets_new/",
                "/data/stiff-physics/franka_sim/assets/",
                "/data/stiff-physics/assets/", "assets/"):
        if p.startswith(pre):
            return _ASSETS_DIR + p[len(pre):]
    return p if os.path.isabs(p) else _ASSETS_DIR + p


def prepare_scene(name):
    import h5py
    p = dict(name=name)
    if name in ("foldshirt", "beaker"):
        ep = _ASSETS_DIR + "trajectories/" + (
            "episode_fold_shirt_umi.hdf5" if name == "foldshirt"
            else "episode_grasp_beaker_umi.hdf5")
        with h5py.File(ep, "r") as f:
            actions = f["actions"][:]
            robot_init = np.asarray(f.attrs["robot_init_pose"])
            oi = json.loads(f.attrs["object_init_info"]); ec = json.loads(f.attrs["env_cfg"])
        # action layout [L_arm(0:7), gripL(7), R_arm(8:15), gripR(15)]
        p.update(arm_l=list(range(0, 7)), arm_r=list(range(8, 15)), grip_l=7, grip_r=15)
        p["ground_offset"] = float(ec.get("ground_offset", 0.75))
        # friction_rate = IPC Coulomb coefficient mu for object/self contact (the
        # gripper-finray <-> object friction). CASE39_FRICTION overrides the
        # episode value — raise it (e.g. 1.0-1.5) if the cup/object slips.
        p["friction"] = float(os.environ.get("CASE39_FRICTION", str(ec.get("friction_rate", 0.8))))
        p["precond"] = 0 if name == "beaker" else 1
        p["abs_dhat"] = 0.00239 if name == "beaker" else 0.0019
        if name == "foldshirt":
            ck = next(k for k, v in oi.items() if v.get("body_type") != "ABD")
            p["fem_obj"] = dict(path=_ASSETS_DIR + "objects/m-panda_single/scaled.obj",
                                T0=np.asarray(oi[ck]["initial_pose"]).reshape(4, 4), young=1e2)
            p["abd_obj"] = None
        else:
            import trimesh
            bk = next(k for k, v in oi.items() if v.get("body_type") == "ABD")
            mesh = trimesh.load(_resolve_mesh(oi[bk]["collision_mesh"]), force='mesh')
            p["abd_obj"] = dict(verts=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
                                T0=np.asarray(oi[bk]["initial_pose"]).reshape(4, 4),
                                young=float(os.environ.get("BEAKER_YOUNG", "1e5")),
                                label=os.path.basename(_resolve_mesh(oi[bk]["collision_mesh"])))
            p["fem_obj"] = None
        p["actions"] = actions; p["robot_init"] = robot_init
        # UMI episodes: arm base from the recorded robot_init_pose.
        p["arm_tf0"] = make_arm_tf(robot_init[:3], ARM_SCALE)
    elif name == "cupshirt":
        traj = _ASSETS_DIR + "trajectories/qpos_case39.h5"
        with h5py.File(traj, "r") as f:
            actions = f["qpos"][:] if "qpos" in f else f["actions"][:]
        # qpos_case39 layout [L_arm(0:7), R_arm(7:14), gripL(14), gripR(15)]
        p.update(arm_l=list(range(0, 7)), arm_r=list(range(7, 14)), grip_l=14, grip_r=15)
        p["robot_init"] = np.array([-0.8, 0.0, 0.0])
        # case_39 cup-grasp scene: arm base at y=-3.0 (NOT the UMI make_arm_tf
        # placement) so the arm reaches UP to the cup at [0.67,-0.2,-0.4] /
        # ground -1.67 — matches replay_case39.py / forcegrip exactly. Using the
        # UMI base here leaves the gripper ~3m above the cup (never grasps).
        arm_tf = np.eye(4)
        arm_tf[:3, :3] = ARM_SCALE * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
        arm_tf[1, 3] = -3.0
        p["arm_tf0"] = arm_tf
        p["ground_offset"] = float(os.environ.get("CASE39_GROUND_OFFSET", "-1.67"))
        p["friction"] = float(os.environ.get("CASE39_FRICTION", "0.8"))
        p["precond"] = 1; p["abs_dhat"] = 0.00239
        cup_scale = float(os.environ.get("CASE39_CUP_SCALE", "0.8"))
        cup_xyz = np.array([float(s) for s in os.environ.get("CASE39_CUP_XYZ", "0.67,-0.2,-0.4").split(",")])
        cup_T = np.eye(4); cup_T[:3, :3] *= cup_scale; cup_T[:3, 3] = cup_xyz
        p["abd_obj"] = dict(msh=_ASSETS_DIR + "sim_data/tetmesh/softgriper_cup.msh", T0=cup_T,
                            young=float(os.environ.get("CUP_YOUNG", "1e8")), label="cup")
        shirt_xyz = np.array([float(s) for s in os.environ.get("CASE39_SHIRT_XYZ", "0.67,0,0").split(",")])
        shirt_T = np.eye(4); shirt_T[:3, 3] = shirt_xyz
        p["fem_obj"] = dict(path=_ASSETS_DIR + "triMesh/shirt_6436v.obj", T0=shirt_T, young=1e2)
        p["actions"] = actions
    else:
        raise ValueError(f"unknown scene '{name}'")

    g = np.concatenate([p["actions"][:, p["grip_l"]], p["actions"][:, p["grip_r"]]])
    p["binary"] = len(np.unique(np.round(g, 3))) <= 2
    return p


# ----------------------------------------------------------------------------
# World build (N envs; N=1 = single env).
# ----------------------------------------------------------------------------
def build_world(eng, prep, num_envs, spacing, sides, ge):
    offs = make_env_offsets(num_envs, spacing)
    arm_tf0 = prep["arm_tf0"]   # per-scene arm base (UMI episode pose / case_39 y=-3.0)
    rigid_young = float(os.environ.get("CASE39UMI_RIGID_YOUNG", "1e8"))
    fem_young = float(os.environ.get("CASE36_FEM_YOUNG", "1e7"))
    up = urdf_path()
    envs = []

    # Phase A: per env: URDF + 4 finray rigid roots + optional rigid (ABD) object
    for env_tf in offs:
        eng.native.load_urdf(up, env_tf @ arm_tf0, True, False, 1e7, {})
        env_abd = [r for r in eng.get_load_records() if r.body_type == 0]
        finger_recs = {}
        for r in reversed(env_abd):
            if r.label in FINGER_LABELS and r.label not in finger_recs:
                finger_recs[r.label] = r
            if len(finger_recs) == 4:
                break
        if len(finger_recs) != 4:
            raise RuntimeError(f"expected 4 finger ABDs, got {len(finger_recs)}")
        arm_ids = [r.body_offset for r in env_abd if r.body_offset >= ge['abd_cursor']]
        for bid in arm_ids:
            eng.add_ground_collision_skip(bid)
        grippers = []
        for label in FINGER_LABELS:
            side = 'L' if label.endswith('leftfinger') else 'R'
            gripper_T = np.asarray(eng.native.get_urdf_link_transform(label)) @ sides[side]['vis'][label]
            eng.load_mesh(sides[side]['rigid_msh'], dimensions=3, body_type="ABD",
                          transform=gripper_T, young_modulus=rigid_young, boundary_type="Free")
            rr = eng.get_load_records()[-1]
            grippers.append(dict(label=label, side=side, finger_id=finger_recs[label].body_offset,
                                 gripper_T=gripper_T, rigid_abd_id=rr.body_offset,
                                 rigid_abd_v_off=rr.vertex_offset))
        env = dict(env_tf=env_tf, arm_ids=arm_ids, grippers=grippers)
        if prep["abd_obj"]:
            ao = prep["abd_obj"]; T = env_tf @ ao["T0"]
            if "msh" in ao:
                eng.load_mesh(ao["msh"], dimensions=3, body_type="ABD", transform=T,
                              young_modulus=ao["young"], boundary_type="Free")
            else:
                eng.load_mesh_from_data(vertices=ao["verts"], faces=ao["faces"], verts_per_face=3,
                                        dimensions=3, body_type="ABD", transform=T,
                                        young_modulus=ao["young"], boundary_type="Free")
            env["abd_obj_rec"] = eng.get_load_records()[-1]
        ge['abd_cursor'] = max(r.body_offset for r in eng.get_load_records() if r.body_type == 0) + 1
        envs.append(env)

    # Phase B: per env: 4 finray FEM trusses + optional cloth (FEM) + stitch + fixed joints
    stitch_cursor = 0
    for env in envs:
        for g in env['grippers']:
            s = sides[g['side']]
            eng.native.load_mesh_from_data(s['verts'], s['tets'], 4, 3, 1, g['gripper_T'], fem_young, 0)
            fr = eng.get_load_records()[-1]
            g['fem_rec'] = fr; g['fem_v_off'] = fr.vertex_offset; g['fem_body_offset'] = fr.body_offset
        if prep["fem_obj"]:
            fo = prep["fem_obj"]
            eng.load_mesh(fo["path"], dimensions=2, body_type="FEM",
                          transform=env['env_tf'] @ fo["T0"], young_modulus=fo["young"])
            env["fem_obj_rec"] = eng.get_load_records()[-1]
        for g in env['grippers']:
            rvidx = sides[g['side']]['rigid_v_idx']
            g['stitch_start'] = stitch_cursor; g['n_stitch'] = len(rvidx)
            for i in range(len(rvidx)):
                eng.add_stitch_spring(g['fem_v_off'] + int(rvidx[i]), g['rigid_abd_v_off'] + i,
                                      g['rigid_abd_id'], rest_offset_world=(0., 0., 0.))
            stitch_cursor += len(rvidx)
            anchor = g['gripper_T'][:3, 3]
            g['fj_idx'] = eng.native.add_fixed_joint(
                parent_body=g['finger_id'], child_body=g['rigid_abd_id'], world_anchor=anchor,
                world_normal=np.array([1., 0., 0.]), world_bitangent=np.array([0., 0., 1.]))

    # exclusions
    n_abd_total = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    for env in envs:
        _exclusions(eng, env, n_abd_total, prep)
    # P1 env-group isolation
    if num_envs > 1 and int(os.environ.get("CASE39ME_ISOLATE", "1")):
        n_fem_total = sum(1 for r in eng.get_load_records() if r.body_type == 1)
        m_abd, m_fem = n_abd_total // num_envs, n_fem_total // num_envs
        groups = [cid // m_abd for cid in range(n_abd_total)] + [f // m_fem for f in range(n_fem_total)]
        eng.native.set_body_groups(groups)
    return envs, n_abd_total


def _exclusions(eng, env, n_abd_total, prep):
    grippers, arm_ids = env['grippers'], env['arm_ids']
    for g in grippers:
        g['fem_global_id'] = n_abd_total + g['fem_body_offset']
    for g in grippers:
        eng.native.add_collision_exclusion(g['rigid_abd_id'], g['fem_global_id'])
        eng.native.add_collision_exclusion(g['fem_global_id'], g['finger_id'])
        eng.native.add_collision_exclusion(g['rigid_abd_id'], g['finger_id'])
        for arm_id in arm_ids:
            if arm_id == g['finger_id']:
                continue
            eng.native.add_collision_exclusion(g['rigid_abd_id'], arm_id)
            eng.native.add_collision_exclusion(g['fem_global_id'], arm_id)
    finger_offsets = {g['finger_id'] for g in grippers}
    # rigid object (cup/beaker): exclude from NON-finger arm bodies (finray still collides = grasp)
    if env.get("abd_obj_rec") is not None:
        oid = env["abd_obj_rec"].body_offset
        for arm_id in arm_ids:
            if arm_id not in finger_offsets:
                eng.native.add_collision_exclusion(arm_id, oid)
    # cloth (FEM object): exclude from ALL arm bodies (finray still collides = grasp/fold)
    if env.get("fem_obj_rec") is not None:
        cgid = n_abd_total + env["fem_obj_rec"].body_offset
        for arm_id in arm_ids:
            eng.native.add_collision_exclusion(arm_id, cgid)
        env["fem_obj_gid"] = cgid
    for g in grippers:
        eng.add_ground_collision_skip(g['fem_global_id'])
        eng.add_ground_collision_skip(g['rigid_abd_id'])
    pre = lambda l: 'left' if l.startswith('left_') else 'right'
    for i, gi in enumerate(grippers):
        for gj in grippers[i + 1:]:
            if pre(gi['label']) != pre(gj['label']):
                continue
            for a in (gi['rigid_abd_id'], gi['fem_global_id']):
                for b in (gj['rigid_abd_id'], gj['fem_global_id'], gj['finger_id']):
                    eng.native.add_collision_exclusion(a, b)
            eng.native.add_collision_exclusion(gi['finger_id'], gj['rigid_abd_id'])
            eng.native.add_collision_exclusion(gi['finger_id'], gj['fem_global_id'])


# ----------------------------------------------------------------------------
# Joint slicing + drive groups (per env per side).
# ----------------------------------------------------------------------------
def slice_env_joints(robot, n):
    nr, npz = len(robot.revolute_joints), len(robot.prismatic_joints)
    assert nr % n == 0 and npz % n == 0, f"{nr} rev / {npz} pri not divisible by {n}"
    rpe, ppe = nr // n, npz // n
    out = []
    for e in range(n):
        rb, pb = range(e * rpe, (e + 1) * rpe), range(e * ppe, (e + 1) * ppe)
        out.append(dict(
            left_rev=[i for i in rb if robot.revolute_joints[i].name.startswith('left_arm_joint')],
            right_rev=[i for i in rb if robot.revolute_joints[i].name.startswith('right_arm_joint')],
            left_pri=[i for i in pb if robot.prismatic_joints[i].name.startswith(('left_arm', 'leftarm'))],
            right_pri=[i for i in pb if robot.prismatic_joints[i].name.startswith(('right_arm', 'rightarm'))]))
    return out


def build_drive_groups(envs, ejs):
    """One group per (env, side): the prismatic joints + finray FEM recs + stitch
    segment indices for that hand, so the gauge modes can read its force/stretch."""
    groups = []
    for e, (env, ej) in enumerate(zip(envs, ejs)):
        for side, pris in (('L', ej['left_pri']), ('R', ej['right_pri'])):
            gs = [g for g in env['grippers'] if (g['label'].startswith('left_') if side == 'L'
                                                 else g['label'].startswith('right_'))]
            groups.append(dict(key=(e, side), pris=pris,
                               fems=[(g['fem_rec'].vertex_offset, g['fem_rec'].vertex_count) for g in gs],
                               seg_idx=[g['_seg_idx'] for g in gs]))
    return groups


# ----------------------------------------------------------------------------
# Grip mapping + the 3 drive modes.
# ----------------------------------------------------------------------------
def _open_close(robot, pi):
    lo = robot.prismatic_joints[pi].lower_limit
    hi = robot.prismatic_joints[pi].upper_limit
    op = lo if abs(lo) > abs(hi) else hi   # fully-open end (farthest from 0)
    cl = hi if abs(lo) > abs(hi) else lo   # near-0 (closed) end
    return op, cl


def map_grip(grip, binary):
    """grip [-1,+1] -> opening fraction s in [0,1] (0 closed, 1 open)."""
    if binary:
        return 1.0 if grip >= 0 else 0.0
    return min(max((grip + 1.0) * 0.5, 0.0), 1.0)


def drive_side(eng, robot, grp, grip, binary, mode, gstate, P, all_stretch):
    """Drive one hand's prismatic joints per the selected gripper mode."""
    pis = grp['pris']
    if not pis:
        return
    if mode == "pos":
        s = map_grip(grip, binary)
        for pi in pis:
            op, cl = _open_close(robot, pi)
            robot.set_prismatic_position(pi, cl + s * (op - cl), millimeters=False)
        return

    if mode == "force":
        # forcebarrier (v0.6.4 final force scheme): REAL force drive — close with a
        # constant prismatic FORCE (strength=0), and a one-sided IPC barrier armed
        # at the closed limit cl (arm_force_barrier, once) guarantees the opening
        # never overshoots past cl no matter the force. Open = position-hold.
        for pi in pis:
            op, cl = _open_close(robot, pi)
            cd = 1.0 if (cl - op) > 0 else -1.0
            if grip < 0:                                   # CLOSE: pure force, barrier stops at cl
                eng.native.set_prismatic_strength(pi, 0.0)
                eng.native.set_prismatic_force(pi, cd * P['barrier_force'])
            else:                                          # OPEN: position-hold to open end
                eng.native.set_prismatic_force(pi, 0.0)
                eng.native.set_prismatic_strength(pi, P['pos_k'])
                eng.native.set_prismatic_target(pi, op)
        return

    # stitch: spring-deformation-gauged POSITION drive. grip>=0 -> open & reset;
    # grip<0 -> march the opening toward closed until the finray stitch-spring
    # stretch crosses the threshold, then latch (hold, no further creep).
    st = gstate.setdefault(grp['key'], {'s': 1.0, 'latched': False})
    if grip >= 0:
        st['s'] = 1.0; st['latched'] = False
    else:
        gauge = max((all_stretch[i] for i in grp['seg_idx']), default=0.0) if all_stretch is not None else 0.0
        if gauge >= P['stitch_thresh']:
            st['latched'] = True
        if not st['latched']:
            st['s'] = max(0.0, st['s'] - P['close_ds'])
    for pi in pis:
        op, cl = _open_close(robot, pi)
        robot.set_prismatic_position(pi, cl + st['s'] * (op - cl), millimeters=False)


def drive_frame(eng, robot, ejs, groups, prep, raw, mode, gstate, P, stitch_seg):
    """Apply one trajectory frame to all envs: arm revolutes + grippers."""
    al, ar, gl, gr = prep["arm_l"], prep["arm_r"], prep["grip_l"], prep["grip_r"]
    for ej in ejs:
        for i, ri in enumerate(ej['left_rev']):
            robot.set_revolute_position(ri, float(raw[al[i]]), degree=False)
        for i, ri in enumerate(ej['right_rev']):
            robot.set_revolute_position(ri, float(raw[ar[i]]), degree=False)
    all_stretch = None
    if mode == "stitch" and stitch_seg is not None:
        all_stretch = eng.native.get_stitch_max_stretch_batched(stitch_seg[0], stitch_seg[1])
    for grp in groups:
        grip = float(raw[gl]) if grp['key'][1] == 'L' else float(raw[gr])
        drive_side(eng, robot, grp, grip, prep["binary"], mode, gstate, P, all_stretch)


# ----------------------------------------------------------------------------
# Common engine setup.
# ----------------------------------------------------------------------------
def make_engine(prep, num_envs):
    cfg = Config(
        dt=0.020, cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1e4, poisson_rate=0.49,
        friction_rate=prep["friction"], relative_dhat=1e-3,
        joint_strength_ratio=100.0, revolute_driving_strength_ratio=100.0,
        prismatic_strength_ratio=float(os.environ.get("CASE39_PRISMATIC_CONSTRAINT_K", "2000")),
        semi_implicit_enabled=bool(int(os.environ.get("CASE39_SEMI", "0"))),
        semi_implicit_beta_tol=5e-2, semi_implicit_min_iter=1,
        newton_tol=float(os.environ.get("CASE39_NEWTON_TOL", "5e-2")),
        newton_iter_cap=int(os.environ.get("CASE39_NEWTON_CAP", "50")),
        preconditioner_type=int(os.environ.get("CASE39_PRECOND", str(prep["precond"]))),
        ground_offset=prep["ground_offset"], assets_dir=_ASSETS_DIR)
    cfg._cfg.collision_detection_buff_scale = float(os.environ.get("CASE39ME_BUFF_SCALE", "4.0"))
    cfg._cfg.linear_system_buff_scale = float(os.environ.get("CASE39ME_LSYS_SCALE", "2.0"))
    cfg._cfg.triplet_internal_margin = float(os.environ.get("CASE39ME_TRIPLET_MARGIN", "4.0"))
    cfg._cfg.absolute_dhat = float(os.environ.get("CASE39ME_ABS_DHAT", str(prep["abs_dhat"])))
    eng = Engine(cfg)
    if int(os.environ.get("CASE39_QUIET", "1")):
        eng.set_log_level(0)
    return eng


def _drive_params():
    return dict(
        pos_k=float(os.environ.get("POS_K", os.environ.get("GRIP_K", "15.0"))),
        close_ds=float(os.environ.get("GRIP_CLOSE_DS", "0.03")),          # stitch: opening-fraction/frame while closing
        stitch_thresh=float(os.environ.get("GRIP_STITCH_THRESH", "5e-6")),  # stitch: latch stretch (m)
        # forcebarrier (real force drive + no-overshoot IPC barrier at the closed limit)
        barrier_force=float(os.environ.get("GRIP_BARRIER_FORCE", "60.0")),  # closing force (N)
        barrier_dhat=float(os.environ.get("GRIP_BARRIER_DHAT", "0.002")),   # barrier standoff from cl (m)
        barrier_kappa=float(os.environ.get("GRIP_BARRIER_KAPPA", "1e3")),   # barrier stiffness
    )


def arm_force_barrier(eng, robot, P, arm=True):
    """forcebarrier: arm (or disarm, kappa<=0) the one-sided IPC barrier at each
    prismatic joint's CLOSED limit cl, so a pure-force close can never overshoot
    past cl. Call once after finalize (or on mode switch). Needs the force-control
    engine build (set_prismatic_limit_barrier)."""
    kappa = P['barrier_kappa'] if arm else 0.0
    for pi in range(len(robot.prismatic_joints)):
        op, cl = _open_close(robot, pi)
        bdir = 1.0 if (op - cl) > 0 else -1.0      # gap = (d-cl)*bdir > 0 while open
        eng.native.set_prismatic_limit_barrier(pi, cl, bdir, P['barrier_dhat'], kappa)


def reset_prismatic_drive(eng, robot, P):
    """Return every prismatic joint to plain position drive (force off, strength
    pos_k) — used when switching AWAY from force mode in the UI."""
    for pi in range(len(robot.prismatic_joints)):
        eng.native.set_prismatic_force(pi, 0.0)
        eng.native.set_prismatic_strength(pi, P['pos_k'])


def _setup_after_finalize(eng, envs, P):
    if int(os.environ.get("CASE36_DISABLE_GRAVITY", "1")):
        for env in envs:
            for a in env['arm_ids']:
                eng.native.set_body_apply_gravity(a, False)
            for g in env['grippers']:
                eng.native.set_body_apply_gravity(g['rigid_abd_id'], False)
    for env in envs:
        for g in env['grippers']:
            eng.native.set_fixed_joint_strength(g['fj_idx'], float(os.environ.get("CASE36_FJ_KAPPA", "1e3")))
    eng.native.set_max_revolute_step_per_frame(float(os.environ.get("CASE36_MAX_RAD_PER_FRAME", "0.04")))
    robot = Robot(eng)
    for i in range(len(robot.prismatic_joints)):
        eng.native.set_prismatic_strength(i, P['pos_k'])
    return robot


def _stitch_seg_arrays(envs):
    starts, counts, idx = [], [], 0
    for env in envs:
        for g in env['grippers']:
            g['_seg_idx'] = idx
            starts.append(g['stitch_start']); counts.append(g['n_stitch']); idx += 1
    return (np.array(starts, np.int32), np.array(counts, np.int32))


# ----------------------------------------------------------------------------
# Harnesses.
# ----------------------------------------------------------------------------
def run_replay(scene_name, default_envs=1):
    mode = os.environ.get("GRIP_MODE", "pos")
    num_envs = int(os.environ.get("CASE39ME_NUM_ENVS", str(default_envs)))
    spacing = float(os.environ.get("CASE39ME_SPACING", "4.0"))
    prep = prepare_scene(scene_name)
    P = _drive_params()
    print(f"[umi:{scene_name}] mode={mode} envs={num_envs} frames={len(prep['actions'])} "
          f"grip={'binary' if prep['binary'] else 'continuous'} arm={os.path.basename(urdf_path())} "
          f"pos_k={P['pos_k']}", flush=True)
    eng = make_engine(prep, num_envs)
    sides = load_finray_sides()
    ge = dict(abd_cursor=0)
    t0 = time.perf_counter()
    envs, n_abd = build_world(eng, prep, num_envs, spacing, sides, ge)
    stitch_seg = _stitch_seg_arrays(envs)
    eng.finalize()
    print(f"[umi:{scene_name}] built {num_envs} envs in {time.perf_counter()-t0:.1f}s ({n_abd} ABD)", flush=True)
    robot = _setup_after_finalize(eng, envs, P)
    if mode == "force":
        arm_force_barrier(eng, robot, P)   # no-overshoot IPC barrier at the closed limit
    ejs = slice_env_joints(robot, num_envs)
    groups = build_drive_groups(envs, ejs)
    gstate = {}
    actions, L = prep["actions"], len(prep["actions"])
    phase = int(os.environ.get("CASE39ME_PHASE", "0"))

    if int(os.environ.get("CASE39ME_HEADLESS", "0")):
        f0 = int(os.environ.get("CASE39_FRAME_START", "0"))
        f1 = min(int(os.environ.get("CASE39_FRAME_END", str(L))), L)
        ms = []
        for fr in range(f0, f1):
            # per-env trajectory phase offset: drive each group's env from its own frame
            if phase:
                for grp in groups:
                    e = grp['key'][0]
                    raw = actions[(fr + e * phase) % L]
                    _drive_one_group(eng, robot, grp, prep, raw, mode, gstate, P, stitch_seg)
                # arms per env
                for e, ej in enumerate(ejs):
                    raw = actions[(fr + e * phase) % L]
                    _drive_arm(robot, ej, prep, raw)
            else:
                drive_frame(eng, robot, ejs, groups, prep, actions[fr], mode, gstate, P, stitch_seg)
            t = time.perf_counter(); eng.step(); ms.append((time.perf_counter() - t) * 1000.0)
            if fr % 20 == 0:
                print(f"[umi:{scene_name}] frame {fr:4d} step={ms[-1]:6.0f}ms", flush=True)
        mm = float(np.mean(ms)) if ms else float('nan')
        print(f"\n[umi:{scene_name}] {num_envs} envs, {len(ms)} frames: mean {mm:.1f}ms "
              f"({1000.0/mm:.2f} fps) = {mm/num_envs:.1f} ms/env  [mode={mode}]", flush=True)
        return

    import polyscope as ps, polyscope.imgui as psim
    v = eng.get_vertices(); fa = eng.get_surface_faces()
    ps.init(); ps.set_up_dir("y_up"); ps.set_ground_plane_mode("none")
    st = dict(idx=0, run=False, ms=0.,
              mesh=ps.register_surface_mesh("scene", v, fa, color=(0.6, 0.7, 0.8)), v=v, f=fa)

    def cb():
        if st['run']:
            if psim.Button("Pause"):
                st['run'] = False
        else:
            if psim.Button("Start" if st['idx'] == 0 else "Resume"):
                st['run'] = True
        psim.SameLine()
        if psim.Button("Reset"):
            st['idx'] = 0; st['run'] = False
        psim.Text(f"{scene_name}  mode={mode}  envs={num_envs}  frame {st['idx']}/{L}")
        psim.Text(f"step {st['ms']:6.1f} ms")
        if not st['run'] or st['idx'] >= L:
            return
        drive_frame(eng, robot, ejs, groups, prep, actions[st['idx']], mode, gstate, P, stitch_seg)
        t = time.perf_counter(); eng.step(); st['ms'] = (time.perf_counter() - t) * 1000.0
        v = eng.get_vertices(); fa = eng.get_surface_faces()
        if v.shape[0] != st['v'].shape[0] or fa.shape != st['f'].shape:
            st['mesh'] = ps.register_surface_mesh("scene", v, fa, color=(0.6, 0.7, 0.8)); st['v'], st['f'] = v, fa
        else:
            st['mesh'].update_vertex_positions(v)
        st['idx'] += 1

    ps.set_user_callback(cb); ps.show()


def _drive_arm(robot, ej, prep, raw):
    al, ar = prep["arm_l"], prep["arm_r"]
    for i, ri in enumerate(ej['left_rev']):
        robot.set_revolute_position(ri, float(raw[al[i]]), degree=False)
    for i, ri in enumerate(ej['right_rev']):
        robot.set_revolute_position(ri, float(raw[ar[i]]), degree=False)


def _drive_one_group(eng, robot, grp, prep, raw, mode, gstate, P, stitch_seg):
    all_stretch = None
    if mode == "stitch" and stitch_seg is not None:
        all_stretch = eng.native.get_stitch_max_stretch_batched(stitch_seg[0], stitch_seg[1])
    grip = float(raw[prep["grip_l"]]) if grp['key'][1] == 'L' else float(raw[prep["grip_r"]])
    drive_side(eng, robot, grp, grip, prep["binary"], mode, gstate, P, all_stretch)


def run_ui(scene_name):
    """Single-env interactive UI (v0.6.4 case_umi_finray_force_ui paradigm + live
    gripper-mode switch): every ARM revolute joint is a slider (deg); each GRIPPER
    is driven by per-side OPEN/CLOSE BUTTONS (L / R); the gripper control mode
    (pos / stitch / force) is a live dropdown. The arm starts at the trajectory's
    frame-0 pose; "Reset pose" returns to it. No replay — you drive everything."""
    import polyscope as ps, polyscope.imgui as psim
    prep = prepare_scene(scene_name)
    P = _drive_params()
    print(f"[umi-ui:{scene_name}] grip={'binary' if prep['binary'] else 'continuous'} "
          f"arm={os.path.basename(urdf_path())}", flush=True)
    eng = make_engine(prep, 1)
    sides = load_finray_sides()
    envs, n_abd = build_world(eng, prep, 1, 4.0, sides, dict(abd_cursor=0))
    stitch_seg = _stitch_seg_arrays(envs)
    eng.finalize()
    robot = _setup_after_finalize(eng, envs, P)
    ejs = slice_env_joints(robot, 1)
    ej = ejs[0]
    groups = build_drive_groups(envs, ejs)
    grp_by_side = {g['key'][1]: g for g in groups}
    gstate = {}
    actions = prep["actions"]
    a0 = actions[0]
    _drive_arm(robot, ej, prep, a0)   # start at the trajectory's frame-0 arm pose
    MODES = ["pos", "stitch", "force"]
    if MODES[MODES.index(os.environ.get("GRIP_MODE", "pos"))] == "force":
        arm_force_barrier(eng, robot, P)

    v = eng.get_vertices(); fa = eng.get_surface_faces()
    ps.init(); ps.set_up_dir("y_up"); ps.set_ground_plane_mode("none")
    ui = dict(run=False, mode_i=MODES.index(os.environ.get("GRIP_MODE", "pos")),
              Lc=False, Rc=False, ms=0., show_edges=False,
              mesh=ps.register_surface_mesh("scene", v, fa, color=(0.6, 0.7, 0.8)), v=v, f=fa)

    def cb():
        # run / pause / reset-pose
        if ui['run']:
            if psim.Button("Pause"):
                ui['run'] = False
        else:
            if psim.Button("Run"):
                ui['run'] = True
        psim.SameLine()
        if psim.Button("Reset pose"):
            _drive_arm(robot, ej, prep, a0)
            ui['Lc'] = ui['Rc'] = False; gstate.clear()
        psim.SameLine()
        ch_e, val_e = psim.Checkbox("show edges", ui['show_edges'])
        if ch_e:
            ui['show_edges'] = val_e; ui['mesh'].set_edge_width(0.5 if val_e else 0.0)
        psim.Text(f"{scene_name}   step {ui['ms']:6.1f} ms")

        # gripper control MODE (live switch). Switching resets latch state, returns
        # the joints to plain position drive, and arms/disarms the forcebarrier.
        cm, mi = psim.Combo("gripper mode", ui['mode_i'], MODES)
        if cm:
            ui['mode_i'] = mi; gstate.clear()
            reset_prismatic_drive(eng, robot, P)
            arm_force_barrier(eng, robot, P, arm=(MODES[mi] == "force"))

        # per-arm OPEN/CLOSE buttons (the prismatic gripper, L / R separate)
        psim.Separator()
        psim.Text("Grippers (per arm)")
        if psim.Button("L CLOSE"):
            ui['Lc'] = True
        psim.SameLine()
        if psim.Button("L OPEN"):
            ui['Lc'] = False; gstate.pop((0, 'L'), None)
        psim.SameLine()
        psim.TextUnformatted("left [CLOSED]" if ui['Lc'] else "left [open]")
        if psim.Button("R CLOSE"):
            ui['Rc'] = True
        psim.SameLine()
        if psim.Button("R OPEN"):
            ui['Rc'] = False; gstate.pop((0, 'R'), None)
        psim.SameLine()
        psim.TextUnformatted("right [CLOSED]" if ui['Rc'] else "right [open]")

        # arm joint sliders (read live target, push on change), in degrees
        psim.Separator(); psim.Text("Left arm joints (deg)")
        for ri in ej['left_rev']:
            ji = robot.revolute_joints[ri]
            chg, val = psim.SliderFloat(ji.name, robot.get_revolute_target_deg(ri),
                                        ji.lower_limit_deg, ji.upper_limit_deg)
            if chg:
                robot.set_revolute_position(ri, val, degree=True)
        psim.Separator(); psim.Text("Right arm joints (deg)")
        for ri in ej['right_rev']:
            ji = robot.revolute_joints[ri]
            chg, val = psim.SliderFloat(ji.name, robot.get_revolute_target_deg(ri),
                                        ji.lower_limit_deg, ji.upper_limit_deg)
            if chg:
                robot.set_revolute_position(ri, val, degree=True)

        if not ui['run']:
            return
        # re-assert the per-side grippers every step (force/stitch are continuous):
        # closed -> grip=-1, open -> grip=+1, via the selected mode.
        mode = MODES[ui['mode_i']]
        all_stretch = (eng.native.get_stitch_max_stretch_batched(stitch_seg[0], stitch_seg[1])
                       if mode == "stitch" else None)
        drive_side(eng, robot, grp_by_side['L'], -1.0 if ui['Lc'] else 1.0, prep["binary"], mode, gstate, P, all_stretch)
        drive_side(eng, robot, grp_by_side['R'], -1.0 if ui['Rc'] else 1.0, prep["binary"], mode, gstate, P, all_stretch)
        t = time.perf_counter(); eng.step(); ui['ms'] = (time.perf_counter() - t) * 1000.0
        v = eng.get_vertices(); fa = eng.get_surface_faces()
        if v.shape[0] != ui['v'].shape[0] or fa.shape != ui['f'].shape:
            ui['mesh'] = ps.register_surface_mesh("scene", v, fa, color=(0.6, 0.7, 0.8))
            ui['mesh'].set_edge_width(0.5 if ui['show_edges'] else 0.0); ui['v'], ui['f'] = v, fa
        else:
            ui['mesh'].update_vertex_positions(v)

    ps.set_user_callback(cb); ps.show()
