#!/usr/bin/env python3
"""Multi-environment CUP(beaker)+SHIRT replay with the UMI FINRAY soft gripper.

The SECOND beaker-class scene (the ~1018-frame one that has BOTH a shirt and a
cup/beaker), multi-env sibling of replay_case39_UMI_obb_cup_shirt_forcegrip.py:
OBB arm + UMI finray STRATEGY_F hybrid soft gripper grasps a rigid cup (ABD)
while a shirt (cloth FEM) sits in the scene, replaying the 1018-frame cup-grasp
trajectory qpos_case39.h5. Differs from:
  * replay_beaker_UMI_finray_multienv.py  — 94-frame beaker-only episode, no cloth.
  * replay_foldshirt_UMI_finray_multienv.py — shirt only, no cup.

Gripper = PURE POSITION CONTROL (each prismatic joint driven straight to the
commanded opening, POS_K stiffness; grasp compliance from the FEM fin-ray truss).
NOTE: qpos_case39.h5 action layout is [L_arm(0:7), R_arm(7:14), gripL(14),
gripR(15)] (grips are binary -1/+1), NOT the UMI [L,gripL,R,gripR] layout.

N envs tiled in ONE merged world; P1 set_body_groups isolation + per-env
exclusions. Per-env line-search via STIFF_PERENV_ALPHA=1 (+ _MASK).

Usage:
    PYTHONPATH=. CASE39ME_HEADLESS=1 CASE39ME_NUM_ENVS=4 \
        python examples/replay_cupshirt_UMI_finray_multienv.py [qpos.h5]
"""
import sys, os, math, time, re
from pathlib import Path

_ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from stiff_physics import Engine, Config
from stiff_physics.robot import Robot

_URDF_DIR = _ASSETS_DIR + "sim_data/urdf/ridgeback_dual_panda_UMI/"
URDF_PATH = (_URDF_DIR + "ridgeback_dual_panda2.urdf"
             if os.environ.get("GRIP_URDF", "obb") == "detailed"
             else _URDF_DIR + "ridgeback_dual_panda2_OBB.urdf")
_FEM_DIR = "sim_data/" + os.environ.get("CASE39UMI_FINRAY_DIR", "umi_hybrid_sf_v800")
CUP_MSH = _ASSETS_DIR + "sim_data/tetmesh/softgriper_cup.msh"
SHIRT_OBJ = _ASSETS_DIR + "triMesh/shirt_6436v.obj"
DEFAULT_TRAJ = _ASSETS_DIR + "trajectories/qpos_case39.h5"
ARM_SCALE = 1.0
FINGER_LABELS = ['left_arm_leftfinger', 'left_arm_rightfinger',
                 'right_arm_leftfinger', 'right_arm_rightfinger']


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


def link_visual_origin(urdf_path, link_name):
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
    xyz = np.array([float(x) for x in om_xyz.group(1).split()]) if om_xyz else np.zeros(3)
    rpy = np.array([float(x) for x in om_rpy.group(1).split()]) if om_rpy else np.zeros(3)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def load_finray_sides():
    sides = {}
    for side in ('L', 'R'):
        unified_npz, rigid_msh, rigid_remap = _sf_paths(side)
        d = np.load(unified_npz)
        rr = np.load(rigid_remap)
        sides[side] = dict(
            verts=np.ascontiguousarray(d['vertices'], np.float64),
            tets=np.ascontiguousarray(d['tets'], np.int32),
            rigid_v_idx=np.asarray(rr['rigid_v_idx'], np.int64),
            rigid_msh=rigid_msh,
            vis={lbl: link_visual_origin(URDF_PATH, lbl) for lbl in FINGER_LABELS},
        )
    return sides


def build_env_abd(eng, env_tf, sides, arm_tf0, cup_T0, ge):
    """Phase A: OBB URDF + 4 finray rigid roots + 1 cup ABD (all contiguous)."""
    arm_tf = env_tf @ arm_tf0
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})
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
    rigid_young = float(os.environ.get("CASE39UMI_RIGID_YOUNG", "1e8"))
    grippers = []
    for label in FINGER_LABELS:
        side = 'L' if label.endswith('leftfinger') else 'R'
        finger_T = np.asarray(eng.native.get_urdf_link_transform(label))
        gripper_T = finger_T @ sides[side]['vis'][label]
        eng.load_mesh(sides[side]['rigid_msh'], dimensions=3, body_type="ABD",
                      transform=gripper_T, young_modulus=rigid_young,
                      boundary_type="Free")
        rr = eng.get_load_records()[-1]
        grippers.append(dict(label=label, side=side,
                             finger_id=finger_recs[label].body_offset,
                             gripper_T=gripper_T, rigid_abd_id=rr.body_offset,
                             rigid_abd_v_off=rr.vertex_offset))
    cup_T = env_tf @ cup_T0
    eng.load_mesh(CUP_MSH, dimensions=3, body_type="ABD",
                  transform=cup_T, young_modulus=float(os.environ.get("CUP_YOUNG", "1e8")),
                  boundary_type="Free")
    cup_rec = eng.get_load_records()[-1]
    ge['abd_cursor'] = max(r.body_offset for r in eng.get_load_records()
                           if r.body_type == 0) + 1
    return dict(arm_ids=arm_ids, grippers=grippers,
                cup_id=cup_rec.body_offset, cup_rec=cup_rec)


def build_env_fem(eng, env, env_tf, sides, shirt_T0, ge):
    """Phase B: 4 finray FEM trusses + shirt, then gap-0 stitch + fixed joints."""
    fem_young = float(os.environ.get("CASE36_FEM_YOUNG", "1e7"))
    grippers = env['grippers']
    for g in grippers:
        s = sides[g['side']]
        eng.native.load_mesh_from_data(s['verts'], s['tets'], 4, 3, 1,
                                       g['gripper_T'], fem_young, 0)
        fr = eng.get_load_records()[-1]
        g['fem_rec'] = fr
        g['fem_v_off'] = fr.vertex_offset
        g['fem_body_offset'] = fr.body_offset
    shirt_T = env_tf @ shirt_T0
    eng.load_mesh(SHIRT_OBJ, dimensions=2, body_type="FEM", transform=shirt_T,
                  young_modulus=ge['shirt_young'])
    env['shirt_rec'] = eng.get_load_records()[-1]
    for g in grippers:
        rvidx = sides[g['side']]['rigid_v_idx']
        for i in range(len(rvidx)):
            eng.add_stitch_spring(g['fem_v_off'] + int(rvidx[i]),
                                  g['rigid_abd_v_off'] + i, g['rigid_abd_id'],
                                  rest_offset_world=(0., 0., 0.))
        anchor = g['gripper_T'][:3, 3]
        g['fj_idx'] = eng.native.add_fixed_joint(
            parent_body=g['finger_id'], child_body=g['rigid_abd_id'],
            world_anchor=anchor, world_normal=np.array([1., 0., 0.]),
            world_bitangent=np.array([0., 0., 1.]))


def exclusions_for_env(eng, env, n_abd_total):
    grippers, arm_ids = env['grippers'], env['arm_ids']
    for g in grippers:
        g['fem_global_id'] = n_abd_total + g['fem_body_offset']
    shirt_gid = n_abd_total + env['shirt_rec'].body_offset
    for g in grippers:
        eng.native.add_collision_exclusion(g['rigid_abd_id'], g['fem_global_id'])
        eng.native.add_collision_exclusion(g['fem_global_id'], g['finger_id'])
        eng.native.add_collision_exclusion(g['rigid_abd_id'], g['finger_id'])
        for arm_id in arm_ids:
            if arm_id == g['finger_id']:
                continue
            eng.native.add_collision_exclusion(g['rigid_abd_id'], arm_id)
            eng.native.add_collision_exclusion(g['fem_global_id'], arm_id)
    # cup + shirt: exclude from non-finger arm bodies (spurious OBB contact); the
    # finray rigid/FEM are NOT excluded -> they collide = the actual grasp/fold.
    finger_offsets = {g['finger_id'] for g in grippers}
    for arm_id in arm_ids:
        if arm_id not in finger_offsets:
            eng.native.add_collision_exclusion(arm_id, env['cup_id'])
        eng.native.add_collision_exclusion(arm_id, shirt_gid)
    for g in grippers:
        eng.add_ground_collision_skip(g['fem_global_id'])
        eng.add_ground_collision_skip(g['rigid_abd_id'])
    pre = lambda l: 'left' if l.startswith('left_') else 'right'
    for i, gi in enumerate(grippers):
        for gj in grippers[i + 1:]:
            if pre(gi['label']) != pre(gj['label']):
                continue
            gi_b = (gi['rigid_abd_id'], gi['fem_global_id'])
            gj_b = (gj['rigid_abd_id'], gj['fem_global_id'], gj['finger_id'])
            for a in gi_b:
                for b in gj_b:
                    eng.native.add_collision_exclusion(a, b)
            eng.native.add_collision_exclusion(gi['finger_id'], gj['rigid_abd_id'])
            eng.native.add_collision_exclusion(gi['finger_id'], gj['fem_global_id'])
    env['shirt_global_id'] = shirt_gid


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


def apply_frame(robot, ej, raw, close_r=0.0):
    # qpos_case39 layout [L_arm(0:7), R_arm(7:14), gripL(14), gripR(15)];
    # grips are binary -1(close)/+1(open). PURE POSITION CONTROL with the UMI
    # finray's MIRRORED prismatic limits (open = end farthest from 0, close = ~0).
    for i, ri in enumerate(ej['left_rev']):
        robot.set_revolute_position(ri, float(raw[i]), degree=False)
    for i, ri in enumerate(ej['right_rev']):
        robot.set_revolute_position(ri, float(raw[7 + i]), degree=False)
    for grip, pris in ((float(raw[14]), ej['left_pri']),
                       (float(raw[15]), ej['right_pri'])):
        for pi in pris:
            lo = robot.prismatic_joints[pi].lower_limit
            hi = robot.prismatic_joints[pi].upper_limit
            op = lo if abs(lo) > abs(hi) else hi
            cl = hi if abs(lo) > abs(hi) else lo
            gp = op if grip >= 0 else (op + (1.0 - close_r) * (cl - op))
            robot.set_prismatic_position(pi, gp, millimeters=False)


def main():
    traj = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else DEFAULT_TRAJ
    num_envs = int(os.environ.get("CASE39ME_NUM_ENVS", "4"))
    spacing = float(os.environ.get("CASE39ME_SPACING", "4.0"))
    close_r = float(os.environ.get("CASE39_CLOSE_RATIO", "0.0"))
    pos_k = float(os.environ.get("POS_K", os.environ.get("GRIP_K", "15.0")))

    import h5py
    with h5py.File(traj, "r") as f:
        actions = f["qpos"][:] if "qpos" in f else f["actions"][:]
    robot_init_pose = np.array([-0.8, 0.0, 0.0])  # case_39 cup-grasp base pose

    cup_scale = float(os.environ.get("CASE39_CUP_SCALE", "0.8"))
    cup_xyz = np.array([float(s) for s in os.environ.get("CASE39_CUP_XYZ", "0.67,-0.2,-0.4").split(",")])
    cup_T0 = np.eye(4); cup_T0[:3, :3] *= cup_scale; cup_T0[:3, 3] = cup_xyz
    shirt_scale = float(os.environ.get("CASE39_SHIRT_SCALE", "1.0"))
    shirt_xyz = np.array([float(s) for s in os.environ.get("CASE39_SHIRT_XYZ", "0.67,0,0").split(",")])
    shirt_T0 = np.eye(4); shirt_T0[:3, :3] *= shirt_scale; shirt_T0[:3, 3] = shirt_xyz
    print(f"[cs-umi] traj={os.path.basename(traj)} frames={len(actions)} envs={num_envs} "
          f"arm={os.path.basename(URDF_PATH)} pos_k={pos_k} (cup@{cup_xyz} shirt@{shirt_xyz})", flush=True)

    cfg = Config(
        dt=0.020, cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1e4, poisson_rate=0.49,
        friction_rate=float(os.environ.get("CASE39_FRICTION", "0.8")),
        relative_dhat=1e-3,
        joint_strength_ratio=100.0, revolute_driving_strength_ratio=100.0,
        prismatic_strength_ratio=float(os.environ.get("CASE39_PRISMATIC_CONSTRAINT_K", "2000")),
        semi_implicit_enabled=bool(int(os.environ.get("CASE39_SEMI", "0"))),
        semi_implicit_beta_tol=5e-2, semi_implicit_min_iter=1,
        newton_tol=float(os.environ.get("CASE39_NEWTON_TOL", "5e-2")),
        newton_iter_cap=int(os.environ.get("CASE39_NEWTON_CAP", "50")),
        preconditioner_type=int(os.environ.get("CASE39_PRECOND", "1")),
        ground_offset=float(os.environ.get("CASE39_GROUND_OFFSET", "-1.67")),
        assets_dir=_ASSETS_DIR)
    cfg._cfg.collision_detection_buff_scale = float(os.environ.get("CASE39ME_BUFF_SCALE", "4.0"))
    cfg._cfg.linear_system_buff_scale = float(os.environ.get("CASE39ME_LSYS_SCALE", "2.0"))
    cfg._cfg.triplet_internal_margin = float(os.environ.get("CASE39ME_TRIPLET_MARGIN", "4.0"))
    cfg._cfg.absolute_dhat = float(os.environ.get("CASE39ME_ABS_DHAT", "0.00239"))
    eng = Engine(cfg)
    if int(os.environ.get("CASE39_QUIET", "1")):
        eng.set_log_level(0)

    sides = load_finray_sides()
    ge = dict(fem_young=1e7, shirt_young=1e2, abd_cursor=0)
    arm_tf0 = make_arm_tf(robot_init_pose[:3], ARM_SCALE)

    print(f"\n[cs-umi] === building {num_envs} envs ===", flush=True)
    offs = make_env_offsets(num_envs, spacing)
    t0 = time.perf_counter()
    envs = [build_env_abd(eng, o, sides, arm_tf0, cup_T0, ge) for o in offs]
    for env, o in zip(envs, offs):
        build_env_fem(eng, env, o, sides, shirt_T0, ge)
    n_abd_total = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    for env in envs:
        exclusions_for_env(eng, env, n_abd_total)
    if num_envs > 1 and int(os.environ.get("CASE39ME_ISOLATE", "1")):
        n_fem_total = sum(1 for r in eng.get_load_records() if r.body_type == 1)
        m_abd, m_fem = n_abd_total // num_envs, n_fem_total // num_envs
        groups = [cid // m_abd for cid in range(n_abd_total)] + \
                 [f // m_fem for f in range(n_fem_total)]
        eng.native.set_body_groups(groups)
        print(f"[cs-umi] env isolation ON: {n_abd_total} ABD + {n_fem_total} FEM "
              f"-> {num_envs} groups", flush=True)
    eng.finalize()
    print(f"[cs-umi] finalized {num_envs} envs in {time.perf_counter()-t0:.1f}s "
          f"({n_abd_total} ABD)", flush=True)

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
    # PURE POSITION CONTROL: drive every prismatic joint to its commanded opening.
    for i in range(len(robot.prismatic_joints)):
        eng.native.set_prismatic_strength(i, pos_k)
    ejs = slice_env_joints(robot, num_envs)
    print(f"[cs-umi] {len(robot.revolute_joints)} rev + {len(robot.prismatic_joints)} pri\n", flush=True)

    phase = int(os.environ.get("CASE39ME_PHASE", "0"))
    L = len(actions)

    if int(os.environ.get("CASE39ME_HEADLESS", "0")):
        f0 = int(os.environ.get("CASE39_FRAME_START", "0"))
        f1 = min(int(os.environ.get("CASE39_FRAME_END", str(L))), L)
        cup_ranges = [(env['cup_rec'].vertex_offset, env['cup_rec'].vertex_count) for env in envs]
        ms = []
        for fr in range(f0, f1):
            for e, ej in enumerate(ejs):
                apply_frame(robot, ej, actions[(fr + e * phase) % L], close_r)
            t = time.perf_counter(); eng.step(); ms.append((time.perf_counter() - t) * 1000.0)
            if fr % 20 == 0:
                v = eng.get_vertices()
                cy = [float(v[o:o + c, 1].mean()) for (o, c) in cup_ranges]
                print(f"[cs-umi-hl] frame {fr:4d} step={ms[-1]:6.0f}ms cup_y/env={['%+.3f'%z for z in cy]}", flush=True)
        mm = float(np.mean(ms))
        print(f"\n[cs-umi-hl] {num_envs} envs, {len(ms)} frames: mean {mm:.1f}ms "
              f"({1000.0/mm:.2f} fps) = {mm/num_envs:.1f} ms/env", flush=True)
        return

    import polyscope as ps, polyscope.imgui as psim
    v = eng.get_vertices(); fa = eng.get_surface_faces()
    ps.init(); ps.set_up_dir("y_up"); ps.set_ground_plane_mode("shadow_only")
    st = dict(idx=0, run=False, ms=0., fps=0.,
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
            st['idx'] = 0; st['run'] = False; st['fps'] = 0.
        eqms = st['ms'] / num_envs if num_envs else st['ms']
        psim.Text(f"frame {st['idx']}/{L}   envs {num_envs}")
        psim.Text(f"step {st['ms']:6.1f} ms    FPS {st['fps']:5.2f}")
        psim.Text(f"per-env-equiv {eqms:6.1f} ms  ({(1000.0/eqms) if eqms>0 else 0:5.1f} env-steps/s)")
        if not st['run'] or st['idx'] >= L:
            return
        for e, ej in enumerate(ejs):
            apply_frame(robot, ej, actions[(st['idx'] + e * phase) % L], close_r)
        t = time.perf_counter(); eng.step(); st['ms'] = (time.perf_counter() - t) * 1000.0
        inst = 1000.0 / st['ms'] if st['ms'] > 0 else 0.0
        st['fps'] = inst if st['fps'] == 0. else 0.9 * st['fps'] + 0.1 * inst
        v = eng.get_vertices(); fa = eng.get_surface_faces()
        if v.shape[0] != st['v'].shape[0] or fa.shape != st['f'].shape:
            st['mesh'] = ps.register_surface_mesh("scene", v, fa, color=(0.6, 0.7, 0.8))
            st['v'], st['f'] = v, fa
        else:
            st['mesh'].update_vertex_positions(v)
        st['idx'] += 1

    ps.set_user_callback(cb); ps.show()


if __name__ == "__main__":
    main()
