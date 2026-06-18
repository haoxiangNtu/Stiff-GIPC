#!/usr/bin/env python3
"""Multi-environment BEAKER-GRASP replay with the UMI FINRAY soft gripper.

Sibling of replay_foldshirt_UMI_finray_multienv.py: same v0.6.4 UMI finray
STRATEGY_F hybrid soft gripper (rigid mount root + FEM fin-ray truss, gap-0
stitch) driven by PURE POSITION CONTROL (continuous grip command [-1,+1] mapped
linearly to finger opening and each prismatic joint driven straight to it; POS_K
= position-drive stiffness; grasp compliance from the FEM truss, no force-ctrl),
but it GRASPS a rigid 100 ml BEAKER (ABD) instead of folding a shirt. No cloth.

  * Arm   : OBB coarse collision (ridgeback_dual_panda2_OBB.urdf) by default.
  * Grip  : 4 finray fingers (rigid ABD root young 1e8 + FEM truss young 1e7).
  * Object: the episode's recorded beaker (ABD, young 1e5), one per env, loaded
            from object_init_info via trimesh. The finray fingers DO collide with
            it (that is the grasp); only the non-finger arm bodies are excluded.
  * N envs tiled in ONE merged world; P1 set_body_groups isolation + per-env
    exclusions. Contact-light (rigid object, no cloth) -> defaults preconditioner
    type 0 (no MAS), the config it was recorded with.

Per-env line-search (S1-S4) via STIFF_PERENV_ALPHA=1 (+ _MASK).

Usage:
    PYTHONPATH=. CASE39ME_HEADLESS=1 CASE39ME_NUM_ENVS=4 \
        python examples/replay_beaker_UMI_finray_multienv.py [episode.hdf5]
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
URDF_PATH = (_URDF_DIR + "ridgeback_dual_panda2.urdf"
             if os.environ.get("GRIP_URDF", "obb") == "detailed"
             else _URDF_DIR + "ridgeback_dual_panda2_OBB.urdf")
_FEM_DIR = "sim_data/" + os.environ.get("CASE39UMI_FINRAY_DIR", "umi_hybrid_sf_v800")
DEFAULT_EP = _ASSETS_DIR + "trajectories/episode_grasp_beaker_umi.hdf5"
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


def build_env_abd(eng, env_tf, sides, arm_tf0, beaker, ge):
    """Phase A: OBB URDF + 4 finray rigid roots + 1 beaker ABD (all contiguous)."""
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
    # beaker ABD (the grasp target) at this env's recorded pose
    beaker_T = env_tf @ beaker['T0']
    eng.load_mesh_from_data(vertices=beaker['verts'], faces=beaker['faces'],
                            verts_per_face=3, dimensions=3, body_type="ABD",
                            transform=beaker_T, young_modulus=float(os.environ.get("BEAKER_YOUNG", "1e5")),
                            boundary_type="Free")
    beaker_rec = eng.get_load_records()[-1]
    ge['abd_cursor'] = max(r.body_offset for r in eng.get_load_records()
                           if r.body_type == 0) + 1
    return dict(arm_ids=arm_ids, grippers=grippers,
                beaker_id=beaker_rec.body_offset, beaker_rec=beaker_rec)


def build_env_fem(eng, env, sides, ge):
    """Phase B: 4 finray FEM trusses, then gap-0 stitch + fixed joints. No cloth."""
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
    for g in grippers:
        eng.native.add_collision_exclusion(g['rigid_abd_id'], g['fem_global_id'])
        eng.native.add_collision_exclusion(g['fem_global_id'], g['finger_id'])
        eng.native.add_collision_exclusion(g['rigid_abd_id'], g['finger_id'])
        for arm_id in arm_ids:
            if arm_id == g['finger_id']:
                continue
            eng.native.add_collision_exclusion(g['rigid_abd_id'], arm_id)
            eng.native.add_collision_exclusion(g['fem_global_id'], arm_id)
    # beaker: exclude from non-finger arm bodies (spurious OBB contact); the
    # finray rigid/FEM are NOT excluded -> they collide = the actual grasp.
    finger_offsets = {g['finger_id'] for g in grippers}
    for arm_id in arm_ids:
        if arm_id not in finger_offsets:
            eng.native.add_collision_exclusion(arm_id, env['beaker_id'])
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
    # action layout [L_arm(0:7), gripL(7), R_arm(8:15), gripR(15)].
    for i, ri in enumerate(ej['left_rev']):
        robot.set_revolute_position(ri, float(raw[i]), degree=False)
    for i, ri in enumerate(ej['right_rev']):
        robot.set_revolute_position(ri, float(raw[8 + i]), degree=False)
    # PURE POSITION CONTROL. Grip command is CONTINUOUS in [-1,+1] (+1 fully open,
    # -1 fully closed) -> map linearly to finger opening, drive each prismatic
    # joint straight to it. UMI finray fingers have MIRRORED limits (joint1
    # [0,+0.041], joint2 [-0.041,0]); open = end farthest from 0, close = near-0.
    # Joint tracks position firmly; grasp compliance is from the FEM truss.
    for grip, pris in ((float(raw[7]), ej['left_pri']),
                       (float(raw[15]), ej['right_pri'])):
        s = min(max((grip + 1.0) * 0.5, 0.0), 1.0)   # 0 = closed, 1 = open
        for pi in pris:
            lo = robot.prismatic_joints[pi].lower_limit
            hi = robot.prismatic_joints[pi].upper_limit
            op = lo if abs(lo) > abs(hi) else hi      # fully-open end
            cl = hi if abs(lo) > abs(hi) else lo      # near-0 (closed) end
            gp = cl + s * (op - cl)                   # linear close->open
            robot.set_prismatic_position(pi, gp, millimeters=False)


def main():
    ep = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else DEFAULT_EP
    num_envs = int(os.environ.get("CASE39ME_NUM_ENVS", "4"))
    spacing = float(os.environ.get("CASE39ME_SPACING", "4.0"))
    close_r = float(os.environ.get("CASE39_CLOSE_RATIO", "0.0"))
    pos_k = float(os.environ.get("POS_K", os.environ.get("GRIP_K", "15.0")))  # prismatic position-drive stiffness

    import h5py, trimesh
    with h5py.File(ep, "r") as f:
        actions = f["actions"][:]
        robot_init_pose = np.asarray(f.attrs["robot_init_pose"])
        oi = json.loads(f.attrs["object_init_info"])
        ec = json.loads(f.attrs["env_cfg"])
    # beaker (the single ABD object)
    bkey = next(k for k, v in oi.items() if v.get("body_type") == "ABD")
    beaker_mesh = oi[bkey]["collision_mesh"]
    for _pre in ("/data/stiff-physics/franka_sim/assets_new/",
                 "/data/stiff-physics/franka_sim/assets/",
                 "/data/stiff-physics/assets/", "assets/"):
        if beaker_mesh.startswith(_pre):
            beaker_mesh = _ASSETS_DIR + beaker_mesh[len(_pre):]
            break
    else:
        if not os.path.isabs(beaker_mesh):
            beaker_mesh = _ASSETS_DIR + beaker_mesh
    bmesh = trimesh.load(beaker_mesh, force='mesh')
    beaker = dict(verts=np.asarray(bmesh.vertices), faces=np.asarray(bmesh.faces),
                  T0=np.asarray(oi[bkey]["initial_pose"]).reshape(4, 4))
    print(f"[bk-umi] episode={os.path.basename(ep)} frames={len(actions)} envs={num_envs} "
          f"arm={os.path.basename(URDF_PATH)} beaker={os.path.basename(beaker_mesh)} "
          f"({len(beaker['verts'])}v) pos_k={pos_k}", flush=True)

    cfg = Config(
        dt=0.020, cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1e4, poisson_rate=0.49,
        friction_rate=float(os.environ.get("CASE39_FRICTION", str(ec.get("friction_rate", 0.8)))),
        relative_dhat=1e-3,
        joint_strength_ratio=100.0, revolute_driving_strength_ratio=100.0,
        prismatic_strength_ratio=float(os.environ.get("CASE39_PRISMATIC_CONSTRAINT_K", "2000")),
        semi_implicit_enabled=bool(int(os.environ.get("CASE39_SEMI", "0"))),
        semi_implicit_beta_tol=5e-2, semi_implicit_min_iter=1,
        newton_tol=float(os.environ.get("CASE39_NEWTON_TOL", "5e-2")),
        newton_iter_cap=int(os.environ.get("CASE39_NEWTON_CAP", "50")),
        # contact-light rigid-object scene -> no MAS by default (recorded config)
        preconditioner_type=int(os.environ.get("CASE39_PRECOND", "0")),
        ground_offset=float(ec.get("ground_offset", 0.75)), assets_dir=_ASSETS_DIR)
    cfg._cfg.collision_detection_buff_scale = float(os.environ.get("CASE39ME_BUFF_SCALE", "4.0"))
    cfg._cfg.linear_system_buff_scale = float(os.environ.get("CASE39ME_LSYS_SCALE", "2.0"))
    cfg._cfg.triplet_internal_margin = float(os.environ.get("CASE39ME_TRIPLET_MARGIN", "4.0"))
    cfg._cfg.absolute_dhat = float(os.environ.get("CASE39ME_ABS_DHAT", "0.00239"))
    eng = Engine(cfg)
    if int(os.environ.get("CASE39_QUIET", "1")):
        eng.set_log_level(0)

    sides = load_finray_sides()
    ge = dict(fem_young=1e7, abd_cursor=0)
    arm_tf0 = make_arm_tf(robot_init_pose[:3], ARM_SCALE)

    print(f"\n[bk-umi] === building {num_envs} envs ===", flush=True)
    offs = make_env_offsets(num_envs, spacing)
    t0 = time.perf_counter()
    envs = [build_env_abd(eng, o, sides, arm_tf0, beaker, ge) for o in offs]
    for env in envs:
        build_env_fem(eng, env, sides, ge)
    n_abd_total = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    for env in envs:
        exclusions_for_env(eng, env, n_abd_total)
    if num_envs > 1 and int(os.environ.get("CASE39ME_ISOLATE", "1")):
        n_fem_total = sum(1 for r in eng.get_load_records() if r.body_type == 1)
        m_abd, m_fem = n_abd_total // num_envs, n_fem_total // num_envs
        groups = [cid // m_abd for cid in range(n_abd_total)] + \
                 [f // m_fem for f in range(n_fem_total)]
        eng.native.set_body_groups(groups)
        print(f"[bk-umi] env isolation ON: {n_abd_total} ABD + {n_fem_total} FEM "
              f"-> {num_envs} groups", flush=True)
    eng.finalize()
    print(f"[bk-umi] finalized {num_envs} envs in {time.perf_counter()-t0:.1f}s "
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
    # PURE POSITION CONTROL: drive every prismatic joint to its commanded opening
    # (pos_k = position-drive stiffness). No external-force / force-limited path.
    for i in range(len(robot.prismatic_joints)):
        eng.native.set_prismatic_strength(i, pos_k)
    ejs = slice_env_joints(robot, num_envs)
    print(f"[bk-umi] {len(robot.revolute_joints)} rev + {len(robot.prismatic_joints)} pri\n", flush=True)

    phase = int(os.environ.get("CASE39ME_PHASE", "0"))
    L = len(actions)

    if int(os.environ.get("CASE39ME_HEADLESS", "0")):
        f0 = int(os.environ.get("CASE39_FRAME_START", "0"))
        f1 = min(int(os.environ.get("CASE39_FRAME_END", str(L))), L)
        bk_ranges = [(env['beaker_rec'].vertex_offset, env['beaker_rec'].vertex_count) for env in envs]
        ms = []
        for fr in range(f0, f1):
            for e, ej in enumerate(ejs):
                apply_frame(robot, ej, actions[(fr + e * phase) % L], close_r)
            t = time.perf_counter(); eng.step(); ms.append((time.perf_counter() - t) * 1000.0)
            if fr % 10 == 0:
                v = eng.get_vertices()
                by = [float(v[o:o + c, 1].mean()) for (o, c) in bk_ranges]
                print(f"[bk-umi-hl] frame {fr:4d} step={ms[-1]:6.0f}ms beaker_y/env={['%+.3f'%z for z in by]}", flush=True)
        mm = float(np.mean(ms))
        print(f"\n[bk-umi-hl] {num_envs} envs, {len(ms)} frames: mean {mm:.1f}ms "
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
