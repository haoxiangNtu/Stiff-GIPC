#!/usr/bin/env python3
"""Multi-environment FOLD-SHIRT trajectory replay (case_39 gripper, NO cup).

Sibling of replay_case39_multienv.py for the cloth-GRASPING (fold-shirt) scene:
same dual-panda + hybrid soft grippers, but it GRASPS/FOLDS a shirt (no cup),
replaying an imitation-learning episode (.hdf5 with `actions` (T,16) +
`robot_init_pose` + `object_init_info` attrs, e.g. /tmp/replay_0528/
episode_00000.hdf5). Cloth mesh = assets/objects/m-panda_single/scaled.obj
(2799 verts).

IMPORTANT — action layout for THIS dataset is [L_arm(0:7), L_grip(7),
R_arm(8:15), R_grip(15)] (the "7+1,7+1" convention), NOT [L_arm,R_arm,gripL,gripR]
like qpos_case39. Grip cols 7 & 15 are binary -1(close)/+1(open). Mapping them
wrong leaves the grippers permanently open (they approach but never close).

Measured ceiling (24GB, buff=4): ~20-22 envs (N=20 @21.4GB, N=22 @23.0GB edge,
N=24 OOM), ~40ms/env — comparable to replay_case39_multienv.

Usage:
    PYTHONPATH=. CASE39ME_HEADLESS=1 CASE39ME_NUM_ENVS=8 \
        python examples/replay_foldshirt_multienv.py [episode.hdf5]
"""
import sys, os, math, time, json
from pathlib import Path

_ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "Assets") + "/"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from stiff_physics import Engine, Config
from stiff_physics.robot import Robot

URDF_PATH   = _ASSETS_DIR + "sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_softgripper.urdf"
RIGID_MSH   = _ASSETS_DIR + "sim_data/hybrid_d/STRATEGY_F_rigid.msh"
RIGID_REMAP = _ASSETS_DIR + "sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"
UNIFIED_NPZ = _ASSETS_DIR + "sim_data/hybrid_d/STRATEGY_F_unified.npz"
# bundled episode (same schema as the /tmp/replay_0528 recordings: actions (T,16)
# + robot_init_pose + object_init_info attrs); argv[1] still overrides.
DEFAULT_EP  = str(Path(__file__).resolve().parent.parent / "Assets" / "trajectories"
                  / "episode_fold_shirt_umi.hdf5")
ARM_SCALE = 1.0
FINGER_LABELS = ['left_arm_leftfinger','left_arm_rightfinger',
                 'right_arm_leftfinger','right_arm_rightfinger']


def make_arm_tf(pos, scale=1.0):
    R = Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix()
    tf = np.eye(4); tf[:3,:3] = scale * R; tf[:3,3] = R @ np.asarray(pos)
    return tf


def make_env_offsets(n, spacing):
    cols = int(math.ceil(math.sqrt(n))); rows = int(math.ceil(n/cols))
    offs = []
    for e in range(n):
        r, c = divmod(e, cols)
        o = np.eye(4); o[0,3] = (c-(cols-1)/2.0)*spacing; o[2,3] = (r-(rows-1)/2.0)*spacing
        offs.append(o)
    return offs


def build_env_abd(eng, env_tf, hybrid, arm_tf0, ge):
    """Phase A: URDF (at robot_init_pose) + 4 gripper rigid ABDs. NO cup."""
    collision_origin = hybrid['collision_origin']
    arm_tf = env_tf @ arm_tf0
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})
    env_abd = [r for r in eng.get_load_records() if r.body_type == 0]
    finger_recs = {}
    for r in reversed(env_abd):
        if r.label in FINGER_LABELS and r.label not in finger_recs:
            finger_recs[r.label] = r
        if len(finger_recs) == 4: break
    if len(finger_recs) != 4:
        raise RuntimeError(f"expected 4 finger ABDs, got {len(finger_recs)}")
    arm_ids = [r.body_offset for r in env_abd if r.body_offset >= ge['abd_cursor']]
    for bid in arm_ids:
        eng.add_ground_collision_skip(bid)
    grippers = []
    for label in FINGER_LABELS:
        finger_T = eng.native.get_urdf_link_transform(label)
        gripper_T = finger_T @ collision_origin
        eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
                      transform=gripper_T, young_modulus=1e8, boundary_type="Free")
        rr = eng.get_load_records()[-1]
        grippers.append(dict(label=label, finger_id=finger_recs[label].body_offset,
                             gripper_T=gripper_T, abd_id=rr.body_offset, abd_v_off=rr.vertex_offset))
    ge['abd_cursor'] = max(r.body_offset for r in eng.get_load_records() if r.body_type==0) + 1
    return dict(arm_ids=arm_ids, grippers=grippers)


def build_env_fem(eng, env, env_tf, hybrid, cloth_obj, cloth_T0, ge):
    """Phase B: 4 gripper FEM softpads + cloth, then stitch springs + fixed joints."""
    rigid_v_idx, n_rigid_v = hybrid['rigid_v_idx'], hybrid['n_rigid_v']
    hv, ht = hybrid['verts'], hybrid['tets']
    grippers = env['grippers']
    for g in grippers:
        eng.native.load_mesh_from_data(hv, ht, 4, 3, 1, g['gripper_T'], ge['fem_young'], 0)
        fr = eng.get_load_records()[-1]
        g['fem_rec'] = fr; g['fem_v_off'] = fr.vertex_offset; g['fem_body_offset'] = fr.body_offset
    cloth_T = env_tf @ cloth_T0
    eng.load_mesh(cloth_obj, dimensions=2, body_type="FEM", transform=cloth_T,
                  young_modulus=ge['cloth_young'])
    env['cloth_rec'] = eng.get_load_records()[-1]
    for g in grippers:
        for i in range(n_rigid_v):
            eng.add_stitch_spring(g['fem_v_off'] + int(rigid_v_idx[i]),
                                  g['abd_v_off'] + i, g['abd_id'], rest_offset_world=(0.,0.,0.))
        anchor = g['gripper_T'][:3,3]
        g['fj_idx'] = eng.native.add_fixed_joint(
            parent_body=g['finger_id'], child_body=g['abd_id'], world_anchor=anchor,
            world_normal=np.array([1.,0.,0.]), world_bitangent=np.array([0.,0.,1.]))


def exclusions_for_env(eng, env, n_abd_total):
    grippers, arm_ids = env['grippers'], env['arm_ids']
    for g in grippers:
        g['fem_global_id'] = n_abd_total + g['fem_body_offset']
    cloth_gid = n_abd_total + env['cloth_rec'].body_offset
    for g in grippers:
        eng.native.add_collision_exclusion(g['abd_id'], g['fem_global_id'])
        eng.native.add_collision_exclusion(g['fem_global_id'], g['finger_id'])
        eng.native.add_collision_exclusion(g['abd_id'], g['finger_id'])
        for arm_id in arm_ids:
            if arm_id == g['finger_id']: continue
            eng.native.add_collision_exclusion(g['abd_id'], arm_id)
            eng.native.add_collision_exclusion(g['fem_global_id'], arm_id)
    finger_offsets = {g['finger_id'] for g in grippers}
    for arm_id in arm_ids:
        if arm_id in finger_offsets: continue
        eng.native.add_collision_exclusion(arm_id, cloth_gid)
    for g in grippers:
        eng.add_ground_collision_skip(g['fem_global_id']); eng.add_ground_collision_skip(g['abd_id'])
    pre = lambda l: 'left' if l.startswith('left_') else 'right'
    for i, gi in enumerate(grippers):
        for gj in grippers[i+1:]:
            if pre(gi['label']) != pre(gj['label']): continue
            for a in (gi['abd_id'], gi['fem_global_id']):
                for b in (gj['abd_id'], gj['fem_global_id']):
                    eng.native.add_collision_exclusion(a, b)
    env['cloth_global_id'] = cloth_gid


def slice_env_joints(robot, n):
    nr, npz = len(robot.revolute_joints), len(robot.prismatic_joints)
    assert nr % n == 0 and npz % n == 0
    rpe, ppe = nr//n, npz//n
    out = []
    for e in range(n):
        rb, pb = range(e*rpe,(e+1)*rpe), range(e*ppe,(e+1)*ppe)
        out.append(dict(
            left_rev =[i for i in rb if robot.revolute_joints[i].name.startswith('left_arm_joint')],
            right_rev=[i for i in rb if robot.revolute_joints[i].name.startswith('right_arm_joint')],
            left_pri =[i for i in pb if robot.prismatic_joints[i].name.startswith('left_arm')],
            right_pri=[i for i in pb if robot.prismatic_joints[i].name.startswith('right_arm')]))
    return out


def apply_frame(robot, ej, raw, close_r):
    # This episode's action layout is [L_arm(0:7), L_grip(7), R_arm(8:15), R_grip(15)]
    # (the "7+1, 7+1" convention) — NOT [L_arm, R_arm, gripL, gripR]. Grip cols 7
    # and 15 are binary-ish -1(close)/+1(open); col 14 is a right-arm joint.
    for i, ri in enumerate(ej['left_rev']):  robot.set_revolute_position(ri, float(raw[i]), degree=False)
    for i, ri in enumerate(ej['right_rev']): robot.set_revolute_position(ri, float(raw[8+i]), degree=False)
    for grip, pris in ((float(raw[7]), ej['left_pri']),
                       (float(raw[15]), ej['right_pri'])):
        if not pris: continue
        lo = robot.prismatic_joints[pris[0]].lower_limit; hi = robot.prismatic_joints[pris[0]].upper_limit
        gp = hi if grip >= 0 else (lo + close_r*(hi-lo))
        for pi in pris: robot.set_prismatic_position(pi, gp, millimeters=False)


def main():
    ep = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else DEFAULT_EP
    num_envs = int(os.environ.get("CASE39ME_NUM_ENVS", "4"))
    # [v0.8.5] Mode-driven env-layout policy: in strict mode the layout is a
    # FIXED strategy, not a scene choice — physics loads CO-LOCATED (so envs
    # are bitwise comparable), the engine separates envs for the broad phase
    # via d_env_offset, and visual separation is applied at DISPLAY time only
    # (never touching physics). Explicit CASE39ME_SPACING/BVH_OFFSET still
    # override for experiments.
    _mode = os.environ.get("STIFF_MULTIENV_MODE", "merged").strip().lower()
    _strict = _mode in ("strict", "2", "deterministic", "c")
    if _strict and "CASE39ME_SPACING" not in os.environ        and "CASE39ME_BVH_OFFSET" not in os.environ:
        os.environ["CASE39ME_SPACING"] = "0"
        os.environ["CASE39ME_BVH_OFFSET"] = "4.0"
        print("[fs] strict mode -> co-located physics + BVH-domain separation "
              "+ display-only visual offsets (fixed policy)", flush=True)
    spacing  = float(os.environ.get("CASE39ME_SPACING", "4.0"))
    close_r  = float(os.environ.get("CASE39_CLOSE_RATIO", "0.0"))

    import h5py
    with h5py.File(ep, "r") as f:
        actions = f["actions"][:]
        robot_init_pose = np.asarray(f.attrs["robot_init_pose"])
        oi = json.loads(f.attrs["object_init_info"])
        ec = json.loads(f.attrs["env_cfg"])
    # cloth (first/only FEM object)
    cloth_key = next(k for k,v in oi.items() if v.get("body_type") != "ABD")
    cloth_T0 = np.asarray(oi[cloth_key]["initial_pose"]).reshape(4,4)
    cloth_obj = _ASSETS_DIR + "objects/m-panda_single/scaled.obj"
    print(f"[fs] episode={ep}  frames={len(actions)}  envs={num_envs}  cloth={cloth_obj.split('/')[-1]}", flush=True)

    cfg = Config(
        dt=0.020, cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1e4, poisson_rate=0.49,
        friction_rate=float(os.environ.get("CASE39_FRICTION", str(ec.get("friction_rate",0.4)))),
        relative_dhat=1e-3,
        joint_strength_ratio=100.0, revolute_driving_strength_ratio=100.0,
        prismatic_strength_ratio=2000.0, semi_implicit_enabled=False,
        semi_implicit_beta_tol=5e-2, semi_implicit_min_iter=1, newton_tol=5e-2,
        newton_iter_cap=50, preconditioner_type=1,
        ground_offset=float(ec.get("ground_offset", 0.75)), assets_dir=_ASSETS_DIR)
    cfg._cfg.collision_detection_buff_scale = float(os.environ.get("CASE39ME_BUFF_SCALE", "4.0"))
    cfg._cfg.linear_system_buff_scale       = float(os.environ.get("CASE39ME_LSYS_SCALE", "2.0"))
    cfg._cfg.triplet_internal_margin        = float(os.environ.get("CASE39ME_TRIPLET_MARGIN", "4.0"))
    # absolute_dhat pinned to single-env value so contact does not inflate with N.
    cfg._cfg.absolute_dhat = float(os.environ.get("CASE39ME_ABS_DHAT", "0.0019"))
    eng = Engine(cfg)
    if num_envs > 1:
        print(f"[fs] NOTE: {num_envs} envs in ONE merged world; GPU-memory bound, "
              "contact pinned by absolute_dhat. Heavier per-env than case_39 (active "
              "grasping) but lighter cloth (2799v).", flush=True)

    # shared hybrid data
    _co = np.eye(4)
    _co[:3,:3] = Rotation.from_euler('xyz',[-1.57079632679,0.20245819348,-1.57079632679]).as_matrix()
    _co[:3,3] = [-0.0165,0.0165,0.12773331296]
    rr = np.load(RIGID_REMAP, allow_pickle=True); hd = np.load(UNIFIED_NPZ)
    hybrid = dict(collision_origin=_co, rigid_v_idx=rr['rigid_v_idx'], n_rigid_v=len(rr['rigid_v_idx']),
                  verts=np.ascontiguousarray(hd['vertices'],np.float64),
                  tets=np.ascontiguousarray(hd['tets'],np.int32))
    ge = dict(fem_young=1e7, cloth_young=1e2, abd_cursor=0)
    arm_tf0 = make_arm_tf(robot_init_pose[:3], ARM_SCALE)

    print(f"\n[fs] === building {num_envs} envs ===", flush=True)
    offs = make_env_offsets(num_envs, spacing)
    t0 = time.perf_counter()
    envs = [build_env_abd(eng, o, hybrid, arm_tf0, ge) for o in offs]
    for env, o in zip(envs, offs):
        build_env_fem(eng, env, o, hybrid, cloth_obj, cloth_T0, ge)
    n_abd_total = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    for env in envs:
        exclusions_for_env(eng, env, n_abd_total)
    # [P1 env isolation] tag each collision body with its env group -> cross-env
    # pairs guaranteed excluded regardless of spacing. Bodies loaded env-by-env
    # so ids are contiguous per env. CASE39ME_ISOLATE=0 to disable (A/B).
    if num_envs > 1 and int(os.environ.get("CASE39ME_ISOLATE", "1")):
        n_fem_total = sum(1 for r in eng.get_load_records() if r.body_type == 1)
        m_abd, m_fem = n_abd_total // num_envs, n_fem_total // num_envs
        groups = [cid // m_abd for cid in range(n_abd_total)] + \
                 [f // m_fem for f in range(n_fem_total)]
        eng.native.set_body_groups(groups)
        print(f"[fs] env isolation ON: {n_abd_total} ABD + {n_fem_total} FEM "
              f"-> {num_envs} groups", flush=True)
    eng.finalize()
    print(f"[fs] finalized {num_envs} envs in {time.perf_counter()-t0:.1f}s ({n_abd_total} ABD)", flush=True)

    # [release gate] CASE39ME_BVH_OFFSET=<spacing>: the strict-determinism layout
    # — envs load CO-LOCATED (CASE39ME_SPACING=0) so their local vertices are
    # bit-comparable, while the ENGINE separates them for the broad-phase via
    # d_env_offset (BVH sees offset copies; physics stays on local vertices).
    # This is the v0.8.3 bitwise-trio configuration.
    bvh_off = float(os.environ.get("CASE39ME_BVH_OFFSET", "0"))
    disp_off = None
    if bvh_off > 0.0 and num_envs > 1 and 'groups' in locals():
        # Display-only per-vertex offsets. Bodies are loaded in TWO phases
        # (all-ABD then all-FEM), so never assume env-contiguous records —
        # use the same per-body groups list that set_body_groups() gets
        # (records order == body order).
        recs_all = eng.get_load_records()
        assert len(recs_all) == len(groups), (len(recs_all), len(groups))
        doffs = make_env_offsets(num_envs, bvh_off)
        import numpy as _np
        disp_off = _np.zeros((int(eng.get_vertices().shape[0]), 3), dtype=_np.float64)
        for r, g in zip(recs_all, groups):
            o3 = (float(doffs[g][0, 3]), 0.0, float(doffs[g][2, 3]))
            a, b = int(r.vertex_offset), int(r.vertex_offset) + int(r.vertex_count)
            disp_off[a:b] = o3
    if bvh_off > 0.0 and num_envs > 1:
        boffs = make_env_offsets(num_envs, bvh_off)
        flat = []
        for o in boffs:
            flat += [float(o[0, 3]), 0.0, float(o[2, 3])]  # up-axis stays 0
        eng.native.set_env_offsets(flat)
        print(f"[fs] BVH env offsets set (spacing={bvh_off}, physics co-located)", flush=True)

    if int(os.environ.get("CASE36_DISABLE_GRAVITY","1")):
        for env in envs:
            for a in env['arm_ids']: eng.native.set_body_apply_gravity(a, False)
            for g in env['grippers']: eng.native.set_body_apply_gravity(g['abd_id'], False)
    for env in envs:
        for g in env['grippers']: eng.native.set_fixed_joint_strength(g['fj_idx'], 1e3)
    eng.native.set_max_revolute_step_per_frame(0.04)
    robot = Robot(eng)
    for i in range(len(robot.prismatic_joints)): eng.native.set_prismatic_strength(i, 15.0)
    ejs = slice_env_joints(robot, num_envs)
    print(f"[fs] {len(robot.revolute_joints)} rev + {len(robot.prismatic_joints)} pri\n", flush=True)

    # [multi-env] CASE39ME_PHASE: drive env e from frame (fr + e*phase) so the envs
    # sit at DIFFERENT points of the trajectory simultaneously (heterogeneous
    # difficulty). phase=0 -> all envs same frame (identical envs, default). This is
    # the condition under which per-env line-search (STIFF_PERENV_ALPHA) helps.
    phase = int(os.environ.get("CASE39ME_PHASE", "0"))
    L = len(actions)
    if phase:
        print(f"[fs] CASE39ME_PHASE={phase} -> envs offset across trajectory "
              f"(env e at frame fr+{phase}*e); heterogeneous difficulty", flush=True)

    if int(os.environ.get("CASE39ME_HEADLESS","0")):
        f0 = int(os.environ.get("CASE39_FRAME_START","0"))
        f1 = min(int(os.environ.get("CASE39_FRAME_END", str(len(actions)))), len(actions))
        cloth_ranges = [(env['cloth_rec'].vertex_offset, env['cloth_rec'].vertex_count) for env in envs]
        ms = []
        for fr in range(f0, f1):
            for e, ej in enumerate(ejs):
                apply_frame(robot, ej, actions[(fr + e*phase) % L], close_r)
            t = time.perf_counter(); eng.step(); ms.append((time.perf_counter()-t)*1000.0)
            if fr % 20 == 0:
                v = eng.get_vertices()
                cz = [float(v[o:o+c,1].mean()) for (o,c) in cloth_ranges]
                print(f"[fs-hl] frame {fr:4d} step={ms[-1]:6.0f}ms cloth_y/env={['%+.3f'%z for z in cz]}", flush=True)
        mm = float(np.mean(ms))
        print(f"\n[fs-hl] {num_envs} envs, {len(ms)} frames: mean {mm:.1f}ms ({1000.0/mm:.2f} fps) = {mm/num_envs:.1f} ms/env", flush=True)
        # [release gate] final-state vertex dump for the strict bitwise trio
        # (cross-env / batch-invariance / run-to-run). Engine-local vertices are
        # co-located across envs in strict layout, so slices compare bitwise.
        dump = os.environ.get("CASE39ME_DUMP_VERTS")
        if dump:
            np.save(dump, np.asarray(eng.get_vertices()))
            recs = np.asarray([(r.vertex_offset, r.vertex_count, r.body_type)
                               for r in eng.get_load_records()], dtype=np.int64)
            np.save(dump.replace(".npy", "_recs.npy"), recs)
            print(f"[fs-hl] verts dumped -> {dump} (+recs)", flush=True)
        return

    import polyscope as ps, polyscope.imgui as psim
    _disp = (lambda x: x + disp_off) if disp_off is not None else (lambda x: x)
    v = _disp(eng.get_vertices()); fa = eng.get_surface_faces()
    ps.init(); ps.set_up_dir("y_up"); ps.set_ground_plane_mode("shadow_only")
    st = dict(idx=0, run=False, ms=0., fps=0., mesh=ps.register_surface_mesh("scene", v, fa, color=(0.6,0.7,0.8)), v=v, f=fa)
    def cb():
        if st['run']:
            if psim.Button("Pause"): st['run']=False
        else:
            if psim.Button("Start" if st['idx']==0 else "Resume"): st['run']=True
        psim.SameLine()
        if psim.Button("Reset"): st['idx']=0; st['run']=False; st['fps']=0.
        eqms = st['ms']/num_envs if num_envs else st['ms']
        psim.Text(f"frame {st['idx']}/{len(actions)}   envs {num_envs}")
        psim.Text(f"step {st['ms']:6.1f} ms    FPS {st['fps']:5.2f}")
        psim.Text(f"per-env-equiv {eqms:6.1f} ms  ({(1000.0/eqms) if eqms>0 else 0:5.1f} env-steps/s)")
        if not st['run'] or st['idx']>=len(actions): return
        for e, ej in enumerate(ejs):
            apply_frame(robot, ej, actions[(st['idx'] + e*phase) % L], close_r)
        t=time.perf_counter(); eng.step(); st['ms']=(time.perf_counter()-t)*1000.0
        inst = 1000.0/st['ms'] if st['ms']>0 else 0.0
        st['fps'] = inst if st['fps']==0. else 0.9*st['fps'] + 0.1*inst
        v=_disp(eng.get_vertices()); fa=eng.get_surface_faces()
        if v.shape[0]!=st['v'].shape[0] or fa.shape!=st['f'].shape:
            st['mesh']=ps.register_surface_mesh("scene", v, fa, color=(0.6,0.7,0.8)); st['v'],st['f']=v,fa
        else: st['mesh'].update_vertex_positions(v)
        st['idx']+=1
    ps.set_user_callback(cb); ps.show()


if __name__ == "__main__":
    main()
