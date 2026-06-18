#!/usr/bin/env python3
"""Multi-environment cloth-fold trajectory replay (case_39 scene × N envs).

This is the multi-environment sibling of ``replay_case39.py``.  It tiles N
copies of the full case_39 scene (dual-panda + hybrid grippers + cup + shirt)
on a horizontal grid and replays the SAME recorded qpos trajectory in every
environment at once, visualizing all of them in a single polyscope window.

How multi-env works in StiffGIPC (important — read this)
--------------------------------------------------------
StiffGIPC is NOT a batched/vectorized solver like libuipc.  There is exactly
ONE world, ONE global BVH and ONE Newton/PCG solver.  "Multi-env" here means:
*build N spatially-separated copies of the scene into the same world, then step
them all in a single ``eng.step()``*.  Consequences you must keep in mind:

  * Envs are separated by a horizontal grid offset (X/Z only, shared ground at
    ``ground_offset``).  Because they are far apart, broad-phase never generates
    cross-env contacts, so NO cross-env collision exclusions are needed — only
    the per-env-internal exclusions are replicated.
  * There is no per-env solver isolation.  If ONE env's contact configuration
    pushes the *global* Newton loop to its iteration cap, EVERY env's frame
    stalls together.  This is the multi-env analogue of the single-scene
    cap-hit problem — watch ``newton_iter_cap`` and per-frame ms, not just fps.
  * GPU memory is the binding limit (measured, RTX 4090 24GB), and after the
    engine fixes it scales ~LINEARLY with N: ~6GB base + ~0.8GB/env.
    MEASURED MAX (through the full grasp trajectory): N=20 @ 22.4GB (369ms/step,
    2.71fps); N=16 @ 18.9GB (278ms, 3.6fps). N=22 is the settling edge (23.7GB);
    N>=24 OOMs. RL throughput ~55 env-steps/sec and SATURATED (serial merged
    solve — multi-env amortizes fixed overhead, 14/s@N=1 -> ~55/s@N>=16, then
    flat; no batch speedup). Defaults RIGHT-SIZED from STIFF_CP_STATS:
    CCD/contact/triplet run ~10-20% used even at N=20 grasp (~5x headroom).
    Note: geometry INSTANCING (libuipc-style) would NOT help — uipc doesn't
    instance the shirt either.
  * Knobs (CASE39ME_BUFF_SCALE / LSYS_SCALE / TRIPLET_MARGIN / ABS_DHAT / SPACING)
    trade memory vs robustness. ABS_DHAT is the key one: it pins contact
    thickness to the single-env value (~2.4mm) so contact does NOT inflate with
    scene size — without it contact grows super-linearly and N>=6 crashes.

Note: in-process multi-env cloth needed engine fixes (see docs/internal/
FIX_multienv_cloth_finalize_crash.txt): load_triMesh offset/bending bug
(crash >2 cloths), configurable triplet margin, and absolute_dhat. This example
needs that fixed build.

Usage:
    # Default 4 envs, bundled trajectory, GUI
    python examples/replay_case39_multienv.py

    # 16 envs, full grasp, ~19GB on 24GB (RL-scale)
    CASE39ME_NUM_ENVS=16 CASE39ME_HEADLESS=1 python examples/replay_case39_multienv.py

    # custom trajectory + spacing
    CASE39ME_SPACING=5.0 python examples/replay_case39_multienv.py /path/qpos.h5

    # headless smoke (build + a few steps, prints per-env cup-y + timing)
    CASE39ME_HEADLESS=1 CASE39ME_FRAME_END=120 python examples/replay_case39_multienv.py
"""
import sys, os, math, time
from pathlib import Path

_ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation

from stiff_physics import Engine, Config
from stiff_physics.robot import Robot

URDF_PATH   = _ASSETS_DIR + "sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_softgripper.urdf"
RIGID_MSH   = _ASSETS_DIR + "sim_data/hybrid_d/STRATEGY_F_rigid.msh"
RIGID_REMAP = _ASSETS_DIR + "sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"
UNIFIED_NPZ = _ASSETS_DIR + "sim_data/hybrid_d/STRATEGY_F_unified.npz"
CUP_MSH     = _ASSETS_DIR + "sim_data/tetmesh/softgriper_cup.msh"
SHIRT_OBJ   = _ASSETS_DIR + "triMesh/shirt_6436v.obj"

ARM_SCALE = 1.0
FINGER_LABELS = [
    'left_arm_leftfinger',  'left_arm_rightfinger',
    'right_arm_leftfinger', 'right_arm_rightfinger',
]


def make_arm_tf(scale: float) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix()
    tf[1, 3] = -3.0
    return tf


def make_env_offsets(num_envs: int, spacing: float) -> list[np.ndarray]:
    """N pure-translation transforms on a near-square X/Z grid, centered."""
    cols = int(math.ceil(math.sqrt(num_envs)))
    rows = int(math.ceil(num_envs / cols))
    offsets = []
    for e in range(num_envs):
        r, c = divmod(e, cols)
        off = np.eye(4)
        off[0, 3] = (c - (cols - 1) / 2.0) * spacing   # X
        off[2, 3] = (r - (rows - 1) / 2.0) * spacing   # Z
        offsets.append(off)
    return offsets


def build_env_abd(eng, env_tf, hybrid, cup_T0, ge):
    """Phase A: load this env's ABD bodies (URDF + gripper rigid + cup).

    The engine REQUIRES all ABD bodies to be loaded before any FEM body
    ("ABD mesh shouldn't be loaded after FEM mesh"), so ABD loading for ALL
    envs must happen before any FEM loading.  Finger transforms are read here,
    right after this env's URDF load, because ``urdf_link_transforms`` is a
    name-keyed map that the next env's identically-named URDF would overwrite.
    """
    collision_origin = hybrid['collision_origin']

    arm_tf = env_tf @ make_arm_tf(ARM_SCALE)
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})

    env_abd = [r for r in eng.get_load_records() if r.body_type == 0]
    finger_recs = {}
    for r in reversed(env_abd):
        if r.label in FINGER_LABELS and r.label not in finger_recs:
            finger_recs[r.label] = r
        if len(finger_recs) == 4:
            break
    if len(finger_recs) != 4:
        raise RuntimeError(f"expected 4 finger ABDs in env, got {len(finger_recs)}")

    arm_ids = [r.body_offset for r in env_abd if r.body_offset >= ge['abd_cursor']]
    for bid in arm_ids:
        eng.add_ground_collision_skip(bid)

    grippers = []
    for label in FINGER_LABELS:
        finger_T = eng.native.get_urdf_link_transform(label)   # reflects THIS env
        gripper_T = finger_T @ collision_origin
        eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
                      transform=gripper_T, young_modulus=1e8, boundary_type="Free")
        rigid_rec = eng.get_load_records()[-1]
        grippers.append(dict(
            label=label, finger_id=finger_recs[label].body_offset,
            gripper_T=gripper_T,
            abd_id=rigid_rec.body_offset, abd_v_off=rigid_rec.vertex_offset,
        ))

    cup_T = env_tf @ cup_T0
    eng.load_mesh(CUP_MSH, dimensions=3, body_type="ABD",
                  transform=cup_T, young_modulus=1e8, boundary_type="Free")
    cup_rec = eng.get_load_records()[-1]

    ge['abd_cursor'] = max(r.body_offset for r in eng.get_load_records()
                           if r.body_type == 0) + 1
    return dict(arm_ids=arm_ids, grippers=grippers,
                cup_id=cup_rec.body_offset, cup_rec=cup_rec)


def build_env_fem(eng, env, env_tf, hybrid, shirt_T0, ge):
    """Phase B: load this env's FEM bodies (gripper softpads + shirt) and add
    the per-env stitch springs + fixed joints (both can be added pre-finalize
    once the referenced ABD + FEM bodies exist)."""
    rigid_v_idx  = hybrid['rigid_v_idx']
    n_rigid_v    = hybrid['n_rigid_v']
    hybrid_verts = hybrid['verts']
    hybrid_tets  = hybrid['tets']
    grippers = env['grippers']

    for g in grippers:
        eng.native.load_mesh_from_data(
            hybrid_verts, hybrid_tets, 4, 3, 1, g['gripper_T'], ge['fem_young'], 0)
        fem_rec = eng.get_load_records()[-1]
        g['fem_rec'] = fem_rec
        g['fem_v_off'] = fem_rec.vertex_offset
        g['fem_body_offset'] = fem_rec.body_offset

    shirt_T = env_tf @ shirt_T0
    eng.load_mesh(SHIRT_OBJ, dimensions=2, body_type="FEM",
                  transform=shirt_T, young_modulus=ge['shirt_young'])
    env['shirt_rec'] = eng.get_load_records()[-1]

    _no_stitch = int(os.environ.get("CASE39ME_NO_STITCH", "0"))
    _no_fj     = int(os.environ.get("CASE39ME_NO_FJ", "0"))
    for g in grippers:
        if not _no_stitch:
            for i in range(n_rigid_v):
                eng.add_stitch_spring(
                    g['fem_v_off'] + int(rigid_v_idx[i]),
                    g['abd_v_off'] + i,
                    g['abd_id'],
                    rest_offset_world=(0.0, 0.0, 0.0))
        if not _no_fj:
            anchor = g['gripper_T'][:3, 3]
            g['fj_idx'] = eng.native.add_fixed_joint(
                parent_body=g['finger_id'], child_body=g['abd_id'],
                world_anchor=anchor,
                world_normal=np.array([1.0, 0.0, 0.0]),
                world_bitangent=np.array([0.0, 0.0, 1.0]),
            )


def exclusions_for_env(eng, env, n_abd_total):
    """Replicate the case_39 per-env-internal collision exclusions."""
    grippers = env['grippers']
    arm_ids  = env['arm_ids']
    cup_id   = env['cup_id']
    for g in grippers:
        g['fem_global_id'] = n_abd_total + g['fem_body_offset']
    shirt_global_id = n_abd_total + env['shirt_rec'].body_offset

    for g in grippers:
        eng.native.add_collision_exclusion(g['abd_id'], g['fem_global_id'])
        eng.native.add_collision_exclusion(g['fem_global_id'], g['finger_id'])
        eng.native.add_collision_exclusion(g['abd_id'], g['finger_id'])
        for arm_id in arm_ids:
            if arm_id == g['finger_id']:
                continue
            eng.native.add_collision_exclusion(g['abd_id'], arm_id)
            eng.native.add_collision_exclusion(g['fem_global_id'], arm_id)

    finger_offsets = {g['finger_id'] for g in grippers}
    for arm_id in arm_ids:
        if arm_id in finger_offsets:
            continue
        eng.native.add_collision_exclusion(arm_id, cup_id)
        eng.native.add_collision_exclusion(arm_id, shirt_global_id)

    for g in grippers:
        eng.add_ground_collision_skip(g['fem_global_id'])
        eng.add_ground_collision_skip(g['abd_id'])

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

    env['shirt_global_id'] = shirt_global_id


def slice_env_joints(robot, num_envs):
    """Split the flat joint lists into per-env index blocks.

    Every env loads the identical URDF, so joints arrive in equal-size blocks
    in load order. Within each block we re-apply the case_39 name filter.
    """
    n_rev = len(robot.revolute_joints)
    n_pri = len(robot.prismatic_joints)
    assert n_rev % num_envs == 0 and n_pri % num_envs == 0, \
        f"joint counts {n_rev}rev/{n_pri}pri not divisible by {num_envs} envs"
    rpe, ppe = n_rev // num_envs, n_pri // num_envs
    per_env = []
    for e in range(num_envs):
        rev_block = range(e * rpe, (e + 1) * rpe)
        pri_block = range(e * ppe, (e + 1) * ppe)
        left_rev  = [i for i in rev_block
                     if robot.revolute_joints[i].name.startswith('left_arm_joint')]
        right_rev = [i for i in rev_block
                     if robot.revolute_joints[i].name.startswith('right_arm_joint')]
        left_pri  = [i for i in pri_block
                     if robot.prismatic_joints[i].name.startswith('left_arm')]
        right_pri = [i for i in pri_block
                     if robot.prismatic_joints[i].name.startswith('right_arm')]
        assert len(left_rev) == 7 and len(right_rev) == 7, \
            f"env {e}: expected 7+7 revolute, got {len(left_rev)}+{len(right_rev)}"
        per_env.append(dict(left_rev=left_rev, right_rev=right_rev,
                            left_pri=left_pri, right_pri=right_pri))
    return per_env


def apply_frame(robot, ej, raw, close_r):
    """Drive one env's joints from a single qpos frame (16-vector)."""
    q_left, q_right = raw[0:7], raw[7:14]
    grip_L = float(raw[14]) if len(raw) > 14 else 0.0
    grip_R = float(raw[15]) if len(raw) > 15 else 0.0
    for i, rev_idx in enumerate(ej['left_rev']):
        robot.set_revolute_position(rev_idx, float(q_left[i]), degree=False)
    for i, rev_idx in enumerate(ej['right_rev']):
        robot.set_revolute_position(rev_idx, float(q_right[i]), degree=False)
    for side, grip, pris in (('L', grip_L, ej['left_pri']),
                             ('R', grip_R, ej['right_pri'])):
        if not pris:
            continue
        lo = robot.prismatic_joints[pris[0]].lower_limit
        hi = robot.prismatic_joints[pris[0]].upper_limit
        gp = hi if grip >= 0 else (lo + close_r * (hi - lo))
        for pi in pris:
            robot.set_prismatic_position(pi, gp, millimeters=False)


def main():
    default_qpos = _ASSETS_DIR + "trajectories/qpos_case39.h5"
    qpos_path = sys.argv[1] if len(sys.argv) > 1 else default_qpos

    num_envs = int(os.environ.get("CASE39ME_NUM_ENVS", "4"))
    spacing  = float(os.environ.get("CASE39ME_SPACING", "4.0"))
    close_r  = float(os.environ.get("CASE39_CLOSE_RATIO", "0.5"))

    # ---- Multi-env status (fixed 2026-06-04) --------------------------------
    # In-process multi-env cloth used to crash: finalize() heap-corrupted once
    # the world held >2 FEM cloth bodies.  Root cause was two bugs in
    # StiffGIPC/load_mesh.cpp::load_triMesh (a) the running vertex base used
    # `vertexOffset += vertexNum` instead of `= vertexNum` (every other loader
    # used `=`), so the 3rd+ cloth's triangle indices overshot vertexNum and
    # wrote OOB in getVertNeighbors(); (b) bending edges were re-extracted over
    # ALL accumulated triangles each load.  Both are fixed in the engine now, so
    # N copies of the full cloth-fold scene step correctly in ONE engine.
    #
    # Remaining limit is GPU MEMORY, not correctness: this is a single merged
    # world (one global BVH + one Newton/PCG), so buffers scale with total scene
    # size.  On a 24GB GPU the full dual-arm+gripper+cup+shirt scene fits ~6 envs
    # (N>=9 OOMs).  Lower CASE39ME_BUFF_SCALE if you OOM; raise it if you see
    # CCD-pair-overflow.  And note: no per-env solver isolation — if ONE env's
    # contact pushes the global Newton loop to its cap, ALL envs' frame stalls.
    if num_envs > 1:
        print(f"[me] NOTE: {num_envs} envs share ONE merged world (single global "
              "BVH + Newton/PCG). With the absolute-dHat fix (contact thickness no "
              "longer inflates with scene size) + right-sized buffers, contact and "
              "memory scale ~LINEARLY: per-env ~1.7GB. N=8 runs the full grasp at "
              "~19GB on 24GB; ceiling ~10 (N>=12 OOMs). One env hitting the Newton "
              "cap still stalls all (no per-env solver isolation).", flush=True)

    import h5py
    with h5py.File(qpos_path, "r") as f:
        qpos_all = f["qpos"][:]
    print(f"[me] Loaded {len(qpos_all)} qpos frames {qpos_all.shape} from {qpos_path}")
    print(f"[me] num_envs={num_envs} spacing={spacing}m", flush=True)

    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100,
        soft_motion_rate=float(os.environ.get("CASE36_SOFT_RATE", "1e4")),
        poisson_rate=0.49,
        friction_rate=float(os.environ.get("CASE39_FRICTION", "0.8")),
        relative_dhat=float(os.environ.get("CASE39ME_REL_DHAT", "1e-3")),
        joint_strength_ratio=float(os.environ.get("CASE38_JOINT_K", "100")),
        revolute_driving_strength_ratio=float(os.environ.get("CASE36_PD_K", "100")),
        prismatic_strength_ratio=float(os.environ.get("CASE39_PRISMATIC_CONSTRAINT_K", "2000")),
        semi_implicit_enabled=bool(int(os.environ.get("CASE39_SEMI", "0"))),
        semi_implicit_beta_tol=5e-2, semi_implicit_min_iter=1,
        newton_tol=float(os.environ.get("CASE39_NEWTON_TOL", "5e-2")),
        newton_iter_cap=int(os.environ.get("CASE39_NEWTON_CAP", "50")),
        preconditioner_type=int(os.environ.get("CASE39_PRECOND", "1")),
        ground_offset=-1.67,
        assets_dir=_ASSETS_DIR,
    )
    # collision-detection CCD pair buffer scale. replay_case39.py uses 64 (huge
    # single-env headroom); multi-env multiplies the global buffers by env count,
    # so 64 OOMs a 24GB GPU at N>=3. Tune down for multi-env (8-16 is plenty when
    # envs are spatially separated and never cross-contact).
    # Right-sized buffer scales (measured via STIFF_CP_STATS — see docs/internal/
    # FIX_multienv_cloth_finalize_crash.txt). Once absolute_dhat makes contact
    # scale linearly, the CCD/contact/triplet buffers run ~10-20% used at these
    # values even at N=16 grasp (~5× headroom). Lower for more envs / raise if a
    # heavier trajectory overflows.
    cfg._cfg.collision_detection_buff_scale = float(
        os.environ.get("CASE39ME_BUFF_SCALE", "1.5"))
    # Global linear-system triplet buffer. The base size already scales with the
    # (merged) scene, but when contact ramps up — especially the cup-grasp window
    # — the per-frame collision triplet count needs headroom or the DtoH copy in
    # GlobalLinearSystem::convert overruns its device buffer and segfaults
    # (observed ~frame 29 / grasp at default 1.0). 3-4 covers the cloth-fold
    # trajectory; too high OOMs (base is already N-scaled). Tune per N/GPU.
    cfg._cfg.linear_system_buff_scale = float(
        os.environ.get("CASE39ME_LSYS_SCALE", "1.5"))
    # Internal Hessian-triplet margin (hardcoded 32 in the engine historically).
    # Measured ~0 chain-rule extension for this scene, and the internal triplets
    # are written exactly 1x/frame, so 32x is the dominant per-env over-reserve
    # and the wall for scaling envs (at N=8 it alone wants ~16GB). 4x keeps a
    # safe margin here; raise toward 32 only for Strategy-D rigid-heavy hybrids.
    cfg._cfg.triplet_internal_margin = float(
        os.environ.get("CASE39ME_TRIPLET_MARGIN", "3.0"))
    # ROOT-CAUSE FIX for super-linear multi-env contact: dHat is normally derived
    # from the merged-scene bbox diagonal, which grows with env count/spacing, so
    # the contact thickness balloons (2.4mm@N=1 -> 8.7mm@N=4) and contact pairs
    # grow super-linearly. Pin it to the single-env value (~2.4mm for this scene,
    # measured via the [dhat] log at N=1) so contact — and thus memory — scales
    # LINEARLY with N. Set 0 to fall back to the legacy bbox behavior.
    cfg._cfg.absolute_dhat = float(os.environ.get("CASE39ME_ABS_DHAT", "0.00239"))
    eng = Engine(cfg)

    # shared hybrid / mesh data (loaded once, reused per env)
    _co_rpy = np.array([-1.57079632679, 0.20245819348, -1.57079632679])
    _co_xyz = np.array([-0.0165, 0.0165, 0.12773331296])
    collision_origin = np.eye(4)
    collision_origin[:3, :3] = Rotation.from_euler('xyz', _co_rpy).as_matrix()
    collision_origin[:3, 3] = _co_xyz
    rigid_remap = np.load(RIGID_REMAP, allow_pickle=True)
    rigid_v_idx = rigid_remap['rigid_v_idx']
    hybrid_data = np.load(UNIFIED_NPZ)
    hybrid = dict(
        collision_origin=collision_origin,
        rigid_v_idx=rigid_v_idx, n_rigid_v=len(rigid_v_idx),
        verts=np.ascontiguousarray(hybrid_data['vertices'], dtype=np.float64),
        tets=np.ascontiguousarray(hybrid_data['tets'], dtype=np.int32),
    )
    ge = dict(
        fem_young=float(os.environ.get("CASE36_FEM_YOUNG", "1e7")),
        shirt_young=float(os.environ.get("CASE38_SHIRT_YOUNG", "1e2")),
        abd_cursor=0,
    )

    cup_scale = float(os.environ.get("CASE39_CUP_SCALE", "0.8"))
    cup_xyz   = np.array([float(s) for s in
        os.environ.get("CASE39_CUP_XYZ", "0.67,-0.2,-0.4").split(",")])
    cup_T0 = np.eye(4); cup_T0[:3, :3] *= cup_scale; cup_T0[:3, 3] = cup_xyz
    shirt_scale = float(os.environ.get("CASE39_SHIRT_SCALE", "1.0"))
    shirt_xyz   = np.array([float(s) for s in
        os.environ.get("CASE39_SHIRT_XYZ", "0.67,0.00,0.00").split(",")])
    shirt_T0 = np.eye(4); shirt_T0[:3, :3] *= shirt_scale; shirt_T0[:3, 3] = shirt_xyz

    # ---- build N envs (ABD for all envs first, then FEM for all envs) ----
    print(f"\n[me] === building {num_envs} envs ===", flush=True)
    env_offsets = make_env_offsets(num_envs, spacing)
    t_build = time.perf_counter()
    # Phase A: all ABD bodies (engine forbids loading ABD after any FEM)
    envs = [build_env_abd(eng, off, hybrid, cup_T0, ge) for off in env_offsets]
    for e, env in enumerate(envs):
        print(f"[me] env {e} ABD: {len(env['arm_ids'])} arm bodies, cup={env['cup_id']}",
              flush=True)
    # Phase B: all FEM bodies + per-env stitch springs / fixed joints
    for env, off in zip(envs, env_offsets):
        build_env_fem(eng, env, off, hybrid, shirt_T0, ge)

    n_abd_total = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    if not int(os.environ.get("CASE39ME_NO_EXCL", "0")):
        for env in envs:
            exclusions_for_env(eng, env, n_abd_total)
    else:
        for env in envs:
            for g in env['grippers']:
                g['fem_global_id'] = n_abd_total + g['fem_body_offset']
            env['shirt_global_id'] = n_abd_total + env['shirt_rec'].body_offset

    # [P1 env isolation] Tag each collision body with its env group so that
    # cross-env pairs are guaranteed excluded (independent of spacing). Bodies
    # are loaded env-by-env, so ABD ids [0,n_abd) and FEM ids [n_abd,n_abd+n_fem)
    # are contiguous per env -> group = id // (count_per_env). Default ON for N>1;
    # CASE39ME_ISOLATE=0 falls back to spacing-only (for A/B testing).
    if num_envs > 1 and int(os.environ.get("CASE39ME_ISOLATE", "1")):
        n_fem_total = sum(1 for r in eng.get_load_records() if r.body_type == 1)
        m_abd, m_fem = n_abd_total // num_envs, n_fem_total // num_envs
        groups = [cid // m_abd for cid in range(n_abd_total)] + \
                 [f // m_fem for f in range(n_fem_total)]
        eng.native.set_body_groups(groups)
        print(f"[me] env isolation ON: {n_abd_total} ABD + {n_fem_total} FEM bodies "
              f"-> {num_envs} groups ({m_abd} ABD + {m_fem} FEM each)", flush=True)

    fgrav = int(os.environ.get("CASE36_DISABLE_GRAVITY", "1"))
    eng.finalize()
    print(f"[me] finalized {num_envs} envs in {time.perf_counter()-t_build:.1f}s "
          f"({n_abd_total} ABD bodies total)", flush=True)

    # ---- post-finalize per-env setup ----
    if fgrav:
        for env in envs:
            for arm_id in env['arm_ids']:
                eng.native.set_body_apply_gravity(arm_id, False)
            for g in env['grippers']:
                eng.native.set_body_apply_gravity(g['abd_id'], False)
    fj_kappa = float(os.environ.get("CASE36_FJ_KAPPA", "1e3"))
    for env in envs:
        for g in env['grippers']:
            if 'fj_idx' in g:
                eng.native.set_fixed_joint_strength(g['fj_idx'], fj_kappa)
    eng.native.set_max_revolute_step_per_frame(
        float(os.environ.get("CASE36_MAX_RAD_PER_FRAME", "0.04")))

    robot = Robot(eng)
    prismatic_mult = float(os.environ.get("CASE36_PRISMATIC_K", "15"))
    for i in range(len(robot.prismatic_joints)):
        eng.native.set_prismatic_strength(i, prismatic_mult)
    env_joints = slice_env_joints(robot, num_envs)
    print(f"[me] {len(robot.revolute_joints)} revolute + "
          f"{len(robot.prismatic_joints)} prismatic across {num_envs} envs\n", flush=True)

    # ============ HEADLESS smoke / timing ============
    if int(os.environ.get("CASE39ME_HEADLESS", "0")):
        f_start = int(os.environ.get("CASE39_FRAME_START", "0"))
        f_end = min(int(os.environ.get("CASE39_FRAME_END", str(len(qpos_all)))),
                    len(qpos_all))
        cup_ranges = [(env['cup_rec'].vertex_offset, env['cup_rec'].vertex_count)
                      for env in envs]
        ms_log = []
        # [P3a] CASE39ME_PHASE>0 makes envs HETEROGENEOUS: env e is driven from
        # frame (fr + e*PHASE), so envs are at different trajectory points (=
        # different convergence difficulty) — the diverse-RL-env regime where
        # per-env early-exit pays off. 0 = identical envs (default).
        phase = int(os.environ.get("CASE39ME_PHASE", "0"))
        for fr in range(f_start, f_end):
            raw = qpos_all[fr]
            for e, ej in enumerate(env_joints):
                r = qpos_all[(fr + e * phase) % len(qpos_all)] if phase else raw
                apply_frame(robot, ej, r, close_r)
            t0 = time.perf_counter()
            eng.step()
            ms_log.append((time.perf_counter() - t0) * 1000.0)
            if fr % 20 == 0:
                v = eng.get_vertices()
                cups = [float(v[o:o+c, 1].mean()) for (o, c) in cup_ranges]
                print(f"[me-hl] frame {fr:4d}  step={ms_log[-1]:6.0f}ms  "
                      f"cup_y/env={['%+.3f'%y for y in cups]}", flush=True)
        mean_ms = float(np.mean(ms_log))
        print(f"\n[me-hl] === {num_envs} envs, {len(ms_log)} frames ===")
        print(f"[me-hl] mean step {mean_ms:.1f} ms  ({1000.0/mean_ms:.2f} fps)  "
              f"= {mean_ms/num_envs:.1f} ms/env-equiv", flush=True)
        return

    # ============ GUI replay ============
    import polyscope as ps
    import polyscope.imgui as psim

    verts = eng.get_vertices()
    faces = eng.get_surface_faces()
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_program_name(f"replay_case39 × {num_envs} envs")

    state = dict(idx=0, running=False, last_ms=0.0,
                 mesh=ps.register_surface_mesh("scene", verts, faces,
                                               color=(0.6, 0.7, 0.8)),
                 verts=verts, faces=faces)

    # color each env's gripper FEM pads (red=stitched region, blue=bulk) so the
    # N copies are visually distinguishable from the gray arm/cup/shirt.
    base_color = np.array([0.6, 0.7, 0.8])
    red, blue = np.array([0.85, 0.25, 0.25]), np.array([0.25, 0.45, 0.85])
    colors = np.tile(base_color, (verts.shape[0], 1))
    vr = hybrid_data['vertex_region']
    for env in envs:
        for g in env['grippers']:
            fo, fc = g['fem_v_off'], g['fem_rec'].vertex_count
            colors[fo:fo+fc] = np.where((vr[:fc] == 1)[:, None], red, blue)
    state['mesh'].add_color_quantity("region (red=stitched, blue=bulk)",
                                     colors, defined_on='vertices', enabled=True)

    def callback():
        if state['running']:
            if psim.Button("Pause"):
                state['running'] = False
        else:
            if psim.Button("Start" if state['idx'] == 0 else "Resume"):
                state['running'] = True
        psim.SameLine()
        if psim.Button("Reset"):
            state['idx'], state['running'] = 0, False
        psim.Text(f"frame {state['idx']:>4d} / {len(qpos_all)}   envs {num_envs}")
        psim.Text(f"step {state['last_ms']:6.1f} ms   "
                  f"FPS {(1000.0/state['last_ms']) if state['last_ms']>0 else 0.0:5.1f}")

        if not state['running'] or state['idx'] >= len(qpos_all):
            return
        raw = qpos_all[state['idx']]
        for ej in env_joints:
            apply_frame(robot, ej, raw, close_r)
        t0 = time.perf_counter()
        eng.step()
        state['last_ms'] = (time.perf_counter() - t0) * 1000.0
        v = eng.get_vertices()
        f = eng.get_surface_faces()
        if v.shape[0] != state['verts'].shape[0] or f.shape != state['faces'].shape:
            state['mesh'] = ps.register_surface_mesh("scene", v, f, color=(0.6, 0.7, 0.8))
            state['verts'], state['faces'] = v, f
        else:
            state['mesh'].update_vertex_positions(v)
        state['idx'] += 1

    ps.set_user_callback(callback)
    ps.show()
    print(f"[me] window closed at frame {state['idx']}/{len(qpos_all)}")


if __name__ == "__main__":
    main()
