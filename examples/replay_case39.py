#!/usr/bin/env python3
"""Replay a pre-recorded qpos trajectory on the full case_39 scene
(dual-panda + hybrid grippers + cup + shirt) and visualize with polyscope.

Defaults are tuned so the cup is actually gripped and lifted:
    CASE39_PRECOND=1     MAS preconditioner on (~2x faster vs MAS-off on hybrid
                         FEM, requires the v0.6.1 stitch-index metis-remap fix)
    CASE39_FRICTION=0.8  high enough for the cup not to slip out
    CASE39_CLOSE_RATIO=0.5  half-close (avoids over-close IPC<->prismatic jitter)

Usage:
    # Use the bundled trajectory (assets/trajectories/qpos_case39.h5)
    python examples/replay_case39.py

    # Use a custom trajectory file
    python examples/replay_case39.py /path/to/my_qpos.h5

    # Headless sweep (no GUI), prints cup-drop / lift-max metrics
    CASE39_HEADLESS=1 python examples/replay_case39.py
"""
import sys, os, math, time, re
from pathlib import Path

_ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation

from stiff_physics import Engine, Config
from stiff_physics.robot import Robot
import polyscope as ps
import polyscope.imgui as psim

URDF_PATH     = _ASSETS_DIR + "sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_softgripper.urdf"
ORIGINAL_URDF = _ASSETS_DIR + "sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_full.urdf"
RIGID_MSH     = _ASSETS_DIR + "sim_data/hybrid_d/STRATEGY_F_rigid.msh"
RIGID_REMAP   = _ASSETS_DIR + "sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"
UNIFIED_NPZ   = _ASSETS_DIR + "sim_data/hybrid_d/STRATEGY_F_unified.npz"
CUP_MSH       = _ASSETS_DIR + "sim_data/tetmesh/softgriper_cup.msh"
SHIRT_OBJ     = _ASSETS_DIR + "triMesh/shirt_6436v.obj"
# Original (pre-merged) part2 / part3 source meshes — used as reference point
# clouds to classify each tet of the unified mesh into part2 vs part3 by
# centroid proximity, so we can assign different Young's modulus per part.
PART2_MSH     = _ASSETS_DIR + "sim_data/tetmesh/softgriper_part2.msh"
PART3_MSH     = _ASSETS_DIR + "sim_data/tetmesh/softgriper_part3.msh"

ARM_SCALE = 1.0
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
    tf[1, 3] = -3.0
    return tf


def _parse_xyz_rpy(s):
    return np.array([float(x) for x in s.split()], dtype=float)


def parse_link_world_tf(urdf_path: str, link_name: str, base_tf: np.ndarray) -> np.ndarray:
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
    default_qpos = _ASSETS_DIR + "trajectories/qpos_case39.h5"
    qpos_path = sys.argv[1] if len(sys.argv) > 1 else default_qpos

    import h5py
    with h5py.File(qpos_path, "r") as f:
        qpos_all = f["qpos"][:]
    print(f"Loaded {len(qpos_all)} qpos frames from {qpos_path}  shape={qpos_all.shape}")

    # --- Physics config (case_39 values) ---
    prismatic_constraint_K = float(os.environ.get("CASE39_PRISMATIC_CONSTRAINT_K", "2000"))
    joint_K        = float(os.environ.get("CASE38_JOINT_K", "100"))
    revolute_K     = float(os.environ.get("CASE36_PD_K", "100"))
    prismatic_mult = float(os.environ.get("CASE36_PRISMATIC_K", "15"))


    # joint_strength_ratio=1e2,
    # revolute_driving_strength_ratio=1e2,
    # prismatic_strength_ratio=2000,
    # semi_implicit_enabled=False, semi_implicit_beta_tol=5e-2,
    # semi_implicit_min_iter=1, newton_tol=5e-2,
    # preconditioner_type=0,
    cfg = Config(
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
        preconditioner_type=int(os.environ.get("CASE39_PRECOND", "1")), ground_offset=-1.67,
        assets_dir=_ASSETS_DIR,
    )
    cfg._cfg.collision_detection_buff_scale = 64.0
    eng = Engine(cfg)
    print("\n[replay_case39] === building case_39 scene ===", flush=True)

    # --- 1. Load URDF ---
    arm_tf = make_arm_tf(ARM_SCALE)
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})
    n_urdf = eng.abd_body_count
    urdf_recs = list(eng.get_load_records())
    finger_recs = {r.label: r for r in urdf_recs if r.body_type == 0
                   and r.label in FINGER_LABELS}
    if len(finger_recs) != 4:
        raise RuntimeError(f"expected 4 finger ABDs, got {len(finger_recs)}")
    print(f"[replay_case39] URDF: {n_urdf} ABD bodies", flush=True)

    for b in range(n_urdf):
        eng.add_ground_collision_skip(b)

    # --- 2. Per-finger hybrid transforms (finger_full mode) ---
    _co_rpy = np.array([-1.57079632679, 0.20245819348, -1.57079632679])
    _co_xyz = np.array([-0.0165, 0.0165, 0.12773331296])
    collision_origin = np.eye(4)
    collision_origin[:3, :3] = Rotation.from_euler('xyz', _co_rpy).as_matrix()
    collision_origin[:3, 3] = _co_xyz

    rigid_remap_data = np.load(RIGID_REMAP, allow_pickle=True)
    rigid_v_idx = rigid_remap_data['rigid_v_idx']
    n_rigid_v = len(rigid_v_idx)
    hybrid_data = np.load(UNIFIED_NPZ)
    hybrid_verts = np.ascontiguousarray(hybrid_data['vertices'], dtype=np.float64)
    hybrid_tets  = np.ascontiguousarray(hybrid_data['tets'], dtype=np.int32)

    soft_T = {}
    for finger_label in FINGER_LABELS:
        finger_T = eng.native.get_urdf_link_transform(finger_label)
        soft_T[finger_label] = finger_T @ collision_origin

    grippers = []

    # Pass 1: hybrid rigid ABDs
    for label in FINGER_LABELS:
        finger_rec = finger_recs[label]
        gripper_T = soft_T[label]
        eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
                      transform=gripper_T, young_modulus=1e8, boundary_type="Free")
        rigid_rec = eng.get_load_records()[-1]
        grippers.append(dict(
            label=label, finger_id=finger_rec.body_offset,
            gripper_T=gripper_T,
            abd_id=rigid_rec.body_offset, abd_v_off=rigid_rec.vertex_offset,
        ))

    # Pass 1.5: cup ABD
    cup_scale = float(os.environ.get("CASE39_CUP_SCALE", "0.8"))
    cup_xyz   = np.array([float(s) for s in
        os.environ.get("CASE39_CUP_XYZ", "0.67,-0.2,-0.4").split(",")])
    cup_T = np.eye(4); cup_T[:3, :3] *= cup_scale; cup_T[:3, 3] = cup_xyz
    eng.load_mesh(CUP_MSH, dimensions=3, body_type="ABD",
                  transform=cup_T, young_modulus=1e8, boundary_type="Free")
    cup_rec = eng.get_load_records()[-1]
    cup_id = cup_rec.body_offset
    print(f"[replay_case39] cup body_id={cup_id} scale={cup_scale} at {cup_xyz}", flush=True)

    # Pass 2: hybrid FEM
    fem_young = float(os.environ.get("CASE36_FEM_YOUNG", "1e7"))
    fj_kappa  = float(os.environ.get("CASE36_FJ_KAPPA",  "1e3"))
    for g in grippers:
        eng.native.load_mesh_from_data(
            hybrid_verts, hybrid_tets, 4, 3, 1, g['gripper_T'], fem_young, 0)
        fem_rec = eng.get_load_records()[-1]
        g['fem_rec'] = fem_rec
        g['fem_v_off'] = fem_rec.vertex_offset
        g['fem_load_idx'] = len(eng.get_load_records()) - 1  # for per-tet young setter

    # Pass 2.6: per-region Young's modulus override.
    # part2blobal (the unified mesh) was built by merging the original part2 +
    # part3 source meshes.  Reverse-classify each unified tet by centroid
    # proximity to the original part2.msh / part3.msh point clouds, then
    # assign different Young's modulus per part.  Default behavior is to set
    # the same young on both (no-op).  Set CASE39_YOUNG_PART2=1e8 to stiffen
    # the contact face (part2) while keeping part3 soft.
    young_part2 = float(os.environ.get("CASE39_YOUNG_PART2", str(fem_young)))
    young_part3 = float(os.environ.get("CASE39_YOUNG_PART3", str(fem_young)))
    _vert_is_part2 = None   # per-vert label for viz (None = use raw vertex_region)
    if young_part2 != young_part3:
        import meshio
        from scipy.spatial import cKDTree
        p2 = meshio.read(PART2_MSH).points  # local coords
        p3 = meshio.read(PART3_MSH).points  # local coords
        kd2, kd3 = cKDTree(p2), cKDTree(p3)
        tet_centroids = hybrid_verts[hybrid_tets].mean(axis=1)  # (n_tets, 3) local
        d2, _ = kd2.query(tet_centroids)
        d3, _ = kd3.query(tet_centroids)
        in_part2 = d2 < d3                                     # bool per tet
        per_tet_young = np.where(in_part2, young_part2, young_part3).astype(np.float64)
        print(f"[per-region young] {int(in_part2.sum())} part2 tets ({young_part2:.0e}), "
              f"{int((~in_part2).sum())} part3 tets ({young_part3:.0e})", flush=True)
        for g in grippers:
            eng.native.set_per_tet_young_for_body(g['fem_load_idx'], per_tet_young.tolist())
        # Per-vert part2/part3 for viz: majority vote over incident tets.
        v_p2 = np.zeros(hybrid_verts.shape[0], dtype=int)
        v_tot = np.zeros(hybrid_verts.shape[0], dtype=int)
        for ti, tv in enumerate(hybrid_tets):
            inc = int(in_part2[ti])
            for vi in tv:
                v_p2[vi] += inc
                v_tot[vi] += 1
        _vert_is_part2 = (v_p2 * 2 > v_tot).astype(float)  # 1.0 = part2 (stiff), 0.0 = part3 (soft)

    # Pass 2.5: shirt
    shirt_scale = float(os.environ.get("CASE39_SHIRT_SCALE", "1.0"))
    shirt_xyz   = np.array([float(s) for s in
        os.environ.get("CASE39_SHIRT_XYZ", "0.67,0.00,0.00").split(",")])
    shirt_T = np.eye(4); shirt_T[:3, :3] *= shirt_scale; shirt_T[:3, 3] = shirt_xyz
    eng.load_mesh(SHIRT_OBJ, dimensions=2, body_type="FEM",
                  transform=shirt_T,
                  young_modulus=float(os.environ.get("CASE38_SHIRT_YOUNG", "1e2")))
    shirt_rec = eng.get_load_records()[-1]
    print(f"[replay_case39] shirt verts={shirt_rec.vertex_count} scale={shirt_scale} at {shirt_xyz}",
          flush=True)

    n_abd_total = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    for g in grippers:
        g['fem_global_id'] = n_abd_total + g['fem_rec'].body_offset

    # Pass 3: stitch springs + fixed joints
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

    # --- 4. Collision exclusions (same as case_39) ---
    arm_ids = [r.body_offset for r in urdf_recs if r.body_type == 0]
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

    n_abd_total_for_shirt = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    shirt_global_id = n_abd_total_for_shirt + shirt_rec.body_offset
    for arm_id in arm_ids:
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

    eng.finalize()

    # --- 5. Post-finalize setup ---
    if int(os.environ.get("CASE36_DISABLE_GRAVITY", "1")):
        for arm_id in arm_ids:
            eng.native.set_body_apply_gravity(arm_id, False)
        for g in grippers:
            eng.native.set_body_apply_gravity(g['abd_id'], False)

    for g in grippers:
        eng.native.set_fixed_joint_strength(g['fj_idx'], fj_kappa)
    eng.native.set_max_revolute_step_per_frame(
        float(os.environ.get("CASE36_MAX_RAD_PER_FRAME", "0.04")))

    robot = Robot(eng)
    for i in range(len(robot.prismatic_joints)):
        eng.native.set_prismatic_strength(i, prismatic_mult)

    # Map joint indices: qpos = [L×7, R×7, grip_L, grip_R]
    left_rev_indices  = [i for i, ji in enumerate(robot.revolute_joints)
                         if ji.name.startswith('left_arm_joint')]
    right_rev_indices = [i for i, ji in enumerate(robot.revolute_joints)
                         if ji.name.startswith('right_arm_joint')]
    left_pris_indices  = [i for i, ji in enumerate(robot.prismatic_joints)
                          if ji.name.startswith('left_arm')]
    right_pris_indices = [i for i, ji in enumerate(robot.prismatic_joints)
                          if ji.name.startswith('right_arm')]

    assert len(left_rev_indices) == 7 and len(right_rev_indices) == 7, \
        f"expected 7+7 revolute joints, got {len(left_rev_indices)}+{len(right_rev_indices)}"

    print(f"[replay_case39] {len(robot.revolute_joints)} revolute, "
          f"{len(robot.prismatic_joints)} prismatic", flush=True)
    print(f"[replay_case39] finalized — starting replay of {len(qpos_all)} frames\n", flush=True)

    # ============ HEADLESS grip sweep mode ============
    # CASE39_HEADLESS=1 -> run tight loop, log cup-y vs gripper-y per frame,
    # print a single-number "max drop" metric for the grasp window.  Lets us
    # factor-sweep (friction, close_ratio, fem_young, ...) without GUI.
    if int(os.environ.get("CASE39_HEADLESS", "0")):
        cup_off, cup_cnt = cup_rec.vertex_offset, cup_rec.vertex_count
        grp_ranges = [(g['fem_rec'].vertex_offset, g['fem_rec'].vertex_count)
                      for g in grippers]
        cup_y_log, grp_y_log, ms_log = [], [], []
        _close_r = float(os.environ.get("CASE39_CLOSE_RATIO", "0.5"))
        _frame_start = int(os.environ.get("CASE39_FRAME_START", "0"))
        _frame_end = int(os.environ.get("CASE39_FRAME_END", str(len(qpos_all))))
        _frame_end = min(_frame_end, len(qpos_all))
        for f in range(_frame_start, _frame_end):
            raw = qpos_all[f]
            for i, rev_idx in enumerate(left_rev_indices):
                robot.set_revolute_position(rev_idx, float(raw[i]), degree=False)
            for i, rev_idx in enumerate(right_rev_indices):
                robot.set_revolute_position(rev_idx, float(raw[7 + i]), degree=False)
            grip_L = float(raw[14]) if len(raw) > 14 else 0.0
            grip_R = float(raw[15]) if len(raw) > 15 else 0.0
            if left_pris_indices:
                lo = robot.prismatic_joints[left_pris_indices[0]].lower_limit
                hi = robot.prismatic_joints[left_pris_indices[0]].upper_limit
                gp = hi if grip_L >= 0 else (lo + _close_r * (hi - lo))
                for pi in left_pris_indices:
                    robot.set_prismatic_position(pi, gp, millimeters=False)
            if right_pris_indices:
                lo = robot.prismatic_joints[right_pris_indices[0]].lower_limit
                hi = robot.prismatic_joints[right_pris_indices[0]].upper_limit
                gp = hi if grip_R >= 0 else (lo + _close_r * (hi - lo))
                for pi in right_pris_indices:
                    robot.set_prismatic_position(pi, gp, millimeters=False)
            t0 = time.perf_counter()
            eng.step()
            ms_log.append((time.perf_counter() - t0) * 1000.0)
            v = eng.get_vertices()
            cup_y_log.append(float(v[cup_off:cup_off + cup_cnt, 1].mean()))
            gs = 0.0
            for (off, cnt) in grp_ranges:
                gs += float(v[off:off + cnt, 1].mean())
            grp_y_log.append(gs / len(grp_ranges))
            if f % 100 == 0:
                print(f"[hl] frame {f:4d}: cup_y={cup_y_log[-1]:+.4f} grp_y={grp_y_log[-1]:+.4f} rel={cup_y_log[-1]-grp_y_log[-1]:+.4f} step={ms_log[-1]:.0f}ms",
                      flush=True)
        cup_a = np.array(cup_y_log)
        grp_a = np.array(grp_y_log)
        rel   = cup_a - grp_a
        mean_ms = float(np.mean(ms_log))
        # Map full-trajectory grasp window [451, 960) into the actual run range.
        # If we ran a sub-range (e.g. for NCU profiling), only print mean_ms.
        abs_s, abs_e = 451, 960
        run_lo, run_hi = _frame_start, _frame_end
        if run_hi <= abs_s or run_lo >= abs_e:
            print(f"\n[hl] (frame range {run_lo}-{run_hi} outside grasp window — skipping grip summary)", flush=True)
            print(f"[hl] mean step: {mean_ms:.1f} ms ({1000.0/mean_ms:.1f} fps)", flush=True)
            return
        s = max(abs_s, run_lo) - run_lo
        e = min(abs_e, run_hi) - run_lo
        rel_s, rel_e = rel[s], rel[e - 1]
        rel_min = rel[s:e].min()
        drop    = rel_s - rel_min
        print(f"\n[hl] === GRIP SUMMARY ===", flush=True)
        print(f"[hl] grasp window: frames {s}-{e-1}", flush=True)
        print(f"[hl] rel_y at grasp start (451): {rel_s:+.5f}", flush=True)
        print(f"[hl] rel_y at grasp end   ({e-1}): {rel_e:+.5f}", flush=True)
        print(f"[hl] rel_y min during grasp     : {rel_min:+.5f}", flush=True)
        print(f"[hl] cup_max_drop_vs_gripper    : {drop:.5f} m  (smaller = better grip)",
              flush=True)
        # === BETTER metrics: is cup actually lifted off ground? ===
        cup_y_grasp = cup_a[s:e]
        lift_max = float(cup_y_grasp.max() - cup_y_grasp[0])     # max rise above grasp-start
        lift_mean = float(cup_y_grasp.mean() - cup_y_grasp[0])   # mean rise above grasp-start
        ground = -1.67
        time_off_ground = int(np.sum(cup_y_grasp > ground + 0.02)) # frames cup is > 2cm above ground
        frac_off_ground = time_off_ground / max(1, len(cup_y_grasp))
        print(f"[hl] cup_lift_max               : {lift_max:.4f} m  (larger = lifted higher)",
              flush=True)
        print(f"[hl] cup_lift_mean              : {lift_mean:.4f} m  (larger = more time lifted)",
              flush=True)
        print(f"[hl] frac_grasp_off_ground(>2cm): {frac_off_ground:.3f}  (1.0 = always off ground)",
              flush=True)
        print(f"[hl] mean step: {mean_ms:.1f} ms ({1000/mean_ms:.1f} fps)", flush=True)
        return

    # --- 6. Polyscope + replay loop (same scene/data as the viser version,
    #     swapping web rendering for a local GL window for visual comparison) ---
    verts = eng.get_vertices()
    faces = eng.get_surface_faces()

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_program_name("replay_case39 (polyscope)")

    state = dict(
        idx=0, running=False, last_ms=0.0,
        mesh=ps.register_surface_mesh("scene", verts, faces, color=(0.6, 0.7, 0.8)),
        verts=verts, faces=faces,
    )

    # --- Per-vertex region overlay (hybrid_data['vertex_region']) for visualizing
    #     part2 vs part3 of each gripper FEM softpad.  Toggle "region" in
    #     polyscope's quantities panel to see the colormap.
    #     -1 = not a gripper FEM vert (arm/cup/shirt/finger);
    #      0 = gripper FEM region 0 (300 verts/gripper, larger group);
    #      1 = gripper FEM region 1 (150 verts/gripper, matches rigid-stitch count).
    # Per-vertex COLOR quantity: only gripper FEM verts get red/blue, everything
    # else stays the base mesh color (URDF arm, cup, shirt, finger ABDs all
    # rendered in their default gray-blue).
    base_color = np.array([0.6, 0.7, 0.8])
    red        = np.array([0.85, 0.25, 0.25])    # stiff (part2)
    blue       = np.array([0.25, 0.45, 0.85])    # soft  (part3)
    colors = np.tile(base_color, (verts.shape[0], 1))
    if _vert_is_part2 is not None:
        # Per-region young active -> color by centroid-based part2/part3 split.
        for g in grippers:
            fo = g['fem_v_off']; fc = g['fem_rec'].vertex_count
            grip_col = np.where(_vert_is_part2[:fc, None].astype(bool), red, blue)
            colors[fo:fo + fc] = grip_col
        state['mesh'].add_color_quantity(
            "part (red=stiff, blue=soft)", colors, defined_on='vertices', enabled=True)
    else:
        # Per-region young not active -> fall back to raw vertex_region (the
        # rigid-stitch vs non-stitch split inside each gripper FEM).
        vr = hybrid_data['vertex_region']
        for g in grippers:
            fo = g['fem_v_off']; fc = g['fem_rec'].vertex_count
            grip_col = np.where((vr[:fc] == 1)[:, None], red, blue)
            colors[fo:fo + fc] = grip_col
        state['mesh'].add_color_quantity(
            "region (red=1 stitched, blue=0 bulk)", colors, defined_on='vertices',
            enabled=True)

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

        if not state['running'] or state['idx'] >= len(qpos_all):
            return

        raw = qpos_all[state['idx']]
        q_left  = raw[0:7]
        q_right = raw[7:14]
        grip_L  = float(raw[14]) if len(raw) > 14 else 0.0
        grip_R  = float(raw[15]) if len(raw) > 15 else 0.0

        for i, rev_idx in enumerate(left_rev_indices):
            robot.set_revolute_position(rev_idx, q_left[i], degree=False)
        for i, rev_idx in enumerate(right_rev_indices):
            robot.set_revolute_position(rev_idx, q_right[i], degree=False)
        # CASE39_CLOSE_RATIO: 0.0 (default) = close all the way to lo (most over-
        # close, max squeeze, possible IPC↔prismatic tug-of-war jitter under
        # loose Newton). 1.0 = stay open (no close). 0.3-0.5 = partial close,
        # reduces constraint tension; needs higher friction/grip_K to compensate.
        _close_r = float(os.environ.get("CASE39_CLOSE_RATIO", "0.5"))
        if left_pris_indices:
            lo = robot.prismatic_joints[left_pris_indices[0]].lower_limit
            hi = robot.prismatic_joints[left_pris_indices[0]].upper_limit
            grip_pos = hi if grip_L >= 0 else (lo + _close_r * (hi - lo))
            for pi in left_pris_indices:
                robot.set_prismatic_position(pi, grip_pos, millimeters=False)
        if right_pris_indices:
            lo = robot.prismatic_joints[right_pris_indices[0]].lower_limit
            hi = robot.prismatic_joints[right_pris_indices[0]].upper_limit
            grip_pos = hi if grip_R >= 0 else (lo + _close_r * (hi - lo))
            for pi in right_pris_indices:
                robot.set_prismatic_position(pi, grip_pos, millimeters=False)

        t0 = time.perf_counter()
        eng.step()
        state['last_ms'] = (time.perf_counter() - t0) * 1000.0

        v = eng.get_vertices()
        f = eng.get_surface_faces()
        # Re-register if topology changed; else just update vertex positions.
        if v.shape[0] != state['verts'].shape[0] or f.shape != state['faces'].shape:
            state['mesh'] = ps.register_surface_mesh(
                "scene", v, f, color=(0.6, 0.7, 0.8))
            state['verts'], state['faces'] = v, f
        else:
            state['mesh'].update_vertex_positions(v)

        state['idx'] += 1

    ps.set_user_callback(callback)
    ps.show()
    print(f"Replay finished or window closed at frame {state['idx']}/{len(qpos_all)}")


if __name__ == "__main__":
    main()
