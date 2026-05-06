#!/usr/bin/env python3
"""Mobile arm + FEM soft gripper + cup grip demo (case 27 family).

4 FEM soft pads (softgriper_part2_blobal.msh, ~30k tet each → ~120k total)
are stitched to the 4 ABD finger backbones via add_stitch_spring(). The
gripper closes via prismatic finger_joint and the cup (ABD) is gripped
through deformable FEM contact. Arm revolute joints can move the gripper;
the FEM softpads follow translation+rotation through stitch (rest_offset=0).

Run:
    cd /home/ps/Downloads/Stiff-GIPC-dailyv2
    ./run examples/case_27_mobile_s1_softgripper_cup.py

Note: 120k FEM tet is heavy. Step time will be slow (~hundreds of ms).
Reduce by switching SOFT_FEM_MESH to softgriper_part3.msh (7.5k tet/finger)
or softgriper_part2.msh (12k tet/finger).
"""
import sys, os, math, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot

ASSETS_DIR    = str(Path(__file__).resolve().parent.parent / "assets") + "/"
URDF_RELPATH  = "sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_softgripper.urdf"
ORIGINAL_URDF = "sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile.urdf"
# Use SOFT_FEM=part3 (7.5k tet/finger=30k total, fast) by default;
# set FEM_BLOBAL=1 env to switch to part2_blobal (30k/finger=120k total, slow but full geometry).
if os.environ.get("FEM_TINY") == "1":
    SOFT_FEM_MESH = "sim_data/tetmesh/cube.msh"  # 5 tet, debug
elif os.environ.get("FEM_BLOBAL") == "1":
    SOFT_FEM_MESH = "sim_data/tetmesh/softgriper_part2_blobal.msh"
else:
    SOFT_FEM_MESH = "sim_data/tetmesh/softgriper_part3.msh"
CUP_MESH      = "sim_data/tetmesh/softgriper_cup.msh"
TABLE_MESH    = "sim_data/tetmesh/cube.msh"
SCALE = 0.3


def parse_xyz_rpy(s):
    return np.array([float(x) for x in s.split()], dtype=float)


def parse_soft_material_world_tfs(urdf_path, scale, base_tf):
    """Parse URDF to compute world TFs of 4 soft_material links."""
    src = open(urdf_path).read()
    joints = {}
    for m in re.finditer(r'<joint\s+name="([^"]+)"[^>]*>(.*?)</joint>', src, re.DOTALL):
        body = m.group(2)
        pm = re.search(r'<parent\s+link="([^"]+)"', body)
        cm = re.search(r'<child\s+link="([^"]+)"', body)
        if not (pm and cm):
            continue
        om_xyz = re.search(r'<origin[^/>]*xyz="([^"]+)"', body)
        om_rpy = re.search(r'<origin[^/>]*rpy="([^"]+)"', body)
        joints[cm.group(1)] = dict(
            parent=pm.group(1),
            xyz=parse_xyz_rpy(om_xyz.group(1)) if om_xyz else np.zeros(3),
            rpy=parse_xyz_rpy(om_rpy.group(1)) if om_rpy else np.zeros(3),
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

    targets = ['left_arm_leftfinger_soft_material', 'left_arm_rightfinger_soft_material',
               'right_arm_leftfinger_soft_material', 'right_arm_rightfinger_soft_material']
    out = []
    for name in targets:
        T_local = world_tf(name)
        # NOTE: don't pre-scale T_local. base_tf has SCALE in its 3x3, and
        # 4x4 mat-mul already scales the translation: base_tf @ T_local has
        # base_tf[:3,:3] @ T_local[:3,3] = (SCALE*R) @ t_link as desired.
        out.append(base_tf @ T_local)
    return out


def make_arm_tf(scale):
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix()
    tf[1, 3] = -0.9
    return tf


def all_host_vertices(eng):
    """Read ALL host vertices once via batch getter (fast, no per-vertex
    round-trip). Returns (N, 3) numpy array. Pre-finalize compatible."""
    return eng.native.get_vertices_host()


def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    config = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1e6,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
        joint_strength_ratio=100.0, revolute_driving_strength_ratio=100.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.5,
        assets_dir=ASSETS_DIR,
    )
    eng = Engine(config)
    assets_dir = eng.native.get_assets_dir()

    # 1. Arm URDF (37 ABD bodies, no soft_material)
    arm_tf = make_arm_tf(SCALE)
    eng.native.load_urdf(assets_dir + URDF_RELPATH, arm_tf, True, False, 1e7)
    arm_count = eng.abd_body_count
    print(f"[softgripper] arm: {arm_count} ABD bodies", flush=True)
    for b in range(arm_count):
        eng.add_ground_collision_skip(b)

    # 2. Identify finger backbone records
    records_after_arm = list(eng.get_load_records())
    finger_records = [r for r in records_after_arm
                      if r.body_type == 0
                      and 'finger' in r.label.lower()
                      and 'soft_material' not in r.label.lower()]
    print(f"[softgripper] finger backbones: {len(finger_records)}",
          [r.label for r in finger_records], flush=True)

    skip_cup_table = os.environ.get("NO_CUP_TABLE") == "1"
    cup_body_id = -1
    table_body_id = -1
    if skip_cup_table:
        print(f"[softgripper] NO_CUP_TABLE=1 skip cup+table", flush=True)
    else:
        # 3. Cup ABD (must be loaded before FEM!)
        cup_scale = 0.2
        cup_tf = np.eye(4)
        cup_tf[:3, :3] *= cup_scale
        cup_tf[0, 3] = 0.0
        cup_tf[1, 3] = -0.45
        cup_tf[2, 3] = 0.0
        eng.load_mesh(CUP_MESH, dimensions=3, body_type="ABD",
                      transform=cup_tf, young_modulus=1e8)
        cup_body_id = eng.abd_body_count - 1
        print(f"[softgripper] cup body_id={cup_body_id}", flush=True)
        # 4. Table fixed ABD
        table_tf = np.eye(4)
        np.fill_diagonal(table_tf[:3, :3], [0.5, 0.02, 0.5])
        table_tf[1, 3] = -0.5
        eng.load_mesh(TABLE_MESH, dimensions=3, body_type="ABD",
                      transform=table_tf, young_modulus=1e9, boundary_type="Fixed")
        table_body_id = eng.abd_body_count - 1
        print(f"[softgripper] table body_id={table_body_id}", flush=True)

    # 5. Exclusions: non-finger arm vs cup, all arm vs table
    finger_offsets = {fr.body_offset for fr in finger_records}
    if not skip_cup_table:
        for r in records_after_arm:
            if r.body_type != 0:
                continue
            if r.body_offset not in finger_offsets:
                eng.add_collision_exclusion(r.body_offset, cup_body_id)
            eng.add_collision_exclusion(r.body_offset, table_body_id)

    # 6. Load 4 FEM softpads (LAST — after all ABD)
    soft_tfs = parse_soft_material_world_tfs(assets_dir + ORIGINAL_URDF, SCALE, arm_tf)
    # Order soft_tfs to MATCH finger_records by name (parse returns left_arm_*
    # first; finger_records order is whatever URDF importer used). Build name
    # -> tf dict for re-ordering.
    soft_tf_targets = ['left_arm_leftfinger_soft_material', 'left_arm_rightfinger_soft_material',
                       'right_arm_leftfinger_soft_material', 'right_arm_rightfinger_soft_material']
    soft_tfs_by_name = dict(zip(soft_tf_targets, soft_tfs))

    fem_records = []
    finger_to_fem = []  # parallel to finger_records
    if os.environ.get("NO_FEM") != "1":
        n_fem_load = int(os.environ.get("FEM_N", "4"))
        offset_y = float(os.environ.get("FEM_OFFSET_Y", "0"))
        for fr in finger_records[:n_fem_load]:
            soft_name = fr.label + '_soft_material'
            T = soft_tfs_by_name[soft_name].copy()
            T[1, 3] += offset_y  # debug: move FEM away from arm initial position
            eng.load_mesh(SOFT_FEM_MESH, dimensions=3, body_type="FEM",
                          transform=T, young_modulus=1e6)
            fem_rec = eng.get_load_records()[-1]
            fem_records.append(fem_rec)
            finger_to_fem.append((fr, fem_rec))
            print(f"[softgripper] {fr.label} <- FEM V={fem_rec.vertex_count}", flush=True)
    else:
        print(f"[softgripper] NO_FEM=1, skipping FEM softpads", flush=True)

    # 7. Stitch springs (batch read all host vertices once for fast lookup)
    all_verts = all_host_vertices(eng)
    total_pairs = 0
    for f_rec, e_rec in finger_to_fem:
        f_verts = all_verts[f_rec.vertex_offset:f_rec.vertex_offset + f_rec.vertex_count]
        e_verts = all_verts[e_rec.vertex_offset:e_rec.vertex_offset + e_rec.vertex_count]
        # Find Y-overlap zone (in world frame)
        f_y_min, f_y_max = f_verts[:, 1].min(), f_verts[:, 1].max()
        e_y_min, e_y_max = e_verts[:, 1].min(), e_verts[:, 1].max()
        y_overlap_lo = max(f_y_min, e_y_min)
        y_overlap_hi = min(f_y_max, e_y_max)
        if y_overlap_hi < y_overlap_lo:
            print(f"[softgripper] no Y overlap for {f_rec.label}, skip",
                  f"finger Y=[{f_y_min:.4f},{f_y_max:.4f}] FEM Y=[{e_y_min:.4f},{e_y_max:.4f}]",
                  flush=True)
            continue
        # Take both meshes' verts inside the overlap zone
        f_in = np.where((f_verts[:, 1] >= y_overlap_lo) & (f_verts[:, 1] <= y_overlap_hi))[0]
        e_in = np.where((e_verts[:, 1] >= y_overlap_lo) & (e_verts[:, 1] <= y_overlap_hi))[0]
        if len(f_in) == 0 or len(e_in) == 0:
            continue
        # Mutual-NN match within overlap zone
        f_pts = f_verts[f_in]; e_pts = e_verts[e_in]
        f_tree = cKDTree(f_pts); e_tree = cKDTree(e_pts)
        d_e2f, idx_e2f = f_tree.query(e_pts)
        d_f2e, idx_f2e = e_tree.query(f_pts)
        thresh = 0.05 * SCALE  # 15mm at scale=0.3 (loose - finger and FEM may not perfectly align)
        n_pairs = 0
        for i_e in range(len(e_pts)):
            j_f = idx_e2f[i_e]
            if idx_f2e[j_f] == i_e and d_e2f[i_e] < thresh:
                fem_global = e_rec.vertex_offset + e_in[i_e]
                abd_global = f_rec.vertex_offset + f_in[j_f]
                if os.environ.get("NO_STITCH") != "1":
                    eng.add_stitch_spring(fem_global, abd_global, f_rec.body_offset)
                n_pairs += 1
        print(f"[softgripper] stitch {f_rec.label} -> FEM#{fem_records.index(e_rec)}: "
              f"{n_pairs} pairs (overlap Y=[{y_overlap_lo:.4f},{y_overlap_hi:.4f}], "
              f"min_d={d_e2f.min():.4f}, mean_d={d_e2f.mean():.4f})",
              flush=True)
        total_pairs += n_pairs
    print(f"[softgripper] total stitch pairs: {total_pairs}", flush=True)

    eng.finalize()
    robot = Robot(eng)
    print(f"[softgripper] finalized: verts={len(eng.get_vertices())}", flush=True)

    # AUTO-STEP mode for testing (no GUI loop)
    auto_n = int(os.environ.get("AUTO_STEP", "0"))
    if auto_n > 0:
        import time
        for i in range(auto_n):
            t0 = time.perf_counter()
            eng.step()
            print(f"[softgripper] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
        return

    # GUI
    verts = eng.get_vertices()
    faces = eng.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts, faces, smooth_shade=True)
    mesh.set_color((0.6, 0.7, 0.8))

    running = [False]
    step_count = [0]

    def callback():
        psim.SetNextWindowPos((330, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((380, 0), psim.ImGuiCond_Once)
        ret = psim.Begin("FEM Soft Gripper + Cup", True)
        if (ret[0] if isinstance(ret, tuple) else ret):
            if running[0]:
                if psim.Button("Pause"): running[0] = False
            else:
                if psim.Button("Run"): running[0] = True
            psim.SameLine()
            psim.Text(f"Step: {step_count[0]}")
            psim.Separator()
            if robot.revolute_joints:
                psim.Text(f"Revolute Joints ({len(robot.revolute_joints)})")
                for i, ji in enumerate(robot.revolute_joints):
                    lo = math.degrees(ji.lower_limit); hi = math.degrees(ji.upper_limit)
                    cur = robot.get_revolute_target_deg(i)
                    changed, new_val = psim.SliderFloat(ji.name, cur, lo, hi)
                    if changed: robot.set_revolute_position(i, new_val, degree=True)
            if robot.prismatic_joints:
                psim.Spacing()
                psim.Text(f"Prismatic Joints ({len(robot.prismatic_joints)})")
                for i, ji in enumerate(robot.prismatic_joints):
                    lo_mm = ji.lower_limit * 1000.0
                    hi_mm = ji.upper_limit * 1000.0
                    cur_mm = robot.get_prismatic_target_mm(i)
                    changed, new_val = psim.SliderFloat(f"{ji.name} (mm)", cur_mm, lo_mm, hi_mm)
                    if changed: robot.set_prismatic_position(i, new_val, millimeters=True)
            psim.Spacing()
            if psim.Button("Reset All Joints"):
                robot.reset_all()
        psim.End()
        if running[0]:
            eng.step()
            step_count[0] += 1
            mesh.update_vertex_positions(eng.get_vertices())

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
