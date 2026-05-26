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
    """Parse URDF to compute world TFs of 4 soft_material links.

    NOTE: each *_soft_material link in this URDF has a mesh-in-link origin
    on its <visual>/<collision> tag (rpy=(-pi/2, 0.20, -pi/2), xyz=
    (-0.0165, 0.0165, 0.128)). The URDF importer already applies this when
    loading ABD finger backbones, so ABD fingers stand upright. We must
    apply the same transform to the FEM softpad mesh, otherwise it lies
    horizontally instead of along the finger axis.
    """
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

    def visual_origin_in_link(link_name):
        """Extract the <visual>/<collision> origin of a link as a 4x4 mat.
        Returns identity if no origin is found."""
        m = re.search(
            r'<link\s+name="' + re.escape(link_name) + r'"[^>]*>(.*?)</link>',
            src, re.DOTALL)
        if not m: return np.eye(4)
        body = m.group(1)
        # Prefer collision origin; fall back to visual.
        section = (re.search(r'<collision[^>]*>(.*?)</collision>', body, re.DOTALL)
                   or re.search(r'<visual[^>]*>(.*?)</visual>', body, re.DOTALL))
        if not section: return np.eye(4)
        sbody = section.group(1)
        om_xyz = re.search(r'<origin[^/>]*xyz="([^"]+)"', sbody)
        om_rpy = re.search(r'<origin[^/>]*rpy="([^"]+)"', sbody)
        if not (om_xyz or om_rpy): return np.eye(4)
        xyz = parse_xyz_rpy(om_xyz.group(1)) if om_xyz else np.zeros(3)
        rpy = parse_xyz_rpy(om_rpy.group(1)) if om_rpy else np.zeros(3)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
        T[:3, 3] = xyz
        return T

    targets = ['left_arm_leftfinger_soft_material', 'left_arm_rightfinger_soft_material',
               'right_arm_leftfinger_soft_material', 'right_arm_rightfinger_soft_material']
    out = []
    for name in targets:
        T_link_in_world = world_tf(name)
        T_mesh_in_link = visual_origin_in_link(name)
        T_mesh_in_world = T_link_in_world @ T_mesh_in_link
        # NOTE: don't pre-scale here. base_tf has SCALE in its 3x3, and
        # 4x4 mat-mul already scales the translation: base_tf @ T has
        # base_tf[:3,:3] @ T[:3,3] = (SCALE*R) @ t as desired.
        out.append(base_tf @ T_mesh_in_world)
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
        # soft_motion_rate matches set_case11_gripper (gl_main.cu:2183):
        # 1e4 is calibrated equivalent to UIPC's kappa*dt^2 = 1e8 * 0.01^2 = 1e4.
        # The original "FEM tip not tracking" issue we saw was actually a
        # stitch *geometry* problem (matching restricted to Y-overlap zone
        # → only top of finger had stitches → tip was unconstrained),
        # not a stitch *stiffness* problem. The fix below (full-length
        # mutual-NN matching, see Stitch springs section) distributes
        # stitch points along the entire finger like case_11 native does.
        cloth_density=200, strain_rate=100, soft_motion_rate=1e4,
        # relative_dhat=1e-4 (not 1e-3) is required because the FEM softpad
        # mesh (softgriper_part3.msh, scaled by 0.3) has median edge length
        # ~0.38mm. With the default 1e-3 and the full-scene bbox dominated by
        # the ridgeback base (~0.7m diag), gapl = sqrt(dhat) ~0.87mm > median
        # edge length, so EE self-collision detection finds far more pairs
        # than _collisionPair[] can hold and overflows (illegal mem access in
        # mlbvh.cu _selfQuery_ee). 1e-4 brings gapl to ~0.087mm, well below
        # the FEM mesh resolution.
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-4,
        joint_strength_ratio=100.0, revolute_driving_strength_ratio=100.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.5,
        assets_dir=ASSETS_DIR,
    )
    # CCD buffer scale (default 6, bumped to 64 for FEM_BLOBAL aggressive
    # joint motions). compute-sanitizer pinpointed _selfQuery_vf_ccd at
    # mlbvh.cu:1681 overflowing _ccd_collisionPair[] when left_arm_joint6
    # ramps to ~30deg with the full part2_blobal mesh (34k verts, 30k tet
    # / softpad). buff=16 wasn't enough; buff=64 (~620 MB CCD buffer)
    # covers a 60deg single-step SHOCK in earlier tests. The proper fix
    # is engine-side atomic cap (task #66 in tracker) — until that lands,
    # large buffers are the practical workaround.
    config._cfg.collision_detection_buff_scale = 64.0
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
    USE_HYBRID_D = os.environ.get("USE_HYBRID_D", "0") == "1"
    if os.environ.get("NO_FEM") != "1":
        n_fem_load = int(os.environ.get("FEM_N", "4"))
        if USE_HYBRID_D:
            # Strategy D: skip the standard softpad load.  We will build a
            # NEW merged tet mesh per finger (finger volume + soft envelope)
            # and load it later, in the hybrid_d branch.  Setting n=0 here
            # avoids loading the original softpad (which would be
            # geometrically duplicated by the merged mesh's rigid region).
            n_fem_load = 0
            print(f"[softgripper] USE_HYBRID_D=1, skipping standard softpad "
                  f"load (will build merged meshes per finger below)", flush=True)
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

    # 7. Stitch springs — each FEM SURFACE vertex pulls toward its nearest
    # ABD finger backbone vertex (one-way NN, sparse coverage).
    #
    # Why not mutual-NN like case_11 native (gl_main.cu:2034-2063)?
    # case_11 uses part1.msh (ABD) + part2_blobal.msh (FEM) — they are
    # the hard/soft regions of the SAME source mesh, so mutual-NN gives
    # full-length coverage. Our demo uses finger_clean.obj (ABD) which is
    # geometrically *smaller* than part2_blobal.msh (FEM) — mutual-NN's
    # reverse-check fails for the FEM tip (the FEM tip's nearest ABD vertex
    # is at the finger *top*, but that ABD vertex's nearest FEM vertex is
    # also at the FEM top, not the tip). Result: stitch all crowd at the
    # finger top, FEM tip drifts under arm rotation.
    #
    # One-way NN: every FEM SURFACE vertex picks its nearest ABD vertex
    # (within thresh), so stitch distribution follows FEM geometry, not
    # mutual matching. Multiple FEM verts can share an ABD anchor —
    # physically OK (multiple springs anchored at one ABD point).
    # We use surface verts only (not all FEM verts) to keep stitch_count
    # bounded; interior FEM verts are dragged by FEM elasticity from the
    # surface anchors.
    all_verts = all_host_vertices(eng)
    total_pairs = 0
    stitch_viz_pairs = []  # list of (fem_global, abd_global) for GUI viz

    USE_HYBRID = os.environ.get("USE_HYBRID", "0") == "1"

    if USE_HYBRID_D:
        # [Strategy D — true hybrid mesh path]
        # For each finger ABD body, build a merged tet mesh (finger convex
        # hull + spherical soft envelope) using TetGen with region attrs.
        # Load it as a NEW FEM body and pin its rigid region to the finger
        # ABD body via add_fem_pin_to_abd (auto local_pos via finalize).
        # This is "Strategy D" per the hybrid mesh design notes — clean
        # interface, no overlapping geometry, substantial rigid region.
        import sys as _sys, tempfile as _tf
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
        import trimesh as _tm
        from tools.build_hybrid_mesh_d import build_hybrid_d as _build_d

        # Tunable: TetGen volume targets.  Default tuned for ~5k tets/finger
        # → 20k tets total.  Larger max_vol / coarser quality = fewer tets.
        # WARNING: total tets cap ~ 30k with current engine triplet buffer
        # (Phase 4 makes EVERY tet contribute 10 triplets, then M3.5
        # extension uses 16× as ext_capacity, easily overflowing GPU buffer
        # above ~30-50k total tets).  Use very coarse defaults; refine
        # once Phase 4 is upgraded to skip triplet writes for rigid tets.
        d_rigid_max = float(os.environ.get("HYBRID_D_RIGID_MAX_VOL", "5e-6"))
        d_soft_max  = float(os.environ.get("HYBRID_D_SOFT_MAX_VOL",  "1e-5"))
        d_quality   = float(os.environ.get("HYBRID_D_QUALITY", "2.5"))
        d_envelope_factor = float(os.environ.get("HYBRID_D_ENV_FACTOR", "0.55"))

        d_tmp = _P(_tf.mkdtemp(prefix="hybrid_d_"))
        d_total_pins = 0
        d_total_tets = 0
        d_finger_count = int(os.environ.get("HYBRID_D_FINGER_COUNT", "4"))
        d_use_sphere_env = os.environ.get("HYBRID_D_SPHERE_ENV", "0") == "1"

        # Pre-load softpad outer surface ONCE (shared across fingers, in
        # softpad's local frame).  Real softpad geometry beats a sphere.
        import meshio as _meshio0
        _softpad_msh = _meshio0.read(SOFT_FEM_MESH if SOFT_FEM_MESH.startswith('/')
                                     else assets_dir + SOFT_FEM_MESH)
        _softpad_local_verts = np.asarray(_softpad_msh.points, dtype=np.float64)
        _tet_blocks = [c for c in _softpad_msh.cells if c.type == "tetra"]
        _softpad_local_tets = np.vstack([c.data for c in _tet_blocks])
        # Extract softpad boundary surface
        from tools.build_hybrid_mesh_d import _extract_boundary_faces as _extract_bdry
        _softpad_local_faces = _extract_bdry(_softpad_local_tets)
        _softpad_local_mesh = _tm.Trimesh(vertices=_softpad_local_verts,
                                          faces=_softpad_local_faces,
                                          process=True)
        print(f"[hybrid_d] softpad outer surface extracted: "
              f"V={len(_softpad_local_mesh.vertices)} "
              f"F={len(_softpad_local_mesh.faces)} "
              f"watertight={_softpad_local_mesh.is_watertight}", flush=True)

        for fr in finger_records[:d_finger_count]:
            f_world_verts = all_verts[fr.vertex_offset:
                                      fr.vertex_offset + fr.vertex_count]
            f_tris = eng.native.get_abd_surface_body_triangles(fr.body_offset)
            finger_world = _tm.Trimesh(vertices=f_world_verts, faces=f_tris,
                                       process=True)

            # Soft envelope: real softpad mesh transformed to current
            # finger's world position.  The softpad is much SHORTER than
            # finger (case_27: finger Z=54mm, softpad Z=16mm) — softpad
            # only covers the fingertip region.  So we CLIP finger to the
            # softpad's bbox before convex-hulling — gives a reasonable
            # "rigid region" that fits inside softpad.
            soft_name = fr.label + '_soft_material'
            soft_T = soft_tfs_by_name[soft_name]
            softpad_world = _softpad_local_mesh.copy()
            softpad_world.apply_transform(soft_T)

            if d_use_sphere_env:
                # Override: use sphere envelope (debug)
                finger_hull = finger_world.convex_hull
                center = finger_hull.centroid
                diag = float(np.linalg.norm(finger_hull.bounds[1] - finger_hull.bounds[0]))
                radius = diag * d_envelope_factor
                env = _tm.creation.icosphere(subdivisions=2, radius=radius)
                env.apply_translation(center)
                print(f"  [hybrid_d] {fr.label}: SPHERE envelope (debug)", flush=True)
            else:
                # Use real softpad as envelope; clip finger to softpad bbox
                # so rigid region fits inside soft region.
                env = softpad_world
                pad_min = softpad_world.bounds[0]
                pad_max = softpad_world.bounds[1]
                # Shrink bbox slightly so we don't end up with rigid verts
                # exactly on softpad surface
                shrink = 0.001  # 1mm inset
                inner_min = pad_min + shrink
                inner_max = pad_max - shrink
                # Keep only finger verts inside this shrunk bbox
                fv = finger_world.vertices
                inside_box = ((fv >= inner_min) & (fv <= inner_max)).all(axis=1)
                n_in = int(inside_box.sum())
                if n_in < 4:
                    print(f"  [hybrid_d] {fr.label}: only {n_in}/<{len(fv)}> "
                          f"finger verts inside softpad bbox; FALLBACK to sphere",
                          flush=True)
                    finger_hull = finger_world.convex_hull
                    center = finger_hull.centroid
                    diag = float(np.linalg.norm(finger_hull.bounds[1] - finger_hull.bounds[0]))
                    radius = diag * d_envelope_factor
                    env = _tm.creation.icosphere(subdivisions=2, radius=radius)
                    env.apply_translation(center)
                else:
                    finger_hull = _tm.Trimesh(
                        vertices=fv[inside_box],
                        faces=None,  # convex_hull
                    ).convex_hull
                    print(f"  [hybrid_d] {fr.label}: using REAL softpad envelope; "
                          f"clipped finger to {n_in}/{len(fv)} verts inside softpad bbox; "
                          f"clipped hull V={len(finger_hull.vertices)} "
                          f"F={len(finger_hull.faces)}", flush=True)

            hull_path = d_tmp / f"hull_{fr.label}.stl"
            env_path  = d_tmp / f"env_{fr.label}.stl"
            finger_hull.export(str(hull_path))
            env.export(str(env_path))

            fields = _build_d(
                abd_surface_path=str(hull_path),
                soft_mesh_path=str(env_path),
                abd_body_id=fr.body_offset,
                rigid_max_vol=d_rigid_max,
                soft_max_vol=d_soft_max,
                quality=d_quality,
                young_modulus=1e6,
                verbose=False,
            )

            verts = fields["vertices"]
            tets  = fields["tets"]
            d_total_tets += len(tets)

            # Write merged mesh to a temp .msh and use load_mesh path
            # (load_mesh_from_data has multi-body issues — see HANDOVER).
            import meshio as _meshio
            msh_path = d_tmp / f"merged_{fr.label}.msh"
            _meshio.write(
                str(msh_path),
                _meshio.Mesh(
                    points=np.asarray(verts, dtype=np.float64),
                    cells=[("tetra", np.asarray(tets, dtype=np.int32))],
                ),
                file_format="gmsh22",
                binary=False,
            )
            eng.load_mesh(str(msh_path), dimensions=3, body_type="FEM",
                          transform=np.eye(4), young_modulus=1e6)
            new_fem_rec = eng.get_load_records()[-1]
            new_fem_offset = new_fem_rec.vertex_offset

            # Pin rigid verts to finger ABD body.  Use legacy add_fem_pin_to_abd
            # (anchor=fem_global) so finalize auto-derives local_pos from each
            # vertex's current world position vs finger ABD body's q at finalize.
            rigid_mask = fields["vertex_region"] == 1
            for i in np.nonzero(rigid_mask)[0]:
                fem_global = new_fem_offset + int(i)
                eng.native.add_fem_pin_to_abd(
                    fem_global, fem_global, fr.body_offset,
                    rest_offset_world=(0.0, 0.0, 0.0),
                )
            n_rigid = int(rigid_mask.sum())
            n_iface = int((fields['tet_region'] == 0).sum() -
                          ((rigid_mask[tets].sum(axis=1) == 0) &
                           (fields['tet_region'] == 0)).sum())
            n_pure_rigid_t = int((fields['tet_region'] == 2).sum())
            n_pure_fem_t = int((rigid_mask[tets].sum(axis=1) == 0).sum())
            d_total_pins += n_rigid

            # Make sure finger ABD body and this new FEM body don't IPC-collide
            # (their geometries overlap by design).  Disable via env if buggy.
            if os.environ.get("HYBRID_D_NO_EXCL", "0") != "1":
                try:
                    eng.native.add_collision_exclusion(fr.body_offset, new_fem_rec.body_offset)
                except Exception as _e:
                    print(f"  [hybrid_d] WARN add_collision_exclusion failed: {_e}", flush=True)

            print(f"[hybrid_d] {fr.label}: built merged mesh "
                  f"V={verts.shape[0]} T={tets.shape[0]} "
                  f"(rigid_t={n_pure_rigid_t}, FEM_t={n_pure_fem_t}, "
                  f"interface_t={n_iface}); pinned {n_rigid} rigid verts",
                  flush=True)
        print(f"[hybrid_d] TOTAL: {d_total_tets} tets, {d_total_pins} pins "
              f"across {len(finger_records[:4])} fingers", flush=True)
        # Skip the regular stitch / hard-pin / hybrid-B loop
        finger_to_fem_loop = []
    elif USE_HYBRID:
        # [Hybrid mesh path]  Replaces stitch / hard-pin: every FEM softpad
        # vertex inside the ABD finger surface is bulk-pinned to that finger.
        # The M3.5 chain-rule kernel routes the rigid region's FEM Hessian to
        # ABD body DOFs; Phase 4 skips elasticity for all-rigid tets.  This
        # gives true continuous elastic coupling at the ABD↔FEM interface,
        # avoiding the discrete-pin mesh-resolution issues that bottleneck
        # USE_HARD_PIN=1 under joint motion.
        import trimesh as _tm
        for f_rec, e_rec in finger_to_fem:
            f_world_verts = all_verts[f_rec.vertex_offset:
                                      f_rec.vertex_offset + f_rec.vertex_count]
            f_tris = eng.native.get_abd_surface_body_triangles(f_rec.body_offset)
            finger_world = _tm.Trimesh(vertices=f_world_verts, faces=f_tris,
                                       process=False)
            # process=False to keep vertex order matching f_world_verts;
            # is_watertight checks the topology, which doesn't need processing.
            e_verts = all_verts[e_rec.vertex_offset:
                                e_rec.vertex_offset + e_rec.vertex_count]
            if not finger_world.is_watertight:
                # Try repaired copy for the contains test (still need rtree)
                finger_for_test = finger_world.copy()
                finger_for_test.process(validate=True)
            else:
                finger_for_test = finger_world
            inside = finger_for_test.contains(e_verts)
            n_rigid = int(inside.sum())
            # Pass anchor = fem_global (any positive value works — anchor_vec
            # is stored but unused in finalize's local_pos computation, which
            # reads fem_world_pos from the FEM vertex itself.  The -1
            # sentinel branch I added in Phase 3 is reserved for the
            # add_fem_pins_with_local_pos bulk API, NOT this per-pin call.)
            for i in np.nonzero(inside)[0]:
                fem_global = e_rec.vertex_offset + int(i)
                eng.native.add_fem_pin_to_abd(
                    fem_global, fem_global, f_rec.body_offset,
                    rest_offset_world=(0.0, 0.0, 0.0),
                )
                stitch_viz_pairs.append((fem_global, f_rec.vertex_offset))
            total_pairs += n_rigid
            print(f"[hybrid] {f_rec.label}: {n_rigid}/{len(e_verts)} verts "
                  f"pinned (inside finger surface, watertight={finger_world.is_watertight})",
                  flush=True)
        print(f"[hybrid] total {total_pairs} hybrid pins across "
              f"{len(finger_to_fem)} fingers", flush=True)
        # Skip the regular stitch/hard-pin loop below
        finger_to_fem_loop = []
    else:
        finger_to_fem_loop = finger_to_fem

    for f_rec, e_rec in finger_to_fem_loop:
        f_verts = all_verts[f_rec.vertex_offset:f_rec.vertex_offset + f_rec.vertex_count]
        e_verts = all_verts[e_rec.vertex_offset:e_rec.vertex_offset + e_rec.vertex_count]
        # One-way NN: every FEM vertex finds its nearest ABD vertex (within
        # thresh). Multiple FEM verts can share an ABD anchor — physically
        # OK (multiple springs anchored at one ABD point). Sub-sampling
        # FEM verts (every Nth) keeps stitch_count bounded — too many
        # springs makes the Hessian condition number balloon and the
        # solver becomes slow.
        # Sub-sampling stride: smaller -> more stitches -> better tracking
        # but slower PCG (Hessian condition number grows). FEM_BLOBAL has
        # 8029 verts/finger; sub=128 gives ~60 stitch/finger spread along
        # the full length, step ~200ms. Use STITCH_SUB env to override.
        sub_n = int(os.environ.get('STITCH_SUB', '128'))
        e_idx_local = np.arange(0, len(e_verts), sub_n)
        e_sub = e_verts[e_idx_local]
        f_tree = cKDTree(f_verts)
        d_e2f, idx_e2f = f_tree.query(e_sub)
        thresh = 0.020  # 20mm — generous; FEM tip can be ~10mm from ABD
        n_pairs = 0
        used_fem_y = []
        for k in range(len(e_sub)):
            if d_e2f[k] >= thresh: continue
            i_e = int(e_idx_local[k])
            j_f = int(idx_e2f[k])
            fem_global = e_rec.vertex_offset + i_e
            abd_global = f_rec.vertex_offset + j_f
            # CRITICAL: must pass rest_offset = current (fem_pos - abd_pos)
            # so the stitch spring's *natural length* is the current spatial
            # offset. Without this, default rest_offset=(0,0,0) tells the
            # spring to pull FEM vertex onto the same world position as the
            # ABD anchor — which in one-way-NN multi-to-one matching causes
            # every FEM vertex to get yanked toward the same ABD vertex,
            # crumpling the FEM mesh into a ball. case_11 set_case11_gripper
            # at gl_main.cu:2165 does the same:
            #     rest_off = fp - ap
            fem_pos = e_verts[i_e]
            abd_pos = f_verts[j_f]
            rest_off = (fem_pos - abd_pos).tolist()
            if os.environ.get("NO_STITCH") != "1":
                # USE_HARD_PIN=1 enables M1 substitution-method hard pin:
                # pinned FEM vertex's world pos = q.t + R(q) * fem_local_pos
                # each step. PCG sees these as Fixed boundary (Δx≈0); the
                # apply_fem_pins kernel writes positions from ABD's q.
                # FEM follows ABD translation AND rotation exactly.
                if os.environ.get("USE_HARD_PIN", "0") == "1":
                    eng.native.add_fem_pin_to_abd(fem_global, abd_global,
                                                  f_rec.body_offset,
                                                  rest_offset_world=rest_off)
                else:
                    eng.add_stitch_spring(fem_global, abd_global, f_rec.body_offset,
                                          rest_offset_world=rest_off)
            stitch_viz_pairs.append((fem_global, abd_global))
            used_fem_y.append(e_sub[k, 1])
            n_pairs += 1
        if n_pairs > 0:
            uy = np.asarray(used_fem_y)
            print(f"[softgripper] stitch {f_rec.label} -> FEM#{fem_records.index(e_rec)}: "
                  f"{n_pairs} pairs  Y-extent=[{uy.min():.4f}, {uy.max():.4f}] "
                  f"(spread={1000*(uy.max()-uy.min()):.1f}mm, sub={sub_n})  "
                  f"d_min={d_e2f.min()*1000:.2f}mm  d_mean(used)={1000*d_e2f[d_e2f<thresh].mean():.2f}mm",
                  flush=True)
        else:
            print(f"[softgripper] no stitch matches for {f_rec.label} (thresh={thresh*1000:.0f}mm)", flush=True)
        total_pairs += n_pairs
    print(f"[softgripper] total stitch pairs: {total_pairs}", flush=True)

    # 8. Exclude every arm ABD body vs the FEM "slot" so the IPC initial-
    # intersect check doesn't reject FEM softpads that are stitched flush
    # against (or slightly overlapping) the ABD finger backbones. Engine
    # collapses every FEM body's body_id to -1, which the matrix maps to
    # slot (abd_count + fem_count - 1); excluding any arm body against this
    # slot also excludes it from all 4 FEM softpads, but that's fine —
    # arm should never collide with any softpad. Cup + table are kept
    # un-excluded so gripping physics still works.
    # [multi-FEM-bodyid] Engine v0.5+ assigns each FEM body its own body_id
    # (no longer aliased to -1), so we can address each one individually
    # in the exclusion matrix.
    #
    # NOTE: BodyLoadRecord.body_offset is a *type-local* index — for ABD
    # bodies it equals the global body_id (since ABDs come first), but for
    # FEM bodies it's 0..fem_count-1 (sub-id). The exclusion matrix is
    # indexed by the *global* body_id, so we add abd_body_count to FEM
    # offsets to get the global id.
    if eng.fem_body_count > 0:
        records = list(eng.get_load_records())
        fem_records   = [r for r in records if r.body_type == 1]
        arm_records   = [r for r in records
                         if r.body_type == 0
                         and 'cup'  not in r.label.lower()
                         and 'cube' not in r.label.lower()]
        abd_count = eng.abd_body_count
        fem_global_ids = [abd_count + fr.body_offset for fr in fem_records]
        # 1. arm × all FEM softpads — arm shouldn't collide with FEM (stitch
        # springs already attach them, geometry overlaps).
        for ar in arm_records:
            for fid in fem_global_ids:
                eng.add_collision_exclusion(ar.body_offset, fid)
        # 2. FEM_i × FEM_j (i != j) — 4 softpads sit on 4 separate fingers
        # at >10cm separation, no physical reason to check inter-FEM.
        # Each FEM's *own* self-collision is preserved (we don't add the
        # (i, i) self-exclusion).
        for i, fid_i in enumerate(fem_global_ids):
            for fid_j in fem_global_ids[i+1:]:
                eng.add_collision_exclusion(fid_i, fid_j)
        print(f"[softgripper] excluded {len(arm_records) * len(fem_records)} arm-FEM "
              f"pairs + {len(fem_records)*(len(fem_records)-1)//2} FEM-FEM pairs "
              f"(FEM global body ids: {fem_global_ids})",
              flush=True)

    eng.finalize()
    robot = Robot(eng)
    print(f"[softgripper] finalized: verts={len(eng.get_vertices())}", flush=True)

    # Both stitch (USE_HARD_PIN=0) and hard pin (USE_HARD_PIN=1) couple FEM
    # softpads to ABD finger backbones.  When the user drags a joint slider
    # fast, the engine's default per-step revolute clamp (0.1 rad ≈ 5.7°)
    # rotates the finger that much in one step.  At softgriper_part2_blobal
    # mesh resolution (median edge ≈ 0.1mm scaled), even 5.7° causes the
    # softpad surface to self-intersect (free FEM neighbors lag behind the
    # finger-coupled vertices).  The IPC barrier then shrinks line-search
    # alpha repeatedly → CCD pair buffer accumulates → systemd-oomd kills
    # the process.
    #
    # Safe default 0.3°/step works for both modes.  Override with
    # MAX_REVOLUTE_DEG=N env var (HARD_PIN_MAX_DEG kept as alias).
    max_deg = float(os.environ.get("MAX_REVOLUTE_DEG",
                    os.environ.get("HARD_PIN_MAX_DEG", "0.3")))
    max_rad = math.radians(max_deg)
    eng.native.set_max_revolute_step_per_frame(max_rad)
    print(f"[softgripper] revolute angle clamp set to "
          f"{max_deg} deg / step ({max_rad:.4f} rad)", flush=True)

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

    # Stitch springs viz: one curve-network edge per (FEM vertex, ABD anchor)
    # pair, showing the stitch in red. Updated each step.
    stitch_curve = None
    if stitch_viz_pairs:
        n_pairs = len(stitch_viz_pairs)
        stitch_node_idx = np.array(stitch_viz_pairs, dtype=np.int64).reshape(-1)
        # node array layout: [(fem0, abd0, fem1, abd1, ...)] then update by indexing verts
        stitch_nodes = verts[stitch_node_idx]
        stitch_edges = np.array([(2*i, 2*i+1) for i in range(n_pairs)], dtype=int)
        stitch_curve = ps.register_curve_network(
            "stitch springs", stitch_nodes, stitch_edges)
        stitch_curve.set_color((1.0, 0.2, 0.2))
        stitch_curve.set_radius(0.0008, relative=False)

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
            v_now = eng.get_vertices()
            mesh.update_vertex_positions(v_now)
            if stitch_curve is not None:
                stitch_curve.update_node_positions(v_now[stitch_node_idx])

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
