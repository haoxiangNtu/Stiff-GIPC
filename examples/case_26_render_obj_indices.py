#!/usr/bin/env python3
"""Case 26 variant — per-body colored rendering of XArm7 + falling shirt.

Same physics setup as `case_26_arm_cloth_semi_implicit.py` (default MAS
preconditioner, identical Config), but each body is registered as its own
polyscope mesh so you can see them in distinct colors and verify per-body
motion under joint sliders.

Two rendering patterns are used:

1. Shirt (FEM cloth)
     - Slice `engine.get_vertices()[v_off : v_off+v_cnt]` for the shirt
       body's range (located via `BodyLoadRecord`).
     - Feed `engine.get_surface_faces()` filtered to that range
       (converted to local indices) — works regardless of MAS reordering,
       since the engine's face table tracks whatever vertex order the
       engine itself uses.

2. Arm links (ABD rigid)
     - URDF-loaded ABD bodies never go through metis regardless of
       `preconditioner_type`, so vertex order within each link is the
       link's collision .obj order.
     - Since `sim_engine.cu` was patched to emit per-link vertex ranges
       (derived from `point_id_to_body_id`), each link now has its own
       BodyLoadRecord with distinct vertex_offset / vertex_count. We
       render each link as a separate polyscope mesh by slicing
       `get_vertices()[rec.vertex_offset : rec.vertex_offset+vertex_count]`
       and filtering `get_surface_faces()` down to that link's range.

Earlier versions of this script demonstrated a separate "obj index alignment"
trick (parse the .obj a second time and feed its original face table) which
required `preconditioner_type=0` to disable MAS vertex remap. That was
removed because it produced a different chaotic physics trajectory than
case_26_arm_cloth_semi_implicit.py — same equations, different PCG inner
solver, same family of valid solutions but different folding pattern.

If you need exact `.obj` vertex-index mapping (for downstream rendering
pipelines that depend on it), use `Config(preconditioner_type=0)` and
expect different cloth folding from the basic case_26 — see
`docs/internal/RENDERING_HANDOVER.md` (gitignored).

Run:
    python examples/case_26_render_obj_indices.py
"""

import colorsys
import math
import os
from pathlib import Path
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot


# ---------------------------------------------------------------------------
# Self-contained .obj parser (mirrors what StiffGIPC does internally so the
# indexing matches exactly). Handles `v`, `f i`, `f i/t`, `f i/t/n`, and
# polygon fan-triangulation the same way as load_mesh.cpp / obj_mesh_loader.cpp.
# ---------------------------------------------------------------------------
def load_obj_verts_faces(path: str) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r") as f:
        for raw in f:
            parts = raw.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idxs = [int(tok.split("/")[0]) - 1 for tok in parts[1:]]
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    return (np.asarray(verts, dtype=np.float64),
            np.asarray(faces, dtype=np.int32))


def _make_arm_transform(scale: float = 0.3) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    tf[1, 3] = -0.9
    return tf


def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    # Resolve assets dir relative to this script's location (e.g. when the
    # repo is cloned to ~/Downloads/test-stiff-physics/, this picks up
    # ~/Downloads/test-stiff-physics/assets/). The wheel-shipped data dir
    # only contains scene/abd_system_config.json — it does NOT carry the
    # URDF + mesh assets, which live in the public-repo's assets/ dir.
    ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"

    config = Config(
        dt=0.01,
        cloth_thickness=1e-3,
        cloth_young_modulus=1e4,
        bend_young_modulus=1e3,
        cloth_density=200,
        strain_rate=100,
        soft_motion_rate=1.0,
        poisson_rate=0.49,
        friction_rate=0.4,
        relative_dhat=1e-3,
        joint_strength_ratio=1000.0,
        revolute_driving_strength_ratio=1000.0,
        semi_implicit_enabled=True,
        semi_implicit_beta_tol=1e-3,
        semi_implicit_min_iter=1,
        preconditioner_type=1,
        assets_dir=ASSETS_DIR,
    )

    engine = Engine(config)
    assets_dir = engine.native.get_assets_dir()

    # --- Robot arm (ABD) ---
    arm_urdf = os.path.join(assets_dir, "sim_data/urdf/xarm/xarm7_with_gripper.urdf")
    arm_tf = _make_arm_transform(0.3)
    engine.native.load_urdf(arm_urdf, arm_tf, True, False, 1e7)
    arm_body_count = engine.abd_body_count
    for bid in range(arm_body_count):
        engine.add_ground_collision_skip(bid)

    # --- Shirt (FEM cloth) ---
    shirt_rel = "triMesh/shirt_6436v.obj"
    shirt_path = os.path.join(assets_dir, shirt_rel)
    shirt_scale = 0.5
    shirt_tf = np.eye(4)
    shirt_tf[:3, :3] *= shirt_scale
    shirt_tf[0, 3] = 0.25
    shirt_tf[1, 3] = 0.3
    engine.load_mesh(shirt_rel, dimensions=2, body_type="FEM",
                     transform=shirt_tf, young_modulus=1e2)

    engine.finalize()
    robot = Robot(engine)

    # ------------------------------------------------------------------
    # Build obj-index-aware renderers for each body.
    #
    # `get_all_load_records()` returns one BodyLoadRecord per load call:
    #   - rec.body_type     : 0 = ABD, 1 = FEM
    #   - rec.vertex_offset : first engine vertex id of this body
    #   - rec.vertex_count  : number of vertices in this body
    #   - rec.label         : path / identifier passed at load time
    # For URDF imports, one record per ABD link is emitted, all with the
    # same label = urdf path.
    # ------------------------------------------------------------------
    records = engine.native.get_all_load_records()
    all_verts0 = engine.get_vertices()   # snapshot at rest pose
    surf_faces_all = engine.get_surface_faces().astype(np.int64)

    # ---- 1) Shirt (FEM): per-body slice + engine surface faces ----
    # Use engine's own face table (filtered to shirt vertex range) instead of
    # the original .obj face table. This makes rendering work regardless of
    # preconditioner_type — MAS may reorder FEM vertices, but the engine's
    # surface faces are always in sync with whatever vertex order it uses.
    # (Earlier version of this script paired engine_verts with .obj_faces and
    # required preconditioner_type=0 to keep them aligned; that path produced
    # a different chaotic trajectory than case_26_arm_cloth_semi_implicit.py.)
    shirt_rec = next(r for r in records
                     if r.body_type == 1 and shirt_rel in r.label)
    v_off = shirt_rec.vertex_offset
    v_cnt = shirt_rec.vertex_count

    # Sanity check: shirt vertex count should still match the .obj count
    # regardless of MAS — MAS reorders, doesn't add/drop vertices.
    obj_verts, _obj_faces_unused = load_obj_verts_faces(shirt_path)
    assert v_cnt == len(obj_verts), (
        f"shirt vertex count mismatch: engine {v_cnt} vs .obj {len(obj_verts)}."
        " Topology has been altered (e.g., .stl coord-dedup); not just reordered."
    )

    # Filter engine surface faces to the shirt's vertex range, convert to
    # local (within-shirt) indexing.
    shirt_mask = ((surf_faces_all >= v_off)
                  & (surf_faces_all < v_off + v_cnt)).all(axis=1)
    shirt_faces_local = (surf_faces_all[shirt_mask] - v_off).astype(np.int32)

    engine_shirt_v0 = all_verts0[v_off:v_off + v_cnt]
    shirt_mesh = ps.register_surface_mesh(
        "shirt_engine_faces",
        engine_shirt_v0,
        shirt_faces_local,
        smooth_shade=True,
    )
    shirt_mesh.set_back_face_policy("identical")

    # Per-vertex RGB from normalized rest-pose XYZ — colors are locked to
    # material points (vertex IDs in engine space), so as the shirt deforms
    # each point keeps its color.
    rest_min = engine_shirt_v0.min(axis=0)
    rest_max = engine_shirt_v0.max(axis=0)
    rest_rgb = (engine_shirt_v0 - rest_min) / np.maximum(rest_max - rest_min, 1e-9)
    shirt_mesh.add_color_quantity(
        "rest_xyz_as_rgb",
        rest_rgb,
        defined_on="vertices",
        enabled=True,
    )

    # ---- 2) Arm (ABD): per-link rendering via vertex_offset slicing ----
    #
    # Each link gets its own polyscope mesh so you can verify per-link
    # motion independently under joint sliders (a whole-block render
    # wouldn't let you tell whether body assignments are correct).
    arm_records = [r for r in records if r.body_type == 0]
    n_links = len(arm_records)
    arm_meshes = []  # list of (mesh, v_off, v_cnt)
    for i, rec in enumerate(arm_records):
        lo, hi = rec.vertex_offset, rec.vertex_offset + rec.vertex_count
        mask = ((surf_faces_all >= lo) & (surf_faces_all < hi)).all(axis=1)
        link_faces_local = (surf_faces_all[mask] - lo).astype(np.int32)
        if len(link_faces_local) == 0:
            continue
        link_verts = all_verts0[lo:hi]
        m = ps.register_surface_mesh(
            f"arm_body{rec.body_offset:02d}_v[{lo}:{hi})",
            link_verts,
            link_faces_local,
            smooth_shade=True,
        )
        # VHACD-decomposed collision meshes often have mixed winding —
        # show both sides so inverted triangles don't render as black.
        m.set_back_face_policy("identical")
        # HSV hue ramp across links — saturated enough to tell them apart,
        # deterministic across runs.
        hue = (i / max(n_links, 1))
        m.set_color(colorsys.hsv_to_rgb(hue, 0.65, 0.9))
        arm_meshes.append((m, lo, hi))

    # ---- UI / run loop ----
    running = [False]
    step_count = [0]

    def _begin_window(title: str) -> bool:
        result = psim.Begin(title, True)
        return result[0] if isinstance(result, tuple) else result

    def callback():
        psim.SetNextWindowPos((330, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((380, 0), psim.ImGuiCond_Once)

        if _begin_window("Case 26 — Render by OBJ Indices"):
            if running[0]:
                if psim.Button("Pause"):
                    running[0] = False
            else:
                if psim.Button("Run"):
                    running[0] = True
            psim.SameLine()
            psim.Text(f"Step: {step_count[0]}")
            psim.Separator()

            if robot.revolute_joints:
                psim.Text("Revolute Joints")
                psim.Separator()
                for i, ji in enumerate(robot.revolute_joints):
                    lo_deg = math.degrees(ji.lower_limit)
                    hi_deg = math.degrees(ji.upper_limit)
                    cur = robot.get_revolute_target_deg(i)
                    changed, new_val = psim.SliderFloat(ji.name, cur, lo_deg, hi_deg)
                    if changed:
                        robot.set_revolute_position(i, new_val, degree=True)

            psim.Spacing()
            if psim.Button("Reset All Joints"):
                robot.reset_all()

        psim.End()

        if running[0]:
            engine.step()
            step_count[0] += 1
            all_verts = engine.get_vertices()
            shirt_mesh.update_vertex_positions(all_verts[v_off:v_off + v_cnt])
            for m, lo, hi in arm_meshes:
                m.update_vertex_positions(all_verts[lo:hi])

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
