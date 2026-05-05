#!/usr/bin/env python3
"""Demo: per-body data extraction via BodyView API.

Proves that engine.get_bodies() / get_abd_body(id) / get_fem_body(id) /
get_vertex_body_ids() correctly partition the global mesh into individual
ABD bodies + FEM bodies, with stable URDF link names as labels.

Visual proof: each body is registered as a separate Polyscope mesh with
its own color. If the indexing is wrong, colors will mix or bodies will
move together; if right, each arm link moves independently as the
recorded trajectory plays back.

Console proof: prints all three usage patterns side by side.

Run from gripperstr worktree:
    PYTHONPATH=. python examples/demo_body_view.py
        [--qpos /tmp/replay_user/qpos.h5]   (optional, plays trajectory)
        [--no-cloth]                         (skip the FEM shirt body)
"""

import argparse, math, sys, time, colorsys
from pathlib import Path
import numpy as np

import polyscope as ps
import polyscope.imgui as psim

from stiff_physics.engine import Engine, Config, BodyView
from stiff_physics.robot import Robot

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"
DEFAULT_QPOS = "/tmp/replay_user/qpos.h5"


def make_arm_tf(scale: float = 0.3):
    from scipy.spatial.transform import Rotation
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    tf[1, 3] = -0.95
    return tf


def hsv_palette(n: int) -> list[tuple[float, float, float]]:
    """Evenly spaced colors around the hue wheel, fixed saturation+value."""
    return [colorsys.hsv_to_rgb(i / max(n, 1), 0.7, 0.95) for i in range(n)]


def print_body_inventory(engine: Engine):
    """Pattern 1: iterate engine.get_bodies()"""
    print("\n" + "=" * 72)
    print("Pattern 1 — engine.get_bodies()  (mixed ABD+FEM in load order)")
    print("=" * 72)
    bodies = engine.get_bodies()
    print(f"{'idx':>3}  {'kind':>4}  {'body_id':>7}  {'label':<32}  {'verts':>6}  {'faces':>6}")
    for i, b in enumerate(bodies):
        n_v = b.vertex_count
        n_f = b.get_surface_faces().shape[0]
        # Truncate label for display
        lbl = b.label
        if len(lbl) > 32:
            lbl = "..." + lbl[-29:]
        print(f"{i:>3}  {b.kind:>4}  {b.body_id:>7}  {lbl:<32}  {n_v:>6}  {n_f:>6}")
    return bodies


def print_targeted_access(engine: Engine):
    """Pattern 2: engine.get_abd_body(id) / get_fem_body(id)"""
    print("\n" + "=" * 72)
    print("Pattern 2 — engine.get_abd_body(id) / get_fem_body(id)  (targeted)")
    print("=" * 72)
    # Sample a few bodies
    for body_id in [0, 8, 14]:  # base, gripper_base, right_finger
        try:
            b = engine.get_abd_body(body_id)
            verts = b.get_vertices()
            print(f"  get_abd_body({body_id}) → '{b.label}'  "
                  f"verts={verts.shape}  centroid={verts.mean(axis=0).round(3)}")
        except IndexError as e:
            print(f"  get_abd_body({body_id}) → MISSING ({e})")
    try:
        b = engine.get_fem_body(0)
        verts = b.get_vertices()
        print(f"  get_fem_body(0)         → '{Path(b.label).name}'  "
              f"verts={verts.shape}  centroid={verts.mean(axis=0).round(3)}")
    except IndexError:
        print("  get_fem_body(0)         → MISSING (no FEM bodies in scene)")


def print_mask_usage(engine: Engine):
    """Pattern 3: engine.get_vertex_body_ids() for boolean masks"""
    print("\n" + "=" * 72)
    print("Pattern 3 — engine.get_vertex_body_ids()  (per-vertex mask)")
    print("=" * 72)
    ids = engine.get_vertex_body_ids()
    verts = engine.get_vertices()
    n_total = len(verts)
    n_abd = int((ids[:, 0] == 0).sum())
    n_fem = int((ids[:, 0] == 1).sum())
    print(f"  ids shape = {ids.shape}  (column 0 = body_type, column 1 = body_offset)")
    print(f"  total verts:        {n_total}")
    print(f"  ABD verts (mask):   {n_abd}    "
          f"({n_abd / n_total * 100:5.1f}%)   "
          f"verts[(ids[:,0]==0)] → arm")
    print(f"  FEM verts (mask):   {n_fem}    "
          f"({n_fem / n_total * 100:5.1f}%)   "
          f"verts[(ids[:,0]==1)] → cloth")
    if n_abd > 0:
        link3_mask = (ids[:, 0] == 0) & (ids[:, 1] == 3)
        n_link3 = int(link3_mask.sum())
        if n_link3:
            print(f"  link3 verts (mask): {n_link3}     "
                  f"verts[(ids[:,0]==0) & (ids[:,1]==3)] → just one link")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", default=DEFAULT_QPOS)
    ap.add_argument("--no-cloth", action="store_true")
    ap.add_argument("--scale", type=float, default=0.3)
    args = ap.parse_args()
    use_cloth = not args.no_cloth

    # Load qpos if available so the demo animates
    qpos_all = None
    qpos_path = Path(args.qpos)
    if qpos_path.exists():
        try:
            import h5py
            with h5py.File(qpos_path, "r") as f:
                qpos_all = f["qpos"][:]
            print(f"Loaded {len(qpos_all)} qpos frames from {qpos_path}")
        except Exception as e:
            print(f"WARNING: failed to load qpos ({e}); will run static")
    else:
        print(f"NOTE: {qpos_path} not found; running static (no animation)")

    cfg = Config(
        dt=0.020, cloth_thickness=1e-3, cloth_young_modulus=1e4,
        bend_young_modulus=1e3, cloth_density=200, strain_rate=100,
        soft_motion_rate=1.0, poisson_rate=0.49, friction_rate=0.4,
        relative_dhat=1e-3,
        joint_strength_ratio=200.0, revolute_driving_strength_ratio=200.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2, pcg_tol=1e-4,
        assets_dir=ASSETS_DIR,
    )

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_window_size(1280, 720)
    ps.set_autocenter_structures(False)
    ps.set_autoscale_structures(False)

    engine = Engine(cfg)
    assets = engine.native.get_assets_dir()
    engine.native.load_urdf(
        assets + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
        make_arm_tf(args.scale), True, False, 1e7,
    )
    for bid in range(engine.abd_body_count):
        engine.add_ground_collision_skip(bid)
    if use_cloth:
        shirt_tf = np.eye(4); shirt_tf[:3, :3] *= 0.5
        shirt_tf[0, 3] = 0.25; shirt_tf[1, 3] = 0.3
        engine.load_mesh("triMesh/shirt_6436v.obj", dimensions=2,
                         body_type="FEM", transform=shirt_tf, young_modulus=1e2)
    engine.finalize()
    robot = Robot(engine)
    if hasattr(robot, "set_gripper_strength"):
        robot.set_gripper_strength(0.1)

    # ---- Demonstrate all three usage patterns at startup ----
    bodies = print_body_inventory(engine)
    print_targeted_access(engine)
    print_mask_usage(engine)
    print()

    # ---- Visual proof: register each body as its own Polyscope mesh ----
    # If indexing is wrong, faces will reach into other bodies' verts
    # (visible as random connecting triangles) or bodies will not animate
    # independently. If correct, each link is its own colored mesh.
    palette = hsv_palette(len(bodies))
    ps_meshes = []
    for b, color in zip(bodies, palette):
        verts = b.get_vertices()
        faces = b.get_surface_faces()        # default local_indices=True
        if len(faces) == 0:
            ps_meshes.append(None)
            continue
        name = f"{b.kind}_{b.body_id}_{Path(b.label).stem[:20]}"
        # xarm7 collision .obj has inconsistent face winding (~10 of link3's
        # 100 faces point inward; xarm_gripper_base_link has NEGATIVE signed
        # volume — see physics impact note in commit message). Smooth shading
        # interpolates inward normals into vertex normals making those areas
        # render transparent; flat shading + identical back-face renders both
        # sides correctly.
        m = ps.register_surface_mesh(name, verts, faces, smooth_shade=False)
        m.set_color(color)
        m.set_back_face_policy("identical")
        ps_meshes.append(m)

    ps.reset_camera_to_home_view()
    ps.look_at(camera_location=(0.45, -0.05, 0.45), target=(0.05, -0.65, 0.0))

    state = {"frame": 0, "running": qpos_all is not None,
             "ms": 0.0, "loops": 0}

    def cb():
        psim.SetNextWindowPos((1280 - 480 - 14, 14), psim.ImGuiCond_Always)
        psim.SetNextWindowSize((480, 0), psim.ImGuiCond_Always)
        opened = psim.Begin("BodyView Demo", True)
        if isinstance(opened, tuple):
            opened = opened[0]
        if opened:
            getattr(psim, "SetWindowFontScale", lambda x: None)(1.5)
            psim.TextColored((0.7, 0.95, 0.7, 1.0),
                             f"{len(bodies)} bodies registered as separate meshes")
            getattr(psim, "SetWindowFontScale", lambda x: None)(1.0)
            psim.Separator()
            n_abd = sum(1 for b in bodies if b.kind == "ABD")
            n_fem = sum(1 for b in bodies if b.kind == "FEM")
            psim.Text(f"  ABD: {n_abd}   FEM: {n_fem}")
            psim.Separator()
            if qpos_all is not None:
                psim.Text(f"Frame {state['frame']}/{len(qpos_all)}   "
                          f"loops={state['loops']}   step={state['ms']:.1f} ms")
                if state["running"]:
                    if psim.Button("Pause"):
                        state["running"] = False
                else:
                    if psim.Button("Run"):
                        state["running"] = True
            else:
                psim.Text("(static — no qpos file)")
            psim.Separator()
            psim.TextColored((0.6, 0.8, 1.0, 1.0), "Body legend:")
            for b in bodies[:8]:
                psim.Text(f"  {b.kind} {b.body_id:>2}  {b.label[:32]}")
            if len(bodies) > 8:
                psim.Text(f"  ... +{len(bodies) - 8} more")
        psim.End()

        if state["running"] and qpos_all is not None:
            f = state["frame"]
            if f >= len(qpos_all):
                state["frame"] = 0; state["loops"] += 1; f = 0
            a = qpos_all[f]
            for i in range(1, 8):
                robot.set_revolute_position(i, float(a[i - 1]), degree=False)
            for jid in [0, 8, 9, 10, 11, 12]:
                robot.set_revolute_position(jid, float(a[7]), degree=False)
            t0 = time.perf_counter()
            engine.step()
            state["ms"] = (time.perf_counter() - t0) * 1000.0
            state["frame"] = f + 1
            # Update each body's mesh from its own slice — cleanest API usage
            for b, m in zip(bodies, ps_meshes):
                if m is not None:
                    m.update_vertex_positions(b.get_vertices())

    ps.set_user_callback(cb)
    ps.show()


if __name__ == "__main__":
    main()
