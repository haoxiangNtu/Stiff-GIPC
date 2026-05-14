#!/usr/bin/env python3
"""Demo: load Stanford bunny OBJ as an ABD body.

Two bunnies loaded from `assets/triMesh/bunny_surface.obj` are stacked
under gravity onto the ground.  Same surface-ABD path as
demo_cube_abd_obj.py, but with the irregular bunny geometry instead of
a simple cube — verifies that the surface-ABD code handles complex
closed meshes correctly.

The bunny.obj was extracted once from bunny0.msh (1869 verts / 7356 tets)
via per-tet outward-face detection, normalized to ~0.3m diagonal, and
written as a 1078-vertex / 2152-triangle closed mesh.

Parameters intentionally match Kemeng's case_0_box_pipe.py defaults
(dt=0.01, young=1e5 per body, friction=0.4 default, no semi-implicit).
This is the same "good" config that fixed the earlier sticky-bunny issue.

Usage (after `pip install stiff_physics`):
    python <path-to>/demo_bunny_abd_obj.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from pathlib import Path
from stiff_physics.engine import Engine, Config

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"
BUNNY_OBJ  = "triMesh/bunny_surface.obj"


def load_obj_triangles(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Tiny .obj parser — see demo_cube_abd_obj.py docstring."""
    verts, tris = [], []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) != 3:
                    raise ValueError(
                        f"{path}: only triangle faces supported, got {len(parts)}")
                tri = [int(p.split("/")[0]) - 1 for p in parts]
                tris.append(tri)
    return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int32)


def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    config = Config(
        dt=0.01,
        ground_offset=-0.2,
        assets_dir=ASSETS_DIR,
    )
    engine = Engine(config)

    bunny_v, bunny_f = load_obj_triangles(ASSETS_DIR + BUNNY_OBJ)
    print(f"[demo_bunny_abd_obj] loaded {BUNNY_OBJ}: "
          f"V={len(bunny_v)} T={len(bunny_f)}", flush=True)

    # Sanity check: enclosed volume must be positive for surface-ABD mass calc
    v0 = bunny_v[bunny_f[:, 0]]
    v1 = bunny_v[bunny_f[:, 1]]
    v2 = bunny_v[bunny_f[:, 2]]
    vol = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2)))) / 6.0
    print(f"[demo_bunny_abd_obj] enclosed volume = {vol:.4e} m^3"
          + ("  ✓" if vol > 0 else "  ❌ NEGATIVE, mesh winding wrong"),
          flush=True)

    # Bunny A: lower (lands on ground first)
    T_A = np.eye(4)
    T_A[1, 3] = 0.2
    engine.load_mesh_from_data(
        vertices=bunny_v, faces=bunny_f,
        verts_per_face=3,
        dimensions=3,
        body_type="ABD",
        transform=T_A,
        young_modulus=1e5,
        boundary_type="Free",
    )

    # Bunny B: higher (will land on A)
    T_B = np.eye(4)
    T_B[1, 3] = 0.6
    engine.load_mesh_from_data(
        vertices=bunny_v, faces=bunny_f,
        verts_per_face=3,
        dimensions=3,
        body_type="ABD",
        transform=T_B,
        young_modulus=1e5,
        boundary_type="Free",
    )

    engine.finalize()
    print(f"[demo_bunny_abd_obj] finalized: ABD bodies = {engine.abd_body_count}",
          flush=True)

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts, faces, smooth_shade=True)
    mesh.set_color((0.8, 0.7, 0.7))

    running = [False]
    step = [0]

    def callback():
        psim.SetNextWindowPos((20, 20), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((360, 0), psim.ImGuiCond_Once)
        ret = psim.Begin("Bunny ABD-OBJ demo", True)
        if (ret[0] if isinstance(ret, tuple) else ret):
            if running[0]:
                if psim.Button("Pause"): running[0] = False
            else:
                if psim.Button("Run"): running[0] = True
            psim.SameLine()
            psim.Text(f"Step: {step[0]}")
            psim.Spacing()
            psim.Text("2 bunnies from triMesh/bunny_surface.obj")
            psim.Text("body_type=ABD, verts_per_face=3 -> surface ABD path.")
        psim.End()
        if running[0]:
            engine.step()
            step[0] += 1
            mesh.update_vertex_positions(engine.get_vertices())

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
