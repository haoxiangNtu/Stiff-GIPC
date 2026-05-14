#!/usr/bin/env python3
"""Demo: load a closed-triangle OBJ as an ABD body.

Two unit cubes loaded from `assets/triMesh/cube_outward.obj` are stacked
under gravity onto the ground.  Demonstrates the surface-ABD path:

    engine.load_mesh_from_data(verts, faces, verts_per_face=3,
                               dimensions=3, body_type="ABD", ...)

Internally dispatches to `tetMesh.load_surfaceMesh_ABD()` (C++) — the
mass / volume / gravity are computed via the divergence theorem over
the closed surface, no tetrahedralization required.

Parameters intentionally match Kemeng's case_0_box_pipe.py defaults:
    dt = 0.01
    young_modulus = 1e5 (per ABD body)
    friction_rate = 0.4 (Config default)
    semi_implicit_enabled = False (Config default)

Usage (after `pip install stiff_physics`):
    python -m stiff_physics.examples.demo_cube_abd_obj   # or:
    python <path-to>/demo_cube_abd_obj.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Dev-mode prelude: bypass any editable-install hook so this script uses
# the dailyv2 build.  Released wheel users don't need this (the prelude
# file just doesn't exist in the wheel).

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from pathlib import Path
from stiff_physics.engine import Engine, Config

# Per release handbook §5.1: scripts MUST set ASSETS_DIR explicitly and pass
# it to Config(assets_dir=...).  Don't rely on engine default — that breaks
# in wheel-install mode.
ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"
CUBE_OBJ   = "triMesh/cube_outward.obj"   # relative to ASSETS_DIR


def load_obj_triangles(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Tiny .obj parser supporting `v x y z` and triangle `f i j k` only.

    OBJ uses 1-based indexing; we convert to 0-based.  Handles `f a/b/c`
    style by taking just the vertex index.  Quads/n-gons not supported.
    """
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
                        f"{path}: only triangle faces supported, got {len(parts)} verts")
                tri = [int(p.split("/")[0]) - 1 for p in parts]
                tris.append(tri)
    return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int32)


def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    # ABD-friendly config — same parameters as case_0_box_pipe.py (Kemeng
    # original).  Do NOT enable semi-implicit for impact / dynamic scenes;
    # it terminates Newton early before contact is fully resolved (bunny
    # demo bug we previously hit).
    config = Config(
        dt=0.01,
        ground_offset=-0.2,
        assets_dir=ASSETS_DIR,
    )
    engine = Engine(config)

    # Load OBJ once, scale to ~0.1m cube
    cube_v, cube_f = load_obj_triangles(ASSETS_DIR + CUBE_OBJ)
    cube_v = cube_v * 0.1
    print(f"[demo_cube_abd_obj] loaded {CUBE_OBJ}: V={len(cube_v)} T={len(cube_f)}",
          flush=True)

    # Cube A: lower
    T_A = np.eye(4)
    T_A[1, 3] = 0.0
    engine.load_mesh_from_data(
        vertices=cube_v, faces=cube_f,
        verts_per_face=3,                # ← triggers surface-ABD path
        dimensions=3,
        body_type="ABD",
        transform=T_A,
        young_modulus=1e5,               # ← matches case_0
        boundary_type="Free",
    )

    # Cube B: above, no X offset (centered stack)
    T_B = np.eye(4)
    T_B[1, 3] = 0.5
    engine.load_mesh_from_data(
        vertices=cube_v, faces=cube_f,
        verts_per_face=3,
        dimensions=3,
        body_type="ABD",
        transform=T_B,
        young_modulus=1e5,
        boundary_type="Free",
    )

    engine.finalize()
    print(f"[demo_cube_abd_obj] finalized: ABD bodies = {engine.abd_body_count}",
          flush=True)

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts, faces, smooth_shade=False)
    mesh.set_color((0.6, 0.7, 0.85))

    running = [False]
    step = [0]

    def callback():
        psim.SetNextWindowPos((20, 20), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((340, 0), psim.ImGuiCond_Once)
        ret = psim.Begin("Cube ABD-OBJ demo", True)
        if (ret[0] if isinstance(ret, tuple) else ret):
            if running[0]:
                if psim.Button("Pause"): running[0] = False
            else:
                if psim.Button("Run"): running[0] = True
            psim.SameLine()
            psim.Text(f"Step: {step[0]}")
            psim.Spacing()
            psim.Text("2 cubes from triMesh/cube_outward.obj")
            psim.Text("body_type=ABD, verts_per_face=3 -> surface ABD path.")
            psim.Text("Expected: B falls onto A, clean centered stack.")
        psim.End()
        if running[0]:
            engine.step()
            step[0] += 1
            mesh.update_vertex_positions(engine.get_vertices())

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
