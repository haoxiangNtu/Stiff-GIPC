#!/usr/bin/env python3
"""Prismatic-joint external force (GUI) — matches the public-library setup.

Two small cubes (scaled 0.4) joined by a PRISMATIC joint along the Z axis: LEFT
fixed (x=-0.6), RIGHT free (x=+0.6). Gravity on, no ground. The joint axis is
horizontal (Z), so gravity does not fight the slide. A pure external force along
the axis drives the right cube along the rail: +|F| for frames<=50, then a
larger -|F| (slides one way, then back). Force via set_prismatic_force with
strength=0 (pure force control).

Run: STIFF_SKIP_CCD_SANITY=1 python examples/case_force_prismatic_force_ui.py
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics import Engine, Config
import polyscope as ps, polyscope.imgui as psim

CUBE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ("assets" if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")) else "Assets") + "/sim_data/tetmesh/cube.msh")
def T(x):                                  # scale 0.4 + translate along X (like uipc pre_transform)
    m = np.eye(4); m[0, 0] = m[1, 1] = m[2, 2] = 0.4; m[0, 3] = x; return m

eng = Engine(Config(gravity=(0.0, -9.8, 0.0), dt=0.01, ground_offset=-100.0))
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=T(-0.6),
              young_modulus=1e8, boundary_type="Fixed")           # left fixed
lid = eng.get_load_records()[-1].body_offset
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=T(0.6),
              young_modulus=1e8, boundary_type="Free")            # right free
rid = eng.get_load_records()[-1].body_offset
# prismatic joint along Z (horizontal rail; gravity is perpendicular, like uipc)
j = eng.native.add_prismatic_joint(lid, rid, np.array([0.0, 0.0, 0.0]),
                                   np.array([0.0, 0.0, 1.0]), -2.0, 2.0, "pris")
eng.finalize()
eng.native.set_prismatic_strength(j, 0.0)        # pure force control

st = {"run": False, "frame": 0, "fwd": 1000.0, "rev": 5000.0}
ps.init(); ps.set_up_dir("y_up")
mesh = ps.register_surface_mesh("cubes", eng.get_vertices(), eng.get_surface_faces(),
                                color=(0.6, 0.85, 0.7))

def callback():
    _, st["fwd"] = psim.SliderFloat("phase1 +force (N)", st["fwd"], 0, 4000)
    _, st["rev"] = psim.SliderFloat("phase2 -force (N)", st["rev"], 0, 10000)
    if psim.Button("Run / stop"): st["run"] = not st["run"]
    psim.SameLine(); psim.TextUnformatted("RUNNING" if st["run"] else "paused")
    if st["run"]:
        f = st["fwd"] if st["frame"] <= 50 else -st["rev"]   # reverse at frame 50
        eng.native.set_prismatic_force(j, f)
        eng.step(); st["frame"] += 1
        mesh.update_vertex_positions(eng.get_vertices())
        c = np.asarray(eng.get_vertices()).mean(axis=0)
        phase = "phase1 (+F)" if st["frame"] <= 50 else "phase2 (-F)"
        psim.TextUnformatted(f"frame {st['frame']}  {phase}  F={f:+.0f}")
        psim.TextUnformatted(f"right cube along axis: z={c[2]:+.4f}")

ps.set_user_callback(callback)
ps.show()
