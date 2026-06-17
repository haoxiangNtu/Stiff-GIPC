#!/usr/bin/env python3
"""Single-body external force (GUI) — matches the public-library setup.

A free ABD cube (scaled 0.4), gravity off, no ground, driven by a FULL 12-DOF
external wrench (set_body_external_wrench), exactly like the public library's
body-force test:
  - linear part: force_dir = (cos(orbit), 0, sin(orbit)) * |F|, the direction
    slowly rotates in the xz-plane (it is NOT a centripetal force, so the cube
    drifts along a slowly-curving path — it does not trace a fixed circle);
  - affine part: w[5]=+spin, w[9]=-spin, a torque that spins the cube about Y.
So the correct effect is a slowly drifting AND spinning cube. (The public test
uses |F|=10, spin=0.01, which is very subtle; here both are sliders with larger
defaults so the motion is clearly visible.)

Run: STIFF_SKIP_CCD_SANITY=1 python examples/case_force_body_force_ui.py
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics import Engine, Config
import polyscope as ps, polyscope.imgui as psim

CUBE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ("assets" if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")) else "Assets") + "/sim_data/tetmesh/cube.msh")
T = np.eye(4); T[0, 0] = T[1, 1] = T[2, 2] = 0.4   # scale 0.4 (like uipc pre_transform)

eng = Engine(Config(gravity=(0.0, 0.0, 0.0), dt=0.01, ground_offset=-100.0))
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=T,
              young_modulus=1e8, boundary_type="Free")
bid = eng.get_load_records()[-1].body_offset
eng.finalize()

st = {"run": False, "frame": 0, "mag": 40.0, "orbit_speed": 0.2, "spin": 2.0}  # orbit 0.2 like uipc
ps.init(); ps.set_up_dir("y_up")
mesh = ps.register_surface_mesh("cube", eng.get_vertices(), eng.get_surface_faces(),
                                color=(0.85, 0.6, 0.4))

def callback():
    _, st["mag"]         = psim.SliderFloat("linear |F| (N)", st["mag"], 0, 100)
    _, st["orbit_speed"] = psim.SliderFloat("orbit speed (rad/s)", st["orbit_speed"], 0, 1.0)
    _, st["spin"]        = psim.SliderFloat("spin torque (about Y)", st["spin"], 0, 10)
    if psim.Button("Run / stop"): st["run"] = not st["run"]
    psim.SameLine(); psim.TextUnformatted("RUNNING" if st["run"] else "paused")
    if st["run"]:
        orbit = st["frame"] * st["orbit_speed"] * 0.01
        fx, fz = st["mag"]*math.cos(orbit), st["mag"]*math.sin(orbit)
        w = np.zeros(12)
        w[0], w[2] = fx, fz                 # linear orbiting force (xz-plane)
        w[5], w[9] = st["spin"], -st["spin"]  # affine spin torque about Y
        eng.native.set_body_external_wrench(bid, w)
        eng.step(); st["frame"] += 1
        mesh.update_vertex_positions(eng.get_vertices())
        c = np.asarray(eng.get_vertices()).mean(axis=0)
        psim.TextUnformatted(f"frame {st['frame']}  F=({fx:+.1f},0,{fz:+.1f}) spin={st['spin']:.1f}")
        psim.TextUnformatted(f"cube center ({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})")

ps.set_user_callback(callback)
ps.show()
