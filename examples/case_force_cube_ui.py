#!/usr/bin/env python3
"""GUI: a free rigid (ABD) cube driven by set_body_external_force.

Drag the fx/fy/fz sliders and hit "Run" — the cube accelerates in the force
direction (a = M^-1 F, fed into q_tilde like gravity). Gravity is off so the
force is isolated. This is the per-body external-force primitive.

Run:
    STIFF_SKIP_CCD_SANITY=1 python examples/case_force_cube_ui.py
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics import Engine, Config
import polyscope as ps
import polyscope.imgui as psim

CUBE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ("assets" if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")) else "Assets") + "/sim_data/tetmesh/cube.msh")

eng = Engine(Config(gravity=(0.0, 0.0, 0.0), dt=0.01))
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", young_modulus=1e8, boundary_type="Free")
bid = eng.get_load_records()[-1].body_offset
eng.finalize()

state = {"run": False, "fx": 0.0, "fy": 0.0, "fz": 0.0}

ps.init()
ps.set_up_dir("y_up")
mesh = ps.register_surface_mesh("cube", eng.get_vertices(), eng.get_surface_faces(),
                                color=(0.85, 0.6, 0.4))


def callback():
    chg = False
    c, state["fx"] = psim.SliderFloat("force x (N)", state["fx"], -50, 50); chg |= c
    c, state["fy"] = psim.SliderFloat("force y (N)", state["fy"], -50, 50); chg |= c
    c, state["fz"] = psim.SliderFloat("force z (N)", state["fz"], -50, 50); chg |= c
    if chg:
        eng.native.set_body_external_force(bid, state["fx"], state["fy"], state["fz"])
    if psim.Button("Run / stop"):
        state["run"] = not state["run"]
    psim.SameLine(); psim.TextUnformatted("RUNNING" if state["run"] else "paused")
    if psim.Button("reset force (0,0,0)"):
        state["fx"] = state["fy"] = state["fz"] = 0.0
        eng.native.set_body_external_force(bid, 0.0, 0.0, 0.0)
    if state["run"]:
        eng.step()
        mesh.update_vertex_positions(eng.get_vertices())
        c = np.asarray(eng.get_vertices()).mean(axis=0)
        psim.TextUnformatted(f"cube center: ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})")


ps.set_user_callback(callback)
ps.show()
