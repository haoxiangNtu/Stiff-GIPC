#!/usr/bin/env python3
"""Multi-joint velocity (articulation) control (GUI) — strict map of public test 47.

Mirrors the public library's external-articulation-multijoints test exactly:
  - 3 cubes (scaled 0.4) placed along the Z axis at z = -0.8 / 0.0 / +0.8;
    instance 0 is FIXED, instances 1 and 2 are free. gravity on, no ground.
  - a REVOLUTE joint (axis X, center (0,0,-0.4)) connects cube0 <-> cube1;
  - a PRISMATIC joint (axis Z, center (0,0,0.2)) connects cube1 <-> cube2.
The external articulation drives both at once with per-step increments
(velocity * dt): revolute omega = pi/6 rad/s, prismatic v = 0.1 m/s (the exact
public values). Reproduced with the existing per-step target API.

Run: STIFF_SKIP_CCD_SANITY=1 python examples/case_force_multijoint_velocity_ui.py
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics import Engine, Config
import polyscope as ps, polyscope.imgui as psim

CUBE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ("assets" if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")) else "Assets") + "/sim_data/tetmesh/cube.msh")
def Tz(z):                                 # scale 0.4 + translate along Z (like uipc test 47)
    m = np.eye(4); m[0, 0] = m[1, 1] = m[2, 2] = 0.4; m[2, 3] = z; return m

DT = 0.01
# uipc test 47 sets contact.enable=false; StiffGIPC's exact equivalent is
# skip_all_collision=True (short-circuits all collision paths in the solver).
eng = Engine(Config(gravity=(0.0, -9.8, 0.0), dt=DT, ground_offset=-100.0,
                    skip_all_collision=True))
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=Tz(-0.8),
              young_modulus=1e8, boundary_type="Fixed")     # instance 0 (fixed)
b0 = eng.get_load_records()[-1].body_offset
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=Tz(0.0),
              young_modulus=1e8, boundary_type="Free")       # instance 1
b1 = eng.get_load_records()[-1].body_offset
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=Tz(0.8),
              young_modulus=1e8, boundary_type="Free")       # instance 2
b2 = eng.get_load_records()[-1].body_offset
# revolute cube0 <-> cube1: axis X, center (0,0,-0.4)  (uipc edge {(-0.5,0,-0.4),(0.5,0,-0.4)})
jr = eng.native.add_revolute_joint(b0, b1, np.array([1.0, 0.0, 0.0]),
                                   np.array([0.0, 0.0, -0.4]), -100.0, 100.0, 0.0, "rev")
# prismatic cube1 <-> cube2: axis Z, center (0,0,0.2)  (uipc edge {(0,0,0),(0,0,0.4)})
jp = eng.native.add_prismatic_joint(b1, b2, np.array([0.0, 0.0, 0.2]),
                                    np.array([0.0, 0.0, 1.0]), -2.0, 2.0, "pris")
eng.finalize()

st = {"run": False, "frame": 0, "omega": math.pi / 6, "vlin": 0.1, "ptgt": 0.0}  # uipc 47 exact
ps.init(); ps.set_up_dir("y_up")
mesh = ps.register_surface_mesh("chain", eng.get_vertices(), eng.get_surface_faces(),
                                color=(0.8, 0.75, 0.55))

def callback():
    _, st["omega"] = psim.SliderFloat("revolute w (rad/s)", st["omega"], -2.0, 2.0)
    _, st["vlin"]  = psim.SliderFloat("prismatic v (m/s)", st["vlin"], -1.0, 1.0)
    if psim.Button("Run / stop"): st["run"] = not st["run"]
    psim.SameLine(); psim.TextUnformatted("RUNNING" if st["run"] else "paused")
    if st["run"]:
        ang = float(np.asarray(eng.get_revolute_current_angles())[jr])
        eng.native.set_revolute_target(jr, ang + st["omega"] * DT)   # revolute velocity (pi/6)
        st["ptgt"] += st["vlin"] * DT
        eng.native.set_prismatic_target(jp, st["ptgt"])              # prismatic velocity (0.1)
        eng.step(); st["frame"] += 1
        mesh.update_vertex_positions(eng.get_vertices())
        ang2 = float(np.asarray(eng.get_revolute_current_angles())[jr])
        psim.TextUnformatted(f"frame {st['frame']}")
        psim.TextUnformatted(f"revolute angle={ang2:+.4f} rad   prismatic tgt={st['ptgt']:+.4f} m")

ps.set_user_callback(callback)
ps.show()
