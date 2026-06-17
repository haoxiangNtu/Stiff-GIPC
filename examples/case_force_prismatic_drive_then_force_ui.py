#!/usr/bin/env python3
"""Prismatic joint: position driving then external force (GUI) — public-library setup.

Two small cubes (scaled 0.4) + prismatic joint along Z: LEFT fixed (x=-0.6),
RIGHT free (x=+0.6), gravity on, no ground. The joint axis is horizontal (Z), so
gravity is perpendicular to the slide. Two phases on the SAME joint:
  - frames <= 100 : POSITION driving (set_prismatic_target ramps displacement)
  - frames  > 100 : FORCE control    (strength=0, set_prismatic_force)
The right cube is first servo-driven to the target offset (and holds), then a
pure axis force pushes it further along the rail.

Run: STIFF_SKIP_CCD_SANITY=1 python examples/case_force_prismatic_drive_then_force_ui.py
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
              young_modulus=1e8, boundary_type="Fixed")
lid = eng.get_load_records()[-1].body_offset
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=T(0.6),
              young_modulus=1e8, boundary_type="Free")
rid = eng.get_load_records()[-1].body_offset
# prismatic joint along Z (horizontal rail; gravity perpendicular, like uipc)
j = eng.native.add_prismatic_joint(lid, rid, np.array([0.0, 0.0, 0.0]),
                                   np.array([0.0, 0.0, 1.0]), -2.0, 2.0, "pris")
eng.finalize()
# default prismatic position-driving is rate-limited to 0.002 m/frame (a safety
# clamp for FEM softpads); raise it so the velocity drive can move at m/s rates
eng.native.set_max_prismatic_step_per_frame(0.05)

DT = 0.01
DRIVE_END = 100         # uipc kDrivingEndFrame
# uipc 79 uses drive velocity +-10 m/s (very fast); 1.0 m/s here is watchable
st = {"run": False, "frame": 0, "drive_vel": 1.0, "force": 1000.0}
ps.init(); ps.set_up_dir("y_up")
mesh = ps.register_surface_mesh("cubes", eng.get_vertices(), eng.get_surface_faces(),
                                color=(0.7, 0.7, 0.9))

def callback():
    _, st["drive_vel"] = psim.SliderFloat("phase1 drive |v| (m/s)", st["drive_vel"], 0, 10)
    _, st["force"]     = psim.SliderFloat("phase2 |force| (N)", st["force"], 0, 6000)
    if psim.Button("Run / stop"): st["run"] = not st["run"]
    psim.SameLine(); psim.TextUnformatted("RUNNING" if st["run"] else "paused")
    if st["run"]:
        f = st["frame"]
        if f <= DRIVE_END:                 # phase 1: position driving via velocity ramp (uipc 79)
            vel = -st["drive_vel"] if f <= 50 else st["drive_vel"]   # slide one way, then back
            eng.native.set_prismatic_force(j, 0.0)
            cur = float(np.asarray(eng.get_vertices())[np.asarray(eng.get_vertices())[:, 0] > 0][:, 2].mean())
            eng.native.set_prismatic_target(j, cur + vel * DT)
            phase = f"phase1 DRIVE (v={vel:+.1f})"
        else:                               # phase 2: force, reverses at frame 150 (uipc 79)
            frc = -st["force"] if f <= 150 else st["force"]
            eng.native.set_prismatic_strength(j, 0.0)
            eng.native.set_prismatic_force(j, frc)
            phase = f"phase2 FORCE ({frc:+.0f})"
        eng.step(); st["frame"] += 1
        mesh.update_vertex_positions(eng.get_vertices())
        c = np.asarray(eng.get_vertices()).mean(axis=0)
        psim.TextUnformatted(f"frame {st['frame']}  {phase}")
        psim.TextUnformatted(f"right cube along axis: z={c[2]:+.4f}")

ps.set_user_callback(callback)
ps.show()
