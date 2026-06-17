#!/usr/bin/env python3
"""Revolute joint: position driving then external torque (GUI) — public-library setup.

Two small cubes (scaled 0.4) + revolute joint about Z (LEFT fixed, RIGHT free),
gravity off, no ground. Two phases on the SAME joint:
  - frames <= 100 : POSITION driving  (set_revolute_target ramps the angle)
  - frames  > 100 : TORQUE control    (strength=0, set_revolute_torque)
The right cube is first servo-driven to the target angle (and holds there), then
the torque pushes it past the target / spins it freely.

Run: STIFF_SKIP_CCD_SANITY=1 python examples/case_force_revolute_drive_then_torque_ui.py
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics import Engine, Config
import polyscope as ps, polyscope.imgui as psim

CUBE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ("assets" if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")) else "Assets") + "/sim_data/tetmesh/cube.msh")
def T(x):                                  # scale 0.4 + translate along X (like uipc pre_transform)
    m = np.eye(4); m[0, 0] = m[1, 1] = m[2, 2] = 0.4; m[0, 3] = x; return m

eng = Engine(Config(gravity=(0.0, 0.0, 0.0), dt=0.01, ground_offset=-100.0))
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=T(-0.6),
              young_modulus=1e8, boundary_type="Fixed")
lid = eng.get_load_records()[-1].body_offset
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=T(0.6),
              young_modulus=1e8, boundary_type="Free")
rid = eng.get_load_records()[-1].body_offset
j = eng.native.add_revolute_joint(lid, rid, np.array([0.0, 0.0, 1.0]),
                                  np.array([0.0, 0.0, 0.0]), -6.0, 6.0, 0.0, "rev")
eng.finalize()

DT = 0.01
DRIVE_END = 100         # uipc kDrivingEndFrame
st = {"run": False, "frame": 0, "drive_vel": 5.0, "torque": 1000.0}  # uipc: vel +-5 rad/s, torque +-1000
ps.init(); ps.set_up_dir("y_up")
mesh = ps.register_surface_mesh("cubes", eng.get_vertices(), eng.get_surface_faces(),
                                color=(0.6, 0.8, 0.6))

def callback():
    _, st["drive_vel"] = psim.SliderFloat("phase1 drive |w| (rad/s)", st["drive_vel"], 0, 10)
    _, st["torque"]    = psim.SliderFloat("phase2 |torque| (N*m)", st["torque"], 0, 3000)
    if psim.Button("Run / stop"): st["run"] = not st["run"]
    psim.SameLine(); psim.TextUnformatted("RUNNING" if st["run"] else "paused")
    if st["run"]:
        f = st["frame"]
        if f <= DRIVE_END:                 # phase 1: position driving via velocity ramp (uipc 80)
            vel = -st["drive_vel"] if f <= 50 else st["drive_vel"]   # rotate one way, then back
            eng.native.set_revolute_torque(j, 0.0)
            ang = float(np.asarray(eng.get_revolute_current_angles())[j])
            eng.native.set_revolute_target(j, ang + vel * DT)
            phase = f"phase1 DRIVE (w={vel:+.0f})"
        else:                              # phase 2: torque, reverses at frame 150 (uipc 80)
            tau = -st["torque"] if f <= 150 else st["torque"]
            eng.native.set_revolute_strength(j, 0.0)
            eng.native.set_revolute_torque(j, tau)
            phase = f"phase2 TORQUE ({tau:+.0f})"
        eng.step(); st["frame"] += 1
        mesh.update_vertex_positions(eng.get_vertices())
        ang = float(np.asarray(eng.get_revolute_current_angles())[j])
        psim.TextUnformatted(f"frame {st['frame']}  {phase}")
        psim.TextUnformatted(f"joint angle: {ang:+.4f} rad ({ang*180/np.pi:+.1f} deg)")

ps.set_user_callback(callback)
ps.show()
