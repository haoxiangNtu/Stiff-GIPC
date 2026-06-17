#!/usr/bin/env python3
"""Revolute-joint external torque (GUI) — matches the public-library setup.

Two small cubes (scaled 0.4) joined by a revolute joint about Z: LEFT fixed
(x=-0.6), RIGHT free (x=+0.6). Gravity on, no ground plane (free pendulum).
A pure external torque about the joint axis swings the right cube: -|tau| for
the first 50 frames, then +|tau| — the right cube swings down past the bottom,
then reverses and swings up to the other side. Torque via set_revolute_torque
with strength=0 (pure torque control).

Run: STIFF_SKIP_CCD_SANITY=1 python examples/case_force_revolute_torque_ui.py
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics import Engine, Config
import polyscope as ps, polyscope.imgui as psim

CUBE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ("assets" if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")) else "Assets") + "/sim_data/tetmesh/cube.msh")
def T(x):                                  # scale 0.4 + translate along X (like uipc pre_transform)
    m = np.eye(4); m[0, 0] = m[1, 1] = m[2, 2] = 0.4; m[0, 3] = x; return m

# gravity on; push the implicit ground far below so the cubes swing freely
eng = Engine(Config(gravity=(0.0, -9.8, 0.0), dt=0.01, ground_offset=-100.0))
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=T(-0.6),
              young_modulus=1e8, boundary_type="Fixed")            # left fixed
lid = eng.get_load_records()[-1].body_offset
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=T(0.6),
              young_modulus=1e8, boundary_type="Free")             # right free
rid = eng.get_load_records()[-1].body_offset
j = eng.native.add_revolute_joint(lid, rid, np.array([0.0, 0.0, 1.0]),
                                  np.array([0.0, 0.0, 0.0]), -6.0, 6.0, 0.0, "rev")
eng.finalize()
eng.native.set_revolute_strength(j, 0.0)        # pure torque

st = {"run": False, "frame": 0, "mag": 1000.0}
ps.init(); ps.set_up_dir("y_up")
mesh = ps.register_surface_mesh("cubes", eng.get_vertices(), eng.get_surface_faces(),
                                color=(0.55, 0.7, 0.85))

def callback():
    _, st["mag"] = psim.SliderFloat("|torque| (N*m)", st["mag"], 0, 3000)
    if psim.Button("Run / stop"): st["run"] = not st["run"]
    psim.SameLine(); psim.TextUnformatted("RUNNING" if st["run"] else "paused")
    if st["run"]:
        tau = -st["mag"] if st["frame"] <= 50 else st["mag"]   # reverse at frame 50
        eng.native.set_revolute_torque(j, tau)
        eng.step(); st["frame"] += 1
        mesh.update_vertex_positions(eng.get_vertices())
        ang = float(np.asarray(eng.get_revolute_current_angles())[j])
        phase = "phase1 (-tau)" if st["frame"] <= 50 else "phase2 (+tau)"
        psim.TextUnformatted(f"frame {st['frame']}  {phase}  tau={tau:+.0f}")
        psim.TextUnformatted(f"joint angle: {ang:+.4f} rad ({ang*180/np.pi:+.1f} deg)")

ps.set_user_callback(callback)
ps.show()
