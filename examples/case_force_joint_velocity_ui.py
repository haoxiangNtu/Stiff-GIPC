#!/usr/bin/env python3
"""Joint velocity (articulation) control (GUI) — matches the public-library setup.

The public library's external articulation constraint writes
`delta_theta_tilde = omega * dt` each step: a desired angle INCREMENT per step,
i.e. joint VELOCITY control, enforced by the very same joint PD-penalty used for
position driving. We reproduce it faithfully with the existing API: each frame
set the target to (current_angle + omega*dt) — a closed-loop velocity drive.

Scene: two small cubes (scaled 0.4) + revolute joint about Z, LEFT fixed, RIGHT
free, gravity on, no ground. The right cube spins at a commanded angular velocity
(rad/s); flip the sign to reverse, like the +/- pi/6 per second animator.

Run: STIFF_SKIP_CCD_SANITY=1 python examples/case_force_joint_velocity_ui.py
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics import Engine, Config
import polyscope as ps, polyscope.imgui as psim

CUBE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ("assets" if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")) else "Assets") + "/sim_data/tetmesh/cube.msh")
def T(x):                                  # scale 0.4 + translate along X (like uipc pre_transform)
    m = np.eye(4); m[0, 0] = m[1, 1] = m[2, 2] = 0.4; m[0, 3] = x; return m

DT = 0.01
eng = Engine(Config(gravity=(0.0, -9.8, 0.0), dt=DT, ground_offset=-100.0))
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=T(-0.6),
              young_modulus=1e8, boundary_type="Fixed")
lid = eng.get_load_records()[-1].body_offset
eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=T(0.6),
              young_modulus=1e8, boundary_type="Free")
rid = eng.get_load_records()[-1].body_offset
j = eng.native.add_revolute_joint(lid, rid, np.array([0.0, 0.0, 1.0]),
                                  np.array([0.0, 0.0, 0.0]), -100.0, 100.0, 0.0, "rev")
eng.finalize()

st = {"run": False, "frame": 0, "omega": math.pi / 6, "auto_rev": True, "period": 80}  # pi/6 like uipc
ps.init(); ps.set_up_dir("y_up")
mesh = ps.register_surface_mesh("cubes", eng.get_vertices(), eng.get_surface_faces(),
                                color=(0.55, 0.8, 0.8))

def callback():
    _, st["omega"]    = psim.SliderFloat("angular speed |w| (rad/s)", st["omega"], -3.0, 3.0)
    _, st["auto_rev"] = psim.Checkbox("auto reverse every N frames", st["auto_rev"])
    _, st["period"]   = psim.SliderInt("N (period)", st["period"], 20, 200)
    if psim.Button("Run / stop"): st["run"] = not st["run"]
    psim.SameLine(); psim.TextUnformatted("RUNNING" if st["run"] else "paused")
    if st["run"]:
        # auto-reverse: flip the sign of the commanded velocity every `period`
        # frames, so you SEE forward constant-velocity, then reverse, repeating.
        sign = 1.0
        if st["auto_rev"] and (st["frame"] // st["period"]) % 2 == 1:
            sign = -1.0
        w = sign * st["omega"]
        ang = float(np.asarray(eng.get_revolute_current_angles())[j])
        eng.native.set_revolute_target(j, ang + w * DT)   # delta_theta_tilde
        eng.step(); st["frame"] += 1
        mesh.update_vertex_positions(eng.get_vertices())
        ang2 = float(np.asarray(eng.get_revolute_current_angles())[j])
        meas_w = (ang2 - ang) / DT
        dirn = "FORWARD" if w >= 0 else "REVERSE"
        psim.TextUnformatted(f"frame {st['frame']}  {dirn}  cmd_w={w:+.3f} rad/s")
        psim.TextUnformatted(f"angle={ang2:+.4f} rad   measured_w={meas_w:+.3f} rad/s")

ps.set_user_callback(callback)
ps.show()
