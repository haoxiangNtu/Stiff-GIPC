#!/usr/bin/env python3
"""[M0 sentinel -> v0.8.4.2 regression] D7 ghost-friction sentinel: PASS = post-release travel < 5 mm, no dt growth.

M0-2 D7 repro: finger presses a cube's top and drags it sideways via friction,
then LIFTS OFF while continuing to move. Watch the cube's post-release travel.
Ghost-friction symptom: cube keeps being dragged after contact is gone.
Run at dt=0.01 and dt=0.02 (mechanism predicts worse at larger dt)."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from stiff_physics.engine import Engine, Config

def run(dt):
    cfg = Config(dt=dt, density=1e3, young_modulus=1e7, poisson_rate=0.49,
                 friction_rate=0.6, relative_dhat=1e-3, ground_offset=0.0,
                 prismatic_strength_ratio=4000, prismatic_driving_strength_ratio=200,
                 max_prismatic_step_per_frame=0.004,
                 assets_dir=os.path.join(ROOT, "Assets") + "/")
    eng = Engine(cfg)
    t_cube = np.eye(4); t_cube[:3,:3] *= 0.5; t_cube[1,3] = -0.0495
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD", transform=t_cube)  # body0 cube on ground
    t_anc = np.eye(4); t_anc[:3,:3] *= 0.2; t_anc[1,3] = 0.8
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD", transform=t_anc, boundary_type="Fixed")  # 1 anchor
    t_lnk = np.eye(4); t_lnk[:3,:3] *= 0.2; t_lnk[1,3] = 0.45
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD", transform=t_lnk)  # 2 vertical link
    t_fin = np.eye(4); t_fin[:3,:3] *= 0.4; t_fin[1,3] = 0.235   # finger above cube top (cube top ~0.151)
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD", transform=t_fin)  # 3 finger
    eng.native.add_collision_exclusion(1, 2); eng.native.add_collision_exclusion(2, 3)
    jv = eng.add_prismatic_joint(1, 2, world_center=[0.0, 0.6, 0.0], world_axis=[0.0,-1.0,0.0],
                                 lower_limit=-0.2, upper_limit=0.4)
    jh = eng.add_prismatic_joint(2, 3, world_center=[0.0, 0.35, 0.0], world_axis=[1.0,0.0,0.0],
                                 lower_limit=-0.5, upper_limit=0.5)
    eng.finalize()
    recs = eng.get_load_records(); s0, c0 = recs[0].vertex_offset, recs[0].vertex_count
    P1 = int(round(0.6/dt)); P2 = int(round(0.8/dt)); P3 = int(round(0.4/dt))  # 0.6s/0.8s/0.4s
    cube_x = lambda: float(np.asarray(eng.get_vertices())[s0:s0+c0,0].mean())
    # phase A: press down at 0.2 m/s of target motion (physical time based)
    for fr in range(P1):
        t = fr*dt
        eng.native.set_prismatic_target(jv, min(0.2*t, 0.10))
        eng.step()
    xA = cube_x()
    # phase B: drag sideways at 0.35 m/s while pressed
    for fr in range(P2):
        t = fr*dt
        eng.native.set_prismatic_target(jh, min(0.35*t, 0.25))
        eng.step()
    xB = cube_x()
    # phase C: LIFT OFF fast (vertical target retracts) while continuing sideways
    xs = []
    for fr in range(P3):
        t = fr*dt
        eng.native.set_prismatic_target(jv, max(0.10 - 2.0*t, -0.1))     # fast lift (2 m/s)
        eng.native.set_prismatic_target(jh, 0.25 + min(0.35*t, 0.14))    # keep moving sideways
        eng.step(); xs.append(cube_x())
    drag_during = xB - xA
    post = np.array(xs) - xB
    # settle distance after release (last-5 mean)
    settle = float(post[-5:].mean())
    print(f"[d7 dt={dt}] dragged_while_pressed={drag_during*1000:.1f}mm  post_release_travel={settle*1000:.2f}mm", flush=True)
    return settle

s1 = run(0.01); s2 = run(0.02)
ghost = abs(s2) > 2 * abs(s1) and abs(s2) > 0.002
print(f"[d7] VERDICT: post-release travel dt=0.01: {s1*1000:.2f}mm, dt=0.02: {s2*1000:.2f}mm "
      f"-> {'GHOST DRAG REPRODUCED (grows with dt)' if ghost else ('mild/none' if abs(s2) < 0.005 else 'present')}", flush=True)
ok = (not ghost) and abs(s2) < 0.005
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
