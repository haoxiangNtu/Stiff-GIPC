#!/usr/bin/env python3
"""Mixed position/torque control in ONE sim (mimics libuipc 39 idea).

Two independent revolute pairs (fixed base + free cube + revolute joint about z),
separated in z so they don't interact:
  - Pair 0 (joint 0): POSITION control  -> set_revolute_target(0, +0.8 rad)
  - Pair 1 (joint 1): TORQUE   control  -> strength(1)=0 + set_revolute_torque(1, +T)

Expect: free0 swings to ~0.8 rad and settles; free1 keeps rotating under torque
(angle grows, same sign as T). Confirms both modes coexist per-joint.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics import Engine, Config

CUBE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ("assets" if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")) else "Assets") + "/sim_data/tetmesh/cube.msh")
DT, N = 0.01, 60


def T(x, y, z):
    m = np.eye(4); m[0, 3], m[1, 3], m[2, 3] = x, y, z; return m


cfg = Config(gravity=(0.0, 0.0, 0.0), dt=DT)
eng = Engine(cfg)

bodies = {}
for name, (x, y, z, bt) in {
    "base0": (0.0, 0.0, 3.0, "Fixed"), "free0": (1.2, 0.0, 3.0, "Free"),
    "base1": (0.0, 0.0, -3.0, "Fixed"), "free1": (1.2, 0.0, -3.0, "Free"),
}.items():
    eng.load_mesh(CUBE, dimensions=3, body_type="ABD", transform=T(x, y, z),
                  young_modulus=1e8, boundary_type=bt)
    r = eng.get_load_records()[-1]
    bodies[name] = (r.body_offset, r.vertex_offset, r.vertex_count)

axis = np.array([0.0, 0.0, 1.0]);
j0 = eng.native.add_revolute_joint(bodies["base0"][0], bodies["free0"][0],
                                   axis, np.array([0.6, 0.0, 3.0]),
                                   -3.0, 3.0, 0.0, "pos_joint")
j1 = eng.native.add_revolute_joint(bodies["base1"][0], bodies["free1"][0],
                                   axis, np.array([0.6, 0.0, -3.0]),
                                   -3.0, 3.0, 0.0, "torque_joint")
eng.finalize()
print(f"joint idx: pos={j0} torque={j1}")

# Mode assignment
eng.native.set_revolute_target(j0, 0.8)          # position control
eng.native.set_revolute_strength(j1, 0.0)        # disable PD on torque joint
eng.native.set_revolute_torque(j1, 50.0)         # pure torque control

def angle(name):
    _, voff, vcnt = bodies[name]
    c = np.asarray(eng.get_vertices()[voff:voff+vcnt]).mean(axis=0)
    # free cube starts at x=1.2 (anchor 0.6) -> offset (+0.6,0); angle about z
    return math.atan2(c[1] - 0.0, c[0] - 0.6)

for f in range(N):
    eng.step()

a0, a1 = angle("free0"), angle("free1")
print(f"\nPair0 POSITION (target 0.8): final angle = {a0:.4f} rad")
print(f"Pair1 TORQUE  (+50 Nm):      final angle = {a1:.4f} rad")
print("\n=== Checks ===")
print(f"position joint tracked ~0.8 (|err|<0.15): {abs(a0-0.8)<0.15}  (a0={a0:.3f})")
print(f"torque joint rotated +dir (a1>0.05): {a1>0.05}  (a1={a1:.3f})")
print(f"two modes coexist in one sim: {abs(a0-0.8)<0.15 and a1>0.05}")
