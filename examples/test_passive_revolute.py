#!/usr/bin/env python3
"""[passive joints] Free-swing + limit regression for undriven revolute joints.

Requirement (SimulationRequirement 第一類-2): import articulated objects with
PASSIVE joints. The engine's manual joints are penalty constraints; driving is
opt-in per joint — this test pins down that semantics empirically:

  A) PENDULUM: a cube on a revolute hinge, released HORIZONTAL. If the joint
     is truly passive it swings through the bottom (past -120 deg) and swings
     back up (oscillates). A secretly-driven joint would hold near 0 deg.
  B) LIMITED HINGE: same setup with limits [-0.3, +0.3] rad. The one-sided
     limit penalty must stop the swing near the limit (|theta| < 0.55 rad
     allowing penalty softness) instead of swinging through.

Run:  python3 examples/test_passive_revolute.py
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stiff_physics.engine import Engine, Config

ASSETS = os.path.join(ROOT, "Assets") + "/"
FRAMES = 260


def run(lo, hi, tag):
    cfg = Config(dt=0.01, density=1e3, young_modulus=1e7, friction_rate=0.4,
                 relative_dhat=1e-3, ground_offset=-5.0,   # ground far away
                 joint_strength_ratio=1e3, assets_dir=ASSETS)
    eng = Engine(cfg)
    # cube.msh spans 0.4/side, baked center ~(0, 0.3, 0). scale 0.25 -> 10 cm.
    H = np.array([0.0, 0.5, 0.0])                    # hinge point
    t_anc = np.eye(4); t_anc[:3, :3] *= 0.25
    t_anc[:3, 3] = H + [0.0, 0.10, 0.0] - [0.0, 0.075, 0.0]
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD",
                  transform=t_anc, boundary_type="Fixed")
    # pendulum starts HORIZONTAL: center at H + (0.15, 0, 0)
    t_pen = np.eye(4); t_pen[:3, :3] *= 0.25
    t_pen[:3, 3] = H + [0.15, 0.0, 0.0] - [0.0, 0.075, 0.0]
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD",
                  transform=t_pen)
    eng.add_collision_exclusion(0, 1)                # hinge bodies overlap freely
    eng.add_revolute_joint(0, 1, world_axis=[0.0, 0.0, 1.0],
                           joint_pos=H.tolist(),
                           lower_limit=lo, upper_limit=hi,
                           passive=True)
    eng.finalize()
    recs = eng.get_load_records()
    s1, c1 = recs[1].vertex_offset, recs[1].vertex_count

    thetas = []
    for fr in range(FRAMES):
        eng.step()
        c = np.asarray(eng.get_vertices())[s1:s1+c1].mean(0)
        th = float(np.arctan2(c[1] - H[1], c[0] - H[0]))   # 0 = horizontal start
        thetas.append(th)
    th = np.array(thetas)
    assert np.isfinite(th).all(), f"{tag}: diverged"
    print(f"[{tag}] theta: start {np.degrees(th[0]):+.1f}  min {np.degrees(th.min()):+.1f}  "
          f"final {np.degrees(th[-1]):+.1f}  (deg)")
    return th


ok = True

# A) free pendulum (wide limits = effectively none)
th = run(-3.0, 3.0, "A-free")
if not (np.degrees(th.min()) < -120):
    print("FAIL A: pendulum did not swing past -120 deg — joint is NOT passive"); ok = False
# swings back up after the first pass through the bottom
i_min = int(np.argmin(th))
if not (np.degrees(th[i_min:].max() - th.min()) > 40):
    print("FAIL A: no swing-back — motion looks clamped/dead"); ok = False

# B) limited hinge: swing must stop near the -0.3 rad limit
th = run(-0.3, 0.3, "B-limited")
if not (np.degrees(th.min()) > -32):     # -0.3 rad = -17.2 deg; allow penalty softness to ~-32
    print(f"FAIL B: swing passed the lower limit ({np.degrees(th.min()):.1f} deg) — limits not enforced"); ok = False

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
