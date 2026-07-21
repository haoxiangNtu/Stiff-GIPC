#!/usr/bin/env python3
"""[kick root-cause] A/B for the ABD block-preconditioner accumulate fix.

Scene (mirrors the fd drawer-kicks-toys pathology minimally): a light 5 cm ABD
cube rests on the ground; a 20 cm ABD pusher on a slowly driven prismatic
joint plows into it (~0.07 m/s). The buggy upstream preconditioner scatter
(assignment instead of accumulation: mass seed wiped + duplicate contact
triplets racing) degenerates the light body's 12x12 block under contact, and
inverse(P) amplifies its soft rotation mode into the search direction -> the
toy is kicked at many m/s. With the accumulate fix the toy should track the
pusher speed (libuipc reference behavior: peak ~= pusher speed).

Run:
  STIFF_ABD_PRECOND_LEGACY=1 python3 examples/test_kick_abd_precond.py   # A: old
  python3 examples/test_kick_abd_precond.py                              # B: fixed
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stiff_physics.engine import Engine, Config

FRAMES  = 220
RAMP    = 160
TRAVEL  = 0.16     # m, -> ~0.1 m/s average drive speed
tag = "A-legacy" if os.environ.get("STIFF_ABD_PRECOND_LEGACY") else "B-fixed"

cfg = Config(
    dt=0.01, density=1e3, young_modulus=1e7, poisson_rate=0.49,
    friction_rate=0.4, relative_dhat=1e-3, absolute_dhat=0.0019,
    joint_strength_ratio=1e2, prismatic_strength_ratio=2000,
    prismatic_driving_strength_ratio=1e2,
    max_prismatic_step_per_frame=0.002,
    ground_offset=0.0,
    assets_dir=os.path.join(ROOT, "Assets") + "/",
)
eng = Engine(cfg)

# cube.msh spans 0.4 on a side, centered ~(0, 0.3, 0)ish -> use transforms to place.
# toy: 5 cm light cube resting on ground
t_toy = np.eye(4); t_toy[:3, :3] *= 0.125; t_toy[1, 3] = -0.0105  # base ~2 mm above ground
eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD", transform=t_toy)
# anchor: fixed cube well above, holds the prismatic joint
t_anc = np.eye(4); t_anc[:3, :3] *= 0.125; t_anc[0, 3] = -0.25; t_anc[1, 3] = 0.30
eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD", transform=t_anc,
              boundary_type="Fixed")
# pusher: 20 cm cube, base ~5 mm above ground, to the -x side of the toy
t_push = np.eye(4); t_push[:3, :3] *= 0.5; t_push[0, 3] = -0.20; t_push[1, 3] = -0.045
eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD", transform=t_push)

j = eng.add_prismatic_joint(1, 2, world_center=[-0.20, 0.105, 0.0],
                            world_axis=[1.0, 0.0, 0.0],
                            lower_limit=0.0, upper_limit=0.2)
eng.finalize()

recs = eng.get_load_records()
s0, c0 = recs[0].vertex_offset, recs[0].vertex_count      # toy
s2, c2 = recs[2].vertex_offset, recs[2].vertex_count      # pusher

V = np.asarray(eng.get_vertices())
prev = V[s0:s0+c0].copy()
peak, peak_fr = 0.0, -1
for fr in range(FRAMES):
    eng.native.set_prismatic_target(0, TRAVEL * min(1.0, (fr + 1) / RAMP))
    eng.step()
    V = np.asarray(eng.get_vertices())
    T = V[s0:s0+c0]
    if not np.isfinite(T).all():
        print(f"[{tag}] fr={fr} NON-FINITE toy state"); break
    spd = float(np.linalg.norm(T - prev, axis=1).max() / 0.01)
    prev = T.copy()
    if spd > peak: peak, peak_fr = spd, fr
    if spd > 0.5 or fr % 25 == 0:
        c = T.mean(0); px = V[s2:s2+c2, 0].mean(); tmin = T[:,1].min()
        print(f"[{tag}] fr={fr:3d} toy_v={spd:8.3f} m/s toy=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) miny={tmin:+.4f} pusher_x={px:+.3f}",
              flush=True)

print(f"[{tag}] SUMMARY PEAK toy vertex speed = {peak:.3f} m/s at frame {peak_fr} "
      f"(drive ~0.067 m/s; libuipc reference peak ~0.075 m/s; kick pathology = several m/s)")
ok = np.isfinite(peak) and peak < 0.5
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
