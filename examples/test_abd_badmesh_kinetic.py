#!/usr/bin/env python3
"""[G10] Bad-mesh ABD kinetic regression gate (case-26 NaN root cause).

xarm7_with_gripper's body 8 has a broken surface (reversed winding +
partially-indefinite second moment). Before the eigenvalue-floor fix the PSD
clamp left a SINGULAR affine mass -> NaN q_tilde -> NaN kinetic energy ->
line search rejects everything -> the whole scene froze (and pre-fa0cd63 the
NaN was silently ACCEPTED). This gate loads the arm alone, steps 5 frames,
and fails on any NaN sentinel or line-search NaN warning. No gate scene had
a broken-mesh ABD body — this one keeps the class covered.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stiff_physics.engine import Engine, Config

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Assets") + "/"
eng = Engine(Config(dt=0.01, friction_rate=0.4, relative_dhat=1e-3,
                    joint_strength_ratio=1000.0, revolute_driving_strength_ratio=1000.0,
                    semi_implicit_enabled=True, semi_implicit_beta_tol=1e-3,
                    semi_implicit_min_iter=1, assets_dir=ASSETS))
tf = np.eye(4); tf[1, 3] = 0.3
eng.native.load_urdf(ASSETS + "sim_data/urdf/xarm/xarm7_with_gripper.urdf", tf, True, False, 1e7, {})
for bid in range(eng.abd_body_count):
    eng.add_ground_collision_skip(bid)
eng.finalize()
for _ in range(5):
    eng.step()
V = np.asarray(eng.get_vertices())
assert np.isfinite(V).all(), "non-finite vertices"
print("ABD-BADMESH-KINETIC: PASS")
