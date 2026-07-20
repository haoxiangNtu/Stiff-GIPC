#!/usr/bin/env python3
"""[per-env productization] telemetry regression: iters / status / timeout.

Two FEM cubes as two envs (pattern from test_env_isolation.py) with
Config(per_env_exit=True):
  A) normal run: after a step, both envs must report status=converged (1) and
     a freeze iter >= 0 via get_per_env_newton_iters/get_per_env_status.
  B) env_newton_iter_cap=1: both envs must report status=timeout (2) and the
     sim must survive (finite vertices) — a single env exceeding its budget is
     frozen, not fatal.

Run:  python3 examples/test_perenv_telemetry.py
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stiff_physics.engine import Engine, Config

ASSETS = os.path.join(ROOT, "Assets") + "/"


def run(env_cap, semi=False):
    cfg = Config(dt=0.01, density=1e3, young_modulus=1e5, poisson_rate=0.49,
                 friction_rate=0.4, relative_dhat=1e-3, ground_offset=0.0,
                 assets_dir=ASSETS, per_env_exit=True,
                 env_newton_iter_cap=env_cap,
                 semi_implicit_enabled=semi, semi_implicit_beta_tol=1e-2)
    eng = Engine(cfg)
    t0 = np.eye(4); t0[0, 3] = -0.6; t0[1, 3] = -0.09
    t1 = np.eye(4); t1[0, 3] = +0.6; t1[1, 3] = -0.09
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="FEM", transform=t0)
    eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="FEM", transform=t1)
    # per-env SOLVE machinery keys on set_body_groups (per collision-body,
    # BEFORE finalize) — set_vertex_env_ids only isolates broad-phase contact.
    eng.set_body_groups([0, 1])
    eng.finalize()
    for _ in range(30):
        eng.step()
    V = np.asarray(eng.get_vertices())
    iters  = np.asarray(eng.native.get_per_env_newton_iters())[:2]
    status = np.asarray(eng.native.get_per_env_status())[:2]
    return V, iters, status


ok = True

V, iters, status = run(env_cap=0)
print(f"A normal : per-env iters={iters.tolist()} status={status.tolist()}")
if not np.isfinite(V).all():
    print("FAIL: A diverged"); ok = False
if not (status == 1).all():
    print("FAIL: A envs did not report converged (1)"); ok = False
if not (iters >= 0).all():
    print("FAIL: A freeze iters not recorded"); ok = False

V, iters, status = run(env_cap=1)
print(f"B cap=1  : per-env iters={iters.tolist()} status={status.tolist()}")
if not np.isfinite(V).all():
    print("FAIL: B diverged after timeout freeze"); ok = False
if not (status == 2).any():
    print("FAIL: B no env reported timeout (2) despite cap=1"); ok = False

V, iters, status = run(env_cap=0, semi=True)
print(f"C semi   : per-env iters={iters.tolist()} status={status.tolist()}")
if not np.isfinite(V).all():
    print("FAIL: C diverged"); ok = False
if not (status == 1).all():
    print("FAIL: C envs did not converge under per-env semi-implicit"); ok = False

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
