#!/usr/bin/env python3
"""Finalize-time isolation audit: isolated/strict are PHYSICAL isolation only.

A diverging env aborts the whole batch unless the host telemetry path is on
(Config(per_env_exit=True), env_newton_iter_cap>0, or STIFF_PERENV_TELEM=1).
Engine.finalize() prints a one-time [WARN] in the unguarded case. This test runs
six configurations, one subprocess each (per_env_exit is process-scoped), and
checks the warning fires exactly when expected. ~1 min on a 4090.

Usage: PYTHONPATH=. python examples/test_isolated_quarantine_warn.py
"""
import os, sys, subprocess, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHILD = r'''
import os, sys, math, numpy as np, io, contextlib
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/build")
from stiff_physics import Engine, Config
mode, pee, cap = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3])
d = np.load(ROOT + "/Assets/duck/duck_tet.npz")
verts, cells = d["verts"].astype(np.float64), d["cells"].astype(np.int32)
cfg = Config(dt=0.02, poisson_rate=0.45, friction_rate=0.4, relative_dhat=1e-3,
             newton_tol=5e-2, newton_iter_cap=50, preconditioner_type=1,
             ground_offset=0.0, assets_dir=ROOT + "/Assets/",
             multienv_mode=mode, per_env_exit=pee)
cfg._cfg.absolute_dhat = 0.0019
if cap > 0:
    cfg._cfg.env_newton_iter_cap = cap
eng = Engine(cfg)
Ts = []
for i in range(2):
    T = np.eye(4); T[0, 3] = i * 0.5; T[1, 3] = 0.15; Ts.append(T)
eng.native.load_mesh_instanced(verts, cells, 4, 3, 0, Ts, 1e8, 0)
if mode != "merged":
    eng.set_body_groups([0, 1])
eng.finalize()
eng.step()
print("CHILD_OK", flush=True)
'''.replace("ROOT", repr(ROOT))
cases = [  # (label, mode, per_env_exit, cap, extra_env, expect_warn)
    ("isolated, defaults",            "isolated", "0", 0, {}, True),
    ("strict, defaults",              "strict",   "0", 0, {}, True),
    ("isolated + per_env_exit=True",  "isolated", "1", 0, {}, False),
    ("isolated + env_newton_iter_cap","isolated", "0", 100, {}, False),
    ("isolated + STIFF_PERENV_TELEM", "isolated", "0", 0, {"STIFF_PERENV_TELEM": "1"}, False),
    ("merged, defaults",              "merged",   "0", 0, {}, False),
]
allok = True
for label, mode, pee, cap, extra, expect in cases:
    env = dict(os.environ); env.update(extra)
    r = subprocess.run([sys.executable, "-c", CHILD, mode, pee, str(cap)],
                       env=env, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    fired = "physical isolation only" in out
    ok = (fired == expect) and ("CHILD_OK" in out)
    allok &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {label:32s} warn={'yes' if fired else 'no ':3s} expected={'yes' if expect else 'no'}  rc={r.returncode}")
    if not ok:
        print("   --- tail ---"); print("\n".join(out.strip().splitlines()[-8:]))
print("ALL_PASS" if allok else "SOME_FAIL")
sys.exit(0 if allok else 1)
