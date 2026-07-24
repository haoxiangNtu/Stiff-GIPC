"""[v0.8.5.1] Mid-run env-quarantine gate: a HEALTHY env that becomes
ground-infeasible DURING the run must be frozen alone, not kill the process.

Complements test_env_quarantine.py (timeout/NaN freeze of a poisoned-from-the-
start env; initial-state d-floor violations still throw by design). This gate
exercises the new iron-law demotion in throwIfGroundDistanceInvalid: with
per-env machinery live (isolated/strict + per_env_exit), a mid-run infeasible
vertex maps to its env, the env is persistently quarantined (status 3,
alpha pinned 0 every later iteration), and the healthy envs keep simulating.

Scene: N=4 envs, one 30x30 cloth each (strict co-located). Settle 10 frames,
then TELEPORT env0's cloth 10 cm below the ground plane — a state the log
barrier cannot represent (dist <= 0). Continue 15 frames.

Acceptance:
  A) no exception after the teleport (the old behavior was a process-killing
     throw from the next buildCP);
  B) env0 is quarantined: per-env status == 3, and its cloth stays where it
     was put (frozen: displacement < 1 mm across the post-freeze frames);
  C) healthy envs (1..3) stay finite and resting near the ground (no blowup,
     no NaN, cloth stays in a sane height band).
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("STIFF_LOG_LEVEL", "0")
from stiff_physics.engine import Engine, Config

ASSETS = next((os.path.join(ROOT, n) + "/" for n in ("Assets", "assets")
               if os.path.isdir(os.path.join(ROOT, n, "triMesh"))),
              os.path.join(ROOT, "Assets") + "/")
N      = 4
SETTLE = 10
AFTER  = 15
CLOTH  = "triMesh/cloth_30x30.obj"

cfg = Config(dt=0.01, cloth_thickness=1e-3, cloth_young_modulus=1e4,
             bend_young_modulus=1e3, cloth_density=200, strain_rate=100,
             poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
             ground_offset=0.0, assets_dir=ASSETS,
             multienv_mode="strict", preconditioner_type=1,
             per_env_exit=True, env_newton_iter_cap=30,
             collision_detection_buff_scale=8.0, linear_system_buff_scale=4.0)
eng = Engine(cfg)
tf = np.eye(4); tf[:3, :3] *= 0.4; tf[1, 3] = 0.02   # 2 cm above ground
for _ in range(N):
    eng.load_mesh(CLOTH, dimensions=2, body_type="FEM", transform=tf,
                  young_modulus=1e4)
eng.native.set_body_groups(list(range(N)))
eng.finalize()

nv   = np.asarray(eng.get_vertices()).shape[0]
per  = nv // N
sl   = [slice(i * per, (i + 1) * per) for i in range(N)]
print(f"[midrun-quar] {N} envs x {per} verts, settle {SETTLE} + after {AFTER}")

for _ in range(SETTLE):
    eng.step()

P = np.asarray(eng.get_vertices()).copy()
print(f"[midrun-quar] settled: env0 meanY={P[sl[0], 1].mean():.4f}")
P[sl[0], 1] -= 0.10   # env0 cloth 10 cm below the ground plane
eng.native.teleport_fem_vertices(P, None)

frozen_ref = None
ok_exception = True
try:
    for fr in range(AFTER):
        eng.step()
        if fr == 2:   # position after the quarantine has certainly engaged
            frozen_ref = np.asarray(eng.get_vertices())[sl[0]].copy()
except Exception as ex:
    ok_exception = False
    print(f"FAIL: exception after mid-run infeasibility (old behavior): {ex}")

verdicts = []
if ok_exception:
    V = np.asarray(eng.get_vertices())
    # B) env0 quarantined + frozen
    st = None
    for api in ("get_per_env_status",):
        f = getattr(eng.native, api, None)
        if f is not None:
            st = list(f())
            break
    q_ok = (st is not None and len(st) >= 1 and st[0] == 3)
    drift = float(np.abs(V[sl[0]] - frozen_ref).max()) if frozen_ref is not None else 1e9
    frozen_ok = drift < 1e-3
    below_ok  = float(V[sl[0], 1].mean()) < -0.05
    verdicts.append(("env0 status==3 (quarantined)", q_ok, f"status={st}"))
    verdicts.append(("env0 frozen (drift<1mm)", frozen_ok, f"drift={drift:.2e} m"))
    verdicts.append(("env0 still below ground", below_ok,
                     f"meanY={float(V[sl[0],1].mean()):.4f}"))
    # C) healthy envs sane
    for e in range(1, N):
        blk = V[sl[e]]
        fin = bool(np.isfinite(blk).all())
        band = bool((blk[:, 1] > -0.005).all() and (blk[:, 1] < 0.5).all())
        verdicts.append((f"env{e} finite+resting", fin and band,
                         f"finite={fin} yMin={blk[:,1].min():.4f} yMax={blk[:,1].max():.4f}"))

ok = ok_exception and all(v[1] for v in verdicts)
for name, good, detail in verdicts:
    print(f"  [{'ok' if good else 'BAD'}] {name}: {detail}")
print("MIDRUN-QUARANTINE: " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
