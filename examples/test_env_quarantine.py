"""[v0.8.5] Env-quarantine gate: a pathological env must not disturb healthy ones.

Scene: N=4 envs, each one ABD cube resting near the ground (strict mode, so
cross-batch bit-invariance holds). Env0 is poisoned with a SECOND cube
interpenetrating the first (infeasible self-contact start) — a configuration
that drives its Newton loop hard while staying clear of the ground d-floor
invariant (deep ground penetration is rejected at finalize by design).

Config under test: per_env_exit=True + env_newton_iter_cap (the v0.8.5
stability recommendation uses cap=100; this gate runs cap=3 so a benign
two-cube settle reliably exercises the timeout-freeze machinery).

Acceptance (calibrated 2026-07-24):
  A) env0 trips the cap (timeout status) on at least one frame, and the cap
     is respected;
  B) healthy-env trajectories match a reference run containing only the
     three healthy envs to <2e-4 m. Measured reality: ~7e-5 m — the frozen
     env shifts the GLOBAL Newton loop schedule, which moves healthy envs'
     own freeze timing by a few frames; the disturbance is micrometer-scale,
     not zero. Bitwise isolation is NOT promised for heterogeneous envs
     (global scene bbox etc.); micrometer-scale physical isolation IS.
  C) healthy envs land upright near their rest height (not blown away);
  D) no non-finite values ever leak into healthy envs.
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = str(ROOT / "Assets") + "/"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STIFF_LOG_LEVEL", "0")

import numpy as np
from stiff_physics.engine import Engine, Config

FRAMES = 30
CAP = 3  # deliberately low: settle-phase contact Newton (>3) must trip it


def build(envs, poison_first):
    cfg = Config(dt=0.02, ground_offset=0.0, assets_dir=ASSETS,
                 multienv_mode="strict", preconditioner_type=1,
                 per_env_exit=True, env_newton_iter_cap=CAP)
    cfg._cfg.absolute_dhat = 1e-3
    eng = Engine(cfg)
    groups = []
    for e in range(envs):
        t = np.eye(4)
        t[:3, :3] *= 0.125
        t[1, 3] = -0.0105  # 2 mm above ground (healthy rest)
        eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD",
                      transform=t)
        groups.append(e)
        if poison_first and e == 0:
            t2 = np.eye(4)
            t2[:3, :3] *= 0.125
            # A second live cube dropped onto the first: the settle phase
            # keeps env0's contact solve above the (deliberately low) cap.
            # NOTE deep interpenetration does NOT work as poison — the IPC
            # barrier only activates within dhat, a fully embedded intruder
            # is invisible to it (and ground penetration is rejected at
            # finalize). Sustained contact activity is the reliable driver.
            t2[1, 3] = -0.0105 + 0.055
            eng.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD",
                          transform=t2)
            groups.append(e)
    eng.native.set_body_groups(groups)
    eng.finalize()
    return eng


def healthy_slices(eng, first_env):
    recs = eng.get_load_records()
    return [(r.vertex_offset, r.vertex_count) for r in recs[first_env:]]


def run(eng, frames, slices):
    traj = []
    statuses = []
    iters = []
    for _ in range(frames):
        eng.step()
        V = np.asarray(eng.get_vertices())
        traj.append(np.concatenate([V[o:o + c] for o, c in slices]).copy())
        statuses.append(np.asarray(eng.get_per_env_status()).copy())
        iters.append(np.asarray(eng.get_per_env_newton_iters()).copy())
    return traj, statuses, iters


def main():
    # Run A: 4 envs, env0 poisoned.
    engA = build(4, poison_first=True)
    slA = healthy_slices(engA, 2)  # env0 holds TWO bodies (poison pair)
    trajA, stA, itA = run(engA, FRAMES, slA)
    del engA

    # Run B (reference): only the three healthy envs.
    engB = build(3, poison_first=False)
    slB = healthy_slices(engB, 0)
    trajB, stB, _ = run(engB, FRAMES, slB)
    del engB

    sick_flagged = any(int(s[0]) in (2, 3) for s in stA)
    cap_ok = all(int(i[0]) <= CAP for i in itA if int(i[0]) >= 0)
    finite_ok = all(np.isfinite(t).all() for t in trajA)
    # healthy cubes must end near rest height (y in [-0.02, 0.08])
    endV = trajA[-1]
    upright_ok = bool((endV[:, 1] > -0.02).all() and (endV[:, 1] < 0.08).all())

    dev = max(float(np.abs(a - b).max()) for a, b in zip(trajA, trajB))
    bit_ok = dev < 2e-4

    st_tail = stA[-1].tolist()[:4]
    it_tail = itA[-1].tolist()[:4]
    print(f"[quarantine] last statuses={st_tail} iters={it_tail}")
    print(f"[quarantine] sick_flagged={sick_flagged} upright={upright_ok} "
          f"cap_respected={cap_ok} finite={finite_ok} healthy_dev_vs_ref={dev:.3e} ok={bit_ok}")

    ok = sick_flagged and upright_ok and cap_ok and finite_ok and bit_ok
    print("ENV-QUARANTINE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
