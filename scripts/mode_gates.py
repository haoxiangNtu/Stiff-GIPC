#!/usr/bin/env python3
"""[C3] G9 mode envelope + equivalence gate.

Runs the anchor scene through the PYTHON RESOLVER path (multienv_mode arg only
— all mode STIFF_* env stripped, so the resolver bundle itself is under test)
in all three modes and checks three promises:

  1. strict VHASH == gold: the resolver-produced bundle must reproduce the G1
     anchor bit-for-bit (G1 injects the flags explicitly; agreement pins the
     resolver as a faithful second path).
  2. envelopes: merged/isolated total Newton iterations equal their recorded
     baselines (see BASELINES; re-derive with RECORD=1).
  3. cross-mode equivalence: the three modes solve the SAME physics — the
     pairwise max |pos_a - pos_b| over final vertices must stay under EQ_TOL.
     Gross mode divergence (the foldshirt mu=0.8 class: strict failing all
     envs while other modes pass) turns into a standing red gate instead of
     an ad-hoc discovery.

Audit env (STIFF_MIRROR_AUDIT/STIFF_SLOT_AUDIT) is preserved when set by the
suite; every other STIFF_* var is stripped for the child runs.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = "f7fb5a786c2d7935"
KEEP = {"STIFF_MIRROR_AUDIT", "STIFF_SLOT_AUDIT"}
RECORD = os.environ.get("RECORD") == "1"

# BASELINES recorded 2026-07-26 on RTX 4090 (tier-2 tree, SCENE_N=2, 50f):
#   NEWTON identical across all three modes (53); merged r2r: hash varies
#   (atomic order), NEWTON stable; measured equivalence deltas:
#   isolated-strict 2.26e-9, merged-others 6.02e-5. EQ_TOL = 1e-3 gives 16x
#   headroom over measured while still failing on gross mode divergence.
#   Re-derive with RECORD=1 after any change that legitimately moves them.
BASE = {"merged": 53, "isolated": 53}
EQ_TOL = 1e-3


def run(mode, tag):
    with tempfile.TemporaryDirectory(
        prefix=f"stiffgipc-mode-gate-{tag}."
    ) as tmp:
        dump_path = os.path.join(tmp, "positions.npy")
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("STIFF_") or k in KEEP}
        env.update(SCENE_MODE=mode, SCENE_N="2",
                   DUMP_POS=dump_path, STIFF_LOG_LEVEL="0")
        p = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts/anchor_scene.py")],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        lines = p.stdout.splitlines()
        vh = next((l.split()[1] for l in lines if l.startswith("VHASH")), None)
        nt = next((int(l.split()[1]) for l in lines if l.startswith("NEWTON")), None)
        if vh is None or nt is None:
            sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
            raise SystemExit(f"mode {mode}: run failed")
        positions = np.load(dump_path)
    return vh, nt, positions


res = {}
for m in ("merged", "isolated", "strict"):
    res[m] = run(m, m)
vh_m2, nt_m2, _ = run("merged", "merged2")   # merged run-to-run probe

if RECORD:
    for m, (vh, nt, _) in res.items():
        print(f"RECORD {m}: VHASH={vh} NEWTON={nt}")
    print(f"RECORD merged-r2r: run1={res['merged'][0]}/{res['merged'][1]} "
          f"run2={vh_m2}/{nt_m2}")
    for a, b in (("merged", "isolated"), ("merged", "strict"), ("isolated", "strict")):
        d = float(np.max(np.abs(res[a][2] - res[b][2])))
        print(f"RECORD delta {a}-{b}: {d:.6e}")
    raise SystemExit(0)

ok = True
if res["strict"][0] != GOLD:
    print(f"FAIL strict-resolver anchor: {res['strict'][0]} != {GOLD}"); ok = False
for m in ("merged", "isolated"):
    if res[m][1] != BASE[m]:
        print(f"FAIL {m} envelope: NEWTON={res[m][1]} != {BASE[m]}"); ok = False
if nt_m2 != res["merged"][1]:
    print(f"WARN merged r2r NEWTON differs: {res['merged'][1]} vs {nt_m2}")
for a, b in (("merged", "isolated"), ("merged", "strict"), ("isolated", "strict")):
    d = float(np.max(np.abs(res[a][2] - res[b][2])))
    if d > EQ_TOL:
        print(f"FAIL equivalence {a}-{b}: max|dpos|={d:.3e} > {EQ_TOL:.3e}"); ok = False
    else:
        print(f"  eq {a}-{b}: max|dpos|={d:.3e}")
print("MODE-GATES:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
