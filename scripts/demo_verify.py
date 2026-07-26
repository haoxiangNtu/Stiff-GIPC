#!/usr/bin/env python3
"""[demo-verify] Automated sweep of every headless-able demo (owner directive
2026-07-27: automate everything automatable; UI demos stay owner-tested).

Each entry runs under the given modes with a hard timeout; verdict requires:
  rc == 0, AND no line-search NaN warning, AND no [abd-kinetic-nan] sentinel,
  AND the entry's PASS marker if it defines one.
The NaN detectors are exactly the class that case-26 exposed — any demo
regressing that way now fails this sweep automatically.

Usage: python3 scripts/demo_verify.py [quick]
Env: DEMO_MODES=merged,strict   DEMO_ONLY=name1,name2
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUICK = "quick" in sys.argv[1:]
MODES = os.environ.get("DEMO_MODES", "merged,strict").split(",")
ONLY = set(filter(None, os.environ.get("DEMO_ONLY", "").split(",")))
F = "20" if QUICK else "60"

# name: (script, extra_env, timeout_s, pass_marker_or_None)
DEMOS = {
    "anchor":            ("scripts/anchor_scene.py", {"SCENE_N": "2"}, 600, "SCENE_OK"),
    "towel":             ("examples/recipe_towel_scramble.py", {"CASE39ME_HEADLESS": "1"}, 1800, "PASS"),
    "foldshirt":         ("examples/replay_foldshirt_multienv.py",
                          {"CASE39ME_HEADLESS": "1", "CASE39ME_NUM_ENVS": "2",
                           "CASE39_FRICTION": "0.8", "CASE39_FRAME_END": F}, 2400, None),
    "finray_beaker":     ("examples/replay_beaker_finray_multienv.py",
                          {"CASE39ME_HEADLESS": "1", "CASE39ME_NUM_ENVS": "2",
                           "CASE39_FRAME_END": F}, 1800, None),
    "finray_cupshirt":   ("examples/replay_cupshirt_finray_multienv.py",
                          {"CASE39ME_HEADLESS": "1", "CASE39ME_NUM_ENVS": "2",
                           "CASE39_FRAME_END": F}, 1800, None),
    "finray_foldshirt":  ("examples/replay_foldshirt_finray_multienv.py",
                          {"CASE39ME_HEADLESS": "1", "CASE39ME_NUM_ENVS": "2",
                           "CASE39_FRAME_END": F}, 2400, None),
    "finray_beaker_1env":("examples/replay_beaker_finray.py",
                          {"CASE39_HEADLESS": "1", "CASE39_FRAME_END": F}, 1800, None),
    "midrun_quarantine": ("examples/test_env_midrun_quarantine.py", {}, 1200,
                          "MIDRUN-QUARANTINE: PASS"),
    "abd_badmesh":       ("examples/test_abd_badmesh_kinetic.py", {}, 600,
                          "ABD-BADMESH-KINETIC: PASS"),
    "bench_case26":      ("examples/bench_case26_simple.py", {}, 1800, None),
}

BAD = re.compile(r"budget exhausted.*nan|abd-kinetic-nan|Traceback|CUDA error", re.I)

def run(name, mode):
    script, extra, tmo, marker = DEMOS[name]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("STIFF_") or k in ("STIFF_MIRROR_AUDIT", "STIFF_SLOT_AUDIT")}
    env.update(extra)
    env["STIFF_MULTIENV_MODE"] = mode
    env["STIFF_LOG_LEVEL"] = "0"
    env["PYTHONPATH"] = ROOT
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, script], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=tmo)
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = -9, (e.stdout or "") + (e.stderr or "") + "\nTIMEOUT"
    wall = time.perf_counter() - t0
    bad = BAD.findall(out)
    ok = (rc == 0 and not bad and (marker is None or marker in out))
    why = "" if ok else (f"rc={rc}" + (f" bad={bad[0][:40]}" if bad else "")
                         + ("" if marker is None or marker in out else " no-marker"))
    return {"demo": name, "mode": mode, "ok": ok, "wall_s": round(wall, 1), "why": why}

results = []
for name in DEMOS:
    if ONLY and name not in ONLY:
        continue
    for mode in MODES:
        r = run(name, mode)
        results.append(r)
        print(f"{name:18s} {mode:8s} {'PASS' if r['ok'] else 'FAIL':4s} "
              f"{r['wall_s']:7.1f}s  {r['why']}", flush=True)

json.dump(results, open(os.path.join(ROOT, "demo_verify_results.json"), "w"), indent=1)
bad = [r for r in results if not r["ok"]]
print(f"\nDEMO-VERIFY: {'PASS' if not bad else f'FAIL ({len(bad)}/{len(results)})'}")
sys.exit(0 if not bad else 1)
