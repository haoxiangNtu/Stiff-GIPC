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
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = Path(os.environ.get("GATE_RESULTS_DIR", os.path.join(ROOT, "gate-results")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
QUICK = "quick" in sys.argv[1:]
MODES = tuple(
    mode.strip()
    for mode in os.environ.get("DEMO_MODES", "merged,strict").split(",")
    if mode.strip()
)
ONLY = set(filter(None, os.environ.get("DEMO_ONLY", "").split(",")))
F = "20" if QUICK else "60"
# both headless spellings + short trajectory window for the replay family
HL = {"CASE39_HEADLESS": "1", "CASE39ME_HEADLESS": "1", "CASE39_FRAME_END": F}

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
    # isolation machinery is an ISOLATED/STRICT promise; in merged the engine
    # correctly throws fail-fast (the ratified contract) — not a valid mode here
    "midrun_quarantine": ("examples/test_env_midrun_quarantine.py", {}, 1200,
                          "MIDRUN-QUARANTINE: PASS", ("isolated", "strict")),
    "abd_badmesh":       ("examples/test_abd_badmesh_kinetic.py", {}, 600,
                          "ABD-BADMESH-KINETIC: PASS"),
    "bench_case26":      ("examples/bench_case26_simple.py", {}, 1800, None),
    # ---- replay-trajectory family (owner directive: verify ALL replays) ----
    "case39":            ("examples/replay_case39.py", HL, 2400, None),
    "case39_multienv":   ("examples/replay_case39_multienv.py",
                          dict(HL, CASE39ME_NUM_ENVS="2"), 2400, None),
    "umi_beaker":        ("examples/replay_case39_UMI_beaker.py", HL, 2400, None),
    "umi_cupshirt_fg":   ("examples/replay_case39_UMI_obb_cup_shirt_forcegrip.py", HL, 2400, None),
    "umi_sf":            ("examples/replay_case39_UMI_sf.py", HL, 2400, None),
    "umi_sf_obb":        ("examples/replay_case39_UMI_sf_obb.py", HL, 2400, None),
    "finray_cupshirt_1env": ("examples/replay_cupshirt_finray.py", HL, 1800, None),
    "finray_foldshirt_1env":("examples/replay_foldshirt_finray.py", HL, 2400, None),
    "diag_finray_grip":  ("examples/diag_finray_grip.py", {}, 1800, None),
}

BAD = re.compile(r"budget exhausted.*nan|abd-kinetic-nan|Traceback|CUDA error", re.I)

def captured_text(value):
    """Normalize TimeoutExpired output across Python/platform variants."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def run(name, mode):
    entry = DEMOS[name]
    script, extra, tmo, marker = entry[:4]
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
        rc, out = -9, (
            captured_text(e.stdout) + captured_text(e.stderr) + "\nTIMEOUT"
        )
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
    allowed = DEMOS[name][4] if len(DEMOS[name]) > 4 else None
    for mode in MODES:
        if allowed and mode not in allowed:
            continue
        r = run(name, mode)
        results.append(r)
        print(f"{name:18s} {mode:8s} {'PASS' if r['ok'] else 'FAIL':4s} "
              f"{r['wall_s']:7.1f}s  {r['why']}", flush=True)

results_path = RESULTS_DIR / "demo_verify_results.json"
with results_path.open("w") as handle:
    json.dump(results, handle, indent=1)
bad = [r for r in results if not r["ok"]]
print(f"\nRESULTS -> {results_path}")
print(f"\nDEMO-VERIFY: {'PASS' if not bad else f'FAIL ({len(bad)}/{len(results)})'}")
sys.exit(0 if not bad else 1)
