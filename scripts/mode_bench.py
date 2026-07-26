#!/usr/bin/env python3
"""Three-mode claim/perf matrix (owner directive 2026-07-27).

For each scene x mode(merged/isolated/strict): completion, TOTAL WALL TIME,
FPS, PEAK per-frame Newton iterations, mean step ms. Strict additionally does
a run-to-run repeat on the anchor scene (bitwise claim). Scenes emit
per-frame '[bench] frame N newton TOTAL ms X' lines under STIFF_BENCH_STATS=1
(gated additions to the headless loops; zero effect otherwise).

Local soft-gripper note: the dedicated softgripper/finray examples are
UI-bound (polyscope import at top); foldshirt (gripper-grasps-shirt) carries
the soft-gripper family here, and the A800 plate + ground-pickup episodes
cover the rest remotely. Headless ports of the UI gripper cases are a listed
follow-up.

Usage: python3 scripts/mode_bench.py [quick]   (quick = short frame counts)
Env: BENCH_MODES=merged,isolated,strict  BENCH_SCENES=anchor,towel,foldshirt
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUICK = "quick" in sys.argv[1:]
MODES = os.environ.get("BENCH_MODES", "merged,isolated,strict").split(",")
SCENES = os.environ.get("BENCH_SCENES", "anchor,towel,foldshirt").split(",")
GOLD = "f7fb5a786c2d7935"

def scene_cmd(scene):
    if scene == "anchor":
        return [sys.executable, "scripts/anchor_scene.py"], {
            "SCENE_N": "2", "SCENE_FRAMES": "20" if QUICK else "50"}
    if scene == "towel":
        return [sys.executable, "examples/recipe_towel_scramble.py"], {
            "CASE39ME_HEADLESS": "1"}
    if scene == "foldshirt":
        return [sys.executable, "examples/replay_foldshirt_multienv.py"], {
            "CASE39ME_HEADLESS": "1", "CASE39ME_NUM_ENVS": "4",
            "CASE39_FRICTION": "0.8",
            "CASE39_FRAME_END": "30" if QUICK else "60"}
    if scene.startswith("finray_"):   # soft-gripper family (beaker/cupshirt/foldshirt)
        which = scene.split("_", 1)[1]
        return [sys.executable, f"examples/replay_{which}_finray_multienv.py"], {
            "CASE39ME_HEADLESS": "1", "CASE39ME_NUM_ENVS": "4",
            "CASE39_FRAME_END": "20" if QUICK else "60"}
    raise SystemExit(f"unknown scene {scene}")

def run(scene, mode, tag=""):
    cmd, extra = scene_cmd(scene)
    env = {k: v for k, v in os.environ.items() if not k.startswith("STIFF_")
           or k in ("STIFF_MIRROR_AUDIT", "STIFF_SLOT_AUDIT")}
    env.update(extra)
    env["STIFF_MULTIENV_MODE"] = mode
    env["STIFF_BENCH_STATS"] = "1"
    env["STIFF_LOG_LEVEL"] = "0"
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=3600)
    wall = time.perf_counter() - t0
    out = p.stdout
    frames = re.findall(r"\[bench\] frame (\d+) newton (\d+) ms ([0-9.]+)", out)
    peak_newton = 0
    mean_ms = None
    n_frames = len(frames)
    if frames:
        totals = [int(n) for _, n, _ in frames]
        deltas = [totals[0]] + [b - a for a, b in zip(totals, totals[1:])]
        deltas = [d for d in deltas if d >= 0]  # loop-boundary resets guard
        peak_newton = max(deltas) if deltas else 0
        mss = [float(m) for _, _, m in frames]
        mean_ms = sum(mss) / len(mss)
    vh = next((l.split()[1] for l in out.splitlines() if l.startswith("VHASH")), None)
    verdict = "PASS" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
    if scene == "towel" and p.returncode == 0:
        verdict = "PASS" if any(l.strip() == "PASS" for l in out.splitlines()) else "FAIL(no-PASS)"
    return {"scene": scene, "mode": mode, "tag": tag, "verdict": verdict,
            "wall_s": round(wall, 1),
            "fps": round(n_frames / wall, 2) if n_frames else None,
            "frames": n_frames, "peak_newton": peak_newton,
            "mean_step_ms": round(mean_ms, 1) if mean_ms else None,
            "vhash": vh}

results = []
for scene in SCENES:
    for mode in MODES:
        r = run(scene, mode)
        results.append(r)
        print(f"{scene:10s} {mode:9s} {r['verdict']:10s} wall={r['wall_s']:7.1f}s "
              f"fps={r['fps']} peak_newton={r['peak_newton']} "
              f"mean_step={r['mean_step_ms']}ms vhash={r['vhash']}", flush=True)
    if scene == "anchor" and "strict" in MODES:
        r2 = run("anchor", "strict", tag="r2r")
        results.append(r2)
        a = next(x for x in results if x["scene"] == "anchor" and x["mode"] == "strict" and not x["tag"])
        r2r = "BIT-OK" if (r2["vhash"] == a["vhash"] and a["vhash"]) else "DIFFER"
        gold = "BIT-OK" if a["vhash"] == GOLD else "DIFFER"
        print(f"anchor     strict-r2r {r2r}  vs-gold {gold}", flush=True)

out = os.path.join(ROOT, "bench_results_mode_matrix.json")
json.dump(results, open(out, "w"), indent=1)
print(f"\nRESULTS -> {out}")
bad = [r for r in results if not r["verdict"].startswith("PASS")]
print("MODE-BENCH:", "PASS" if not bad else f"FAIL ({len(bad)} runs)")
sys.exit(0 if not bad else 1)
