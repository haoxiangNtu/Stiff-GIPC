#!/usr/bin/env python3
"""Three-mode correctness and performance release gate.

Correctness failures, the strict run-to-run hash, and the strict gold hash all
participate in the exit status.  Performance uses simulator step timings after
a warm-up window; subprocess wall FPS is reported separately and is never used
as the simulator metric.

The release runner requires an idle GPU and a versioned baseline for its
compute capability.  Normal/manual runs still report performance when a
baseline is unavailable.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUICK = "quick" in sys.argv[1:]
MODES = tuple(
    value.strip()
    for value in os.environ.get(
        "BENCH_MODES", "merged,isolated,strict"
    ).split(",")
    if value.strip()
)
SCENES = tuple(
    value.strip()
    for value in os.environ.get(
        "BENCH_SCENES", "anchor,towel,foldshirt"
    ).split(",")
    if value.strip()
)
REPEATS = int(os.environ.get("BENCH_REPEATS", "1" if QUICK else "3"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
REQUIRE_BASELINE = os.environ.get("BENCH_REQUIRE_BASELINE") == "1"
REQUIRE_IDLE = os.environ.get("BENCH_REQUIRE_IDLE_GPU") == "1"
RECORD = os.environ.get("BENCH_RECORD_BASELINE") == "1"
DEFAULT_TOLERANCE = float(os.environ.get("BENCH_REGRESSION_TOL", "0.15"))
GOLD = "f7fb5a786c2d7935"
BAD_OUTPUT = re.compile(
    r"budget exhausted.*nan|abd-kinetic-nan|Traceback|CUDA error", re.I
)

RESULTS_DIR = Path(os.environ.get("GATE_RESULTS_DIR", ROOT / "gate-results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_PATH = Path(
    os.environ.get(
        "BENCH_BASELINE_FILE", ROOT / "scripts" / "baselines" / "mode_bench.json"
    )
)


def captured_text(value: str | bytes | None) -> str:
    """Normalize TimeoutExpired output across Python/platform variants."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def compute_apps(gpu_uuid: str | None) -> list[str]:
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        records = []
        for line in query.stdout.splitlines():
            if not line.strip():
                continue
            fields = [value.strip() for value in line.split(",", 3)]
            if len(fields) != 4:
                return ["compute-process query returned an invalid row"]
            row_uuid, pid, process_name, used_memory = fields
            if gpu_uuid is None or row_uuid == gpu_uuid:
                records.append(f"{pid}, {process_name}, {used_memory} MiB")
        return records
    except (OSError, subprocess.SubprocessError):
        # An unsupported process query is not evidence of an idle GPU.
        return ["compute-process query unavailable"]


def gpu_info() -> dict[str, str | list[str] | None]:
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        if not rows:
            raise RuntimeError("nvidia-smi returned no GPUs")
        devices = []
        for row in rows:
            fields = [part.strip() for part in row.split(",", 3)]
            if len(fields) != 4:
                raise RuntimeError("nvidia-smi returned an invalid GPU row")
            devices.append(fields)
        selector = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
        if selector:
            if selector == "-1":
                raise RuntimeError("CUDA_VISIBLE_DEVICES disables every GPU")
            if selector.isdigit():
                selected = next(
                    (device for device in devices if device[0] == selector),
                    None,
                )
            else:
                selected = next(
                    (
                        device
                        for device in devices
                        if device[3] == selector or device[3].startswith(selector)
                    ),
                    None,
                )
            if selected is None:
                raise RuntimeError(
                    f"cannot resolve CUDA_VISIBLE_DEVICES selector {selector!r}"
                )
        else:
            selected = devices[0]
        _, name, capability, uuid = selected
        model_key = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        key = "sm" + capability.replace(".", "") + "-" + model_key
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        return {
            "key": None,
            "name": None,
            "uuid": None,
            "compute_apps": [],
            "error": str(exc),
        }

    return {
        "key": key,
        "name": name,
        "uuid": uuid,
        "compute_apps": compute_apps(uuid),
        "error": None,
    }


def scene_cmd(scene: str) -> tuple[list[str], dict[str, str]]:
    if scene == "anchor":
        return [sys.executable, "scripts/anchor_scene.py"], {
            "SCENE_N": "2",
            "SCENE_FRAMES": "20" if QUICK else "50",
        }
    if scene == "towel":
        return [sys.executable, "examples/recipe_towel_scramble.py"], {
            "CASE39ME_HEADLESS": "1"
        }
    if scene == "foldshirt":
        return [sys.executable, "examples/replay_foldshirt_multienv.py"], {
            "CASE39ME_HEADLESS": "1",
            "CASE39ME_NUM_ENVS": "4",
            "CASE39_FRICTION": "0.8",
            "CASE39_FRAME_END": "30" if QUICK else "60",
        }
    if scene.startswith("finray_"):
        which = scene.split("_", 1)[1]
        return [
            sys.executable,
            f"examples/replay_{which}_finray_multienv.py",
        ], {
            "CASE39ME_HEADLESS": "1",
            "CASE39ME_NUM_ENVS": "4",
            "CASE39_FRAME_END": "20" if QUICK else "60",
        }
    raise ValueError(f"unknown scene {scene!r}")


def external_apps(apps: list[str], allowed_pid: int) -> list[str]:
    external = []
    for record in apps:
        try:
            pid = int(record.split(",", 1)[0])
        except (ValueError, IndexError):
            external.append(record)
            continue
        if pid != allowed_pid:
            external.append(record)
    return external


def run(
    scene: str, mode: str, repeat: int, gpu_uuid: str | None
) -> dict:
    cmd, extra = scene_cmd(scene)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("STIFF_")
        or key in ("STIFF_MIRROR_AUDIT", "STIFF_SLOT_AUDIT")
    }
    env.update(extra)
    env.update(
        STIFF_MULTIENV_MODE=mode,
        SCENE_MODE=mode,
        STIFF_BENCH_STATS="1",
        STIFF_LOG_LEVEL="0",
        PYTHONPATH=str(ROOT),
    )
    started = time.perf_counter()
    external_seen: list[str] = []
    stop_monitor = threading.Event()
    process: subprocess.Popen[str] | None = None

    def monitor_gpu() -> None:
        assert process is not None
        while not stop_monitor.is_set():
            for record in external_apps(compute_apps(gpu_uuid), process.pid):
                if record not in external_seen:
                    external_seen.append(record)
            if stop_monitor.wait(5.0):
                return

    try:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        monitor = (
            threading.Thread(target=monitor_gpu, daemon=True)
            if REQUIRE_IDLE
            else None
        )
        if monitor:
            monitor.start()
        stdout, stderr = process.communicate(timeout=3600)
        returncode = process.returncode
        output = stdout + stderr
    except subprocess.TimeoutExpired as exc:
        assert process is not None
        process.kill()
        stdout, stderr = process.communicate()
        returncode = -9
        output = (
            captured_text(stdout or exc.stdout)
            + captured_text(stderr or exc.stderr)
            + "\nTIMEOUT"
        )
    finally:
        stop_monitor.set()
        if "monitor" in locals() and monitor:
            monitor.join(timeout=15)
    wall = time.perf_counter() - started

    frames = re.findall(
        r"\[bench\] frame (\d+) newton (\d+) ms ([0-9.eE+-]+)", output
    )
    totals = [int(total) for _, total, _ in frames]
    deltas = (
        [totals[0]] + [right - left for left, right in zip(totals, totals[1:])]
        if totals
        else []
    )
    deltas = [value for value in deltas if value >= 0]
    all_step_ms = [float(ms) for _, _, ms in frames]
    measured_ms = all_step_ms[WARMUP:]
    mean_step_ms = statistics.fmean(measured_ms) if measured_ms else None
    p95_step_ms = (
        sorted(measured_ms)[int(0.95 * (len(measured_ms) - 1))]
        if measured_ms
        else None
    )
    vhash = next(
        (
            line.split()[1]
            for line in output.splitlines()
            if line.startswith("VHASH ")
        ),
        None,
    )

    reasons: list[str] = []
    if returncode != 0:
        reasons.append(f"rc={returncode}")
    match = BAD_OUTPUT.search(output)
    if match:
        reasons.append(f"bad-output={match.group(0)[:60]}")
    if scene == "towel" and not any(
        line.strip() == "PASS" for line in output.splitlines()
    ):
        reasons.append("missing-PASS")
    if mean_step_ms is None:
        reasons.append(f"need>{WARMUP} bench samples")
    apps_after = (
        external_apps(compute_apps(gpu_uuid), process.pid)
        if REQUIRE_IDLE and process is not None
        else []
    )
    for record in apps_after:
        if record not in external_seen:
            external_seen.append(record)
    if external_seen:
        reasons.append(
            "external GPU compute appeared during run: "
            + "; ".join(external_seen)
        )

    return {
        "scene": scene,
        "mode": mode,
        "repeat": repeat,
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "wall_s": round(wall, 3),
        "wall_fps": round(len(frames) / wall, 3) if frames else None,
        "frames": len(frames),
        "peak_newton": max(deltas) if deltas else 0,
        "mean_step_ms": round(mean_step_ms, 3) if mean_step_ms is not None else None,
        "p95_step_ms": round(p95_step_ms, 3) if p95_step_ms is not None else None,
        "simulation_fps": (
            round(1000.0 / mean_step_ms, 3) if mean_step_ms else None
        ),
        "vhash": vhash,
        "output_tail": output[-2000:] if reasons else "",
    }


def aggregate_runs(runs: list[dict]) -> dict[str, dict[str, dict]]:
    aggregate: dict[str, dict[str, dict]] = {}
    for scene in SCENES:
        aggregate[scene] = {}
        for mode in MODES:
            selected = [
                item
                for item in runs
                if item["scene"] == scene and item["mode"] == mode
            ]
            values = [
                item["mean_step_ms"]
                for item in selected
                if item["mean_step_ms"] is not None
            ]
            aggregate[scene][mode] = {
                "runs": len(selected),
                "median_mean_step_ms": (
                    round(statistics.median(values), 3) if values else None
                ),
                "min_mean_step_ms": round(min(values), 3) if values else None,
                "max_mean_step_ms": round(max(values), 3) if values else None,
            }
    return aggregate


def load_gpu_baseline(gpu_key: str | None) -> tuple[dict | None, str]:
    if gpu_key is None:
        return None, "GPU identity unavailable"
    try:
        document = json.loads(BASELINE_PATH.read_text())
    except FileNotFoundError:
        return None, f"missing {BASELINE_PATH}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid {BASELINE_PATH}: {exc}"
    if document.get("schema_version") != 1:
        return None, "unsupported baseline schema"
    baseline = document.get("gpus", {}).get(gpu_key)
    if baseline is None:
        return None, f"no baseline for {gpu_key}"
    return baseline, ""


def compare_baseline(aggregate: dict, baseline: dict | None) -> tuple[list[dict], list[str]]:
    comparisons: list[dict] = []
    failures: list[str] = []
    if baseline is None:
        return comparisons, failures
    tolerance = float(baseline.get("regression_tolerance", DEFAULT_TOLERANCE))
    reference = baseline.get("mean_step_ms", {})
    for scene in SCENES:
        for mode in MODES:
            measured = aggregate[scene][mode]["median_mean_step_ms"]
            expected = reference.get(scene, {}).get(mode)
            if measured is None or expected is None:
                failures.append(f"baseline missing measurement {scene}/{mode}")
                continue
            limit = float(expected) * (1.0 + tolerance)
            passed = float(measured) <= limit
            comparisons.append(
                {
                    "scene": scene,
                    "mode": mode,
                    "measured_ms": measured,
                    "baseline_ms": expected,
                    "limit_ms": round(limit, 3),
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(
                    f"{scene}/{mode} {measured:.1f}ms > {limit:.1f}ms "
                    f"(baseline {expected:.1f}ms)"
                )
    return comparisons, failures


def main() -> int:
    if not MODES or not SCENES or REPEATS < 1 or WARMUP < 0:
        raise ValueError("non-empty modes/scenes, repeats>=1, and warmup>=0 required")
    if RECORD:
        if (
            not REQUIRE_IDLE
            or QUICK
            or set(MODES) != {"merged", "isolated", "strict"}
            or set(SCENES) != {"anchor", "towel", "foldshirt"}
            or REPEATS < 3
        ):
            raise ValueError(
                "baseline recording requires an idle-GPU full run of all "
                "three release modes/scenes with at least three repeats"
            )

    gpu = gpu_info()
    print(
        f"[bench] gpu={gpu['name'] or 'unknown'} key={gpu['key'] or 'unknown'} "
        f"uuid={gpu['uuid'] or 'unknown'}",
        flush=True,
    )
    infrastructure_failures: list[str] = []
    if REQUIRE_IDLE and gpu["compute_apps"]:
        infrastructure_failures.append(
            "GPU is not idle: " + "; ".join(gpu["compute_apps"])
        )
    if REQUIRE_IDLE and gpu["error"]:
        infrastructure_failures.append(f"GPU preflight failed: {gpu['error']}")
    if infrastructure_failures:
        for failure in infrastructure_failures:
            print(f"INFRA FAIL {failure}", flush=True)
        return 2

    runs: list[dict] = []
    for scene in SCENES:
        scene_repeats = max(REPEATS, 2) if scene == "anchor" and "strict" in MODES else REPEATS
        for repeat in range(scene_repeats):
            order = MODES if repeat % 2 == 0 else tuple(reversed(MODES))
            for mode in order:
                if REQUIRE_IDLE:
                    apps = compute_apps(gpu["uuid"])
                    if apps:
                        print(
                            "INFRA FAIL GPU became busy before "
                            f"{scene}/{mode}/r{repeat + 1}: "
                            + "; ".join(apps),
                            flush=True,
                        )
                        return 2
                result = run(scene, mode, repeat, gpu["uuid"])
                runs.append(result)
                print(
                    f"{scene:10s} {mode:9s} r{repeat + 1} "
                    f"{result['verdict']:4s} wall={result['wall_s']:7.1f}s "
                    f"sim_fps={result['simulation_fps']} wall_fps={result['wall_fps']} "
                    f"peak_newton={result['peak_newton']} "
                    f"mean={result['mean_step_ms']}ms p95={result['p95_step_ms']}ms "
                    f"vhash={result['vhash']}",
                    flush=True,
                )

    failures = [
        f"{item['scene']}/{item['mode']}/r{item['repeat'] + 1}: "
        + ", ".join(item["reasons"])
        for item in runs
        if item["verdict"] != "PASS"
    ]

    if "anchor" in SCENES and "strict" in MODES:
        hashes = [
            item["vhash"]
            for item in runs
            if item["scene"] == "anchor" and item["mode"] == "strict"
        ]
        if len(hashes) < 2 or not hashes[0] or len(set(hashes)) != 1:
            failures.append(f"strict anchor run-to-run mismatch: {hashes}")
        # The quick profile intentionally shortens the anchor trajectory, so
        # it can enforce run-to-run identity but cannot compare the 50-frame
        # release gold. Full/tag runs always use 50 frames and enforce GOLD.
        if not QUICK and (not hashes or any(value != GOLD for value in hashes)):
            failures.append(f"strict anchor differs from gold {GOLD}: {hashes}")

    aggregate = aggregate_runs(runs)
    baseline, baseline_reason = load_gpu_baseline(gpu["key"])
    if baseline is None:
        message = f"PERF BASELINE {'FAIL' if REQUIRE_BASELINE else 'SKIP'}: {baseline_reason}"
        print(message, flush=True)
        if REQUIRE_BASELINE:
            failures.append(baseline_reason)
    comparisons, perf_failures = compare_baseline(aggregate, baseline)
    failures.extend(perf_failures)
    for item in comparisons:
        print(
            f"PERF {'PASS' if item['passed'] else 'FAIL'} "
            f"{item['scene']}/{item['mode']} measured={item['measured_ms']:.1f}ms "
            f"baseline={item['baseline_ms']:.1f}ms limit={item['limit_ms']:.1f}ms",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "gpu": gpu,
        "quick": QUICK,
        "warmup_frames": WARMUP,
        "repeats": REPEATS,
        "runs": runs,
        "aggregate": aggregate,
        "baseline_file": str(BASELINE_PATH),
        "comparisons": comparisons,
        "failures": failures,
    }
    report_path = RESULTS_DIR / "mode_bench_results.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if RECORD and not failures:
        candidate = {
            "schema_version": 1,
            "metric": "median of per-run mean step ms after warmup",
            "gpus": {
                gpu["key"] or "unknown": {
                    "device_name": gpu["name"],
                    "regression_tolerance": DEFAULT_TOLERANCE,
                    "mean_step_ms": {
                        scene: {
                            mode: aggregate[scene][mode]["median_mean_step_ms"]
                            for mode in MODES
                        }
                        for scene in SCENES
                    },
                }
            },
        }
        candidate_path = RESULTS_DIR / "mode_bench_baseline_candidate.json"
        candidate_path.write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n"
        )
        print(f"BASELINE CANDIDATE -> {candidate_path}", flush=True)
    elif RECORD:
        print("BASELINE CANDIDATE NOT WRITTEN: gate failures present", flush=True)

    for failure in failures:
        print(f"FAIL {failure}", flush=True)
    print(f"RESULTS -> {report_path}", flush=True)
    print("MODE-BENCH:", "PASS" if not failures else "FAIL", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
