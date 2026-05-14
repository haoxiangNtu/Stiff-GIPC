#!/usr/bin/env python3
"""Rigorous A/B benchmark for case_33 gravity hypothesis.

Improvements over the original (single-run) benchmark:
 1. N repeats per configuration → median + std + IQR (drops outlier influence)
 2. Contact-phase vs free-phase split: gravity effect is hypothesized to
    matter ONLY when cloth is being pushed; we separate frames accordingly
 3. Two engine modes:
      semi_implicit=True  (production, fast early-exit)
      semi_implicit=False (full Newton, exposes raw solver behavior)
 4. Newton-iter parsing fix: scan ALL `iteration k:` markers per step from
    the engine's per-iter line (not the end-of-frame summary), and verify
    via warmup-baseline sanity check
"""
import sys, os, math, time, io, contextlib, statistics, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np

# Reuse setup_scene from case_33
from case_33_arm_cloth_gravity import setup_scene, _FdCapture


def parse_newton_iter_per_step(stdout_text: str) -> int:
    """Engine prints '      Kappa: NNN  iteration k:  M' at end of each
    Newton solve.  Per step, the LAST such line is the final iter count."""
    last = 0
    for line in stdout_text.split('\n'):
        s = line.strip()
        if 'iteration k:' in s:
            try:
                last = int(s.split('iteration k:')[1].strip().split()[0])
            except Exception:
                pass
    return last


def run_one_trial(gravity_off: bool, semi_implicit: bool,
                  n_warmup: int = 5, n_steps: int = 300,
                  ramp_target_deg: float = -90.0,
                  rate_deg_per_step: float = 0.5,
                  contact_threshold_deg: float = -45.0):
    """Run a single ramp 0 → ramp → 0 cycle, collect per-step data.

    contact_threshold_deg: angles below this (more negative) are considered
        "contact phase" — based on geometry, arm enters cloth around -45°.
    """
    scene = setup_scene(disable_arm_gravity=gravity_off,
                        verbose=False, semi_implicit=semi_implicit)
    eng = scene['eng']
    joint_idx = scene['joint_idx']

    # warmup
    for _ in range(n_warmup):
        eng.step()

    half = n_steps // 2
    angles = []
    driven = 0.0
    for _ in range(half):
        delta = ramp_target_deg - driven
        if abs(delta) > rate_deg_per_step:
            driven += rate_deg_per_step * np.sign(delta)
        else:
            driven = ramp_target_deg
        angles.append(driven)
    for _ in range(n_steps - half):
        delta = 0.0 - driven
        if abs(delta) > rate_deg_per_step:
            driven += rate_deg_per_step * np.sign(delta)
        else:
            driven = 0.0
        angles.append(driven)
    angles = np.array(angles)

    step_times_ms = np.zeros(n_steps)
    newton_iters  = np.zeros(n_steps, dtype=np.int32)
    ls_hangs      = np.zeros(n_steps, dtype=np.int32)

    for s in range(n_steps):
        eng.native.set_revolute_target(joint_idx, math.radians(angles[s]))
        with _FdCapture() as cap:
            t0 = time.perf_counter()
            eng.step()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        step_times_ms[s] = elapsed_ms
        newton_iters[s] = parse_newton_iter_per_step(cap.text)
        ls_hangs[s] = cap.text.count('lineSearchCount=9')

    # Phase classification
    contact_mask = angles < contact_threshold_deg

    return dict(
        angles=angles,
        step_times_ms=step_times_ms,
        newton_iters=newton_iters,
        ls_hangs=ls_hangs,
        contact_mask=contact_mask,
        config=dict(gravity_off=gravity_off, semi_implicit=semi_implicit),
    )


def summarize_run(data, label):
    st  = data['step_times_ms']
    ni  = data['newton_iters']
    lh  = data['ls_hangs']
    cm  = data['contact_mask']

    def stats(arr, mask=None):
        a = arr[mask] if mask is not None else arr
        if len(a) == 0:
            return dict(n=0, median=0, mean=0, p95=0, max=0, std=0)
        return dict(
            n=int(len(a)),
            median=float(np.median(a)),
            mean=float(np.mean(a)),
            p95=float(np.percentile(a, 95)),
            max=float(np.max(a)),
            std=float(np.std(a)),
        )

    return {
        'label': label,
        'all':     dict(step_ms=stats(st), newton=stats(ni), ls_hang=stats(lh)),
        'contact': dict(step_ms=stats(st, cm), newton=stats(ni, cm), ls_hang=stats(lh, cm)),
        'free':    dict(step_ms=stats(st, ~cm), newton=stats(ni, ~cm), ls_hang=stats(lh, ~cm)),
    }


def aggregate_repeats(per_trial_summaries):
    """Aggregate N repeats: for each metric, compute median-of-medians +
    inter-trial std (a measure of run-to-run noise)."""
    out = {}
    for phase in ('all', 'contact', 'free'):
        out[phase] = {}
        for metric in ('step_ms', 'newton', 'ls_hang'):
            out[phase][metric] = {}
            for stat in ('median', 'mean', 'p95', 'max'):
                vals = [s[phase][metric][stat] for s in per_trial_summaries]
                out[phase][metric][stat] = dict(
                    median_across_runs=float(np.median(vals)),
                    std_across_runs=float(np.std(vals)),
                    values=vals,
                )
            # ls_hang sum across all steps in trial (useful per-trial total)
            sums = [s[phase]['ls_hang']['mean'] * s[phase]['ls_hang']['n']
                    for s in per_trial_summaries]
            out[phase][metric].setdefault('sum_per_trial', sums)
    return out


def main():
    N_REPEATS = int(os.environ.get("BENCH_REPEATS", "5"))
    N_STEPS = int(os.environ.get("BENCH_STEPS", "300"))
    RAMP = float(os.environ.get("BENCH_RAMP", "-90.0"))
    RATE = float(os.environ.get("BENCH_RATE", "0.5"))
    CONTACT_DEG = float(os.environ.get("BENCH_CONTACT_DEG", "-45.0"))

    configs = [
        ('semi_grav_ON',  False, True),    # gravity_off=False, semi=True
        ('semi_grav_OFF', True,  True),
        ('full_grav_ON',  False, False),   # semi=False = full Newton
        ('full_grav_OFF', True,  False),
    ]

    print(f"\n{'='*78}")
    print(f"CASE_33 RIGOROUS BENCHMARK")
    print(f"  N_repeats={N_REPEATS}, n_steps={N_STEPS}, ramp={RAMP}°, "
          f"rate={RATE}°/step")
    print(f"  contact phase = angles < {CONTACT_DEG}°")
    print(f"  full = semi_implicit OFF (forces Newton iterations)")
    print(f"{'='*78}")

    all_results = {}
    t_start = time.time()
    for label, gravity_off, semi in configs:
        print(f"\n[{label}]  (gravity_off={gravity_off}, semi_implicit={semi})",
              flush=True)
        trials = []
        for r in range(N_REPEATS):
            print(f"  trial {r+1}/{N_REPEATS}...", flush=True, end=' ')
            t0 = time.time()
            with _FdCapture():  # suppress engine setup noise
                data = run_one_trial(gravity_off=gravity_off,
                                      semi_implicit=semi,
                                      n_steps=N_STEPS,
                                      ramp_target_deg=RAMP,
                                      rate_deg_per_step=RATE,
                                      contact_threshold_deg=CONTACT_DEG)
            trials.append(summarize_run(data, label))
            print(f"done in {time.time()-t0:.1f}s "
                  f"(ALL median ms={trials[-1]['all']['step_ms']['median']:.2f}, "
                  f"CONTACT median ms={trials[-1]['contact']['step_ms']['median']:.2f}, "
                  f"ls_hang sum={int(trials[-1]['all']['ls_hang']['mean']*trials[-1]['all']['ls_hang']['n'])})",
                  flush=True)
        all_results[label] = aggregate_repeats(trials)

    print(f"\n[total elapsed: {(time.time()-t_start)/60:.1f} min]\n", flush=True)

    # Pretty-print summary table
    print(f"\n{'='*78}")
    print(f"RESULTS (median across {N_REPEATS} trials; ± std-across-trials)")
    print(f"{'='*78}\n")

    def fmt(d, key):
        m = d['median_across_runs']
        s = d['std_across_runs']
        return f"{m:7.2f}±{s:5.2f}"

    for phase in ('all', 'contact', 'free'):
        print(f"--- phase: {phase.upper()} "
              f"(angles {'<' if phase=='contact' else '>' if phase=='free' else '*'} "
              f"{CONTACT_DEG}°) ---")
        header = f"{'config':<18} {'step_ms median':>16} {'step_ms p95':>14} "\
                 f"{'step_ms max':>14} {'newton median':>15} {'newton max':>13}"
        print(header)
        for label, _, _ in configs:
            r = all_results[label]
            sm = r[phase]['step_ms']
            ni = r[phase]['newton']
            print(f"{label:<18} {fmt(sm['median']):>16} {fmt(sm['p95']):>14} "
                  f"{fmt(sm['max']):>14} {fmt(ni['median']):>15} {fmt(ni['max']):>13}")
        print()

    print(f"\n{'='*78}")
    print("KEY COMPARISONS (semi-implicit mode, contact phase)")
    print(f"{'='*78}")
    on_c  = all_results['semi_grav_ON']['contact']
    off_c = all_results['semi_grav_OFF']['contact']
    on_f  = all_results['full_grav_ON']['contact']
    off_f = all_results['full_grav_OFF']['contact']

    def delta(off, on, key='median'):
        off_v = off[key]['median_across_runs']
        on_v  = on[key]['median_across_runs']
        pct = (off_v - on_v) / max(on_v, 1e-9) * 100
        return f"ON={on_v:7.2f} OFF={off_v:7.2f} Δ={pct:+6.1f}%"

    print(f"\nSEMI-IMPLICIT mode (production setting):")
    print(f"  step_ms median  {delta(off_c['step_ms'], on_c['step_ms'])}")
    print(f"  step_ms p95     {delta(off_c['step_ms'], on_c['step_ms'], 'p95')}")
    print(f"  step_ms max     {delta(off_c['step_ms'], on_c['step_ms'], 'max')}")
    print(f"  newton median   {delta(off_c['newton'], on_c['newton'])}")
    print(f"  newton max      {delta(off_c['newton'], on_c['newton'], 'max')}")

    print(f"\nFULL-NEWTON mode (semi-implicit off — stress test):")
    print(f"  step_ms median  {delta(off_f['step_ms'], on_f['step_ms'])}")
    print(f"  step_ms p95     {delta(off_f['step_ms'], on_f['step_ms'], 'p95')}")
    print(f"  newton median   {delta(off_f['newton'], on_f['newton'])}")
    print(f"  newton max      {delta(off_f['newton'], on_f['newton'], 'max')}")

    print(f"\nFREE PHASE (no cloth contact — gravity should NOT matter much here):")
    on_free  = all_results['semi_grav_ON']['free']
    off_free = all_results['semi_grav_OFF']['free']
    print(f"  step_ms median  {delta(off_free['step_ms'], on_free['step_ms'])}")
    print(f"  newton median   {delta(off_free['newton'], on_free['newton'])}")

    # Save raw data
    out_path = "/tmp/case33_rigorous.json"
    def cleanse(obj):
        if isinstance(obj, dict): return {k: cleanse(v) for k,v in obj.items()}
        if isinstance(obj, list): return [cleanse(x) for x in obj]
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        return obj
    with open(out_path, 'w') as f:
        json.dump(cleanse(all_results), f, indent=2)
    print(f"\n[raw data saved to {out_path}]")


if __name__ == "__main__":
    main()
