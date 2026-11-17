"""M14: still water in the fixed flask (same fixture and measures as m8_hydrostatic.py), but the in-solve rest drag
(LIQ_REST_K / LIQ_REST_KV) IS allowed, so its speed threshold can be compared against the residual creep.

Island/particle sleep, velocity reset and table adhesion (anything that freezes particles) stay forbidden, as in m8.
Fixture contains recorded vessel geometry and seed particles, not a prescribed
surface or a precomputed pressure force. All motion comes from the coupled solve.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stiff_physics.engine import Engine, Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fixture')
    ap.add_argument('--out', required=True)
    ap.add_argument('--duration', type=float, default=11.)
    ap.add_argument('--dt', type=float, default=.005)
    ap.add_argument('--push', type=float, default=.9)
    ap.add_argument('--mass-floor', type=float, default=.05)
    ap.add_argument('--epsv', type=float, default=1e-3)
    ap.add_argument('--shift', type=float, default=0., help='shift grid origin in x, fraction of dx')
    ap.add_argument('--newton-tol', type=float, default=1e-4)
    ap.add_argument('--pcg-tol', type=float, default=1e-6)
    ap.add_argument('--kappa', type=float, default=1e5)
    ap.add_argument('--jmode', type=int, default=0)
    args = ap.parse_args()
    for key in ('LIQ_ISL_V', 'LIQ_SLEEP_V', 'LIQ_REST_DAMP', 'LIQ_STICK_H'):   # m8 minus LIQ_REST_K (the in-solve rest drag is the subject)
        if float(os.environ.get(key, '0')) != 0:
            raise ValueError(f'{key} must be 0 (no freezing) in this test')
        os.environ[key] = '0'
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.fixture)
    x0 = z['particles'].astype(float)
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=.0005,
                 newton_tol=args.newton_tol, absolute_epsv=args.epsv, pcg_tol=args.pcg_tol,
                 newton_iter_cap=120, pcg_max_iter_ratio=.3, energy_rel_tol=0.,
                 friction_rate=0., gd_friction_rate=0., ground_offset=-.1)
    engine = Engine(cfg); engine.set_log_level(0)
    engine.load_mesh_from_data(z['vertices'], z['faces'], 3, 3, 'ABD', np.eye(4), 1e8, 'Fixed')
    lo, hi = z['lo'].astype(float), z['hi'].astype(float)
    lo[0] += args.shift * .005; hi[0] += args.shift * .005
    engine.add_liquid(x0, .005, lo, hi, rho=1000., kappa=args.kappa,
                      particle_volume=.0025**3, push_apart=args.push, j_mode=args.jmode,
                      mass_floor=args.mass_floor, ground_mu=0., stick_h=0.)
    engine.finalize()
    states, velocities, volumes, times = [], [], [], []
    fallbacks = {}   # engine clamp / fallback events, summed over all steps (traceability: an invisible clamp is not a pass)
    def _accumulate_fallbacks():
        try: st = engine.get_liquid_stats()
        except Exception: return
        for k, v in st.items():
            if k.startswith('fb_'): fallbacks[k] = fallbacks.get(k, 0) + int(v)
    begin = time.monotonic()
    nsteps = round(args.duration / args.dt)
    for step in range(nsteps + 1):
        if step % 8 == 0 or step == nsteps:
            x = engine.get_liquid_particles(); v = engine.get_liquid_velocities(); j = engine.get_liquid_J()
            if not (np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(j).all()):
                raise RuntimeError(f'nonfinite liquid at step {step}')
            states.append(x); velocities.append(v); volumes.append(j); times.append(step * args.dt)
        if step % 200 == 0:
            print(f'[hydro] step={step} t={step*args.dt:.3f} max_v={np.linalg.norm(v,axis=1).max():.6g}', flush=True)
        if step != nsteps: engine.step(); _accumulate_fallbacks()
    # acceptance 1: total liquid -> vessel load at the end (particle barrier on the vessel vertices + wall field constraint
    # reaction) against the water weight. Both exports are engine-side; nothing here is prescribed.
    support = {}
    try:
        nv = len(z['vertices'])
        fb_ = np.asarray(engine.get_vertex_contact_forces(include_ground=False, components="normal"))[:nv]
        fw_ = np.asarray(engine.get_liquid_wall_forces())[:nv] if hasattr(engine, 'get_liquid_wall_forces') else np.zeros_like(fb_)
        weight = len(x0) * 1000. * .0025**3 * 9.81
        fg_ = np.asarray(engine.get_liquid_ghost_reaction()) if hasattr(engine, 'get_liquid_ghost_reaction') else np.zeros(3)
        support = dict(barrier_N=fb_.sum(0).tolist(), wfc_N=fw_.sum(0).tolist(), ghost_N=fg_.tolist(), total_N=(fb_ + fw_).sum(0).tolist(),
                       water_weight_N=float(weight), vertical_ratio=float(-(fb_ + fw_).sum(0)[1] / weight),
                       vertical_ratio_with_ghost=float(-((fb_ + fw_).sum(0)[1] + fg_[1]) / weight))
    except Exception as exc:  # old builds without the export
        support = dict(error=repr(exc))
    x, v, j, t = map(np.asarray, (states, velocities, volumes, times))
    first = np.searchsorted(t, t[-1] - 3.)
    sp = np.linalg.norm(np.diff(x[first:], axis=0), axis=2) / np.diff(t[first:])[:, None]
    drift = np.linalg.norm(x[-1] - x[first], axis=1)
    result = dict(particles=len(x0), end_s=float(t[-1]), wall_s=time.monotonic()-begin,
                  median_speed_mm_s=float(np.median(sp,axis=1).mean()*1000),
                  p90_speed_mm_s=float(np.percentile(sp,90,axis=1).mean()*1000),
                  max_speed_mm_s=float(sp.max()*1000),
                  drift_median_mm=float(np.median(drift)*1000), drift_p90_mm=float(np.percentile(drift,90)*1000),
                  mean_J=float(j[-1].mean()), min_J=float(j[-1].min()),
                  height_p95_mm=float(np.percentile(x[-1,:,1],95)*1000),
                  fallbacks=fallbacks, support=support,
                  newton_iters=int(engine._engine.get_total_newton_iters()),
                  parameters=vars(args), environment={k:v for k,v in os.environ.items() if k.startswith('LIQ_')})
    np.savez_compressed(out/'states.npz', x=x, v=v, J=j, t=t, lo=lo, hi=hi)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2), flush=True)


if __name__ == '__main__':
    main()
