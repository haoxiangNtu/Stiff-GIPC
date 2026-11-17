"""Tunnelling probe: a free rigid box hits a FIXED tessellated plate (checks the 'zero penetration' claim for fast, heavy bodies).

One configuration per process. The box starts 30 mm above the plate with a downward initial velocity (set_abd_body_velocities),
so the impact speed is set directly. Reported: minimum (box bottom - plate top) gap over the run, the time it occurred, the
final resting gap, whether the box ended up below the plate top (tunnelled), and the Newton iterations per step.

    PYTHONPATH=. python examples/nuka/tunnel_probe.py --out out/nuka/tp_a --dt 0.005 --speed 3 --mass 4.3
Optional: --target fem  drops the box on a soft FEM pad lying on the plate (the elastoplastic-pad case).
"""
import argparse, json, os, os.path as osp, sys, time
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
from common import Recorder, Scene  # noqa: E402

PLATE_TOP = 0.07


def box_tets(size, cell):
    """Regular hex grid split into 5 tets each (same scheme as elastoplastic_compression.box_tets)."""
    sx, sy, sz = size
    nx, ny, nz = [max(1, int(round(s / cell))) for s in size]
    xs = np.linspace(-sx / 2, sx / 2, nx + 1); ys = np.linspace(0.0, sy, ny + 1); zs = np.linspace(-sz / 2, sz / 2, nz + 1)
    V = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    idx = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k   # noqa: E731
    T = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
                     idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)]
                if (i + j + k) % 2 == 0:
                    T += [[c[0], c[1], c[3], c[4]], [c[1], c[2], c[3], c[6]], [c[1], c[4], c[5], c[6]], [c[3], c[4], c[6], c[7]], [c[1], c[3], c[4], c[6]]]
                else:
                    T += [[c[0], c[1], c[2], c[5]], [c[0], c[2], c[3], c[7]], [c[0], c[4], c[5], c[7]], [c[2], c[5], c[6], c[7]], [c[0], c[2], c[5], c[7]]]
    return V, np.array(T, np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--speed", type=float, default=3.0, help="impact speed m/s (initial downward velocity)")
    ap.add_argument("--mass", type=float, default=4.3)
    ap.add_argument("--size", type=float, default=0.10)
    ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--dhat", type=float, default=2e-3)
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--pcg-tol", type=float, default=1e-5)
    ap.add_argument("--newton-cap", type=int, default=40)
    ap.add_argument("--duration", type=float, default=0.8)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--target", default="plate", choices=["plate", "fem"])
    ap.add_argument("--pad-young", type=float, default=5e4)
    ap.add_argument("--pad-h", type=float, default=0.06)
    ap.add_argument("--ptype", type=int, default=0)
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--vel-init", action="store_true", help="use set_abd_body_velocities instead of a free fall from v^2/2g (the velocity setter had no effect in build_rg8)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=args.ptype, newton_iter_cap=args.newton_cap, absolute_dhat=args.dhat,
                    newton_tol=args.newton_tol, pcg_tol=args.pcg_tol, friction_rate=args.mu, young_modulus=args.pad_young,
                    poisson_rate=0.3), log_level=args.log_level)
    plate = sc.add_box("plate", (0.0, PLATE_TOP - 0.03, 0.0), (0.5, 0.06, 0.5), fixed=True, cell=args.cell, mu=args.mu)
    sc.eng.add_ground_collision_skip(plate.body_id)
    top = PLATE_TOP if args.target == "plate" else PLATE_TOP + 1e-3 + args.pad_h
    s = args.size
    fall_h = 0.0 if args.vel_init else args.speed ** 2 / (2 * 9.81)
    y0 = top + 0.03 + s / 2 + fall_h
    # ABD bodies must be loaded before FEM bodies
    box = sc.add_box("box", (0.0, y0, 0.0), (s, s, s), fixed=False, density=args.mass / s ** 3, cell=args.cell, mu=args.mu, color=(0.9, 0.4, 0.3))
    pad = None
    if args.target == "fem":
        V, T = box_tets((0.24, args.pad_h, 0.20), 0.02)
        Tp = np.eye(4); Tp[1, 3] = PLATE_TOP + 1e-3
        pad = sc.add_fem("pad", V, T, Tp, young=args.pad_young, density=1000.0)
    sc.finalize()
    ids = np.array([box.body_id], np.int32)
    if args.vel_init:
        Vel = np.zeros((1, 4, 4)); Vel[0, 1, 3] = -args.speed
        sc.eng.set_abd_body_velocities(ids, Vel)
    else:
        args.duration += args.speed / 9.81 + 0.1
        print(f"[probe] free fall from {fall_h*1e3 + 30:.0f} mm above the target -> impact at ~{args.speed:.1f} m/s after {args.speed/9.81:.2f} s", flush=True)
    rec = Recorder(sc) if args.record else None
    n_steps = int(round(args.duration / args.dt))
    rec_every = max(1, int(round(1.0 / (50.0 * args.dt))))
    newton0 = sc.eng.native.get_total_newton_iters()
    min_gap, t_min, through = 1e9, 0.0, False
    gaps = []
    t0 = time.time()
    y_prev = None
    for k in range(n_steps + 1):
        t = k * args.dt
        Vb = np.asarray(sc.eng.get_abd_body(box.body_id).get_vertices(), float)
        bottom = Vb[:, 1].min(); centre = Vb[:, 1].mean()
        if pad is not None:
            P = sc.fem_verts("pad")
            top_now = P[(np.abs(P[:, 0]) < s / 2 + 0.01) & (np.abs(P[:, 2]) < s / 2 + 0.01), 1].max() if np.any((np.abs(P[:, 0]) < s / 2 + 0.01) & (np.abs(P[:, 2]) < s / 2 + 0.01)) else top
            gap_plate = bottom - PLATE_TOP
        else:
            top_now = top; gap_plate = bottom - PLATE_TOP
        gap = bottom - top_now
        gaps.append((t, bottom, gap_plate))
        if gap_plate < min_gap:
            min_gap, t_min = gap_plate, t
        if centre < PLATE_TOP:
            through = True
        if k == 1 and y_prev is not None and args.vel_init:
            print(f"[probe] first-step dy {(centre - y_prev)*1e3:+.2f} mm (expected {-args.speed*args.dt*1e3:+.2f} mm from the initial velocity)", flush=True)
        if k > 0 and (centre - y_prev) / args.dt < -0.5 * args.speed and not getattr(main, "_vhit", False) and bottom - top_now < 0.02:
            main._vhit = True; print(f"[probe] approach speed {(y_prev - centre)/args.dt:.2f} m/s at gap {(bottom-top_now)*1e3:.1f} mm (t {t:.3f} s)", flush=True)
        y_prev = centre
        if k % max(1, int(round(0.05 / args.dt))) == 0:
            nit = sc.eng.native.get_total_newton_iters()
            print(f"[{t:5.3f}s] bottom {bottom*1e3:7.2f} mm  gap-to-plate {gap_plate*1e3:+7.2f} mm{'' if pad is None else f'  gap-to-pad {gap*1e3:+7.2f} mm'}  newton {(nit-newton0)}", flush=True)
            newton0 = nit
        if rec is not None and k % rec_every == 0:
            rec.record(t, 0, "drop")
        sc.step()
    Vb = np.asarray(sc.eng.get_abd_body(box.body_id).get_vertices(), float)
    final_gap = Vb[:, 1].min() - PLATE_TOP
    summ = dict(dt=args.dt, speed=args.speed, mass=args.mass, dhat=args.dhat, newton_tol=args.newton_tol, pcg_tol=args.pcg_tol, newton_cap=args.newton_cap,
                target=args.target, min_gap_mm=min_gap * 1e3, t_min=t_min, final_gap_mm=final_gap * 1e3, through=through, wall_s=time.time() - t0)
    json.dump(dict(summary=summ, gaps=gaps), open(osp.join(args.out, "summary.json"), "w"))
    if rec is not None:
        rec.save(osp.join(args.out, "recording.npz"), dict(example="tunnel_probe", dt=args.dt, fps=50.0, fem_materials={"pad": "rubber"} if pad else {}, fem_colors=[[0.6, 0.7, 0.9]] if pad else []))
    print(f"[tp] dt={args.dt} v={args.speed} m={args.mass} dhat={args.dhat} tol={args.newton_tol}/{args.pcg_tol} cap={args.newton_cap} target={args.target} "
          f"mingap={min_gap*1e3:+.2f}mm@{t_min:.3f}s final={final_gap*1e3:+.2f}mm through={through} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
