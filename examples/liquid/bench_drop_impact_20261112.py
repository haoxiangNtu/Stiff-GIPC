"""Bench (2026-11-12): free-falling liquid particles / small clumps landing on the bare table and in a water-filled box.

The Rokae dropper wet replays hit the Newton iteration budget in single frames ("drain stall"); in the one completed
record with such a frame (sl_w7, frame 3540 = 25.2 s) nothing was near the tip, but the drop expelled 0.19 s earlier from
178 mm above the table was landing. This scene isolates that event with the replay's liquid and solver settings:
a fixed open box with a pool, a free cube on the table (solid DOFs), and drops released from several heights so that
every impact falls in its own frame.

    PYTHONPATH=. python examples/liquid/bench_drop_impact_20261112.py --out out/pump_bench/drop1
"""
import argparse, json, os, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from stiff_physics.engine import Engine, Config  # noqa: E402
from m2_box import open_box, cuboid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/pump_bench/drop")
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--dhat", type=float, default=1e-3)
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--pcg-tol", type=float, default=1e-4)
    ap.add_argument("--newton-cap", type=int, default=300)
    ap.add_argument("--no-cube", action="store_true")
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--table-only", action="store_true", help="only the drops over the bare table")
    ap.add_argument("--pool-only", action="store_true", help="only the drops over the pool")
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dx, dt = args.dx, args.dt
    hs = dx / 2
    cfg = Config(dt=dt, density=1000.0, young_modulus=1e7, poisson_rate=0.45, friction_rate=0.5, gravity=(0.0, -9.81, 0.0),
                 ground_offset=0.0, ground_normal=(0.0, 1.0, 0.0), absolute_dhat=args.dhat, preconditioner_type=0,
                 soft_motion_rate=1e4, newton_tol=args.newton_tol, pcg_tol=args.pcg_tol, newton_iter_cap=args.newton_cap,
                 pcg_max_iter_ratio=0.3, energy_rel_tol=0.0)
    eng = Engine(cfg)
    eng.set_log_level(args.log_level)
    floor_y, cxz, inner, wall, height = 0.006, (0.20, 0.20), 0.06, 0.005, 0.05
    V, F = open_box(cxz, inner, wall, floor_y, height, 0.005)
    eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    cube = None
    if not args.no_cube:
        cv, cf = cuboid([0.30, 0.0012, 0.34], [0.33, 0.0312, 0.37])
        eng.load_mesh_from_data(cv, cf, verts_per_face=3, dimensions=3, body_type="ABD", transform=np.eye(4), young_modulus=1e8, boundary_type="Free")
        cube = int(eng.get_load_records()[-1].body_offset)
        eng.set_abd_body_density(cube, 1000.0)
    rng = np.random.default_rng(0)
    groups = {}
    pts = []
    n = 0

    def add(name, p):
        nonlocal n
        p = np.atleast_2d(np.asarray(p, float))
        groups[name] = (n, n + len(p))
        pts.append(p)
        n += len(p)

    water_top = floor_y + 0.03
    if not args.no_pool:
        m = 0.6 * dx
        h2 = inner / 2
        g = np.meshgrid(np.arange(cxz[0] - h2 + m, cxz[0] + h2 - m, hs), np.arange(floor_y + m, water_top, hs),
                        np.arange(cxz[1] - h2 + m, cxz[1] + h2 - m, hs), indexing="ij")
        p = np.stack([x.ravel() for x in g], 1)
        add("pool", p + rng.uniform(-0.31 * hs, 0.31 * hs, p.shape))
    clump = np.array([[i * hs, j * hs, k * hs] for i in range(2) for j in range(3) for k in range(1)])
    if not args.pool_only:
        # over the bare table (x 0.32 is outside the box); fall time sqrt(2 h / g): 0.12 -> 0.156 s, 0.16 -> 0.181, 0.20 -> 0.202, ...
        add("t_single_120", [0.32, 0.120, 0.10])
        add("t_single_160", [0.32, 0.160, 0.14])
        add("t_single_178", [0.32, 0.178, 0.18])
        add("t_pair_200", [[0.32, 0.200, 0.22], [0.32, 0.200 + hs, 0.22]])
        add("t_clump_230", clump + [0.32, 0.230, 0.26])
        add("t_single_260", [0.32, 0.260, 0.30])
    if not args.table_only and not args.no_pool:
        add("p_single_100", [0.185, water_top + 0.100, 0.185])
        add("p_single_140", [0.215, water_top + 0.140, 0.185])
        add("p_pair_170", [[0.185, water_top + 0.170, 0.215], [0.185, water_top + 0.170 + hs, 0.215]])
        add("p_clump_200", clump + [0.212, water_top + 0.200, 0.212])
    pts = np.vstack(pts)
    lo = np.array([0.10, -0.005, 0.05])
    hi = np.array([0.40, 0.330, 0.40])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=1e5, particle_volume=hs ** 3, node_pool=90000, push_apart=0.9, j_mode=0,
                   stick_h=0.0, mass_floor=0.05, apic=True, apic_ratio=1.0)
    eng.finalize()
    print(f"[bench] {len(pts)} particles; groups {groups}", flush=True)
    log = []
    nt0 = eng._engine.get_total_newton_iters()
    y_prev = {k: None for k in groups}
    for step in range(args.steps):
        ts = time.time()
        eng.step()
        wall_s = time.time() - ts
        nt = eng._engine.get_total_newton_iters()
        it = nt - nt0
        nt0 = nt
        x = np.asarray(eng.get_liquid_particles())
        row = dict(step=step, t=(step + 1) * dt, newton=int(it), wall=wall_s)
        ev = []
        for k, (a, b) in groups.items():
            if k == "pool":
                continue
            y = float(x[a:b, 1].min())
            row[k] = y
            ref = 0.0 if k.startswith("t_") else water_top
            if y_prev[k] is not None and y_prev[k] - ref > 0.004 >= y - ref:
                ev.append(k)
            y_prev[k] = y
        row["events"] = ev
        log.append(row)
        flag = " <== " + ",".join(ev) if ev else ""
        if it >= 8 or ev or step % 20 == 0:
            print(f"step {step:4d} t {row['t']:.3f} newton {it:4d} wall {wall_s*1e3:7.0f} ms{flag}", flush=True)
    json.dump(log, open(os.path.join(args.out, "log.json"), "w"))
    its = np.array([r["newton"] for r in log])
    print(f"[bench] done: newton per frame mean {its.mean():.2f} max {its.max()} frames >= 50: {(its >= 50).sum()} at the cap: {(its >= args.newton_cap).sum()}", flush=True)


if __name__ == "__main__":
    main()
