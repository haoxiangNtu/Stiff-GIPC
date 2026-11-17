"""Throughput probe for RL-style batching: N Unitree Go2 robots (free floating bases, explicit PD at 200 Hz) in ONE engine
instance, spaced along z. Reports wall time per physics step and env-steps per second (one env-step = one robot x one step).

    PYTHONPATH=. python examples/nuka/go2_multi.py --n 16 --duration 2 --out out/nuka/go2_multi16
"""
import argparse, json, os, os.path as osp, sys, time
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
from common import ASSETS, Body, Scene, _load_urdf_capture  # noqa: E402
from go2_policy import DEFAULT_Q, JOINT_NAMES, R_CONV, TAU_LIMIT, parse_inertials, rpy_to_R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--kd", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--dhat", type=float, default=2e-3)
    ap.add_argument("--newton-cap", type=int, default=40)
    ap.add_argument("--newton-tol", type=float, default=1e-4)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    ap.add_argument("--ptype", type=int, default=0)
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--random-targets", type=float, default=0.0, help="amplitude (rad) of random per-robot target perturbations at 50 Hz (exercise the solver)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=args.ptype, newton_iter_cap=args.newton_cap, absolute_dhat=args.dhat,
                    newton_tol=args.newton_tol, pcg_tol=args.pcg_tol, friction_rate=args.mu), log_level=args.log_level)
    eng = sc.eng
    urdf = osp.join(ASSETS, "nuka", "go2", "go2_hull.urdf")
    inert = parse_inertials(urdf)
    init = {n: float(v) for n, v in zip(JOINT_NAMES, DEFAULT_Q)}
    robots = []
    t_load = time.time()
    for r in range(args.n):
        T = np.eye(4); T[:3, :3] = R_CONV; T[1, 3] = 0.376; T[2, 3] = (r - (args.n - 1) / 2) * args.spacing
        n_rec0 = len(eng.get_load_records()); n_rev0 = eng.native.get_num_revolute_joints()
        _load_urdf_capture(eng, urdf, T, False, False, 1e7, init)
        eng.set_log_level(args.log_level)
        bodies = []
        for rec in eng.get_load_records()[n_rec0:]:
            if rec.body_type != 0:
                continue
            b = Body(rec, f"go2_{r}/{rec.label}", "prop"); sc.bodies[b.name] = b; bodies.append(b)
            eng.set_body_friction(b.body_id, args.mu)
            if rec.label in inert:
                m, xyz, rpy, Im = inert[rec.label]
                Tl = np.asarray(eng.native.get_urdf_link_transform(rec.label), float)
                if not np.allclose(Tl[:3, :3], T[:3, :3], atol=1e-6):
                    Tl = T @ Tl
                com_w = (Tl @ np.append(xyz, 1.0))[:3]; Ri = Tl[:3, :3] @ rpy_to_R(rpy)
                eng.set_abd_body_inertia(b.body_id, m, com_w, Ri @ Im @ Ri.T)
        n_rev = eng.native.get_num_revolute_joints()
        names = {eng.get_revolute_joint_info(i).name: i for i in range(n_rev0, n_rev)}
        robots.append(dict(bodies=bodies, jid=np.array([names[n] for n in JOINT_NAMES], int)))
    # robots never touch each other: exclude their mutual collisions to keep the broad phase honest about what an RL batch needs
    for a in range(args.n):
        for b in range(a + 1, args.n):
            for ba in robots[a]["bodies"]:
                for bb in robots[b]["bodies"]:
                    eng.add_collision_exclusion(ba.body_id, bb.body_id)
    sc.finalize()
    print(f"[multi] {args.n} Go2 loaded in {time.time()-t_load:.1f} s: {len(sc.bodies)} bodies, {eng.native.get_num_revolute_joints()} revolute joints, "
          f"{eng.native.get_vertex_count()} vertices", flush=True)
    jid_all = np.concatenate([r["jid"] for r in robots])
    for j in jid_all:
        eng.set_revolute_strength(int(j), 0.0)
    q_prev = None; rng = np.random.default_rng(0)
    q_des = np.tile(DEFAULT_Q, args.n)
    n_steps = int(round(args.duration / args.dt))
    newton0 = eng.native.get_total_newton_iters()
    t0 = time.time(); t_warm = None; wall = []
    for k in range(n_steps):
        q = np.asarray(eng.get_revolute_current_angles_abs(), float)[jid_all]
        qd = np.zeros_like(q) if q_prev is None else (q - q_prev) / args.dt
        q_prev = q
        if args.random_targets > 0 and k % 4 == 0:
            q_des = np.tile(DEFAULT_Q, args.n) + rng.uniform(-args.random_targets, args.random_targets, q_des.shape)
        tau = np.clip(args.kp * (q_des - q) - args.kd * qd, -np.tile(TAU_LIMIT, args.n), np.tile(TAU_LIMIT, args.n))
        for j, t in zip(jid_all, tau):
            eng.native.set_revolute_torque(int(j), float(t))
        ts = time.time(); sc.step(); wall.append(time.time() - ts)
        if k == int(0.5 / args.dt):
            t_warm = time.time(); newton0 = eng.native.get_total_newton_iters(); wall = []
        if k % int(round(0.5 / args.dt)) == 0:
            # base heights of all robots (mean of the base body vertices y) + stability
            ys = [np.asarray(eng.get_abd_body(r["bodies"][0].body_id).get_vertices(), float)[:, 1].mean() for r in robots]
            print(f"[{k*args.dt:5.2f}s] base y mean {np.mean(ys):.3f} min {np.min(ys):.3f} max {np.max(ys):.3f} | step {np.mean(wall[-20:])*1e3 if wall else 0:6.1f} ms "
                  f"| newton/step {(eng.native.get_total_newton_iters()-newton0)/max(1,len(wall)):4.1f} | {(time.time()-t0)/60:4.1f} min", flush=True)
    n_meas = len(wall)
    per_step = float(np.mean(wall)) if wall else float("nan")
    summ = dict(n=args.n, dt=args.dt, steps_measured=n_meas, ms_per_step=per_step * 1e3, env_steps_per_s=args.n / per_step if per_step > 0 else 0.0,
                sim_s_per_wall_s=args.dt / per_step if per_step > 0 else 0.0, newton_per_step=(eng.native.get_total_newton_iters() - newton0) / max(1, n_meas),
                vertices=int(eng.native.get_vertex_count()), wall_total_min=(time.time() - t0) / 60)
    json.dump(summ, open(osp.join(args.out, "summary.json"), "w"))
    print(f"[multi] n={args.n}: {per_step*1e3:.1f} ms/step, {summ['env_steps_per_s']:.0f} env-steps/s, {summ['sim_s_per_wall_s']:.3f}x realtime per env, newton/step {summ['newton_per_step']:.1f}", flush=True)


if __name__ == "__main__":
    main()
