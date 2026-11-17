"""Unitree Go2 on a FREE floating base in StiffGIPC: explicit PD stand and policy sim2sim rollouts (Nuka-Physics policy demos).

Contract (unitree_rl_gym / legged_gym Go2; Nuka sim_val README): physics 200 Hz (dt 5 ms), policy 50 Hz (decimation 4),
PD torque tau = Kp (q_des - q) - Kd qd with Kp 20 / Kd 0.5, torque limits 23.7 (hip, thigh) / 45.4 (calf) N m,
obs(48) = [lin_vel_body*2, ang_vel_body*0.25, projected_gravity, cmd*(2,2,0.25), (q-q_default)*1, qd*0.05, last_action],
q_des = q_default + 0.25 action. Joint order FL, FR, RL, RR x hip, thigh, calf;
q_default = [0.1,0.8,-1.5, -0.1,0.8,-1.5, 0.1,1.0,-1.5, -0.1,1.0,-1.5].

The engine joints are driven in PURE TORQUE mode (set_revolute_strength 0 + set_revolute_torque), the torque being the explicit
PD law evaluated on the engine state before each implicit step, exactly like the training set-up (Isaac Gym PD at 200 Hz).
Base velocities are finite differences of the base body transform. Link masses / COMs / inertias come from the URDF inertial
tags (set_abd_body_inertia), not from the collision hulls.

Modes:
  signtest   fixed base in the air, one joint free, constant torque -> reports the engine torque sign w.r.t. the joint angle
  stand      PD hold of q_default on the ground
  walk       motion.pt (unitree_rl_gym Go2 pre-trained policy, 48 -> 12) with a velocity command
  handstand  Nuka-trained front-handstand policy (48 -> 12, lin-vel slots zero, Kp 40 / Kd 1.92, filtered + hold scaffold)
  backflip   Nuka-trained double-backflip policy (60 -> 12, Genesis joint order, phase features, Kp 70 / Kd 3, scale 0.5)

    PYTHONPATH=. python examples/nuka/go2_policy.py --mode stand --out out/nuka/go2_stand1
"""
import argparse, json, math, os, os.path as osp, sys, time
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
from common import ASSETS, Body, Recorder, Scene, _load_urdf_capture, abd_world_tf  # noqa: E402

R_CONV = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])   # URDF z-up -> engine y-up: y_up = R_CONV @ z_up
LEGS, JNTS = ("FL", "FR", "RL", "RR"), ("hip", "thigh", "calf")
JOINT_NAMES = [f"{l}_{j}_joint" for l in LEGS for j in JNTS]
DEFAULT_Q = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
BACKFLIP_Q = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5])
HANDSTAND_DESIRED_Q = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 2.25, -1.75, 0.0, 2.25, -1.75])
TAU_LIMIT = np.array([23.7, 23.7, 45.43] * 4)
GENESIS_FROM_URDF = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])   # backflip policy order FR, FL, RR, RL (self-inverse)
POLICIES = osp.join(ASSETS, "nuka", "policies")


# ----------------------------------------------------------------------------- helpers
def rpy_to_R(rpy):
    r, p, y = [float(v) for v in rpy]
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def parse_inertials(urdf_path):
    out = {}
    for link in ET.parse(urdf_path).getroot().findall("link"):
        ine = link.find("inertial")
        if ine is None or ine.find("mass") is None:
            continue
        m = float(ine.find("mass").get("value"))
        o = ine.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
        I = ine.find("inertia")
        g = lambda k: float(I.get(k, "0")) if I is not None else 0.0   # noqa: E731
        Im = np.array([[g("ixx"), g("ixy"), g("ixz")], [g("ixy"), g("iyy"), g("iyz")], [g("ixz"), g("iyz"), g("izz")]])
        out[link.get("name")] = (m, xyz, rpy, Im)
    return out


def so3_log(R):
    """Rotation vector of R (robust small-angle form)."""
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


class Go2:
    """The Go2 in the engine: joint ids in policy order, PD torque drive, base state in the z-up policy convention."""

    def __init__(self, sc, args, root_fixed, q_init):
        eng = sc.eng
        urdf = osp.join(ASSETS, "nuka", "go2", "go2_hull.urdf")
        T = np.eye(4); T[:3, :3] = R_CONV; T[1, 3] = args.base_y
        init = {n: float(v) for n, v in zip(JOINT_NAMES, q_init)}
        n_rec0 = len(eng.get_load_records())
        log = _load_urdf_capture(eng, urdf, T, root_fixed, False, 1e7, init)
        eng.set_log_level(args.log_level)
        bad = [l for l in log.splitlines() if "Failed" in l or "error" in l.lower()]
        if bad:
            print("[warn] URDF import:", bad[:3], flush=True)
        self.bodies = []
        for r in eng.get_load_records()[n_rec0:]:
            if r.body_type != 0:
                continue
            b = Body(r, f"go2/{r.label}", "prop", color=(0.80, 0.80, 0.82))
            sc.bodies[b.name] = b; self.bodies.append(b)
            eng.set_body_friction(b.body_id, args.mu)
        self.base = next(b for b in self.bodies if b.rec.label == "base")
        self.p_load = T[:3, 3].copy()           # root link origin at load (the ABD translation itself is the mass centre)
        self.T_load, self.init = T.copy(), init
        # URDF inertials (the importer builds masses from the collision hulls otherwise)
        inert = parse_inertials(urdf)
        Tb = np.asarray(eng.native.get_urdf_link_transform("base"), float)
        pre = np.eye(4) if np.allclose(Tb[:3, :3], T[:3, :3], atol=1e-6) else T
        m_tot = 0.0; n_set = 0
        for b in self.bodies:
            if b.rec.label not in inert:
                continue
            m, xyz, rpy, Im = inert[b.rec.label]
            Tl = pre @ np.asarray(eng.native.get_urdf_link_transform(b.rec.label), float)
            com_w = (Tl @ np.append(xyz, 1.0))[:3]
            Ri = Tl[:3, :3] @ rpy_to_R(rpy)
            if not args.hull_mass:
                eng.set_abd_body_inertia(b.body_id, m, com_w, Ri @ Im @ Ri.T)
            m_tot += m; n_set += 1
        n_rev = eng.native.get_num_revolute_joints()
        rev = {eng.get_revolute_joint_info(i).name: i for i in range(n_rev)}
        self.jid = np.array([rev[n] for n in JOINT_NAMES], int)
        self.q_init = np.asarray(q_init, float).copy()
        self.sc, self.eng, self.dt = sc, eng, sc.dt
        self.q_prev = None; self.qd = np.zeros(12); self.tau = np.zeros(12)
        self.p_prev = None; self.R_prev = None; self.v_w = np.zeros(3); self.w_w = np.zeros(3)
        print(f"[go2] {len(self.bodies)} links, {n_rev} revolute joints, URDF inertials on {n_set} links ({m_tot:.2f} kg"
              f"{', NOT applied (hull masses)' if args.hull_mass else ''}), root {'fixed' if root_fixed else 'FREE'}, base y {args.base_y:.3f}", flush=True)

    # ---- state
    def q(self):
        return np.asarray(self.eng.get_revolute_current_angles_abs(), float)[self.jid]

    def update_state(self):
        """Call once per physics step BEFORE applying torques: joint angles/velocities and base pose/velocities."""
        q = self.q()
        self.qd = np.zeros(12) if self.q_prev is None else (q - self.q_prev) / self.dt
        self.q_prev = q
        C = self.eng.get_abd_body_transforms(np.array([self.base.body_id], np.int32))[0]
        A = abd_world_tf(C)
        M = A @ self.base.C0inv                   # rigid motion of the base relative to the loaded pose (y-up world)
        R_w, p_w = M[:3, :3], (M @ np.append(self.p_load, 1.0))[:3]
        if self.p_prev is not None:
            self.v_w = (p_w - self.p_prev) / self.dt
            self.w_w = so3_log(R_w @ self.R_prev.T) / self.dt
        self.p_prev, self.R_prev = p_w.copy(), R_w.copy()
        self.R_zup = R_CONV.T @ R_w @ R_CONV     # base orientation in the z-up (URDF) world convention
        self.p_zup = R_CONV.T @ p_w
        self.cur_q = q
        return q

    def base_obs(self):
        Rt = self.R_zup.T
        lin_b = Rt @ (R_CONV.T @ self.v_w)
        ang_b = Rt @ (R_CONV.T @ self.w_w)
        grav_b = Rt @ np.array([0.0, 0.0, -1.0])
        return lin_b, ang_b, grav_b

    def tilt_deg(self):
        return math.degrees(math.acos(float(np.clip(self.R_zup[2, 2], -1.0, 1.0))))

    # ---- drive
    def torque_mode(self):
        for j in self.jid:
            self.eng.set_revolute_strength(int(j), 0.0)

    def pd(self, q_des, kp, kd, limit, sign):
        tau = kp * (q_des - self.cur_q) - kd * self.qd
        tau = np.clip(tau, -limit, limit)
        self.tau = tau
        for j, t in zip(self.jid, tau):
            self.eng.native.set_revolute_torque(int(j), float(sign * t))

    def feet_min_y(self):
        ys = []
        for b in self.bodies:
            if b.rec.label.endswith("_foot") or b.rec.label.endswith("_calf"):
                V = np.asarray(self.eng.get_abd_body(b.body_id).get_vertices(), float)
                ys.append(V[:, 1].min())
        return min(ys) if ys else float("nan")


def load_torch_policy(path):
    import torch
    m = torch.jit.load(path, map_location="cpu").eval()

    def run(obs):
        with torch.no_grad():
            return m(torch.from_numpy(np.ascontiguousarray(obs, np.float32)[None]))[0].numpy().astype(np.float64)
    return run


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="stand", choices=["signtest", "stand", "walk", "handstand", "backflip"])
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--decimation", type=int, default=4)
    ap.add_argument("--duration", type=float, default=4.0, help="seconds AFTER the settle phase")
    ap.add_argument("--settle", type=float, default=1.0, help="PD hold of the default pose before the policy starts")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--base-y", type=float, default=0.376)
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--dhat", type=float, default=2e-3)
    ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--kd", type=float, default=0.5)
    ap.add_argument("--tau-sign", type=float, default=1.0, help="engine torque sign w.r.t. the joint angle (see --mode signtest)")
    ap.add_argument("--tau", type=float, default=3.0, help="signtest torque (N m)")
    ap.add_argument("--cmd", type=str, default="0.5,0,0", help="walk command vx, vy, yaw rate")
    ap.add_argument("--hull-mass", action="store_true", help="keep the importer's hull-derived masses (no URDF inertials)")
    ap.add_argument("--newton-cap", type=int, default=40)
    ap.add_argument("--newton-tol", type=float, default=1e-4)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    ap.add_argument("--ptype", type=int, default=0)
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sc = Scene(dict(dt=args.dt, preconditioner_type=args.ptype, newton_iter_cap=args.newton_cap, absolute_dhat=args.dhat,
                    newton_tol=args.newton_tol, pcg_tol=args.pcg_tol, friction_rate=args.mu), log_level=args.log_level)
    mode = args.mode
    q0 = BACKFLIP_Q if mode == "backflip" else DEFAULT_Q
    if mode == "backflip":
        args.base_y = max(args.base_y, 0.35 + 0.03)
    robot = Go2(sc, args, root_fixed=(mode == "signtest"), q_init=q0)
    sc.finalize()
    rec = Recorder(sc)
    dt = args.dt
    log_every = max(1, int(round(0.25 / dt)))
    rec_every = max(1, int(round(1.0 / (args.fps * dt))))
    meta = dict(example="go2_policy", mode=mode, dt=dt, fps=args.fps, kp=args.kp, kd=args.kd, mu=args.mu, dhat=args.dhat,
                fem_materials={}, fem_colors=[], robot=dict(urdf="nuka/go2/go2_hull.urdf", prefix="go2/", T=robot.T_load.tolist(), init=robot.init))
    series = []
    t0 = time.time()

    # ------------------------------------------------------------------ sign test
    if mode == "signtest":
        j_free = 2                                   # FL_calf
        sc.eng.set_revolute_strength(int(robot.jid[j_free]), 0.0)
        n_steps = int(round(0.3 / dt))
        print(f"[signtest] FL_calf free (strength 0), ext torque {args.tau:+.2f} N m, base fixed; other joints held", flush=True)
        for k in range(n_steps + 1):
            q = robot.update_state()
            sc.eng.native.set_revolute_torque(int(robot.jid[j_free]), float(args.tau))
            if k % max(1, int(round(0.025 / dt))) == 0:
                print(f"[{k*dt:5.3f}s] FL_calf q {q[j_free]:+.4f} (dq {q[j_free]-q0[j_free]:+.4f}) qd {robot.qd[j_free]:+.3f} | others |dq|max {np.abs(np.delete(q - q0, j_free)).max():.4f}", flush=True)
            if k % rec_every == 0:
                rec.record(k * dt, 0, "signtest")
            sc.step()
        dq = robot.update_state()[j_free] - q0[j_free]
        print(f"[signtest] torque {args.tau:+.2f} N m -> dq {dq:+.4f} rad after 0.3 s  => tau_sign = {'+1' if dq * args.tau > 0 else '-1'} (if gravity drift is smaller)", flush=True)
        rec.save(osp.join(args.out, "recording.npz"), meta)
        return

    # ------------------------------------------------------------------ PD modes
    robot.torque_mode()
    kp, kd, limit = args.kp, args.kd, TAU_LIMIT
    policy = None; action = np.zeros(12); last_action = np.zeros(12); q_des = q0.copy()
    cmd = np.array([float(v) for v in args.cmd.split(",")])
    if mode == "walk":
        policy = load_torch_policy(osp.join(POLICIES, "motion.pt"))
    elif mode == "handstand":
        policy = load_torch_policy(osp.join(POLICIES, "nuka_go2_front_handstand.pt")); kp, kd, limit = 40.0, 1.92, np.full(12, 33.5)
        filtered = np.zeros(12); startup = int(round(1.5 / (args.decimation * dt))); ramp_n = int(round(0.75 / (args.decimation * dt))); warm_done = False
    elif mode == "backflip":
        policy = load_torch_policy(osp.join(POLICIES, "nuka_go2_backflip.pt")); kp, kd, limit = 70.0, 3.0, np.full(12, 33.5)
        rot_int = 0.0; rot = 0.0; switched = False; peak = 0.0
    n_settle = int(round(args.settle / dt)); n_run = int(round(args.duration / dt)); n_steps = n_settle + n_run
    newton0 = sc.eng.native.get_total_newton_iters()
    print(f"[run] mode {mode}: settle {args.settle:.1f} s (PD on q_default) + {args.duration:.1f} s, dt {dt*1e3:.0f} ms, policy every {args.decimation} steps, "
          f"Kp {kp} Kd {kd}, cmd {cmd.tolist()}", flush=True)
    kpol = 0
    for k in range(n_steps + 1):
        t = k * dt
        q = robot.update_state()
        lin_b, ang_b, grav_b = robot.base_obs()
        active = policy is not None and k >= n_settle
        if active and (k - n_settle) % args.decimation == 0:
            if mode == "walk":
                obs = np.concatenate([lin_b * 2.0, ang_b * 0.25, grav_b, cmd * np.array([2.0, 2.0, 0.25]), (q - DEFAULT_Q), robot.qd * 0.05, last_action])
                action = np.clip(policy(np.clip(obs, -100, 100)), -100, 100)
                q_des = DEFAULT_Q + 0.25 * action
                last_action = action.copy()
            elif mode == "handstand":
                obs = np.concatenate([np.zeros(3), ang_b * 0.25, grav_b, np.zeros(3), (q - DEFAULT_Q), robot.qd * 0.05, action])
                obs = np.clip(obs, -100, 100)
                if kpol >= startup:
                    ptar = np.clip(policy(obs), -100, 100) * 0.25
                    prog = float(np.clip((-grav_b[0] - 0.45) / 0.30, 0.0, 1.0))
                    alpha = 0.05 * (1.0 - prog) + 0.30 * prog
                    filtered = alpha * filtered + (1.0 - alpha) * ptar
                ramp = 1.0 if warm_done else max(0.0, min(1.0, (kpol - startup + 1) / max(1, ramp_n)))
                if ramp >= 1.0:
                    warm_done = True
                if ramp <= 0.0:
                    delta = np.zeros(12)
                else:
                    prog = float(np.clip((-grav_b[0] - 0.45) / 0.30, 0.0, 1.0))
                    hold = 0.08 * ramp * prog
                    delta = ((1.0 - hold) * filtered + hold * (HANDSTAND_DESIRED_Q - DEFAULT_Q)) * ramp
                action = delta / 0.25
                q_des = DEFAULT_Q + delta
            elif mode == "backflip":
                qg = (q - BACKFLIP_Q)[GENESIS_FROM_URDF]; qdg = robot.qd[GENESIS_FROM_URDF]
                ph = math.pi * min(kpol, 150) / 150.0
                phase = np.array([math.sin(ph), math.cos(ph), math.sin(ph / 2), math.cos(ph / 2), math.sin(ph / 4), math.cos(ph / 4)])
                obs = np.concatenate([ang_b * 0.25, grav_b, qg, qdg * 0.05, action, last_action, phase])
                rot_int += float(ang_b[1]) * (args.decimation * dt)
                rot = -math.degrees(rot_int); peak = max(peak, rot)
                exec_action = last_action.copy()
                if not switched:
                    new_action = np.clip(policy(np.clip(obs, -100, 100)), -100, 100)
                    if rot >= 300.0:
                        switched = True; new_action = np.zeros(12)
                else:
                    new_action = np.zeros(12)
                target_gen = BACKFLIP_Q[GENESIS_FROM_URDF] + 0.5 * exec_action
                q_des = target_gen[GENESIS_FROM_URDF]
                last_action = new_action.copy(); action = new_action.copy()
            kpol += 1
        robot.pd(q_des, kp, kd, limit, args.tau_sign)
        if k % log_every == 0:
            nit = sc.eng.native.get_total_newton_iters()
            vx = float((R_CONV.T @ robot.v_w)[0])
            extra = ""
            if mode == "backflip":
                extra = f" rot {rot:6.1f} deg{' SWITCHED' if switched else ''}"
            print(f"[{t:5.2f}s] {'policy' if active else 'settle'} z {robot.p_zup[2]:.3f} tilt {robot.tilt_deg():5.1f} deg vx {vx:+.2f} lin_b {lin_b.round(2).tolist()} |qd|max {np.abs(robot.qd).max():5.1f} "
                  f"|tau|max {np.abs(robot.tau).max():5.1f} |q-q0|max {np.abs(q - q0).max():.2f} feet_y {robot.feet_min_y()*1e3:6.1f} mm newton/step {(nit-newton0)/log_every:4.1f}{extra} | {(time.time()-t0)/60:4.1f} min", flush=True)
            newton0 = nit
        series.append(dict(t=t, z=float(robot.p_zup[2]), x=float(robot.p_zup[0]), y=float(robot.p_zup[1]), tilt=robot.tilt_deg(), vx=float((R_CONV.T @ robot.v_w)[0]),
                           grav=grav_b.round(4).tolist(), q=q.round(4).tolist(), tau=robot.tau.round(3).tolist(), action=action.round(4).tolist()))
        if k % rec_every == 0:
            rec.record(t, 0 if not active else 1, "settle" if not active else mode)
        sc.step()
    meta["wall_min"] = (time.time() - t0) / 60
    rec.save(osp.join(args.out, "recording.npz"), meta)
    zs = np.array([s["z"] for s in series[n_settle:]]); tilts = np.array([s["tilt"] for s in series[n_settle:]]); vxs = np.array([s["vx"] for s in series[n_settle:]])
    summ = dict(mode=mode, z_mean=float(zs.mean()), z_min=float(zs.min()), tilt_mean=float(tilts.mean()), tilt_max=float(tilts.max()),
                vx_mean=float(vxs.mean()), x_travel=float(series[-1]["x"] - series[n_settle]["x"]), wall_min=meta["wall_min"],
                fell=bool(tilts.max() > 60.0 or zs.min() < 0.12))
    if mode == "backflip":
        summ.update(peak_rot_deg=peak, switched=switched)
    json.dump(dict(summary=summ, series=series), open(osp.join(args.out, "metrics.json"), "w"))
    print(f"[done] {n_steps} steps in {meta['wall_min']:.1f} min; summary {json.dumps(summ)}", flush=True)


if __name__ == "__main__":
    main()
