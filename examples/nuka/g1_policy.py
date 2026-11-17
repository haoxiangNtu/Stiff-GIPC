"""Unitree G1 (29 DoF + Dex3 hands) on a FREE floating base in StiffGIPC: PD stand and the G1 Moves 'Shuffle' dance policy
(sim2sim of an externally trained ONNX actor, as in the Nuka-Physics 'G1 dance' demo).

Contract (experientialtech/g1-moves run_policy.py, mjlab training): physics 200 Hz, policy 50 Hz, torque PD with the per-joint
mjlab gains (KP / KD below), joint target = default (all zeros) + action, obs(160) =
[ref_q(29), ref_qd(29), anchor_pos_b(3), anchor_ori_b(6), base_ang_vel_b(3), base_lin_vel_b(3), q(29), qd(29), last_action(29)]
(--obs-order anglin = the dataset player's order; linang = lin before ang, the order documented by Nuka). The motion anchor is the
reference pelvis pose expressed in the robot's pelvis frame. The reference clip J_Dance17_Shuffle is sampled at 60 Hz.
Joint order: L hip p/r/y, L knee, L ankle p/r, R hip p/r/y, R knee, R ankle p/r, waist y/r/p, L shoulder p/r/y, L elbow,
L wrist r/p/y, R shoulder p/r/y, R elbow, R wrist r/p/y. The 14 Dex3 hand joints are position-held at zero.

    PYTHONPATH=. python examples/nuka/g1_policy.py --mode stand --out out/nuka/g1_stand1
    PYTHONPATH=. python examples/nuka/g1_policy.py --mode dance --duration 20 --out out/nuka/g1_dance1
"""
import argparse, json, math, os, os.path as osp, sys, time
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "tools"))
from common import ASSETS, Body, Recorder, Scene, _load_urdf_capture, abd_world_tf  # noqa: E402
from urdf_kin import UrdfKin, so3_log  # noqa: E402
from go2_policy import R_CONV, parse_inertials, rpy_to_R  # noqa: E402

LEGS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARMS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
JOINTS29 = LEGS + WAIST + ARMS
HANDS = [f"{s}_hand_{j}_joint" for s in ("left", "right") for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]
KP = np.array([40.2, 99.1, 40.2, 99.1, 28.6, 28.6, 40.2, 99.1, 40.2, 99.1, 28.6, 28.6, 40.2, 28.6, 28.6,
               14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8, 14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8])
KD = np.array([2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 1.8, 1.8,
               0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1])
# mjlab (pinned e788f29) G1 tracking task: default joint pose = KNEES_BENT keyframe, action scale = 0.25 * effort / stiffness per
# actuator (7520_14: 88 N m / 40.2; 7520_22: 139 / 99.1; 5020: 25 / 14.3; 4010: 5 / 16.8; waist roll/pitch + ankles: 2 x 5020),
# torque clip = the actuator effort limit, motion anchor = torso_link.
DEFAULT_Q = np.zeros(29)
for _i in (0, 6): DEFAULT_Q[_i] = -0.312          # hip pitch
for _i in (3, 9): DEFAULT_Q[_i] = 0.669           # knee
for _i in (4, 10): DEFAULT_Q[_i] = -0.363         # ankle pitch
DEFAULT_Q[15] = DEFAULT_Q[22] = 0.2               # shoulder pitch
DEFAULT_Q[16], DEFAULT_Q[23] = 0.2, -0.2          # shoulder roll
DEFAULT_Q[18] = DEFAULT_Q[25] = 0.6               # elbow
_EFF_LEG = [88.0, 139.0, 88.0, 139.0, 50.0, 50.0]; _EFF_ARM = [25.0] * 5 + [5.0] * 2
EFFORT = np.array(_EFF_LEG * 2 + [88.0, 50.0, 50.0] + _EFF_ARM * 2)
ACTION_SCALE = 0.25 * EFFORT / KP
ANCHOR_LINK = "torso_link"; ANCHOR_REF_INDEX = 15   # torso_link in the clip's 30-body list (MuJoCo tree order)
URDF = osp.join(ASSETS, "nuka", "g1", "g1_hull.urdf")
POLICIES = osp.join(ASSETS, "nuka", "policies")


def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def Rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def foot_clearance_base_z(kin, q, clearance=0.005):
    """Base (pelvis) height in the z-up world that puts the lowest foot-hull vertex `clearance` above the ground."""
    import trimesh
    T = kin.fk(q)
    zmin = 1e9
    for side in ("left", "right"):
        link = f"{side}_ankle_roll_link"
        m = trimesh.load(osp.join(ASSETS, "nuka", "g1", "meshes", f"{link}_hull.stl"), force="mesh")
        V = (T[link] @ np.c_[m.vertices, np.ones(len(m.vertices))].T).T[:, :3]
        zmin = min(zmin, float(V[:, 2].min()))
    return clearance - zmin


class G1:
    def __init__(self, sc, args, q_init, xy0=(0.0, 0.0), yaw0=0.0):
        eng = sc.eng
        self.kin = UrdfKin(URDF)
        base_z = foot_clearance_base_z(self.kin, {n: float(v) for n, v in zip(JOINTS29, q_init)})
        self.R_B0 = Rz(yaw0)
        T = np.eye(4); T[:3, :3] = R_CONV @ self.R_B0; T[:3, 3] = R_CONV @ np.array([xy0[0], xy0[1], base_z])
        init = {n: float(v) for n, v in zip(JOINTS29, q_init) if abs(v) > 0}
        n_rec0 = len(eng.get_load_records())
        log = _load_urdf_capture(eng, URDF, T, False, False, 1e7, init)
        eng.set_log_level(args.log_level)
        bad = [l for l in log.splitlines() if "Failed" in l or "error" in l.lower()]
        if bad:
            print("[warn] URDF import:", bad[:5], flush=True)
        self.bodies = []
        for r in eng.get_load_records()[n_rec0:]:
            if r.body_type != 0:
                continue
            b = Body(r, f"g1/{r.label}", "prop", color=(0.82, 0.82, 0.84))
            sc.bodies[b.name] = b; self.bodies.append(b)
            eng.set_body_friction(b.body_id, args.mu)
        self.base = next(b for b in self.bodies if b.rec.label == "pelvis")
        self.anchor = next(b for b in self.bodies if b.rec.label == ANCHOR_LINK)
        self.p_load = T[:3, 3].copy()
        self.T_load, self.init = T.copy(), init
        inert = parse_inertials(URDF)
        self.inert = inert
        Tb = np.asarray(eng.native.get_urdf_link_transform("pelvis"), float)
        pre = np.eye(4) if np.allclose(Tb[:3, :3], T[:3, :3], atol=1e-6) else T
        m_tot = 0.0
        for b in self.bodies:
            if b.rec.label not in inert:
                continue
            m, xyz, rpy, Im = inert[b.rec.label]
            Tl = pre @ np.asarray(eng.native.get_urdf_link_transform(b.rec.label), float)
            com_w = (Tl @ np.append(xyz, 1.0))[:3]; Ri = Tl[:3, :3] @ rpy_to_R(rpy)
            eng.set_abd_body_inertia(b.body_id, m, com_w, Ri @ Im @ Ri.T); m_tot += m
        n_rev = eng.native.get_num_revolute_joints()
        rev = {eng.get_revolute_joint_info(i).name: i for i in range(n_rev)}
        missing = [n for n in JOINTS29 + HANDS if n not in rev]
        if missing:
            raise RuntimeError(f"joints missing in the engine: {missing}")
        self.jid = np.array([rev[n] for n in JOINTS29], int)
        self.hid = np.array([rev[n] for n in HANDS], int)
        self.effort = EFFORT.copy()
        Ta = pre @ np.asarray(eng.native.get_urdf_link_transform(ANCHOR_LINK), float)
        self.T_anchor_load = Ta.copy()
        self.sc, self.eng, self.dt = sc, eng, sc.dt
        self.q_prev = None; self.qd = np.zeros(29); self.tau = np.zeros(29)
        self.p_prev = None; self.R_prev = None; self.v_w = np.zeros(3); self.w_w = np.zeros(3)
        print(f"[g1] {len(self.bodies)} links, {n_rev} revolute joints (29 driven + 14 hand), URDF mass {m_tot:.2f} kg, base z {base_z:.3f} m, "
              f"xy0 {np.round(xy0, 3).tolist()} yaw0 {math.degrees(yaw0):.1f} deg", flush=True)

    def update_state(self):
        q = np.asarray(self.eng.get_revolute_current_angles_abs(), float)[self.jid]
        self.qd = np.zeros(29) if self.q_prev is None else (q - self.q_prev) / self.dt
        self.q_prev = q
        C = self.eng.get_abd_body_transforms(np.array([self.base.body_id], np.int32))[0]
        A = abd_world_tf(C); M = A @ self.base.C0inv
        R_w, p_w = M[:3, :3], (M @ np.append(self.p_load, 1.0))[:3]
        if self.p_prev is not None:
            self.v_w = (p_w - self.p_prev) / self.dt
            self.w_w = so3_log(R_w @ self.R_prev.T) / self.dt
        self.p_prev, self.R_prev = p_w.copy(), R_w.copy()
        self.R_zup = (R_CONV.T @ R_w @ R_CONV) @ self.R_B0
        self.p_zup = R_CONV.T @ p_w
        # anchor body (torso_link) pose in the z-up world
        Ca = self.eng.get_abd_body_transforms(np.array([self.anchor.body_id], np.int32))[0]
        Ma = abd_world_tf(Ca) @ self.anchor.C0inv
        Ta = Ma @ self.T_anchor_load
        self.R_anchor_zup = R_CONV.T @ Ta[:3, :3]
        self.p_anchor_zup = R_CONV.T @ Ta[:3, 3]
        self.cur_q = q
        return q

    def base_obs(self):
        Rt = self.R_zup.T
        return Rt @ (R_CONV.T @ self.v_w), Rt @ (R_CONV.T @ self.w_w)

    def tilt_deg(self):
        return math.degrees(math.acos(float(np.clip(self.R_zup[2, 2], -1.0, 1.0))))

    def torque_mode(self):
        self.backend = "torque"
        for j in self.jid:
            self.eng.set_revolute_strength(int(j), 0.0)

    def engine_pd_mode(self, kp_scale, kd_scale, kd_cap, ratio):
        """Fully IMPLICIT PD through the engine's revolute drive. The drive is a spring K (theta - target) with torque
        K dtheta / dt^2 and K = ratio * strength * (m_p + m_c) (importer masses = Config.density x hull volume). A damping term
        -Kd qd with the backward-Euler velocity qd = (theta_new - theta_now) / dt is the spring (Kd / dt)(theta_now - theta_new),
        so stiffness + damping = ONE spring K_tot = Kp + Kd / dt toward the blended target
        t* = (Kp q_des + (Kd / dt) theta_now) / K_tot. No explicit torque, no stability limit on Kd."""
        self.backend = "engine"
        self.eng.set_max_revolute_step_per_frame(100.0)     # no per-frame target slew (default 0.1 rad caps the drive torque at K * 0.1)
        import trimesh
        hull_mass = {}
        for b in self.bodies:
            try:
                hull_mass[b.rec.label] = 1000.0 * float(abs(trimesh.load(osp.join(ASSETS, "nuka", "g1", "meshes", f"{b.rec.label}_hull.stl"), force="mesh").volume))
            except Exception:   # noqa: BLE001
                hull_mass[b.rec.label] = 0.0
        self.kp_i = KP * kp_scale; self.kd_i = KD * kd_scale
        self.k_tot = self.kp_i + self.kd_i / self.dt
        self.strength = np.zeros(29)
        for i, n in enumerate(JOINTS29):
            j = self.kin.joints[n]
            m_sum = hull_mass.get(j["parent"], 0.0) + hull_mass.get(j["child"], 0.0)
            self.strength[i] = self.k_tot[i] * self.dt ** 2 / (ratio * m_sum)
            self.eng.set_revolute_strength(int(self.jid[i]), float(self.strength[i]))
        print(f"[g1] implicit engine PD: K_tot = Kp + Kd/dt in [{self.k_tot.min():.0f}, {self.k_tot.max():.0f}] N m/rad, strength [{self.strength.min():.2e}, {self.strength.max():.2e}], ratio {ratio}", flush=True)

    def pd(self, q_des, kp_scale=1.0, kd_scale=1.0):
        if getattr(self, "backend", "torque") == "engine":
            tau_lin = self.kp_i * (q_des - self.cur_q) - self.kd_i * self.qd          # expected torque (for logging / effort clip)
            q_des_c = q_des.copy()
            over = np.abs(tau_lin) > self.effort
            if np.any(over):   # clip to the actuator effort limit by moving the stiffness target
                q_des_c[over] = self.cur_q[over] + (np.sign(tau_lin[over]) * self.effort[over] + self.kd_i[over] * self.qd[over]) / self.kp_i[over]
            target = (self.kp_i * q_des_c + (self.kd_i / self.dt) * self.cur_q) / self.k_tot
            self.tau = np.clip(tau_lin, -self.effort, self.effort)
            for j, t in zip(self.jid, target):
                self.eng.set_revolute_target(int(j), float(t))
        else:
            tau = np.clip(KP * kp_scale * (q_des - self.cur_q) - KD * kd_scale * self.qd, -self.effort, self.effort)
            self.tau = tau
            for j, t in zip(self.jid, tau):
                self.eng.native.set_revolute_torque(int(j), float(t))
        for j in self.hid:
            self.eng.set_revolute_target(int(j), 0.0)

    def drive_check(self):
        """Measured drive torque / angle error for the engine backend (should be ~Kp)."""
        out = []
        for i in (0, 3, 15, 20):
            t = self.eng.get_revolute_drive_torque(int(self.jid[i]))
            out.append(f"{JOINTS29[i].replace('_joint','')}: drive {t:+.1f} vs PD {self.tau[i]:+.1f} N m")
        return "; ".join(out)

    def feet_min_y(self):
        ys = []
        for b in self.bodies:
            if b.rec.label.endswith("ankle_roll_link"):
                V = np.asarray(self.eng.get_abd_body(b.body_id).get_vertices(), float); ys.append(V[:, 1].min())
        return min(ys) if ys else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="stand", choices=["stand", "dance"])
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--decimation", type=int, default=4)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--settle", type=float, default=0.2)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--dhat", type=float, default=2e-3)
    ap.add_argument("--pd-backend", default="engine", choices=["engine", "torque"],
                    help="engine: implicit position drive with K = Kp (strength = Kp dt^2 / (ratio (m_p + m_c))) + explicit damping torque capped for "
                         "stability; torque: fully explicit PD torque (unstable for the light wrist links at dt 5 ms)")
    ap.add_argument("--kd-cap", type=float, default=0.3, help="explicit damping cap: Kd_eff = min(Kd, kd_cap * I_child / dt)")
    ap.add_argument("--drive-ratio", type=float, default=1000.0, help="Config.revolute_driving_strength_ratio used by the Scene scaffolding")
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--kd-scale", type=float, default=1.0)
    ap.add_argument("--obs-order", default="linang", choices=["anglin", "linang"], help="mjlab actor obs: base_lin_vel before base_ang_vel")
    ap.add_argument("--clip", default="J_Dance17_Shuffle")
    ap.add_argument("--speed", type=float, default=1.0, help="reference playback speed")
    ap.add_argument("--init-ref", type=int, default=1, help="dance: start at the reference frame-0 joint pose (1) or at q = 0 (0)")
    ap.add_argument("--action-scale", type=float, default=1.0, help="multiplier on the mjlab per-joint action scale 0.25*effort/stiffness")
    ap.add_argument("--newton-cap", type=int, default=40)
    ap.add_argument("--newton-tol", type=float, default=1e-4)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    ap.add_argument("--ptype", type=int, default=0)
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref = None; xy0 = (0.0, 0.0); yaw0 = 0.0
    if args.mode == "dance":
        ref = np.load(osp.join(POLICIES, f"{args.clip}.npz"))
        ref_fps = float(np.asarray(ref["fps"]).ravel()[0]); n_ref = int(ref["joint_pos"].shape[0])
        p0 = ref["body_pos_w"][0, 0]; R0 = quat_wxyz_to_R(ref["body_quat_w"][0, 0])
        xy0 = (float(p0[0]), float(p0[1])); yaw0 = math.atan2(R0[1, 0], R0[0, 0])
        print(f"[ref] {args.clip}: {n_ref} frames at {ref_fps:.0f} Hz, pelvis0 {np.round(p0, 3).tolist()} yaw0 {math.degrees(yaw0):.1f} deg, "
              f"ref q0 |max| {np.abs(ref['joint_pos'][0]).max():.2f}", flush=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=args.ptype, newton_iter_cap=args.newton_cap, absolute_dhat=args.dhat,
                    newton_tol=args.newton_tol, pcg_tol=args.pcg_tol, friction_rate=args.mu), log_level=args.log_level)
    q0 = np.zeros(29)                      # the policy's default joint position (all zeros)
    q_init = ref["joint_pos"][0].astype(np.float64) if (ref is not None and args.init_ref) else q0.copy()
    robot = G1(sc, args, q_init, xy0, yaw0)
    sc.finalize()
    if args.pd_backend == "engine":
        robot.engine_pd_mode(args.kp_scale, args.kd_scale, args.kd_cap, args.drive_ratio)
    else:
        robot.torque_mode()
    rec = Recorder(sc)
    dt = args.dt
    n_settle = int(round(args.settle / dt)); n_run = int(round(args.duration / dt)); n_steps = n_settle + n_run
    log_every = max(1, int(round(0.25 / dt))); rec_every = max(1, int(round(1.0 / (args.fps * dt))))
    session = None
    if args.mode == "dance":
        import onnxruntime as ort
        session = ort.InferenceSession(osp.join(POLICIES, f"{args.clip}_policy.onnx"), providers=["CPUExecutionProvider"])
        in_name = session.get_inputs()[0].name
    q_des = q_init.copy(); last_action = np.zeros(29); action = np.zeros(29)
    series = []; t0 = time.time(); newton0 = sc.eng.native.get_total_newton_iters(); n_pol = 0
    print(f"[run] mode {args.mode}: settle {args.settle:.1f} s + {args.duration:.1f} s, dt {dt*1e3:.0f} ms, obs order {args.obs_order}, kp x{args.kp_scale} kd x{args.kd_scale}", flush=True)
    for k in range(n_steps + 1):
        t = k * dt
        q = robot.update_state()
        lin_b, ang_b = robot.base_obs()
        active = session is not None and k >= n_settle
        ref_q = None
        if active and (k - n_settle) % args.decimation == 0:
            tm = (k - n_settle) * dt * args.speed
            f = int(tm * ref_fps) % n_ref
            ref_q = ref["joint_pos"][f].astype(np.float64); ref_qd = ref["joint_vel"][f].astype(np.float64)
            p_a = ref["body_pos_w"][f, ANCHOR_REF_INDEX].astype(np.float64); R_a = quat_wxyz_to_R(ref["body_quat_w"][f, ANCHOR_REF_INDEX])
            Rt = robot.R_anchor_zup.T
            anchor_p = Rt @ (p_a - robot.p_anchor_zup)
            R_rel = Rt @ R_a
            anchor_o = R_rel[:, :2].reshape(-1)          # mjlab: mat[..., :2] row-major -> [m00, m01, m10, m11, m20, m21]
            vel = [ang_b, lin_b] if args.obs_order == "anglin" else [lin_b, ang_b]
            obs = np.concatenate([ref_q, ref_qd, anchor_p, anchor_o] + vel + [q - DEFAULT_Q, robot.qd, last_action]).astype(np.float32)
            action = session.run(None, {in_name: obs[None]})[0][0].astype(np.float64)
            last_action = action.copy()
            q_des = DEFAULT_Q + args.action_scale * ACTION_SCALE * action
            if n_pol == 0:
                print(f"[policy] first action |max| {np.abs(action).max():.2f}, q_des-q |max| {np.abs(q_des - q).max():.2f} rad, q_des-ref_q |max| {np.abs(q_des - ref_q).max():.2f}, "
                      f"anchor_p {anchor_p.round(3).tolist()} anchor_o {anchor_o.round(2).tolist()}", flush=True)
            n_pol += 1
        robot.pd(q_des, args.kp_scale, args.kd_scale)
        if k % log_every == 0:
            nit = sc.eng.native.get_total_newton_iters()
            track = "" if ref_q is None else f" track_rmse {np.sqrt(np.mean((q - ref_q) ** 2)):.3f} rad"
            track += f" [qd@{JOINTS29[int(np.argmax(np.abs(robot.qd)))].replace('_joint','')}]"
            if args.pd_backend == "engine" and k in (log_every, 2 * log_every):
                print("[drive] " + robot.drive_check(), flush=True)
            print(f"[{t:5.2f}s] {'policy' if active else 'settle'} z {robot.p_zup[2]:.3f} tilt {robot.tilt_deg():5.1f} deg v_b {lin_b.round(2).tolist()} |qd|max {np.abs(robot.qd).max():5.1f} "
                  f"|tau|max {np.abs(robot.tau).max():6.1f} |q|max {np.abs(q).max():.2f} feet_y {robot.feet_min_y()*1e3:6.1f} mm newton/step {(nit-newton0)/log_every:4.1f}{track} | {(time.time()-t0)/60:4.1f} min", flush=True)
            newton0 = nit
        series.append(dict(t=t, z=float(robot.p_zup[2]), tilt=robot.tilt_deg(), q=q.round(4).tolist(), ref_q=None if ref_q is None else ref_q.round(4).tolist()))
        if k % rec_every == 0:
            rec.record(t, 0 if not active else 1, "settle" if not active else args.mode)
            if k > 0 and k % int(round(2.0 / dt)) == 0:      # checkpoint the recording so a stalled run still leaves a clip
                rec.save(osp.join(args.out, "recording.npz"), dict(example="g1_policy", mode=args.mode, dt=dt, fps=args.fps, fem_materials={}, fem_colors=[], partial=True,
                                                                   robot=dict(urdf="nuka/g1/g1_hull.urdf", prefix="g1/", T=robot.T_load.tolist(), init=robot.init)))
        sc.step()
    wall = (time.time() - t0) / 60
    rec.save(osp.join(args.out, "recording.npz"), dict(example="g1_policy", mode=args.mode, dt=dt, fps=args.fps, fem_materials={}, fem_colors=[], wall_min=wall,
                                                        robot=dict(urdf="nuka/g1/g1_hull.urdf", prefix="g1/", T=robot.T_load.tolist(), init=robot.init)))
    zs = np.array([s["z"] for s in series[n_settle:]]); tilts = np.array([s["tilt"] for s in series[n_settle:]])
    err = [np.sqrt(np.mean((np.array(s["q"]) - np.array(s["ref_q"])) ** 2)) for s in series if s["ref_q"] is not None]
    summ = dict(mode=args.mode, z_min=float(zs.min()), z_mean=float(zs.mean()), tilt_mean=float(tilts.mean()), tilt_max=float(tilts.max()),
                track_rmse_mean=float(np.mean(err)) if err else None, fell=bool(tilts.max() > 45 or zs.min() < 0.45), wall_min=wall, policy_steps=n_pol)
    json.dump(dict(summary=summ, series=series), open(osp.join(args.out, "metrics.json"), "w"))
    print(f"[done] {n_steps} steps in {wall:.1f} min; summary {json.dumps(summ)}", flush=True)


if __name__ == "__main__":
    main()
