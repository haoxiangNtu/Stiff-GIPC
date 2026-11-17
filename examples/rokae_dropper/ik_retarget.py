"""Correct the right arm of the teleop recording where it handles the pipette, so that the hand does what the real
cameras show; everywhere else the recorded joint angles are kept.

Why: replayed through the URDF, the recorded joints move the hand 3-4 cm sideways while the pipette is still in the
flask neck (clearance ~1 cm), whereas both cameras of the recording (and the external video) show the pipette coming
straight up, vertical, and going back straight down into the flask. The difference is a kinematic mismatch between
the URDF / hand mount and the real robot, not a property of the task.

Corrections (all in the robot frame, solved by damped-least-squares IK on the 7 right-arm joints, seeded with the
recorded angles, so the arm configuration stays close to the recording):
  E  extraction  the pipette (rigid in the palm from the grasp) stays vertical on the flask axis while its tip is in or
                 near the flask, height as recorded; blends back to the recorded hand pose in the air above the flask.
  D  drip        constant translation of the hand so that the pipette tip of the recorded drip pose is above the
                 beaker placed where the external video shows it (beside the flask, ~18 cm, same distance from the robot).
  R  return      the pipette vertical on the flask axis again, height as recorded, until the hand has let go.

    python ik_retarget.py [--out Assets/rokae_wuji/teleop_dropper_ik.npz] [--beaker-rel 0.025 0.18]
"""
import argparse
import os.path as osp
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import robot_assets as RA  # noqa: E402
import rokae_dropper as R  # noqa: E402

RJ = [f"AR5-5_08R-W4C4A6-ZY2_joint_{i}" for i in range(1, 8)]


class Chain:
    """FK + geometric Jacobian of the root -> palm chain (only the joints on it)."""

    def __init__(self, kin, tip="right_palm_link"):
        chain, link = [], tip
        while link in kin.child_of:
            j = kin.child_of[link]
            chain.append(j)
            link = j["parent"]
        self.chain = chain[::-1]
        self.idx = {n: k for k, n in enumerate(RJ)}

    def fk(self, qd):
        T = np.eye(4)
        axes, origins = [], []
        for j in self.chain:
            T = T @ j["T"]
            if j["type"] == "revolute":
                if j["name"] in self.idx:
                    axes.append(T[:3, :3] @ j["axis"])
                    origins.append(T[:3, 3].copy())
                Rj = np.eye(4)
                Rj[:3, :3] = Rot.from_rotvec(j["axis"] * qd.get(j["name"], 0.0)).as_matrix()
                T = T @ Rj
        return T, np.array(axes), np.array(origins)

    def jac(self, T, axes, origins):
        p = T[:3, 3]
        Jp = np.cross(axes, p - origins).T
        return np.vstack([Jp, axes.T])


def pose_err(T, Td):
    dp = Td[:3, 3] - T[:3, 3]
    dr = Rot.from_matrix(Td[:3, :3] @ T[:3, :3].T).as_rotvec()
    return np.r_[dp, dr]


def ik(chain, qd, Td, lim, iters=60, lam=1e-3, w_rot=0.3):
    q = np.array([qd[n] for n in RJ])
    q0 = q.copy()
    W = np.diag([1, 1, 1, w_rot, w_rot, w_rot])
    for _ in range(iters):
        d = dict(qd)
        d.update({n: v for n, v in zip(RJ, q)})
        T, ax, org = chain.fk(d)
        e = pose_err(T, Td)
        if np.linalg.norm(e[:3]) < 2e-5 and np.linalg.norm(e[3:]) < 2e-4:
            break
        J = W @ chain.jac(T, ax, org)
        # damped least squares + a small pull towards the recorded configuration (null space)
        H = J.T @ J + lam * np.eye(7)
        dq = np.linalg.solve(H, J.T @ (W @ e) + 1e-4 * (q0 - q))
        q = np.clip(q + dq, [l[0] for l in lim], [l[1] for l in lim])
    d = dict(qd)
    d.update({n: v for n, v in zip(RJ, q)})
    T, _, _ = chain.fk(d)
    return q, pose_err(T, Td)


def smooth(t, a, b):
    """0 before a, 1 after b, C1 smoothstep in between."""
    s = np.clip((t - a) / max(b - a, 1e-9), 0.0, 1.0)
    return s * s * (3 - 2 * s)


def blend(Ta, Tb, beta):
    if beta <= 0.0:
        return Ta
    if beta >= 1.0:
        return Tb
    out = np.eye(4)
    out[:3, 3] = (1 - beta) * Ta[:3, 3] + beta * Tb[:3, 3]
    out[:3, :3] = Slerp([0, 1], Rot.from_matrix([Ta[:3, :3], Tb[:3, :3]]))(beta).as_matrix()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=osp.join(R.REPO, "Assets", "rokae_wuji", "teleop_dropper_ik.npz"))
    ap.add_argument("--beaker-rel", type=float, nargs=2, default=(0.025, 0.18),
                    help="beaker centre minus flask centre (robot x forward, y left), from the external video")
    ap.add_argument("--t-grasp", type=float, default=R.T_GRASP_DROPPER)
    ap.add_argument("--clear", type=float, default=0.02, help="tip height above the flask mouth where E blends out (m)")
    # window timing of the recording (defaults = Rokae_AR5_260615_145413_0); another recording passes its own
    ap.add_argument("--d-in-end", type=float, default=23.0, help="D (drip shift) fully in by this time (s)")
    ap.add_argument("--d-out", type=float, nargs=2, default=(27.5, 28.3), help="D blends out over this window (s)")
    ap.add_argument("--drip-window", type=float, nargs=2, default=(23.5, 27.5), help="recorded squeeze: mean tip position taken over it (s)")
    ap.add_argument("--back-after", type=float, default=28.5, help="search the return (tip back at the mouth) after this time (s)")
    ap.add_argument("--r-out", type=float, nargs=2, default=(34.3, 35.3), help="R blends out over this window (the hand has let go) (s)")
    ap.add_argument("--t-end", type=float, default=36.0, help="end of the timing scan (s)")
    ap.add_argument("--t-out", type=float, default=None, help="override: time the tip clears the mouth (extraction)")
    ap.add_argument("--t-in", type=float, default=None, help="override: time the tip is back at the mouth (return)")
    a = ap.parse_args()

    kin = RA.URDFKin(RA.build()["sim"])
    tele = R.Teleop(kin=kin)
    chain = Chain(kin)
    lim = [kin.joints[n].get("lim", (-np.pi, np.pi)) for n in RJ]
    z = dict(np.load(R.TRAJ))
    t_arm, q_arm = z["arm_t"], z["arm_q"].copy()
    names = [str(n) for n in z["arm_names"]]
    col = [names.index(n) for n in RJ]

    # pipette frame at the grasp: vertical, origin at the bulb centre on the flask axis; rigid in the palm from then on
    f0 = kin.fk(tele.at(a.t_grasp))
    P0 = f0["right_palm_link"]
    c0 = R.bulb_centre(f0)
    pip0 = np.eye(4)
    pip0[:3, 3] = c0
    T_hp = np.linalg.inv(P0) @ pip0                 # pipette in palm
    table_z = R.table_z_from_recording(kin, tele)
    tip_below = R.BULB_EQUATOR                       # bulb centre -> tip
    flask_top = table_z + 0.160
    axis_xy = c0[:2]
    beaker_xy = axis_xy + np.asarray(a.beaker_rel)
    print(f"flask axis {np.round(axis_xy, 4)}, beaker target {np.round(beaker_xy, 4)}, table z {table_z:.4f}, flask mouth z {flask_top:.4f}")

    def palm_rec(t):
        return kin.fk(tele.at(t))["right_palm_link"]

    def pip_rec(t):
        return palm_rec(t) @ T_hp

    # timing of the windows from the recorded motion: the tip leaves the mouth (E) and re-enters it (R)
    ts = np.arange(a.t_grasp, a.t_end, 0.02)
    tipz = np.array([(pip_rec(t) @ np.r_[0, 0, -tip_below, 1])[2] for t in ts])
    above = tipz > flask_top + a.clear
    t_out = ts[np.argmax(above)]                                    # first time the tip is clear (extraction)
    back = np.where(~above & (ts > a.back_after))[0]                 # after the drip (the tip dips below the mouth height over the beaker)
    t_in = ts[back[0]] if len(back) else 33.0                       # tip back at the mouth (return)
    if a.t_out is not None: t_out = a.t_out
    if a.t_in is not None: t_in = a.t_in
    # drip: mean recorded tip position over the squeeze, translated onto the beaker axis
    td = np.arange(a.drip_window[0], a.drip_window[1], 0.1)
    tip_drip = np.mean([(pip_rec(t) @ np.r_[0, 0, -tip_below, 1])[:2] for t in td], axis=0)
    shift_d = np.r_[beaker_xy - tip_drip, 0.0]
    print(f"tip clears the mouth at {t_out:.2f} s, back at the mouth at {t_in:.2f} s; drip shift {np.round(shift_d * 1e3, 1)} mm")

    def vertical_on_axis(t):
        P = pip_rec(t)
        out = np.eye(4)
        out[:3, 3] = np.r_[axis_xy, P[2, 3]]
        return out

    E0, E1 = a.t_grasp, t_out
    E_out = (t_out, t_out + 1.0)                 # blend back in the air
    D_in, D_out = (t_out + 1.0, a.d_in_end), tuple(a.d_out)
    R_in = (D_out[1], max(D_out[1] + 0.3, t_in - 0.3))   # straighten above the mouth before entering (after the drip shift is gone)
    R_out = tuple(a.r_out)                               # the hand has let go: back to the recorded pose
    R_end = R_out[1]

    def palm_des(t):
        Pr = palm_rec(t)
        # E: vertical on axis, fading out after the tip clears
        if E0 <= t <= E_out[1]:
            Pv = vertical_on_axis(t) @ np.linalg.inv(T_hp)
            return blend(Pv, Pr, smooth(t, *E_out))
        if D_in[0] < t < D_out[1]:
            beta = smooth(t, *D_in) * (1.0 - smooth(t, *D_out))
            Ps = Pr.copy()
            Ps[:3, 3] += shift_d * beta
            return Ps
        if R_in[0] <= t <= R_end:
            Pv = vertical_on_axis(t) @ np.linalg.inv(T_hp)
            return blend(Pr, Pv, smooth(t, *R_in) * (1.0 - smooth(t, *R_out)))
        return None

    n_fix, worst = 0, 0.0
    for i, t in enumerate(t_arm):
        if t < E0 or t > R_end:
            continue
        Td = palm_des(t)
        if Td is None:
            continue
        qd = tele.at(t)
        q, e = ik(chain, qd, Td, lim)
        q_arm[i, col] = q
        n_fix += 1
        worst = max(worst, np.linalg.norm(e[:3]))
    z["arm_q"] = q_arm
    z["ik_note"] = np.array(f"right arm corrected by ik_retarget.py: E {E0:.2f}-{E_out[1]:.2f} s, D {D_in[0]:.2f}-{D_out[1]:.2f} s, "
                            f"R {R_in[0]:.2f}-{R_end:.2f} s; beaker rel {tuple(a.beaker_rel)}")
    np.savez(a.out, **z)
    print(f"corrected {n_fix} samples, worst position residual {worst * 1e3:.2f} mm -> {a.out}")

    # check: the rigid-in-palm pipette on the corrected trajectory
    tele2 = R.Teleop(path=a.out, kin=kin)
    for t in (15.5, 17.0, 18.5, 19.5, 20.5, 21.0, 22.0, 25.5, 27.9, 28.3, 28.6, 29.0, 31.0, 32.5, 33.5, 34.5, 35.0, 35.5):
        P = kin.fk(tele2.at(t))["right_palm_link"] @ T_hp
        tip = P @ np.r_[0, 0, -tip_below, 1]
        tilt = np.degrees(np.arccos(np.clip(P[2, 2], -1, 1)))
        dq = np.degrees(np.abs(np.array([tele2.at(t)[n] - tele.at(t)[n] for n in RJ])).max())
        print(f"  t {t:5.2f}  tip xy off flask axis {np.linalg.norm(tip[:2] - axis_xy) * 1e3:6.1f} mm, off beaker {np.linalg.norm(tip[:2] - beaker_xy) * 1e3:6.1f} mm,"
              f" tip z-table {(tip[2] - table_z) * 1e3:6.1f} mm, tilt {tilt:5.1f} deg, max joint change {dq:5.1f} deg")


if __name__ == "__main__":
    main()
