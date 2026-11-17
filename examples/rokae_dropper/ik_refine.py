"""Closed-loop refinement of the right-arm correction from a simulated run of the previous trajectory.

ik_retarget.py assumes the pipette is rigid in the palm as at the grasp. In the simulation the fingers push it 1-2 cm
and ~5 deg while closing and it rubs the neck, so the tip is not on the flask axis. This script measures the pipette
(stem ABD body) and the palm in a run made with trajectory N, computes the world motion that would put the pipette
where it should be (tip on the flask axis / over the beaker, axis vertical), applies the same motion to the commanded
palm pose of trajectory N and solves the 7 right-arm joints again (seeded with N) -> trajectory N+1.

    python ik_refine.py --traj Assets/rokae_wuji/teleop_dropper_ik.npz --run out/rokae_dropper/ik1_dry \
        --out Assets/rokae_wuji/teleop_dropper_ik2.npz
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import robot_assets as RA  # noqa: E402
import rokae_dropper as R  # noqa: E402
from ik_retarget import RJ, Chain, ik, smooth  # noqa: E402


def W(C):
    T = np.array(C, np.float64)
    T[:3, :3] = C[:3, :3].T
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="trajectory the run was made with")
    ap.add_argument("--run", required=True, help="run directory (scene.npz, rest.npz, rec.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.25, help="moving-average window of the correction (s)")
    ap.add_argument("--e0", type=float, default=12.8, help="start of the extraction window (before the fingers close)")
    ap.add_argument("--release", type=float, default=34.3, help="return window ends (hand open, ramp out 1 s)")
    a = ap.parse_args()

    sc = np.load(osp.join(a.run, "scene.npz"), allow_pickle=True)
    rec = np.load(osp.join(a.run, "rec.npz"), allow_pickle=True)
    rest = np.load(osp.join(a.run, "rest.npz"))
    names = [str(n) for n in sc["body_names"]]
    C0inv, link_T0 = sc["C0inv"], sc["link_T0"]
    prop_T = {k: np.array(v) for k, v in json.loads(str(sc["prop_T"])).items()}
    iS, iF, iB, iP = (names.index(n) for n in ("stem", "flask", "beaker", "right_palm_link"))
    st, _ = R.build_pipette()
    sr = rest[f"v_{iS}"].astype(float)
    tip_r = sr[np.asarray(st["tip_idx"])].mean(0)
    top_r = sr[np.asarray(st["top_idx"])].mean(0)
    TZ = float(sc["table_z"])
    B = np.eye(4)
    B[:3, :3] = R.R_WR
    B[1, 3] = -TZ
    Binv = np.linalg.inv(B)

    t = rec["t"]
    keep = np.r_[True, np.diff(t) > 1e-9]
    t = t[keep]
    abd = rec["abd"][keep]
    n = len(t)
    tip = np.zeros((n, 3))
    ax = np.zeros((n, 3))
    for j in range(n):
        M = W(abd[j][iS]) @ C0inv[iS]
        tip[j] = M[:3, :3] @ tip_r + M[:3, 3]
        top = M[:3, :3] @ top_r + M[:3, 3]
        ax[j] = (top - tip[j]) / np.linalg.norm(top - tip[j])
    TF = W(abd[0][iF]) @ C0inv[iF] @ prop_T["flask"]
    TBk = W(abd[0][iB]) @ C0inv[iB] @ prop_T["beaker"]
    f_xz, b_xz = TF[[0, 2], 3], TBk[[0, 2], 3]
    mouth_y = TF[1, 3] + 0.160

    # windows from the simulated motion of trajectory N
    clear = tip[:, 1] > mouth_y + 0.03
    t_out = t[np.argmax(clear & (t > 15.0))]
    r_f = np.hypot(tip[:, 0] - f_xz[0], tip[:, 2] - f_xz[1])
    r_b = np.hypot(tip[:, 0] - b_xz[0], tip[:, 2] - b_xz[1])
    E = (a.e0, t_out, t_out + 1.0)                  # full correction until t_out, ramp out over 1 s
    D = (21.4, 23.0, 27.5, 28.3)
    Rw = (28.3, 29.3, a.release, a.release + 1.0)
    print(f"flask xz {np.round(f_xz, 4)} beaker xz {np.round(b_xz, 4)} mouth y {mouth_y:.4f}; sim tip clears at {t_out:.2f} s")

    def beta(tt):
        if E[0] <= tt <= E[2]:
            return smooth(tt, E[0], E[0] + 0.3) * (1 - smooth(tt, E[1], E[2])), f_xz
        if D[0] < tt < D[3]:
            return smooth(tt, D[0], D[1]) * (1 - smooth(tt, D[2], D[3])), b_xz
        if Rw[0] <= tt <= Rw[3]:
            return smooth(tt, Rw[0], Rw[1]) * (1 - smooth(tt, Rw[2], Rw[3])), f_xz
        return 0.0, None

    # measurements are used only while the pipette sits in the hand undisturbed by glass contact: in E until the tip
    # reaches the mouth (the rub + snap there is what we remove), in R until the tip is back near the mouth; the
    # correction is held at the last valid value after that (the measured pose there is a consequence, not a cause)
    t_exit = t[np.argmax((tip[:, 1] > mouth_y - 0.005) & (t > 15.0))]
    rr = np.where((tip[:, 1] < mouth_y + 0.02) & (t > 29.0))[0]
    t_reent = t[rr[0]] if len(rr) else Rw[2]
    def valid(tt):
        if E[0] <= tt <= E[2]:
            return tt < t_exit
        if Rw[0] <= tt <= Rw[3]:
            return tt < t_reent
        return True
    print(f"E measured until {t_exit:.2f} s, R measured until {t_reent:.2f} s")

    # world correction per recorded frame: rotate about the tip to vertical, move the tip onto the target axis
    dp = np.zeros((n, 3))
    rv = np.zeros((n, 3))
    last = (np.zeros(3), np.zeros(3), 0.0)
    for j in range(n):
        bt, xz = beta(t[j])
        if bt <= 0:
            continue
        if not valid(t[j]):
            d0, r0, b0 = last
            dp[j], rv[j] = d0 / max(b0, 1e-9) * bt, r0 / max(b0, 1e-9) * bt
            continue
        up = np.array([0.0, 1.0, 0.0])
        c = np.cross(ax[j], up)
        s = np.linalg.norm(c)
        if s > 1e-9:
            rv[j] = c / s * np.arctan2(s, ax[j] @ up) * bt * a.gain
        dp[j, [0, 2]] = (xz - tip[j, [0, 2]]) * bt * a.gain
        last = (dp[j].copy(), rv[j].copy(), bt)
    # smooth in time (moving average), the in-hand state is noisy
    k = max(1, int(round(a.smooth / np.median(np.diff(t)))))
    ker = np.ones(k) / k
    dp_s = np.stack([np.convolve(dp[:, i], ker, mode="same") for i in range(3)], 1)
    rv_s = np.stack([np.convolve(rv[:, i], ker, mode="same") for i in range(3)], 1)
    tip_s = np.stack([np.convolve(tip[:, i], ker, mode="same") for i in range(3)], 1)

    kin = RA.URDFKin(RA.build()["sim"])
    teleN = R.Teleop(path=a.traj, kin=kin)
    chain = Chain(kin)
    lim = [kin.joints[nm].get("lim", (-np.pi, np.pi)) for nm in RJ]
    z = dict(np.load(a.traj))
    t_arm, q_arm = z["arm_t"], z["arm_q"].copy()
    an = [str(x) for x in z["arm_names"]]
    col = [an.index(nm) for nm in RJ]

    nfix, worst, dqmax = 0, 0.0, 0.0
    for i, ta in enumerate(t_arm):
        if ta < t[0] or ta > t[-1]:
            continue
        d = np.array([np.interp(ta, t, dp_s[:, m]) for m in range(3)])
        r = np.array([np.interp(ta, t, rv_s[:, m]) for m in range(3)])
        if np.linalg.norm(d) < 1e-6 and np.linalg.norm(r) < 1e-6:
            continue
        p = np.array([np.interp(ta, t, tip_s[:, m]) for m in range(3)])
        Dw = np.eye(4)
        Dw[:3, :3] = Rot.from_rotvec(r).as_matrix()
        Dw[:3, 3] = p + d - Dw[:3, :3] @ p
        qd = teleN.at(ta)
        Pcmd = kin.fk(qd)["right_palm_link"]
        Td = Binv @ Dw @ B @ Pcmd
        q, e = ik(chain, qd, Td, lim)
        dqmax = max(dqmax, np.degrees(np.abs(q - np.array([qd[nm] for nm in RJ])).max()))
        q_arm[i, col] = q
        nfix += 1
        worst = max(worst, np.linalg.norm(e[:3]))
    z["arm_q"] = q_arm
    z["ik_note"] = np.array(str(z.get("ik_note", "")) + f" | refined by ik_refine.py from a simulated run (E {E[0]:.1f}-{E[2]:.1f}, "
                            f"D {D[0]}-{D[3]}, R {Rw[0]}-{Rw[3]} s, gain {a.gain})")
    np.savez(a.out, **z)
    print(f"refined {nfix} samples, worst residual {worst*1e3:.2f} mm, max joint change vs input {dqmax:.1f} deg -> {a.out}")
    for tt in (13.5, 14.5, 15.5, 17.0, 19.0, 20.1, 20.5, 25.0, 29.5, 31.0, 32.5, 33.5):
        j = np.argmin(abs(t - tt))
        tilt = np.degrees(np.arccos(np.clip(ax[j, 1], -1, 1)))
        print(f"  sim N  t {t[j]:5.2f}  tip off flask {r_f[j]*1e3:6.1f} mm  off beaker {r_b[j]*1e3:6.1f} mm  tilt {tilt:5.1f}"
              f"  -> correction {np.linalg.norm(dp_s[j])*1e3:5.1f} mm {np.degrees(np.linalg.norm(rv_s[j])):4.1f} deg")


if __name__ == "__main__":
    main()
