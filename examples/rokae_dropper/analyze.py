"""Checks on a rokae_dropper recording.

    python analyze.py out/rokae_dropper/<run> [--rec rec.npz]

Prints, over time: dropper tip / bulb height, bulb cavity volume, dropper tilt, flask and beaker
displacement, joint tracking error (arm, fingers); and, with water, particles per container.
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)


def abd_world_tf(C):
    T = np.array(C, np.float64)
    T[:3, :3] = C[:3, :3].T
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--rec", default="rec.npz")
    ap.add_argument("--every", type=float, default=0.5, help="print interval (s)")
    a = ap.parse_args()
    sc = np.load(osp.join(a.run, "scene.npz"), allow_pickle=True)
    rec = np.load(osp.join(a.run, a.rec), allow_pickle=True)
    names = [str(n) for n in sc["body_names"]]
    C0inv = sc["C0inv"]
    prop_T = {k: np.array(v) for k, v in json.loads(str(sc["prop_T"])).items()}
    args = json.loads(str(sc["args"]))
    import rokae_dropper as R
    split = "stem" in names
    if split:                                   # rigid glass stem (ABD) + FEM bulb
        st, bu = R.build_pipette()
        cav = np.asarray(bu["cavity_faces"])
        iS = names.index("stem")
        rest = np.load(osp.join(a.run, "rest.npz"))
        stem_rest = rest[f"v_{iS}"].astype(np.float64)
        tip_idx, top_idx = np.asarray(st["tip_idx"]), np.asarray(st["top_idx"])
    else:                                       # older runs: one FEM pipette
        import robot_assets as RA
        kin = RA.URDFKin(RA.build()["sim"])
        tele = R.Teleop(kin=kin)
        d = R.build_dropper()
        m = d.meta
        cav = np.asarray(m["cavity_faces"])
        tip_idx, top_idx = np.asarray(m["tip_idx"]), np.asarray(m["top_idx"])
    t = rec["t"]
    abd = rec["abd"]
    drop = rec["drop"]
    jn = [str(n) for n in rec["joint_names"]]
    qe = np.abs(rec["q_act"] - rec["q_cmd"])
    arm_j = [i for i, n in enumerate(jn) if "AR5" in n]
    fin_r = [i for i, n in enumerate(jn) if n.startswith("right_finger")]
    fin_l = [i for i, n in enumerate(jn) if n.startswith("left_finger")]
    iF, iB = names.index("flask"), names.index("beaker")
    lab = sc["liq_label"]
    liq = rec["liq_i16"] if "liq_i16" in rec.files else None
    origin = sc["liq_origin"]

    def tip_top(j):
        if split:
            M = abd_world_tf(abd[j][iS]) @ C0inv[iS]
            sv = stem_rest @ M[:3, :3].T + M[:3, 3]
            return sv[tip_idx].mean(0), sv[top_idx].mean(0)
        v = drop[j]
        return v[tip_idx].mean(0), v[top_idx].mean(0)

    def V_cav(v, origin_pt):
        A, B, C = v[cav[:, 0]] - origin_pt, v[cav[:, 1]] - origin_pt, v[cav[:, 2]] - origin_pt
        return -np.einsum("ij,ij->i", A, np.cross(B, C)).sum() / 6.0

    pF0 = (abd_world_tf(abd[0][iF]) @ C0inv[iF] @ prop_T["flask"])[:3, 3]
    pB0 = (abd_world_tf(abd[0][iB]) @ C0inv[iB] @ prop_T["beaker"])[:3, 3]
    hdr = "   t    tip_y  top_y   tilt   Vcav(ml)  flask_d  beaker_d beaker_y  arm_err  finR_err finL_err"
    if liq is not None:
        hdr += "  | in flask  in beaker  in bore  table  (blue/yellow)"
    print(hdr)
    step = max(1, int(round(a.every / max(np.median(np.diff(t)), 1e-6))))
    for j in list(range(0, len(t), step)) + [len(t) - 1]:
        v = drop[j]
        tip, top = tip_top(j)
        ax = top - tip
        tilt = np.degrees(np.arccos(ax[1] / np.linalg.norm(ax)))
        bulb = v[:, 1].max()             # top of the bulb
        TF = abd_world_tf(abd[j][iF]) @ C0inv[iF] @ prop_T["flask"]
        TB = abd_world_tf(abd[j][iB]) @ C0inv[iB] @ prop_T["beaker"]
        line = (f"{t[j]:6.2f} {tip[1]*1e3:7.1f} {bulb*1e3:7.1f} {tilt:5.1f} {V_cav(v, top if split else tip)*1e6:8.2f} "
                f"{np.linalg.norm(TF[:3,3]-pF0)*1e3:8.1f} {np.linalg.norm(TB[[0,2],3]-pB0[[0,2]])*1e3:8.1f} {TB[1,3]*1e3:8.1f} "
                f"{np.degrees(qe[j, arm_j].max()):7.2f}d {np.degrees(qe[j, fin_r].max()):7.1f}d {np.degrees(qe[j, fin_l].max()):7.1f}d")
        if liq is not None:
            p = liq[j].astype(np.float64) * 1e-4 + origin
            def inside(T, r, h):
                q = (p - T[:3, 3]) @ T[:3, :3]
                return (np.hypot(q[:, 0], q[:, 2]) < r) & (q[:, 1] > -0.002) & (q[:, 1] < h)
            inF = inside(TF, 0.043, 0.150)
            inB = inside(TB, 0.032, 0.075)
            rel = p - tip
            axn = ax / np.linalg.norm(ax)
            s = rel @ axn
            rad = np.linalg.norm(rel - np.outer(s, axn), axis=1)
            inBore = (s > 0) & (s < np.linalg.norm(ax)) & (rad < 0.0035)
            table = (p[:, 1] < 0.004) & ~inF & ~inB
            def bl(mask):
                return f"{int((mask & (lab == 0)).sum())}/{int((mask & (lab == 1)).sum())}"
            line += f"  | {bl(inF):>11} {bl(inB):>11} {bl(inBore):>8} {bl(table):>8}"
        print(line)


if __name__ == "__main__":
    main()
