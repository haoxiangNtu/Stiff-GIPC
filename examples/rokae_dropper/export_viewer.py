"""Export a rokae_dropper run for the web viewer (docs/rokae_viewer.html on the project site), format 2.

    python export_viewer.py <run_dir> <out.bin> --id NAME [--fstep 1] [--bulk-stride 3] [--bulk-fstep 1] [--t-ref 18.0]

One gzip file: uint32 header length, JSON header, then the arrays listed in header["arrays"] (name, dtype, shape, byte
offset from the start of the array section).
  t          float32 [nf]              recording time of each exported frame
  T          float32 [nf, nb, 3, 4]    rest -> world transform of every rigid body (rounded to 1e-5 so still bodies compress)
  V, F, vo, vn, fo, fn                 rest collision meshes (vertices, faces, per-body offsets / counts)
  bulb       int16 [nf, nv, 3]         FEM bulb vertices in the STEM frame (the stem transform places them), time-delta coded
  bulb_faces uint32 [nface, 3]
  liqA       int16 [nf, nA, 3]         water particles that ever enter the bore or leave their container: all of them, time-delta coded
  liqB       int16 [nfB, nB, 3]        the remaining (bulk) water: every bulk_stride-th particle, every bulk_fstep-th frame
  labA, labB uint8                     0 = flask water (blue), 1 = beaker water (yellow)
header["diag"]: per-frame diagnostics for the viewer's curves, same definitions as analyze.py / slipcheck.py / gripdiag.py:
stem tilt, rotation of the stem relative to the right palm since t_ref, bulb bend, tip height, tip-to-beaker-axis
distance, grip force, bulb cavity volume, blue-water particles in the bore / in the beaker / on the table / out of the flask,
and the air-chamber pump's state (water it wants in the bore, tip submerged or not).
World frame: y up, table at y = 0, metres.
"""
import argparse
import glob
import gzip
import json
import os
import os.path as osp

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(osp.dirname(HERE))
# inner profile (height above the base, radius) of the reference flask: same numbers as rokae_dropper.FLASK_INNER
FLASK_INNER = np.array([(0.0049, 0.0360), (0.0161, 0.0362), (0.1022, 0.0173), (0.1150, 0.0144), (0.1600, 0.0144)])
BEAKER_R_IN = 0.032        # "in the beaker" radius used by analyze.py
BULB_SCALE = 5e-5          # 0.05 mm
LIQ_SCALE = 1e-4           # the recording's own unit (0.1 mm)


def world_tf(C):
    """ABD state -> 4x4 (the engine stores the 3x3 block transposed)."""
    C = np.asarray(C, np.float64)
    T = C.copy()
    T[..., :3, :3] = np.swapaxes(C[..., :3, :3], -1, -2)
    return T


def angle_deg(a, b):
    c = np.einsum("...i,...i->...", a, b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def delta_t(q):
    """Time-delta code an int16 [frames, ...] array (wraps like the decoder's Int16Array sum)."""
    d = q.copy()
    d[1:] = q[1:] - q[:-1]
    return d


def jlist(a, nd=1):
    """JSON-friendly list: rounded, NaN -> None."""
    a = np.asarray(a, np.float64)
    return [None if not np.isfinite(v) else (int(v) if nd == 0 else round(float(v), nd)) for v in a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--id", required=True, help="public name of the record (goes into the header)")
    ap.add_argument("--fstep", type=int, default=1, help="export every n-th recorded frame")
    ap.add_argument("--bulk-stride", type=int, default=3, help="keep every n-th bulk water particle")
    ap.add_argument("--bulk-fstep", type=int, default=1, help="bulk water: every n-th EXPORTED frame")
    ap.add_argument("--t-ref", type=float, default=None, help="reference time of the in-hand rotation (default 18.0 / 17.5 s for recording 2 / 1)")
    ap.add_argument("--mount-mm", type=float, default=0.0, help="hand-mount length correction the run used (WUJI_FLANGE_EXTRA, mm); "
                    "an environment switch, so it is not in the run's own arguments")
    ap.add_argument("--real-ref", default=None, help="npz with positions measured in the real footage (robot frame; vidcal/v3/make_real_ref.py): "
                    "t, bulb[n,3] (NaN where not measured), tips[m,4] = (t, x, y, z), optional tip[n,3] = glass tip of the carried dropper (+ meas, "
                    "stem_len, vis_len). Adds header 'real' and the diagnostics 'bulb_err', 'axis_err' (deg), 'tip_err' (mm)")
    a = ap.parse_args()

    sc = np.load(osp.join(a.run, "scene.npz"), allow_pickle=True)
    rec = np.load(osp.join(a.run, "rec.npz"), allow_pickle=True)
    rest = np.load(osp.join(a.run, "rest.npz"))
    names = [str(n) for n in sc["body_names"]]
    nb = len(names)
    args = json.loads(str(sc["args"]))
    prop_T = {k: np.array(v) for k, v in json.loads(str(sc["prop_T"])).items()}
    recording = 2 if "r2" in osp.basename(str(args.get("traj", ""))) else 1
    t_ref = a.t_ref if a.t_ref is not None else (18.0 if recording == 2 else 17.5)

    global FLASK_INNER
    hs = float(sc["flask_hscale"]) if "flask_hscale" in sc.files else 1.0     # flask scaled to its measured height (RD_FLASK_H)
    if hs != 1.0:
        FLASK_INNER = FLASK_INNER * np.array([hs, 1.0])
    flask_h = 0.160 * hs
    fs = a.fstep
    t = np.asarray(rec["t"], np.float64)[::fs]
    nf = len(t)
    # the first frames hold the start pose (--settle) and are all logged at t0: give them their own times before t0
    n0 = int(np.sum(t == t[0]))
    if n0 > 1:
        dt_rec = float(args.get("dt", 0.005)) * int(args.get("rec_every", 8)) * fs
        t[:n0] = t[0] - dt_rec * np.arange(n0 - 1, -1, -1)
    M = world_tf(rec["abd"][::fs]) @ np.asarray(sc["C0inv"], np.float64)[None]          # [nf, nb, 4, 4] rest -> world
    iS, iP, iF, iB = names.index("stem"), names.index("right_palm_link"), names.index("flask"), names.index("beaker")

    # ---- rest meshes
    verts, faces, vo, fo = [], [], [], []
    for i in range(nb):
        v = rest[f"v_{i}"].astype(np.float32)
        f = rest[f"f_{i}"].astype(np.uint32)
        if names[i] == "dropper":            # older single-body pipette placeholder: animated separately
            v, f = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
        vo.append(sum(len(x) for x in verts))
        fo.append(sum(len(x) for x in faces))
        verts.append(v)
        faces.append(f)
    V, F = np.concatenate(verts), np.concatenate(faces)

    # ---- stem kinematics
    sr = rest[f"v_{iS}"].astype(np.float64)
    tip_rest = sr[np.asarray(sc["stem_tip_idx"])].mean(0)
    top_rest = sr[np.asarray(sc["stem_top_idx"])].mean(0)
    A, p = M[:, iS, :3, :3], M[:, iS, :3, 3]
    tip = np.einsum("fij,j->fi", A, tip_rest) + p
    top = np.einsum("fij,j->fi", A, top_rest) + p
    ax = top - tip
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    up = np.array([0.0, 1.0, 0.0])
    tilt = angle_deg(ax, up[None])

    # rotation relative to the right palm since t_ref (slipcheck.py)
    R, q = M[:, iP, :3, :3], M[:, iP, :3, 3]
    jr = int(np.argmin(np.abs(t - t_ref)))
    a_p = R[jr].T @ ax[jr]
    ax_rigid = np.einsum("fij,j->fi", R, a_p)
    slip = angle_deg(ax, ax_rigid)
    tilt_rigid = angle_deg(ax_rigid, up[None])

    # ---- bulb: world vertices -> stem frame; bend (gripdiag.py), cavity volume (analyze.py)
    drop = np.asarray(rec["drop"][::fs], np.float64)                                     # [nf, nv, 3]
    nv = drop.shape[1]
    Ainv = np.linalg.inv(A)
    bl = np.einsum("fij,fvj->fvi", Ainv, drop - p[:, None, :])
    bc = bl.reshape(-1, 3).mean(0)
    bulb_q = np.round((bl - bc) / BULB_SCALE)
    assert np.abs(bulb_q).max() < 32000, "bulb out of int16 range"
    bulb_q = bulb_q.astype(np.int16)
    yb = np.asarray(sc["bulb_rest"], np.float64)[:, 1]
    neck, cap = yb < np.percentile(yb, 8), yb > np.percentile(yb, 92)
    bend = angle_deg(drop[:, cap].mean(1) - drop[:, neck].mean(1), ax)
    # cavity faces of THIS run's pipette mesh: the cache whose bulb vertices equal the run's rest bulb in the dropper frame
    # (RD_STEM_TOP / RD_COLLAR variants have their own meshes; the default one gave a wrong volume for them before 11-09)
    Td = prop_T.get("dropper", prop_T["stem"])
    rest_local = (np.asarray(sc["bulb_rest"], np.float64) - Td[:3, 3]) @ Td[:3, :3]
    vcav = np.full(nf, np.nan)
    for c in sorted(set(c for root in (REPO, os.getcwd()) for c in glob.glob(osp.join(root, "Assets", "rokae_wuji", "lab_ref", "pipette_split_*.npz")))):
        bd = np.load(c, allow_pickle=True)["bulb"].item()
        if len(bd["verts"]) == nv and np.abs(np.asarray(bd["verts"]) - rest_local).max() < 1e-5:
            cav = np.asarray(bd["cavity_faces"])
            a0, b0, c0 = (drop[:, cav[:, k]] - top[:, None, :] for k in range(3))
            vcav = -np.einsum("fni,fni->f", a0, np.cross(b0, c0)) / 6.0 * 1e6        # ml
            break

    # ---- containers
    TF = M[:, iF] @ prop_T["flask"]
    TB = M[:, iB] @ prop_T["beaker"]
    flask_d = np.linalg.norm(TF[:, :3, 3] - TF[0, :3, 3], axis=1) * 1e3
    beaker_d = np.linalg.norm(TB[:, [0, 2], 3] - TB[0, [0, 2], 3], axis=1) * 1e3
    tip_off = np.hypot(tip[:, 0] - TB[:, 0, 3], tip[:, 2] - TB[:, 2, 3]) * 1e3
    force = np.asarray(rec["force"], np.float64)[::fs]

    # ---- water: counts per place (analyze.py), particles that ever leave their container
    has_liq = "liq_i16" in rec.files and rec["liq_i16"].shape[1] > 0
    diag_liq = {}
    liqA = np.zeros((nf, 0, 3), np.int16)
    liqB = np.zeros((0, 0, 3), np.int16)
    labA = labB = np.zeros(0, np.uint8)
    origin = np.asarray(sc["liq_origin"], np.float64)
    n_trav = 0
    flask_level = None
    if has_liq:
        L = rec["liq_i16"][::fs]
        lab = np.asarray(sc["liq_label"]).astype(np.uint8)
        blue = lab == 0
        npart = L.shape[1]
        away = np.zeros(npart, bool)
        cnt = {k: np.zeros(nf, np.int64) for k in ("bore", "beaker", "table", "out")}
        length = np.linalg.norm(top_rest - tip_rest)
        half = float(args["liquid_dx"]) / 4            # half the particle spacing: surface = top particle centres + half a spacing
        level = np.full(nf, np.nan)                    # liquid surface height in the flask (above the table)
        depth = np.full(nf, np.nan)                    # tip below that surface (mm), while the tip is inside the flask
        for j in range(nf):
            P = L[j].astype(np.float64) * LIQ_SCALE + origin
            qf = (P - TF[j, :3, 3]) @ TF[j, :3, :3]
            qb = (P - TB[j, :3, 3]) @ TB[j, :3, :3]
            rf, rb = np.hypot(qf[:, 0], qf[:, 2]), np.hypot(qb[:, 0], qb[:, 2])
            in_f = (rf < 0.043) & (qf[:, 1] > -0.002) & (qf[:, 1] < 0.150)
            in_b = (rb < BEAKER_R_IN) & (qb[:, 1] > -0.002) & (qb[:, 1] < 0.075)
            rel = P - tip[j]
            s = rel @ ax[j]
            rad = np.linalg.norm(rel - np.outer(s, ax[j]), axis=1)
            in_bore = (s > 0) & (s < length) & (rad < 0.0035)
            table = (P[:, 1] < 0.004) & ~in_f & ~in_b
            cnt["bore"][j] = (in_bore & blue).sum()
            cnt["beaker"][j] = (in_b & blue).sum()
            cnt["table"][j] = (table & blue).sum()
            cnt["out"][j] = (~in_f & blue).sum()
            calm = blue & in_f & (rad > 0.008)                       # flask water away from the tube
            if calm.sum() > 50:
                level[j] = np.percentile(P[calm, 1], 99) + half
                if np.hypot(tip[j, 0] - TF[j, 0, 3], tip[j, 2] - TF[j, 2, 3]) < 0.030 and tip[j, 1] < 0.165:
                    depth[j] = (level[j] - tip[j, 1]) * 1e3
            # tight home tests for the "ever left" set: inside the flask's inner profile / inside the beaker
            home_f = (qf[:, 1] > -0.002) & (qf[:, 1] < flask_h) & (rf < np.interp(qf[:, 1], FLASK_INNER[:, 0], FLASK_INNER[:, 1]) + 0.0015)
            home_b = (rb < 0.0345) & (qb[:, 1] > -0.002) & (qb[:, 1] < 0.105)
            away |= np.where(blue, ~home_f, ~home_b) | in_bore
        n_trav = int(away.sum())
        bulk = np.nonzero(~away)[0][::a.bulk_stride]
        trav = np.nonzero(away)[0]
        liqA, labA = np.ascontiguousarray(L[:, trav]), lab[trav]
        liqB, labB = np.ascontiguousarray(L[::a.bulk_fstep][:, bulk]), lab[bulk]
        diag_liq = {k: jlist(v, 0) for k, v in cnt.items()}
        diag_liq["depth"] = jlist(depth)
        early = (t >= t[0] + 4.0) & (t <= t[0] + 5.0)                  # settled, before the hand arrives
        flask_level = float(np.nanmedian(level[early]) * 1e3) if np.isfinite(level[early]).any() else None
        pump = np.asarray(rec["pump"], np.float64)[::fs]
        vp = (float(args["liquid_dx"]) / 2) ** 3
        if np.isfinite(pump[:, 1]).any():            # air-chamber pump active: its own bore count, target and submerged flag
            diag_liq["bore"] = jlist(pump[:, 1] / vp, 0)
            diag_liq["target"] = jlist(pump[:, 2] / vp, 1)
            diag_liq["sub"] = jlist(pump[:, 4], 0)

    diag = dict(tilt=jlist(tilt), tilt_rigid=jlist(tilt_rigid), slip=jlist(slip), bend=jlist(bend), tip_h=jlist(tip[:, 1] * 1e3),
                tip_off=jlist(tip_off), grip=jlist(force[:, 0]), beaker_grip=jlist(force[:, 4]), vcav=jlist(vcav, 2),
                flask_d=jlist(flask_d), beaker_d=jlist(beaker_d), **diag_liq)
    meta = dict(recording=recording, liquid=bool(has_liq), t_ref=float(t[jr]), beaker_r_in=BEAKER_R_IN * 1e3,
                newton_tol=args.get("newton_tol"), pcg_tol=args.get("pcg_tol"), finger_strength_right=args.get("finger_strength_right"),
                e_bulb=args.get("e_bulb"), table_dz=args.get("table_dz", 0.0), flask_fill=args.get("flask_fill"),
                beaker_offset=args.get("beaker_offset"), dt=args.get("dt"), rec_every=args.get("rec_every"),
                particle_ml=(float(args["liquid_dx"]) / 2) ** 3 * 1e6 if has_liq and args.get("liquid_dx") else None,
                n_left_container=n_trav, flask_level=None if flask_level is None else round(flask_level, 1),
                lean_deg=float(args.get("lean_deg", 0.0) or 0.0), mount_mm=float(a.mount_mm), settle=float(args.get("settle", 0.0) or 0.0),
                flask_level_target=(float(args["flask_level"]) * 1e3 if args.get("flask_level") else None),
                flask_xy_mm=[round(float(v) * 1e3, 1) for v in json.loads(str(sc["cal"]))["flask_xy"]],
                flask_h_mm=round(flask_h * 1e3, 1), table_z=float(sc["table_z"]),
                cmd_offset=json.loads(str(sc["cmd_offset"])) if "cmd_offset" in sc.files else {},
                hand_mount_env=json.loads(str(sc["hand_mount_env"])) if "hand_mount_env" in sc.files else {})

    # ---- positions measured in the real footage: the bulb centre track and the tip at the hover (world frame of the record)
    real = None
    if a.real_ref:
        rr = np.load(a.real_ref, allow_pickle=True)
        tz = float(sc["table_z"])
        def r2w(P):                                    # robot (x, y, z) -> record world (x, z - table, -y)
            P = np.atleast_2d(np.asarray(P, np.float64))
            return np.c_[P[:, 0], P[:, 2] - tz, -P[:, 1]]
        rt, rb = np.asarray(rr["t"], np.float64), np.asarray(rr["bulb"], np.float64)
        ok = np.isfinite(rb[:, 0])
        j0 = np.clip(np.searchsorted(rt, t), 1, len(rt) - 1)
        w = np.clip((t - rt[j0 - 1]) / np.maximum(rt[j0] - rt[j0 - 1], 1e-9), 0.0, 1.0)
        rbw = r2w((1 - w)[:, None] * rb[j0 - 1] + w[:, None] * rb[j0])          # NaN wherever either neighbour is not measured
        rbw[(t < rt[0]) | (t > rt[-1])] = np.nan
        err = np.linalg.norm(drop.mean(1) - rbw, axis=1) * 1e3
        diag["bulb_err"] = jlist(err)
        real = dict(bulb=[None if not np.isfinite(v[0]) else [round(float(x), 4) for x in v] for v in rbw],
                    tips=[[round(float(r[0]), 2)] + [round(float(x), 4) for x in r2w(r[1:4])[0]] for r in np.atleast_2d(rr["tips"])] if ("tips" in rr.files and np.size(rr["tips"])) else [],
                    note=str(rr["note"]) if "note" in rr.files else "")
        if "tip" in rr.files:                          # the carried real dropper: glass tip = bulb mark + stem_len along the measured stem direction
            rtp = np.asarray(rr["tip"], np.float64)
            rtw = r2w((1 - w)[:, None] * rtp[j0 - 1] + w[:, None] * rtp[j0])
            rtw[(t < rt[0]) | (t > rt[-1])] = np.nan
            okr = np.isfinite(rtw[:, 0]) & np.isfinite(rbw[:, 0])
            dreal = rtw - rbw
            dreal /= np.maximum(np.linalg.norm(dreal, axis=1, keepdims=True), 1e-12)
            axis_err = np.where(okr, angle_deg(-ax, dreal), np.nan)          # ax points tip -> top
            tip_err = np.where(okr, np.linalg.norm(tip - rtw, axis=1) * 1e3, np.nan)
            diag["axis_err"] = jlist(axis_err)
            diag["tip_err"] = jlist(tip_err)
            real["tip"] = [None if not np.isfinite(v[0]) else [round(float(x), 4) for x in v] for v in rtw]
            if "meas" in rr.files:
                mm = np.asarray(rr["meas"], np.float64)
                real["tip_meas"] = [int(mm[j] > 0.5 and mm[j - 1] > 0.5) for j in j0]
            real["stem_len"] = float(rr["stem_len"]) if "stem_len" in rr.files else None
            real["vis_len"] = float(rr["vis_len"]) if "vis_len" in rr.files else None
    meta["scene_cal"] = osp.basename(str(args.get("scene_cal") or ""))

    arrays = [("t", t.astype(np.float32)), ("T", np.round(M[:, :, :3, :], 5).astype(np.float32)), ("V", V), ("F", F),
              ("vo", np.array(vo, np.uint32)), ("vn", np.array([len(x) for x in verts], np.uint32)),
              ("fo", np.array(fo, np.uint32)), ("fn", np.array([len(x) for x in faces], np.uint32)),
              ("bulb", delta_t(bulb_q)), ("bulb_faces", np.asarray(sc["dropper_faces"]).astype(np.uint32)),
              ("liqA", delta_t(liqA)), ("labA", labA), ("liqB", delta_t(liqB)), ("labB", labB)]
    hdr = dict(format=2, id=a.id, names=names, kind=[str(k) for k in sc["body_kind"]], nframes=nf, nbodies=nb, stem=iS,
               tip_rest=tip_rest.tolist(), top_rest=top_rest.tolist(), bulb_centre=bc.tolist(), bulb_scale=BULB_SCALE,
               flask=iF, beaker=iB, flask_rest=prop_T["flask"][:3, 3].tolist(), beaker_rest=prop_T["beaker"][:3, 3].tolist(),
               liq_origin=origin.tolist(), liq_scale=LIQ_SCALE, bulk_fstep=a.bulk_fstep, bulk_stride=a.bulk_stride,
               delta=["bulb", "liqA", "liqB"], meta=meta, diag=diag, real=real, arrays=[])
    off = 0
    for nm, arr in arrays:
        arr = np.ascontiguousarray(arr)
        hdr["arrays"].append(dict(name=nm, dtype=str(arr.dtype), shape=list(arr.shape), offset=off))
        off += arr.nbytes
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    raw = b"".join([np.array([len(hb)], np.uint32).tobytes(), hb] + [np.ascontiguousarray(arr).tobytes() for _, arr in arrays])
    gz = gzip.compress(raw, compresslevel=9)
    with open(a.out, "wb") as fh:
        fh.write(gz)
    sizes = " ".join(f"{nm} {np.asarray(arr).nbytes/1e6:.1f}" for nm, arr in arrays if np.asarray(arr).nbytes > 2e5)
    print(f"wrote {a.out}: {len(gz)/1e6:.2f} MB gzip ({len(raw)/1e6:.1f} raw: {sizes}), {nf} frames, bulb {nv} verts, "
          f"water {liqA.shape[1]} bore/left-container + {liqB.shape[1]} bulk x {liqB.shape[0]} frames, recording {recording}, t_ref {t[jr]:.2f}, "
          f"flask level {flask_level if flask_level is None else round(flask_level, 1)} mm")


if __name__ == "__main__":
    main()
