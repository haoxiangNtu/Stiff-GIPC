"""Replay a real teleoperation recording on StiffGIPC: a Rokae AR5 dual-arm robot with two Wuji
five-finger hands takes a rubber-bulb dropper out of a flask, draws up blue water, drips it into a
beaker of yellow water, puts the dropper back; the other hand then picks the beaker up, swirls it
and sets it down. Water = the engine's MPM liquid, solved in the same IPC Newton step as the robot,
the glassware and the FEM dropper.

What is replayed: the measured arm joint angles (100 Hz) and the commanded finger joint angles
(about 70 Hz) of the recording, interpolated to the simulation step; nothing is scripted.

What is modelled:
  * robot      AR5 links and Wuji hand links are ABD bodies on revolute joints driven to the
               recorded angles (finite drive stiffness: fingers yield under contact).
  * dropper    one FEM body: glass tube (E 2e8) + hollow rubber bulb (E 3e5, 1.5 mm wall). The
               fingers squeeze the bulb; the bulb's cavity volume is read from the deformed mesh.
  * pump       air-chamber model: while the tip is under water the trapped air volume is constant,
               so the water column in the bore follows the cavity volume (squeeze -> water out,
               release -> water in); a squeeze with an empty bore bubbles air out, a release with
               the tip in air lets air in. The column is moved with the engine's dropper pump
               (gravity cancelled + velocity tracking in the bore, conveyor = volume error / area).
  * water      engine MPM liquid (implicit surface tension), blue in the flask, yellow in the
               beaker; colour is a per-particle label (no diffusion).

Frames: world is Y-up with the table top at y = 0; the robot description is Z-up (x forward,
y left) and is mapped x -> X, z -> Y (minus the table height), y -> -Z.
"""
from __future__ import annotations
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(osp.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(REPO, "examples", "lab_tactile"))
import robot_assets as RA  # noqa: E402
from hand_properties import load_hand_properties, apply_hand_inertia, drive_strength_ratio

TRAJ = os.environ.get("RD_TRAJ", osp.join(REPO, "Assets", "rokae_wuji", "teleop_dropper.npz"))   # raw recording (RD_TRAJ overrides it for another recording)
ASSETS_ROOT = osp.join(REPO, "Assets")

# ---------------------------------------------------------------- scene calibration (robot frame, m)
TABLE_Z = 0.39             # table height in the robot frame; set from the recording by table_z_from_recording()
# event times of the recording (s); defaults = recording Rokae_AR5_260615_145413_0; other recordings override them with
# RD_T_GRASP / RD_T_DRIP / RD_T_GRASP_BEAKER (ik_retarget.py and the calibration read these names)
T_GRASP_DROPPER = float(os.environ.get("RD_T_GRASP", 15.5))        # right hand closed on the bulb, dropper still standing in the flask
T_DRIP = float(os.environ.get("RD_T_DRIP", 25.5))                  # right hand squeezing over the beaker
T_GRASP_BEAKER = float(os.environ.get("RD_T_GRASP_BEAKER", 39.5))  # left hand closed on the beaker
R_WR = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])     # robot -> world rotation


def robot_to_world(p):
    p = np.asarray(p, float)
    return R_WR @ p + np.array([0.0, -TABLE_Z, 0.0])


def base_T():
    T = np.eye(4)
    T[:3, :3] = R_WR
    T[1, 3] = -TABLE_Z
    return T


# RD_CMD_OFFSET="joint:rad,joint:rad": zero offsets added to recorded finger commands before the URDF limits are applied.
# The recorded values are the teleop glove's commands; the video of recording 150534 shows the right thumb's base
# (right_finger1_joint1) about 0.9 rad less opposed than the command says when read as a URDF angle (11-09 silhouette fit).
CMD_OFFSET = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in os.environ.get("RD_CMD_OFFSET", "").split(",") if ":" in kv}


def parse_finger_drive_scales(entries):
    """Resolve explicit per-joint drive multipliers; these are not force measurements."""
    known = {name for names in RA.FINGER_JOINTS.values() for name in names}
    scales = {}
    for entry in entries:
        try:
            name, raw = entry.split("=", 1)
            value = float(raw)
        except (ValueError, TypeError):
            raise ValueError(f"invalid finger drive scale {entry!r}; use JOINT=FACTOR") from None
        if name not in known:
            raise ValueError(f"unknown finger joint {name!r}")
        if name in scales:
            raise ValueError(f"duplicate finger drive scale for {name}")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"finger drive scale for {name} must be finite and positive")
        scales[name] = value
    return scales


class Teleop:
    """Recorded joint angles, interpolated in time; finger angles clamped to the URDF limits."""

    def __init__(self, path=TRAJ, kin=None):
        z = np.load(path)
        self.t = {k: z[f"{k}_t"] for k in ("arm", "left", "right")}
        self.q = {k: z[f"{k}_q"] for k in ("arm", "left", "right")}
        self.names = {k: [str(n) for n in z[f"{k}_names"]] for k in ("arm", "left", "right")}
        self.lim = {}
        if kin is not None:
            for n, j in kin.joints.items():
                if "lim" in j:
                    self.lim[n] = j["lim"]

    def at(self, t):
        out = {}
        for k in ("arm", "left", "right"):
            tt, qq = self.t[k], self.q[k]
            i = int(np.clip(np.searchsorted(tt, t), 1, len(tt) - 1))
            w = float(np.clip((t - tt[i - 1]) / max(tt[i] - tt[i - 1], 1e-9), 0.0, 1.0))
            q = (1 - w) * qq[i - 1] + w * qq[i]
            for n, v in zip(self.names[k], q):
                lo, hi = self.lim.get(n, (-1e9, 1e9))
                out[n] = float(np.clip(v + CMD_OFFSET.get(n, 0.0), lo + 1e-4, hi - 1e-4))
        return out


# ---------------------------------------------------------------- calibration from the recording

def pinch(f, side):
    return 0.5 * (f[f"{side}_finger1_tip_link"][:3, 3] + f[f"{side}_finger2_tip_link"][:3, 3])


# Where the bulb sits in the right hand (palm frame): the point held from opposite sides by the thumb and the
# index / middle fingers with the smallest gap over the loosest part of the recorded hold (17-23 s), i.e. the
# centre of the recorded pinch (find_grasp_cage.py). The finger-tip midpoint is 1 cm off it.
RIGHT_BULB_IN_PALM = np.array([float(v) for v in os.environ["RD_BULB_IN_PALM"].split()]) if os.environ.get("RD_BULB_IN_PALM") else np.array([0.0739, 0.0197, 0.1161])   # RD_BULB_IN_PALM="x y z" for another recording (find_grasp_cage.py)


def bulb_centre(f):
    return (f["right_palm_link"] @ np.r_[RIGHT_BULB_IN_PALM, 1.0])[:3]


# Where the beaker axis sits in the left hand (palm frame, at the recorded grasp 39.5 s): the vertical line that the
# thumb and the index/middle/ring fingers close around from opposite sides (cos -0.98) within the beaker's height band
# (find_grasp_cage.py --left). The thumb-index midpoint is 1.2 cm off it.
LEFT_BEAKER_IN_PALM = np.array([0.0762, -0.0126, 0.1289])


def beaker_grasp_centre(f):
    return (f["left_palm_link"] @ np.r_[LEFT_BEAKER_IN_PALM, 1.0])[:3]


def calibrate(kin, tele, tube_below_bulb, beaker_offset_robot=(0.0, 0.0)):
    """Object placements implied by the recording (robot frame): the flask under the right hand's
    pinch at the grasp, the beaker under the dropper tip at the drip (dropper rigid in the hand
    from the grasp on), and the left-arm mount offset that puts the left grasp on that beaker."""
    f0 = kin.fk(tele.at(T_GRASP_DROPPER))
    c0 = bulb_centre(f0)
    P0 = f0["right_palm_link"]
    bulb_local = np.linalg.inv(P0) @ np.r_[c0, 1.0]
    down_local = P0[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    f1 = kin.fk(tele.at(T_DRIP))
    P1 = f1["right_palm_link"]
    tip = (P1 @ bulb_local)[:3] + P1[:3, :3] @ down_local * tube_below_bulb
    fl = kin.fk(tele.at(T_GRASP_BEAKER))
    lg = beaker_grasp_centre(fl)
    beaker_xy = tip[:2] + np.asarray(beaker_offset_robot, float)
    shift = np.r_[beaker_xy - lg[:2], 0.0]
    return dict(flask_xy=c0[:2], bulb_z=float(c0[2]), beaker_xy=beaker_xy, left_shift=shift, drip_tip=tip,
                left_grasp=lg)


# ---------------------------------------------------------------- props (lab reference models)
# Assets/rokae_wuji/lab_ref: flask / beaker = the reference glass meshes (real wall thickness, watertight,
# Y-up, base at y = 0); pipette_stem / pipette_bulb = the reference pipette (25 cm, bulb 35 mm) in its own
# frame (tip at the origin, axis +y). Exported from lab_reference_models.zip (Blender 5) by export_lab_ref.py.
LAB_REF = osp.join(ASSETS_ROOT, "rokae_wuji", "lab_ref")
FLASK_FLOOR = 0.0036                  # inner floor height of the reference flask
FLASK_INNER = np.array([(0.0049, 0.0360), (0.0161, 0.0362), (0.1022, 0.0173), (0.1150, 0.0144), (0.1600, 0.0144)])
# RD_FLASK_H: outer height (m) of the flask actually on the bench. The reference mesh is 160 mm; the two-camera
# triangulation of recording 150534 (11-09: centres of the base and rim ellipses) gives 137 mm (a 250 mL Erlenmeyer).
# The mesh, the inner profile and the floor are scaled in height only (the radii are kept).
FLASK_HSCALE = float(os.environ["RD_FLASK_H"]) / 0.160 if os.environ.get("RD_FLASK_H") else 1.0
if FLASK_HSCALE != 1.0:
    FLASK_FLOOR *= FLASK_HSCALE
    FLASK_INNER = FLASK_INNER * np.array([FLASK_HSCALE, 1.0])
BEAKER_FLOOR, BEAKER_R_IN, BEAKER_RIM = 0.0044, 0.0318, 0.0984
# pipette profile (r, y) from the reference model, tip at y = 0
STEM_R_OUT, STEM_R_IN, STEM_TOP = 0.0046, 0.0032, 0.200
BULB_OUTER = [(0.0054, 0.195), (0.0058, 0.201), (0.0060, 0.205), (0.0068, 0.209), (0.0078, 0.213), (0.0086, 0.215),
              (0.0094, 0.217), (0.0104, 0.219), (0.0115, 0.221), (0.0131, 0.223), (0.0145, 0.225), (0.0153, 0.227),
              (0.0162, 0.229), (0.0168, 0.231), (0.0171, 0.233), (0.0172, 0.236), (0.0169, 0.239), (0.0163, 0.241),
              (0.0154, 0.243), (0.0136, 0.245), (0.0113, 0.247), (0.0084, 0.249), (0.0, 0.2495)]
BULB_INNER = [(0.0, 0.2475), (0.0021, 0.247), (0.0081, 0.245), (0.0112, 0.243), (0.0127, 0.241), (0.0138, 0.239),
              (0.0143, 0.237), (0.0144, 0.235), (0.0141, 0.233), (0.0136, 0.231), (0.0129, 0.229), (0.0115, 0.227),
              (0.0102, 0.225), (0.0091, 0.223), (0.0081, 0.221), (0.0071, 0.219), (0.0062, 0.217), (0.0055, 0.215),
              (0.0049, 0.213), (0.0044, 0.209), (STEM_R_IN, 0.204)]
BULB_EQUATOR = 0.235                  # widest ring of the bulb above the tip: what the pinch holds
# RD_STEM_TOP: a shorter stem (the real dropper of recording 150534 measures ~185 mm bulb-centre-to-tip in the video,
# the reference pipette 235 mm): the bulb profile and its neck rings slide down with the stem top.
_STEM_TOP_REF = STEM_TOP
STEM_TOP = float(os.environ.get("RD_STEM_TOP", _STEM_TOP_REF))
_STEM_SHIFT = _STEM_TOP_REF - STEM_TOP
if abs(_STEM_SHIFT) > 1e-9:
    BULB_OUTER = [(r, y - _STEM_SHIFT) for r, y in BULB_OUTER]
    BULB_INNER = [(r, y - _STEM_SHIFT) for r, y in BULB_INNER]
    BULB_EQUATOR -= _STEM_SHIFT
E_BULB_SILICONE = 1.2e6               # silicone squeeze bulb
E_GLASS_STEM = 2e8                    # glass stem (stiff FEM stand-in for glass, as in the lab dropper)
PIPETTE_RHO = 1700.0                  # 17 g glass stem + 10 g silicone bulb over the solid's volume


def load_ref_mesh(name):
    import trimesh
    m = trimesh.load(osp.join(LAB_REF, name + ".obj"), process=False)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32)


def build_flask():
    import lab_assets as LA
    v, f = load_ref_mesh("flask")
    if FLASK_HSCALE != 1.0:
        v = v * np.array([1.0, FLASK_HSCALE, 1.0])
    return LA.Prop("flask", v, f, "ABD", LA.GLASS_RHO, color=(0.80, 0.88, 0.95), meta=dict(height=float(v[:, 1].max())))


def build_beaker():
    import lab_assets as LA
    v, f = load_ref_mesh("beaker")
    return LA.Prop("beaker", v, f, "ABD", LA.GLASS_RHO, color=(0.80, 0.88, 0.95), meta=dict(height=float(v[:, 1].max())))


def flask_r_in(y):
    return np.interp(y, FLASK_INNER[:, 0], FLASK_INNER[:, 1])


def build_dropper(max_volume=2.5e-9):
    """The reference pipette as one FEM solid: glass stem with an open bore (r 3.2 mm) + hollow silicone bulb
    whose cavity joins the bore. (The real stem tapers to a ~1 mm tip; the bore is kept straight so that the
    2.5 mm liquid particles can enter.) Cached in Assets/rokae_wuji/lab_ref (tetgen only to rebuild)."""
    import lab_assets as LA
    cache = osp.join(LAB_REF, f"pipette_fem_{max_volume*1e12:.0f}.npz")
    if osp.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return LA.Prop("dropper", z["verts"], z["faces"], "FEM", float(z["density"]), tets=z["tets"],
                       tet_young=z["tet_young"], color=(0.15, 0.35, 0.85), meta=z["meta"].item())
    outer = [(STEM_R_OUT, 0.0)] + [(STEM_R_OUT, float(y)) for y in np.linspace(0.0, 0.194, 14)[1:]] + BULB_OUTER
    inner = BULB_INNER + [(STEM_R_IN, float(y)) for y in np.linspace(0.200, 0.0, 15)]
    m = LA.revolve(np.array(outer + inner), 20)
    verts, tets = LA._tetrahedralise(m, max_volume)
    faces = LA._boundary_faces(tets)
    cen = verts[tets].mean(1)
    rc = np.hypot(cen[:, 0], cen[:, 2])
    glass = (cen[:, 1] < STEM_TOP - 0.5e-3) & (rc < STEM_R_OUT + 0.3e-3)
    young = np.where(glass, E_GLASS_STEM, E_BULB_SILICONE)
    op = np.array(outer)
    fc = verts[faces].mean(1)
    r_f = np.hypot(fc[:, 0], fc[:, 2])
    r_outer = np.interp(fc[:, 1], op[:, 1], op[:, 0])
    cavity_faces = faces[(r_f < r_outer - 0.7e-3) & (fc[:, 1] > 0.3e-3)]
    rr = np.hypot(verts[:, 0], verts[:, 2])
    tip_idx = np.where(verts[:, 1] < 0.3e-3)[0]
    top_idx = np.where((np.abs(verts[:, 1] - 0.203) < 1.5e-3) & (rr < STEM_R_IN + 0.5e-3))[0]
    a, b, c = verts[cavity_faces[:, 0]], verts[cavity_faces[:, 1]], verts[cavity_faces[:, 2]]
    v_cav = -float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    meta = dict(tube_len=STEM_TOP, tube_r=STEM_R_OUT, tube_r_in=STEM_R_IN, bulb_cy=BULB_EQUATOR, bulb_r=0.0172,
                E_bulb=E_BULB_SILICONE, E_tube=E_GLASS_STEM, hollow=True, hollow_bulb=True,
                tip_idx=tip_idx.astype(np.int32), top_idx=top_idx.astype(np.int32),
                cavity_faces=cavity_faces.astype(np.int32), cavity_volume=v_cav, tip_y=FLASK_FLOOR + 0.5e-3,
                glass_mask=glass)
    d = LA.Prop("dropper", verts, faces, "FEM", PIPETTE_RHO, tets=tets, tet_young=young, color=(0.15, 0.35, 0.85), meta=meta)
    np.savez_compressed(cache, verts=d.verts, faces=d.faces, tets=d.tets, tet_young=d.tet_young, density=d.density,
                        meta=np.array(d.meta, dtype=object))
    return d


NECK_Y = (STEM_TOP - 0.005, STEM_TOP - 0.0025, STEM_TOP)        # where the bulb's neck grips the stem (stitched vertex rings)
# RD_COLLAR: the glass tube reaches this far beyond STEM_TOP into the bulb's neck (a real dropper's tube sits 10-20 mm
# deep in the bulb); the bulb hugs it (inner radius = tube radius) and is stitched to it every 5 mm, so the bulb can only
# bend above the collar instead of at a 5 mm neck (11-07: that neck stored 14 deg of bend in the flask neck -> exit snap).
COLLAR = float(os.environ.get("RD_COLLAR", "0.0"))
SECTIONS = 20


def build_pipette(max_volume=2.5e-9):
    """The reference pipette in two parts (glass is rigid; a 13 k-tet glass FEM made every PCG solve 4 s):
    stem = ABD rigid glass tube (open bore r 3.2 mm, outer 4.6 mm, 200 mm); bulb = FEM silicone shell whose neck
    sits on the top 5 mm of the stem. Both are revolved from the same rings, so the bulb's neck vertices coincide with
    stem vertices: those pairs are stitched (bilateral springs) and the two bodies do not collide with each other.
    Cached in Assets/rokae_wuji/lab_ref (tetgen only to rebuild)."""
    import lab_assets as LA
    tag = "" if abs(_STEM_SHIFT) < 1e-9 else f"_top{STEM_TOP*1e3:.0f}"
    if COLLAR > 0:
        tag += f"_col{COLLAR*1e3:.0f}"
    cache = osp.join(LAB_REF, f"pipette_split_{max_volume*1e12:.0f}{tag}.npz")
    if osp.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["stem"].item(), z["bulb"].item()
    y0, y1, y2 = NECK_Y
    gt = STEM_TOP + COLLAR                                   # glass top
    if COLLAR > 0:
        rings = [y0 + 0.005 * k for k in range(int(round((gt - y0) / 0.005)) + 1)]   # stitched rings every 5 mm up to the glass top
    else:
        rings = list(NECK_Y)
    ys = sorted(set([0.0] + [float(y) for y in np.linspace(0.0, y0 - 0.005, 12)] + rings))
    stem_poly = [(STEM_R_OUT, y) for y in ys] + [(STEM_R_IN, y) for y in ys[::-1]]
    ms = LA.revolve(np.array(stem_poly), SECTIONS)
    sv, sf = np.asarray(ms.vertices), np.asarray(ms.faces, np.int32)
    # bulb: neck on the stem (r = STEM_R_OUT, y STEM_TOP-5mm..glass top), outer profile, cavity down to the glass top
    if COLLAR > 0:
        inner = [p for p in BULB_INNER if p[1] > gt + 1e-4]          # cavity wall above the glass top (dome downwards)
        hug = [(STEM_R_OUT, y) for y in rings[::-1] if y > y0 + 1e-9]   # annular floor at the glass top, then down the glass
        bulb_poly = [(STEM_R_OUT, y0), (0.0054, y0)] + BULB_OUTER + inner + hug
    else:
        bulb_poly = [(STEM_R_OUT, y0), (0.0054, y0)] + BULB_OUTER + BULB_INNER[:-2] + [(STEM_R_OUT, y2 + 0.004), (STEM_R_OUT, y2), (STEM_R_OUT, y1)]
    y2 = gt
    mb = LA.revolve(np.array(bulb_poly), SECTIONS)
    bv, bt = LA._tetrahedralise(mb, max_volume)
    bf = LA._boundary_faces(bt)
    # stitched pairs: bulb vertices on the neck rings that coincide with stem vertices
    from scipy.spatial import cKDTree
    d, j = cKDTree(sv).query(bv)
    rb = np.hypot(bv[:, 0], bv[:, 2])
    on_neck = (d < 1e-7) & (np.abs(rb - STEM_R_OUT) < 1e-6) & (bv[:, 1] < y2 + 1e-4)
    pairs = np.c_[np.nonzero(on_neck)[0], j[on_neck]].astype(np.int32)
    # cavity faces: bulb boundary faces facing the cavity (above the stem top, inside the outer profile)
    op = np.array([(0.0054, y0)] + BULB_OUTER)
    fc = bv[bf].mean(1)
    r_f = np.hypot(fc[:, 0], fc[:, 2])
    r_outer = np.interp(fc[:, 1], op[:, 1], op[:, 0])
    cav = bf[(r_f < r_outer - 0.7e-3) & (fc[:, 1] > y2 + 1e-5)]
    top = np.array([0.0, y2, 0.0])
    a, b, c = bv[cav[:, 0]] - top, bv[cav[:, 1]] - top, bv[cav[:, 2]] - top
    v_cav = -float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    rs = np.hypot(sv[:, 0], sv[:, 2])
    stem = dict(verts=sv, faces=sf, tip_idx=np.where((sv[:, 1] < 1e-6) & (rs < STEM_R_IN + 1e-4))[0].astype(np.int32),
                top_idx=np.where((sv[:, 1] > y2 - 1e-6) & (rs < STEM_R_IN + 1e-4))[0].astype(np.int32))
    bulb = dict(verts=bv, tets=bt, faces=bf, cavity_faces=cav.astype(np.int32), cavity_volume=v_cav, pairs=pairs)
    np.savez_compressed(cache, stem=np.array(stem, dtype=object), bulb=np.array(bulb, dtype=object))
    return stem, bulb


def table_z_from_recording(kin, tele):
    """Table height (robot frame) such that the pipette stands on the flask floor with its bulb equator at the
    recorded pinch centre."""
    f0 = kin.fk(tele.at(T_GRASP_DROPPER))
    return float(bulb_centre(f0)[2]) - (FLASK_FLOOR + 0.5e-3 + BULB_EQUATOR)


# ---------------------------------------------------------------- pump

class AirChamberPump:
    """Water column in the bore driven by the bulb's cavity volume (incompressible trapped air)."""

    G = 9.81

    def __init__(self, scene, tau=0.05, vmax=0.15, track=1.0):
        self.sc = scene
        self.track = float(track)
        st, bu = scene.stem, scene.bulb
        self.tip_idx, self.top_idx = st["v_off"] + st["tip_idx"], st["v_off"] + st["top_idx"]
        self.cav = np.asarray(bu["cavity_faces"], np.int64)
        self.r_in = STEM_R_IN
        self.r_out = STEM_R_OUT
        self.area = np.pi * self.r_in ** 2
        self.vp = float(scene.particle_volume)
        self.tau, self.vmax = tau, vmax
        self.V_air_ref = None
        self.state = {}

    def cavity_volume(self, vb, top):
        """bulb cavity above the stem's top (origin on the stem's top plane, where the cavity's open side lies)"""
        a, b, c = vb[self.cav[:, 0]] - top, vb[self.cav[:, 1]] - top, vb[self.cav[:, 2]] - top
        return -float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def apply(self, liq):
        V_all = np.asarray(self.sc.eng.get_vertices())
        tip, top = V_all[self.tip_idx].mean(0), V_all[self.top_idx].mean(0)
        ax = top - tip
        L = float(np.linalg.norm(ax))
        ax /= L
        bu = self.sc.bulb
        V = self.cavity_volume(V_all[bu["v_off"]:bu["v_off"] + bu["n"]], top)
        # liquid in the bore / just under the tip
        rel = liq - tip
        s = rel @ ax
        rad = np.linalg.norm(rel - np.outer(s, ax), axis=1)
        in_bore = (s > 0) & (s < L) & (rad < self.r_in)
        # tip under water: liquid around the last 10 mm of the stem / just below the tip, outside the bore (the tip may
        # stand on the flask floor, so "below the tip" alone is not enough)
        around = (s > -8e-3) & (s < 10e-3) & (rad < self.r_out + 4e-3) & ~in_bore
        V_col = float(in_bore.sum()) * self.vp
        submerged = int(around.sum()) >= 6
        V_bore = self.area * L
        if self.V_air_ref is None:
            self.V_air_ref = V + V_bore - V_col
        target = V + V_bore - self.V_air_ref
        event = ""
        if target < -0.05 * self.vp:                       # squeeze with an empty bore: air bubbles out
            self.V_air_ref = V + V_bore
            target = 0.0
            event = "bubble"
        target = max(target, 0.0)
        if target > V_bore:
            target = V_bore
        if target > V_col and not submerged:               # release with the tip in air: air enters
            self.V_air_ref = V + V_bore - V_col
            target = V_col
            event = event or "air-in"
        err = target - V_col
        conveyor = float(np.clip(err / self.area / self.tau, -self.vmax, self.vmax))
        # accel = g + full velocity tracking: the sealed column moves with the tube (air cannot get past the water in a
        # 6.4 mm bore under these accelerations); the conveyor alone changes how much water is in the bore
        self.sc.eng.set_liquid_pump(tip, top, self.r_in + 0.5e-3, self.G, conveyor, 0.0, self.track, bore_radius=self.r_in)
        self.state = dict(V_cav=V, V_col=V_col, n_col=int(in_bore.sum()), target=target, conveyor=conveyor,
                          submerged=submerged, event=event, tip=tip, top=top)
        return self.state


# ---------------------------------------------------------------- scene

class RokaeDropperScene:
    def __init__(self, args):
        import lab_assets as LA
        from lab_scene import Body, abd_world_tf, _load_urdf_capture_joint_indices
        from stiff_physics.engine import Config, Engine
        self.Body, self.abd_world_tf = Body, abd_world_tf
        self.args = args
        kin0 = RA.URDFKin(RA.build()["sim"])
        self.tele = Teleop(path=args.traj or TRAJ, kin=kin0)
        global TABLE_Z
        tele_cal = Teleop(path=args.cal_traj, kin=kin0) if args.cal_traj else self.tele
        if args.scene_cal:
            # scene from an external calibration (video_calibration/*.json: table_z, flask_xy, beaker_xy, left_shift in the
            # robot frame) instead of the pinch-cage inference from the finger commands
            import json as _json
            scene_json = _json.load(open(args.scene_cal))
            TABLE_Z = float(scene_json["table_z"]) + args.table_dz
        else:
            TABLE_Z = table_z_from_recording(kin0, tele_cal) + args.table_dz
        self.stem_data, self.bulb_data = build_pipette()
        tip_y = FLASK_FLOOR + 0.5e-3
        # beaker: under the drip tip of a rigid-in-hand pipette, plus the offset measured in the simulation (the pipette
        # pivots in the pinch while it leaves the flask neck and carries ~25 deg instead of ~13 deg); world (dx, dz)
        # -> robot (dx, -dz)
        off = (args.beaker_offset[0], -args.beaker_offset[1])
        if args.scene_cal:
            cal = dict(flask_xy=np.array(scene_json["flask_xy"], float), bulb_z=TABLE_Z + FLASK_FLOOR + BULB_EQUATOR,
                       beaker_xy=np.array(scene_json["beaker_xy"], float) + np.asarray(off, float),
                       left_shift=np.array(scene_json.get("left_shift", [0.0, 0.0, 0.0]), float),
                       drip_tip=np.zeros(3), left_grasp=np.zeros(3))
            print(f"[cal] scene placements from {args.scene_cal}", flush=True)
        else:
            cal = calibrate(kin0, tele_cal, BULB_EQUATOR, off)      # bulb centre -> tip along the tube
        self.cal = cal
        shift = cal["left_shift"] if not args.no_left_shift else np.zeros(3)
        self.urdf = RA.build(left_shift=shift)
        self.kin = RA.URDFKin(self.urdf["sim"])
        print(f"[cal] table z {TABLE_Z:.3f}  flask xy {np.round(cal['flask_xy'], 4)}  bulb z {cal['bulb_z']:.3f} "
              f"(stem {STEM_TOP*1e3:.0f} mm rigid, bulb {len(self.bulb_data['tets'])} tets, cavity {self.bulb_data['cavity_volume']*1e6:.2f} ml)  "
              f"beaker xy {np.round(cal['beaker_xy'], 4)}  left-arm shift {np.round(shift, 4)}", flush=True)

        cfg = Config(dt=args.dt, density=1000.0, young_modulus=1e7, poisson_rate=0.45,
                     friction_rate=0.5, gravity=(0.0, -9.81, 0.0), ground_offset=0.0, ground_normal=(0.0, 1.0, 0.0),
                     joint_strength_ratio=args.joint_ratio, revolute_driving_strength_ratio=args.joint_ratio,
                     max_revolute_step_per_frame=args.rev_step, absolute_dhat=args.dhat, preconditioner_type=args.precond,
                     assets_dir=ASSETS_ROOT + "/", soft_motion_rate=1e4,
                     newton_tol=args.newton_tol, pcg_tol=args.pcg_tol,
                     newton_velocity_tol=args.newton_velocity_tol,
                     newton_iter_cap=args.newton_cap,
                     kappa_mean_mass=args.kappa_mean_mass,
                     pcg_max_iter_ratio=args.pcg_ratio, energy_rel_tol=0.0)
        self.eng = eng = Engine(cfg)
        eng.set_log_level(0)
        self.bodies = {}

        # 1. robot (ABD), loaded at the first replayed pose
        q0 = self.tele.at(args.t0)
        self.joint_constraint_idx = _load_urdf_capture_joint_indices(eng, self.urdf["sim"], base_T(), True, False, 1e7, q0)
        eng.set_log_level(0)
        self.robot_bodies = []
        for r in eng.get_load_records():
            if r.body_type != 0:
                continue
            b = Body(r, r.label, "robot")
            self.bodies[r.label] = b
            self.robot_bodies.append(b)
            eng.add_ground_collision_skip(b.body_id)
        self.link_T0 = {b.name: np.asarray(eng.native.get_urdf_link_transform(b.name)) for b in self.robot_bodies}
        n_rev = eng.native.get_num_revolute_joints()
        self.rev_idx = {eng.get_revolute_joint_info(i).name: i for i in range(n_rev)}
        if args.hand_properties_data is not None:
            apply_hand_inertia(eng, self.bodies, self.link_T0, args.hand_properties_data)
            print(f"[hand-properties] applied authored mass/COM/inertia to {len(args.hand_properties_data['links'])} links; "
                  f"explicit drive stiffness, sha256={args.hand_properties_sha256}", flush=True)
        print(f"[scene] robot: {len(self.robot_bodies)} links, {n_rev} revolute joints", flush=True)

        # 2. props (ABD)
        self.props = {"flask": build_flask(), "beaker": build_beaker()}
        fxy = robot_to_world(np.r_[cal["flask_xy"], TABLE_Z])
        bxy = robot_to_world(np.r_[cal["beaker_xy"], TABLE_Z])
        gap = 1.0e-3
        self.prop_T = {"flask": self._T(fxy + [0, gap, 0]), "beaker": self._T(bxy + [0, gap, 0]),
                       "stem": self._T(fxy + [0, gap + tip_y, 0])}
        if args.lean_deg > 0:
            # the dropper rests LEANING in the flask (video: the open hand slid forward and pushed the bulb before closing):
            # tip on the floor displaced away from the pinch, stem against the far neck wall, bulb toward the pinch
            from scipy.spatial.transform import Rotation as _Rot
            if args.lean_dir_world is not None:
                # explicit world direction of the lean (bulb side), e.g. the video: bulb toward robot -y = world +z
                dirw = np.array([args.lean_dir_world[0], 0.0, args.lean_dir_world[1]], float)
            else:
                cw = robot_to_world(bulb_centre(kin0.fk(tele_cal.at(T_GRASP_DROPPER))))
                dirw = cw - fxy
            dirw[1] = 0.0; dirw = dirw / max(np.linalg.norm(dirw), 1e-9)
            # tip: lean_tip_height above the floor (a short dropper rests with its tip on the flask's cone wall and its
            # stem across the neck rim, not on the floor), displaced away from the lean
            tip_w = fxy + np.array([0, gap + tip_y + args.lean_tip_height, 0]) - dirw * args.lean_tip_offset
            axis = np.cross([0.0, 1.0, 0.0], dirw)
            Tl = np.eye(4); Tl[:3, :3] = _Rot.from_rotvec(axis * np.radians(args.lean_deg)).as_matrix(); Tl[:3, 3] = tip_w
            self.prop_T["stem"] = Tl
            print(f"[scene] dropper starts leaning {args.lean_deg:.1f} deg (bulb toward world dir {np.round(dirw, 2)}), tip {args.lean_tip_height*1e3:.0f} mm above the floor, "
                  f"offset {args.lean_tip_offset*1e3:.0f} mm; bulb centre {(gap + tip_y + args.lean_tip_height + BULB_EQUATOR*np.cos(np.radians(args.lean_deg)))*1e3:.0f} mm above the table", flush=True)
        import lab_assets as LA
        self.props["stem"] = LA.Prop("stem", self.stem_data["verts"], self.stem_data["faces"], "ABD", LA.GLASS_RHO,
                                     color=(0.9, 0.95, 1.0), meta={})
        for name in ("flask", "beaker", "stem"):
            self._add_prop(name)
        self.stem = dict(self.stem_data, v_off=self.bodies["stem"].v_off, body_id=self.bodies["stem"].body_id)

        # 3. silicone bulb (FEM) on the stem's top, stitched to it at the neck rings
        bd = self.bulb_data
        T_d = self.prop_T["stem"]
        self.prop_T["dropper"] = T_d
        bv = bd["verts"] @ T_d[:3, :3].T + T_d[:3, 3]
        eng.load_mesh_from_data(bv, bd["tets"], verts_per_face=4, dimensions=3, body_type="FEM",
                                transform=np.eye(4), young_modulus=args.e_bulb, boundary_type="Free")
        recs = eng.get_load_records()
        rec_idx = len(recs) - 1
        self.bulb = dict(bd, v_off=int(recs[-1].vertex_offset), n=len(bd["verts"]), body_id=rec_idx)
        self.dropper = dict(v_off=self.bulb["v_off"], n=self.bulb["n"], faces=bd["faces"], body_id=rec_idx)
        eng.set_soft_body_density(rec_idx, 1100.0)
        eng.set_body_friction(rec_idx, args.mu_bulb)
        sb = self.stem["body_id"]
        if args.collar_y > 0:
            # glass seat + thick collar: the bulb's lower cone (y < collar_y, pipette frame) is pinned rigidly to the stem
            # (hard FEM->ABD pins, substitution method). The legacy 5 mm stitched neck let the bulb bend 18-33 deg at the
            # neck under the flask-neck lever while the stem was pulled out (slipcheck/ringdiag 11-07); a real dropper's
            # glass sits deep in a thick collar and only the dome is soft.
            from scipy.spatial import cKDTree
            sel = np.nonzero(bd["verts"][:, 1] < args.collar_y)[0]
            _, js_near = cKDTree(self.stem_data["verts"]).query(bd["verts"][sel])
            for ib, js in zip(sel, js_near):
                eng.native.add_fem_pin_to_abd(int(self.bulb["v_off"] + ib), int(self.stem["v_off"] + js), int(sb),
                                              rest_offset_world=(0.0, 0.0, 0.0))
            print(f"[scene] bulb collar: {len(sel)}/{len(bd['verts'])} bulb vertices below y={args.collar_y*1e3:.1f} mm "
                  f"pinned rigidly to the stem (no neck stitches)", flush=True)
        else:
            for ib, js in bd["pairs"]:
                eng.add_stitch_spring(int(self.bulb["v_off"] + ib), int(self.stem["v_off"] + js), int(sb), (0.0, 0.0, 0.0))
        eng.add_collision_exclusion(int(sb), int(rec_idx))
        for b in self.robot_bodies:
            if "finger" in b.name or "palm" in b.name:
                eng.set_body_friction(b.body_id, args.mu_hand)

        # 4. liquid
        self.particle_volume = None
        self.liq_label = None
        if args.liquid:
            self._seed_liquid()

        t0 = time.time()
        eng.finalize()
        print(f"[scene] finalize {time.time()-t0:.1f}s: {eng.vertex_count} verts, {eng.abd_body_count} ABD, "
              f"{eng.fem_body_count} FEM", flush=True)
        ids = np.array([b.body_id for b in self.bodies.values()], np.int32)
        C0 = eng.get_abd_body_transforms(ids)
        for b, c in zip(self.bodies.values(), C0):
            b.C0inv = np.linalg.inv(abd_world_tf(c))
            bv = eng.get_abd_body(b.body_id)
            b.rest_verts = np.asarray(bv.get_vertices(), np.float64).copy()
            b.faces = np.asarray(bv.get_surface_faces(local_indices=True), np.int32).copy()
        for jname, (jidx, jtype) in self.joint_constraint_idx.items():
            if jtype == "Fixed":
                eng.native.set_fixed_joint_strength(jidx, 1000.0)
        for n, i in self.rev_idx.items():
            if n.startswith("right_finger"):
                strength = args.finger_strength if args.finger_strength_right is None else args.finger_strength_right
            elif n.startswith("left_finger"):
                strength = args.finger_strength if args.finger_strength_left is None else args.finger_strength_left
            else:
                continue
            if args.hand_properties_data is not None:
                strength = drive_strength_ratio(args.hand_properties_data, n, args.dt, args.joint_ratio)
                stiffness = args.hand_properties_data['drives'][n]['stiffness_nm_per_rad']
                print(f"[drive-physical] {n}: {stiffness:g} Nm/rad before per-joint multiplier; ratio {strength:g}", flush=True)
            scale = args.finger_drive_scales.get(n, 1.0)
            if getattr(args, "selflock", False) and n in args.selflock_table:
                strength, scale = args.selflock_table[n]["strength"], 1.0
            eng.set_revolute_strength(i, strength * scale)
            if scale != 1.0:
                print(f"[drive] {n}: strength {strength:g} * {scale:g} = {strength*scale:g}", flush=True)


    @staticmethod
    def _T(p):
        T = np.eye(4)
        T[:3, 3] = p
        return T

    def _add_prop(self, name):
        p = self.props[name]
        T = self.prop_T[name]
        wv = p.verts @ T[:3, :3].T + T[:3, 3]
        self.eng.load_mesh_from_data(wv, p.faces, verts_per_face=3, dimensions=3, body_type="ABD",
                                     transform=np.eye(4), young_modulus=1e8, boundary_type="Free")
        rec = self.eng.get_load_records()[-1]
        b = self.Body(rec, name, "prop", color=p.color)
        dens = p.density
        if name == "flask" and not self.args.liquid and self.args.flask_water_mass > 0:
            # dry tests: carry the water's mass in the glass (the flask's resistance to being dragged)
            import trimesh
            vol = trimesh.Trimesh(p.verts, p.faces, process=False).volume
            dens = p.density + self.args.flask_water_mass / vol
        self.eng.set_abd_body_density(b.body_id, dens)
        # glass-glass / glass-hand contacts use mu_glass; glass on the lab bench's anti-static rubber mat uses mu_table
        self.eng.set_body_friction(b.body_id, self.args.mu_glass, ground_mu=self.args.mu_table)
        self.bodies[name] = b

    def _seed_liquid(self):
        a = self.args
        dx = a.liquid_dx
        hs = dx / 2.0
        jit = float(os.environ.get("LAB_LIQ_JITTER", "0.31"))
        rng = np.random.default_rng(0)
        m = 0.6 * dx
        pts_all, lab_all = [], []
        # flask (blue): analytic cone, the standing dropper tube kept free
        Tf = self.prop_T["flask"]
        base, cx, cz = float(Tf[1, 3]), float(Tf[0, 3]), float(Tf[2, 3])
        y_floor = base + FLASK_FLOOR
        R0 = float(FLASK_INNER[:, 1].max())
        Ts = self.prop_T["stem"]
        tube_r = STEM_R_OUT

        def seed_flask(fill, mw, rng_f=None):
            """jittered grid inside the flask up to `fill` above the floor, `mw` away from the glass, the dropper's stem
            (standing or leaning: tested in the stem's own frame) kept free"""
            rng_f = np.random.default_rng(0) if rng_f is None else rng_f
            g = np.meshgrid(np.arange(cx - R0 + mw, cx + R0 - mw, hs), np.arange(y_floor + mw, y_floor + fill, hs),
                            np.arange(cz - R0 + mw, cz + R0 - mw, hs), indexing="ij")
            q = np.stack([x.ravel() for x in g], 1)
            q = q + rng_f.uniform(-jit * hs, jit * hs, q.shape)
            q = q[np.hypot(q[:, 0] - cx, q[:, 2] - cz) < flask_r_in(q[:, 1] - base) - mw]
            q = q[q[:, 1] > y_floor + mw]
            loc = (q - Ts[:3, 3]) @ Ts[:3, :3]                       # stem frame: tip at the origin, axis +y
            in_stem = (np.hypot(loc[:, 0], loc[:, 2]) < tube_r + 1.5e-3) & (loc[:, 1] > -3e-3) & (loc[:, 1] < STEM_TOP + 5e-3)
            return q[~in_stem]

        if a.flask_level > 0:
            # true liquid surface height above the table: the seeded grid keeps a margin to the glass, so a grid filled to
            # the target height settles 20-30 % lower (11-09: "70 mm" seeded -> 48 mm). Seed the particle count of the
            # inner volume up to the target (minus the immersed stem), with a smaller wall margin and a taller column
            mw = float(os.environ.get("RD_SEED_MARGIN", "0.3")) * dx
            ys = np.linspace(y_floor, a.flask_level, 2000)
            a_in = np.pi * flask_r_in(ys - base) ** 2
            v_in = float(np.sum(0.5 * (a_in[1:] + a_in[:-1]) * np.diff(ys)))       # (np.trapz is gone in NumPy 2)
            lean_cos = float(Ts[1, 1])
            v_stem = np.pi * tube_r ** 2 * max(a.flask_level - float(Ts[1, 3]), 0.0) / max(lean_cos, 0.5)
            n_target = int(round((v_in - v_stem) / hs ** 3))
            fill = a.flask_level - y_floor
            p = seed_flask(fill, mw)
            while len(p) < n_target and fill < 0.150:
                fill += hs
                p = seed_flask(fill, mw)
            if len(p) > n_target:
                p = p[np.argsort(p[:, 1], kind="stable")[:n_target]]
            print(f"[liquid] flask: target surface {a.flask_level*1e3:.0f} mm above the table = {v_in*1e6:.1f} mL inner volume - "
                  f"{v_stem*1e6:.1f} mL stem -> {len(p)} particles ({len(p)*hs**3*1e6:.1f} mL), seeded to {fill*1e3:.0f} mm above the floor, "
                  f"wall margin {mw*1e3:.2f} mm", flush=True)
        else:
            p = seed_flask(a.flask_fill, m, rng)                 # shared stream: same particles as before 11-09
            print(f"[liquid] flask: fill to {a.flask_fill*1e3:.0f} mm above the floor -> {len(p)} particles", flush=True)
        pts_all.append(p)
        lab_all.append(np.zeros(len(p), np.int8))
        # beaker (yellow)
        T = self.prop_T["beaker"]
        bx, bz = float(T[0, 3]), float(T[2, 3])
        r_in, yf, y1 = BEAKER_R_IN, float(T[1, 3]) + BEAKER_FLOOR, float(T[1, 3]) + BEAKER_RIM
        h = yf + a.beaker_fill * (y1 - yf)
        g = np.meshgrid(np.arange(bx - r_in + m, bx + r_in - m, hs), np.arange(yf + m, h, hs),
                        np.arange(bz - r_in + m, bz + r_in - m, hs), indexing="ij")
        p = np.stack([x.ravel() for x in g], 1)
        p = p + rng.uniform(-jit * hs, jit * hs, p.shape)
        p = p[np.hypot(p[:, 0] - bx, p[:, 2] - bz) < r_in - m]
        pts_all.append(p)
        lab_all.append(np.ones(len(p), np.int8))
        print(f"[liquid] beaker: r_in {r_in*1e3:.1f} mm floor {yf*1e3:.1f} rim {y1*1e3:.1f} -> fill to {h*1e3:.1f} mm, "
              f"{len(p)} particles", flush=True)
        pts = np.vstack(pts_all)
        self.liq_label = np.concatenate(lab_all)
        o = np.array(a.liquid_domain[:3])
        ext = np.array(a.liquid_domain[3:])
        self.particle_volume = hs ** 3
        self.liq_origin = o
        self.eng.add_liquid(pts, dx, o, o + ext, rho=1000.0, kappa=1e5, particle_volume=hs ** 3,
                            node_pool=a.liquid_node_pool, push_apart=0.9, j_mode=0, stick_h=0.0,
                            mass_floor=0.05, apic=True, apic_ratio=1.0)
        print(f"[liquid] total {len(pts)} particles, dx {dx*1e3:.1f} mm, domain {o} + {ext}", flush=True)

    # ------------------------------------------------------------ queries / driving
    def dropper_verts(self):
        d = self.dropper
        return np.asarray(self.eng.get_vertices())[d["v_off"]:d["v_off"] + d["n"]]

    def drive(self, q):
        for n, v in q.items():
            i = self.rev_idx.get(n)
            if i is not None:
                self.eng.set_revolute_target(i, v)

    def abd_raw(self):
        ids = np.array([b.body_id for b in self.bodies.values()], np.int32)
        return np.asarray(self.eng.get_abd_body_transforms(ids), np.float32)


# ---------------------------------------------------------------- self-locking finger drive (2026-11-11)
# Stall torques (N m) = the per-joint torque ranges of the vendor's system-identified hand model (wuji-hand-description
# v2026.04.27 MJCF, right hand); they are the vendor's values, not a measurement of the recorded session.
SELFLOCK_STALL = {1: (0.4452, 0.4259, 0.1888, 0.1468), 2: (0.6188, 0.1822, 0.2251, 0.2170), 3: (0.6494, 0.1827, 0.2078, 0.2018),
                  4: (0.6389, 0.1832, 0.2249, 0.2044), 5: (0.6441, 0.1798, 0.2384, 0.1866)}
SL_CAL0, SL_CAL1, SL_DITHER = 10, 40, 0.003        # calibration window (steps inside the settle) and setpoint dither (rad)
SELFLOCK_BASE_STRENGTH = 5e-6          # the drive-strength ratio wuji_authored_baseline_drives.json's stiffnesses were measured at


def selflock_tables(args):
    """per right-finger joint: (strength ratio that gives --selflock-k, stall torque, lead = stall / k)"""
    base = json.load(open(osp.join(HERE, "wuji_authored_baseline_drives.json")))["drives"]
    extra = {}
    for e in args.selflock_joint_scale:
        n, v = e.split("=", 1); extra[n] = float(v)
    out = {}
    for f, taus in SELFLOCK_STALL.items():
        for j, tau in enumerate(taus, 1):
            n = f"right_finger{f}_joint{j}"
            k_base = float(base[n]["stiffness_nm_per_rad"])            # N m / rad at SELFLOCK_BASE_STRENGTH with the glove-inflated links
            stall = tau * args.selflock_scale * extra.get(n, 1.0)
            out[n] = dict(strength=SELFLOCK_BASE_STRENGTH * args.selflock_k / k_base, stall=stall, lead=stall / args.selflock_k)
    return out


def _mesh_volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def selflock_predicted_k(sc, args, density=1000.0):
    """drive stiffness (N m / rad) the engine has for the CURRENT strengths: ratio * strength * (m_parent + m_child) / dt^2"""
    base = json.load(open(osp.join(HERE, "wuji_authored_baseline_drives.json")))["drives"]
    mass = {n: density * _mesh_volume(b.rest_verts, b.faces) for n, b in sc.bodies.items() if n.startswith("right_")}
    out = {}
    for n, d in args.selflock_table.items():
        pc = base[n]
        if pc["parent"] in mass and pc["child"] in mass:
            out[n] = args.joint_ratio * d["strength"] * (mass[pc["parent"]] + mass[pc["child"]]) / args.dt ** 2
    return out


def selflock_setpoint(sp, cmd, act, lead):
    """ratchet: follow the command, lead the joint by at most `lead` while advancing, never back off because of a load"""
    if cmd >= sp:
        return min(cmd, max(sp, act + lead))
    return max(cmd, min(sp, act - lead))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=osp.join(REPO, "out", "rokae_dropper", "run"))
    ap.add_argument("--traj", default=None, help="joint trajectory npz (default: the raw recording; ik_retarget.py writes the camera-corrected one)")
    ap.add_argument("--cal-traj", default=None, help="trajectory the object placements are calibrated from (default: --traj); "
                    "keep it fixed while a trajectory is refined (ik_refine.py) so the flask / beaker do not move with it")
    ap.add_argument("--t0", type=float, default=8.0, help="replay start (s in the recording)")
    ap.add_argument("--t1", type=float, default=62.0, help="replay end")
    ap.add_argument("--settle", type=float, default=0.3, help="hold the start pose this long first (s)")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--rec-every", type=int, default=8)
    ap.add_argument("--partial-every", type=int, default=1000, help="write rec_partial.npz every this many steps (for peeking at a running job)")
    ap.add_argument("--precond", type=int, default=0)
    ap.add_argument("--joint-ratio", type=float, default=1000.0)
    ap.add_argument("--finger-strength", type=float, default=5e-5,
                    help="finger joint drive-stiffness multiplier, not a measured force limit in N. Effective small-angle "
                         "stiffness is ratio*strength*(m_parent+m_child)/dt^2. The rate-limited drive target and joint error "
                         "determine torque; contact geometry determines the resulting finger load.")
    ap.add_argument("--rev-step", type=float, default=0.1,
                    help="engine's max_revolute_step_per_frame (rad): every frame the drive setpoint is the joint's last angle plus the "
                         "command error clamped to this. 0.1 (default): a finger whose command lies deeper than 0.1 rad pushes with a "
                         "CONSTANT torque and has no stiffness about where it stalled (the held object can swing between the fingers). "
                         "~1.0: a plain spring between command and joint (force proportional to the command error); lower "
                         "--finger-strength with it (the torque at 0.1 rad is what --finger-strength was tuned for)")
    ap.add_argument("--newton-tol", type=float, default=1e-4)
    ap.add_argument("--newton-velocity-tol", type=float, default=0.0,
                    help="physical Newton direction tolerance in m/s; 0 uses the legacy scene-relative tolerance")
    ap.add_argument("--pcg-tol", type=float, default=1e-6,
                    help="legacy reduction of the squared preconditioned residual (1e-4 means 1e-2 in that norm)")
    ap.add_argument("--newton-cap", type=int, default=80)
    ap.add_argument("--pcg-ratio", type=float, default=0.3)
    ap.add_argument("--mu-hand", type=float, default=1.0)
    ap.add_argument("--kappa-mean-mass", type=float, default=0.0, help="pin the mean vertex mass used for the adaptive contact stiffness kappa (0 = engine value; the liquid's massless slots dilute it, so a wet run has softer contacts than its dry twin unless pinned)")
    ap.add_argument("--dhat", type=float, default=5e-4,
                    help="contact (barrier) distance in metres. 5e-4 (default) is a thin, stiff layer: with a solve that stops on a "
                         "displacement tolerance the contact forces on a light held object are noisy by newtons (0.1 mm of unresolved "
                         "displacement across a stiff barrier) and kick it round in the grasp; 2e-3 acts like the compliant glove fabric")
    ap.add_argument("--mu-bulb", type=float, default=0.9)
    ap.add_argument("--mu-glass", type=float, default=0.5)
    ap.add_argument("--mu-table", type=float, default=0.8, help="glass on the bench's rubber (ESD) mat")
    ap.add_argument("--finger-strength-right", type=float, default=None)
    ap.add_argument("--finger-drive-scale", action="append", default=[], metavar="JOINT=FACTOR",
                    help="multiply one finger joint's configured drive strength; repeat for multiple joints. "
                         "Default is 1 for every joint. This changes the actuator model, not the recorded target angles.")
    ap.add_argument("--finger-strength-left", type=float, default=None)
    ap.add_argument("--hand-properties", default=None,
                    help="JSON with authored hand inertia and independent drive stiffness in Nm/rad; overrides legacy "
                         "finger-strength ratios, preserves recorded target angles and per-joint scale multipliers")
    ap.add_argument("--beaker-offset", type=float, nargs=2, default=(0.0, 0.0),
                    help="world (dx, dz) added to the beaker placement: where the simulated drip tip actually is "
                         "(r1_liq, stiff right hand: +0.049, +0.010)")
    ap.add_argument("--scene-cal", default=None,
                    help="JSON with table_z, flask_xy, beaker_xy, left_shift (robot frame) from the external-video calibration; "
                         "replaces the pinch-cage inference")
    ap.add_argument("--lean-deg", type=float, default=0.0, help="initial lean of the dropper in the flask toward the pinch (deg); 0 = vertical")
    ap.add_argument("--lean-tip-offset", type=float, default=0.03, help="tip displacement on the flask floor away from the pinch (m) when leaning")
    ap.add_argument("--lean-dir-world", type=float, nargs=2, default=None, help="world (dx, dz) direction the bulb leans toward (default: toward the recorded pinch)")
    ap.add_argument("--lean-tip-height", type=float, default=0.0, help="tip height above the flask floor when leaning (m); > 0 for a dropper shorter than the flask is deep")
    ap.add_argument("--collar-y", type=float, default=0.0,
                    help="pin the bulb vertices below this height (pipette frame, m; the stem top is 0.200) rigidly to the "
                         "glass stem = glass seat + thick collar, so the bulb bends only above it; 0 = legacy 5 mm stitched neck")
    ap.add_argument("--e-bulb", type=float, default=E_BULB_SILICONE, help="Young's modulus of the silicone bulb (Pa)")
    ap.add_argument("--flask-water-mass", type=float, default=0.117, help="dry runs only: water mass (kg) added to the flask")
    ap.add_argument("--no-left-shift", action="store_true")
    ap.add_argument("--table-dz", type=float, default=0.0, help="raise the table (m, robot frame) relative to the calibration")
    ap.add_argument("--liquid", action="store_true")
    ap.add_argument("--liquid-dx", type=float, default=0.005)
    ap.add_argument("--liquid-node-pool", type=int, default=90000)
    ap.add_argument("--liquid-domain", type=float, nargs=6, default=(0.54, -0.005, -0.13, 0.32, 0.42, 0.45))
    ap.add_argument("--flask-fill", type=float, default=0.045, help="SEEDING height of the water above the flask floor (m); the "
                    "settled surface is 20-30 %% lower (0.045 -> 36 mm, 0.066 -> 48 mm above the table)")
    ap.add_argument("--flask-level", type=float, default=0.0, help="true liquid surface height above the table (m) after settling; "
                    "> 0 overrides --flask-fill (video: 0.070)")
    ap.add_argument("--beaker-fill", type=float, default=0.38, help="fraction of the beaker height (floor to rim)")
    ap.add_argument("--no-pump", action="store_true")
    ap.add_argument("--pump-track", type=float, default=1.0, help="velocity tracking of the bore water to the tube (0..1)")
    ap.add_argument("--selflock", action="store_true",
                    help="right hand: self-locking position-controlled joints (the Wuji spec: 'self-locking rotary direct drive joints', "
                         "position control, current limit). Each joint is a stiff spring (--selflock-k) to a SETPOINT that follows the "
                         "recorded command; while it advances the setpoint may lead the joint by at most stall_torque / k (the motor "
                         "stalls), and it never retreats because of a load (a self-locking transmission cannot be back-driven), so a "
                         "finger that is pushed back holds with the full spring stiffness instead of yielding at constant torque")
    ap.add_argument("--selflock-k", type=float, default=100.0, help="holding stiffness of a self-locking joint (N m / rad)")
    ap.add_argument("--selflock-scale", type=float, default=1.0, help="multiplies the per-joint stall torques (vendor torque ranges)")
    ap.add_argument("--selflock-joint-scale", action="append", default=[], metavar="JOINT=FACTOR", help="extra stall-torque factor for one joint")
    ap.add_argument("--selflock-backlash", type=float, default=0.0, help="reserved (rad); 0 = none")
    args = ap.parse_args()
    try:
        args.finger_drive_scales = parse_finger_drive_scales(args.finger_drive_scale)
        args.hand_properties_data = None
        args.hand_properties_sha256 = None
        if args.hand_properties:
            args.hand_properties_data, args.hand_properties_sha256 = load_hand_properties(
                args.hand_properties, [name for names in RA.FINGER_JOINTS.values() for name in names])
            for name in args.hand_properties_data['drives']:
                drive_strength_ratio(args.hand_properties_data, name, args.dt, args.joint_ratio)
    except ValueError as error:
        ap.error(str(error))
    if args.newton_velocity_tol < 0 or not np.isfinite(args.newton_velocity_tol):
        ap.error("--newton-velocity-tol must be finite and nonnegative")
    if args.liquid:
        # A launcher must not silently reintroduce rest drag / sleep into this task.
        for key in ("LIQ_ISL_V", "LIQ_SLEEP_V", "LIQ_REST_K", "LIQ_REST_DAMP", "LIQ_STICK_H", "LIQ_NEWTON_CAP"):
            try:
                value = float(os.environ.get(key, "0"))
            except ValueError:
                ap.error(f"{key} must be zero for this physical-baseline replay")
            if not np.isfinite(value) or value != 0.0:
                ap.error(f"{key}={value:g}: sleep, rest drag and liquid-only Newton truncation are disabled for this task")
            os.environ[key] = "0"
    os.makedirs(args.out, exist_ok=True)
    args.selflock_table = selflock_tables(args) if args.selflock else {}
    if args.selflock:
        for n in ("right_finger1_joint2", "right_finger2_joint1", "right_finger2_joint3", "right_finger2_joint4"):
            d = args.selflock_table[n]
            print(f"[selflock] {n}: k {args.selflock_k:g} N m/rad (strength {d['strength']:.3g}), stall {d['stall']:.3f} N m, lead {np.degrees(d['lead']):.2f} deg", flush=True)

    sc = RokaeDropperScene(args)
    eng = sc.eng
    pump = AirChamberPump(sc, track=args.pump_track) if (args.liquid and not args.no_pump) else None
    names = list(sc.bodies)
    # static scene description for rendering
    np.savez_compressed(osp.join(args.out, "scene.npz"),
                        body_names=np.array(names), body_kind=np.array([sc.bodies[n].kind for n in names]),
                        C0inv=np.array([sc.bodies[n].C0inv for n in names]),
                        link_T0=np.array([sc.link_T0.get(n, np.eye(4)) for n in names]),
                        dropper_faces=sc.dropper["faces"], bulb_rest=sc.bulb["verts"] @ sc.prop_T["dropper"][:3, :3].T + sc.prop_T["dropper"][:3, 3],
                        stem_tip_idx=sc.stem["tip_idx"], stem_top_idx=sc.stem["top_idx"],
                        liq_label=sc.liq_label if sc.liq_label is not None else np.zeros(0, np.int8),
                        liq_origin=getattr(sc, "liq_origin", np.zeros(3)),
                        prop_T=json.dumps({k: v.tolist() for k, v in sc.prop_T.items()}),
                        cal=json.dumps({k: np.asarray(v).tolist() for k, v in sc.cal.items()}),
                        table_z=TABLE_Z, urdf_vis=sc.urdf["vis"], urdf_sim=sc.urdf["sim"], args=json.dumps(vars(args)),
                        flask_hscale=FLASK_HSCALE, cmd_offset=json.dumps(CMD_OFFSET),
                        force_measurement="legacy half-sum of per-vertex normal-contact magnitudes; not calibrated finger force",
                        solver_env=json.dumps({k: v for k, v in os.environ.items()
                                              if k.startswith(("LIQ_", "STIFF_", "STIFFGIPC_"))}),
                        hand_mount_env=json.dumps({k: os.environ.get(k, "") for k in ("WUJI_MOUNT", "WUJI_FLANGE_EXTRA", "WUJI_MOUNT_FIT", "WUJI_MOUNT_FIT_LEFT")}))
    np.savez_compressed(osp.join(args.out, "rest.npz"),
                        **{f"v_{i}": sc.bodies[n].rest_verts.astype(np.float32) for i, n in enumerate(names)},
                        **{f"f_{i}": sc.bodies[n].faces for i, n in enumerate(names)})

    rec = dict(t=[], abd=[], drop=[], liq=[], pump=[], q_cmd=[], q_act=[], force=[], stemF=[], q_set=[], tau=[])
    sl_sp = {}
    sl_ksamp = {}
    arm_and_finger = [n for n in sc.rev_idx]
    n_settle = int(round(args.settle / args.dt))
    n_steps = n_settle + int(round((args.t1 - args.t0) / args.dt))
    t_wall = time.time()
    frame_times = []
    for k in range(n_steps + 1):
        t = args.t0 + max(0, k - n_settle) * args.dt
        q = sc.tele.at(t)
        qd = q
        if args.selflock:
            ang_now = eng.get_revolute_current_angles_abs()
            qd = dict(q)
            # stiffness self-calibration during the settle (the hand is in free space). The engine's drive stiffness is
            # ratio * strength * (m_parent + m_child) / dt^2 and the link masses change with the glove thickness, so:
            # (a) predicted from the collision-mesh volumes (density 1000), (b) measured = the engine's own drive torque /
            # (setpoint - angle) while a 3 mrad step on the setpoints makes the joints move (median over the window).
            # One rescale at SL_CAL1 (measured where there are enough samples, predicted otherwise), verified by a second step.
            cal = SL_CAL0 <= k < SL_CAL1 or SL_CAL1 + 10 <= k < SL_CAL1 + 40
            if SL_CAL0 < k <= SL_CAL1 + 40:
                for n in args.selflock_table:
                    i = sc.rev_idx.get(n)
                    if i is None or n not in sl_sp:
                        continue
                    de = sl_sp[n] - float(ang_now[i])
                    if abs(de) > 3e-6:
                        sl_ksamp.setdefault(n, []).append(float(eng.get_revolute_drive_torque(i)) / de)
            if k in (SL_CAL1, SL_CAL1 + 40):
                meas = {n: float(np.median(v)) for n, v in sl_ksamp.items() if len(v) >= 8}
                if k == SL_CAL1:
                    pred = selflock_predicted_k(sc, args)
                    dev = [meas[n] / pred[n] - 1.0 for n in meas if n in pred]
                    print(f"[selflock] stiffness before correction: measured on {len(meas)} joints, predicted from mesh volumes on {len(pred)}; "
                          f"measured/predicted - 1: {min(dev, default=0):+.3f} .. {max(dev, default=0):+.3f}", flush=True)
                    for n, d in args.selflock_table.items():
                        km = meas.get(n, pred.get(n))
                        if km and km > 1e-6:
                            d["strength"] *= args.selflock_k / km
                            eng.set_revolute_strength(sc.rev_idx[n], d["strength"])
                else:
                    v = np.array(list(meas.values()))
                    print(f"[selflock] verified stiffness: {len(v)} joints, {v.min():.2f} .. {v.max():.2f} N m/rad (target {args.selflock_k:g})", flush=True)
                sl_ksamp.clear()
            for n, d in args.selflock_table.items():
                i = sc.rev_idx.get(n)
                if i is None or n not in q:
                    continue
                if k < SL_CAL1 + 40:
                    sp = q[n] + (SL_DITHER if cal else 0.0)          # free space: plain tracking (+ the calibration step)
                else:
                    sp = selflock_setpoint(sl_sp.get(n, q[n]), q[n], float(ang_now[i]), d["lead"])
                sl_sp[n] = sp
                qd[n] = sp
        sc.drive(qd)
        st = {}
        if pump is not None:
            st = pump.apply(np.asarray(eng.get_liquid_particles()))
        if k % args.rec_every == 0:
            rec["t"].append(t)
            rec["abd"].append(sc.abd_raw())
            rec["drop"].append(sc.dropper_verts().astype(np.float32))
            if args.liquid:
                lp = np.asarray(eng.get_liquid_particles())
                rec["liq"].append(np.round((lp - sc.liq_origin) * 1e4).astype(np.int16))   # 0.1 mm units
            rec["pump"].append([st.get("V_cav", np.nan), st.get("V_col", np.nan), st.get("target", np.nan),
                                st.get("conveyor", np.nan), float(st.get("submerged", 0))])
            fc = np.asarray(eng.get_vertex_contact_forces(False, "normal"))
            d0 = sc.dropper["v_off"]
            fd = fc[d0:d0 + sc.dropper["n"]]
            bulb = np.ones(len(fd), bool)
            bb = sc.bodies["beaker"]
            fb = fc[bb.v_off:bb.v_off + len(bb.rest_verts)]
            # Legacy contact-load proxy, not a hardware force limit or a finger-pair measurement:
            # includes all normal contacts on this body (including possible self contacts), excludes friction.
            # Keep the recorded array layout for compatibility with existing viewers.
            rec["force"].append([0.5 * np.linalg.norm(fd[bulb], axis=1).sum(), *fd.sum(0),
                                 0.5 * np.linalg.norm(fb, axis=1).sum(), *fb.sum(0)])
            # liquid / contact load on the glass stem: [wall-field (liquid) force xyz, barrier contact force xyz, wall torque about the stem top xyz]
            sb_ = sc.bodies["stem"]; s0, s1 = sb_.v_off, sb_.v_off + len(sb_.rest_verts)
            fst = fc[s0:s1]
            if args.liquid:
                fw = np.asarray(eng.get_liquid_wall_forces())[s0:s1]
                xs = np.asarray(eng.get_vertices())[s0:s1]
                tq = np.cross(xs - xs[np.argmax(xs[:, 1])], fw).sum(0)
            else:
                fw = np.zeros((1, 3)); tq = np.zeros(3)
            rec["stemF"].append([*fw.sum(0), *fst.sum(0), *tq, float(np.abs(fw).sum()), float(np.linalg.norm(fst, axis=1).sum())])
            ang = eng.get_revolute_current_angles_abs()
            rec["q_cmd"].append([q.get(n, 0.0) for n in arm_and_finger])
            rec["q_act"].append([ang[sc.rev_idx[n]] for n in arm_and_finger])
            rec["q_set"].append([qd.get(n, 0.0) for n in arm_and_finger])
            rec["tau"].append([float(eng.get_revolute_drive_torque(sc.rev_idx[n])) if n.startswith("right_finger") else 0.0 for n in arm_and_finger])
        if k == n_steps:
            break
        ts = time.time()
        eng.step()
        frame_times.append(time.time() - ts)
        if k % 100 == 0:
            ft = np.mean(frame_times[-100:]) if frame_times else 0
            msg = f"[step] k {k}/{n_steps} t {t:.2f} s  {ft*1e3:.0f} ms/frame  wall {(time.time()-t_wall)/60:.1f} min"
            if st:
                msg += (f"  V_cav {st['V_cav']*1e6:.2f} ml  col {st['n_col']} p (target {st['target']/sc.particle_volume:.0f})"
                        f"  conv {st['conveyor']*100:.1f} cm/s  sub {int(st['submerged'])} {st['event']}")
            if rec["force"]:
                fr = rec["force"][-1]
                msg += f"  grip {fr[0]:.1f} N  dropper Fy {fr[2]:+.2f} N  beaker grip {fr[4]:.1f} N"
            print(msg, flush=True)
        if k % args.partial_every == 0 and k > 0:
            _save(args.out, rec, names, arm_and_finger, partial=True)
    _save(args.out, rec, names, arm_and_finger, partial=False)
    print(f"[done] {n_steps} steps in {(time.time()-t_wall)/60:.1f} min", flush=True)


def _save(out, rec, names, joints, partial):
    path = osp.join(out, "rec_partial.npz" if partial else "rec.npz")
    d = dict(t=np.array(rec["t"]), abd=np.array(rec["abd"]), drop=np.array(rec["drop"]),
             pump=np.array(rec["pump"]), force=np.array(rec["force"], np.float32), stemF=np.array(rec["stemF"], np.float32), q_cmd=np.array(rec["q_cmd"], np.float32), q_act=np.array(rec["q_act"], np.float32),
             body_names=np.array(names), joint_names=np.array(joints))
    if rec["liq"]:
        d["liq_i16"] = np.array(rec["liq"])
    if rec.get("q_set"):
        d["q_set"] = np.array(rec["q_set"], np.float32)
        d["tau"] = np.array(rec["tau"], np.float32)
    np.savez(path + ".tmp.npz", **d)
    os.replace(path + ".tmp.npz", path)


if __name__ == "__main__":
    main()
