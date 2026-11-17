#!/usr/bin/env python3
"""Tactile lab-bench demo: FR3 + VBTS finger pads run the five-step protocol
(dropper -> test tube pour -> stir -> shake -> pour to waste) on StiffGIPC,
estimating each grasped object's Young's modulus from the pads.

    # full physics run (headless), writes out/lab_demo.npz
    PYTHONPATH=. python examples/lab_tactile/run_lab_demo.py --out out/lab_tactile

    # kinematics-only dry run (no engine / GPU): validates IK + timeline
    PYTHONPATH=. python examples/lab_tactile/run_lab_demo.py --dry --out out/lab_tactile_dry

Then: render_lab_video.py <npz>  and  export_lab_usd.py <npz>  (pxr python).
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
import ctypes
# [audit] LAB_PROFILE_FRAMES=a:b -> cuProfilerStart before frame a's step, cuProfilerStop before frame b's
_PROF_RANGE = tuple(int(x) for x in os.environ["LAB_PROFILE_FRAMES"].split(":")) if os.environ.get("LAB_PROFILE_FRAMES") else None

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from fr3_kin import Fr3Arm, rot_log  # noqa: E402
from lab_assets import REPO, build_all  # noqa: E402
from lab_rec import save_recording  # noqa: E402
from lab_scene import (BASE_T, FINGER_L, FINGER_R, HAND, PAD_T, PAD_Z_ON_FINGER, URDF,  # noqa: E402
                       LabScene, YoungEstimator, cradle_prop, planned_placements, T_of)
from lab_task import all_stages, minjerk, top_down_R  # noqa: E402

HOME_TCP = np.array([0.42, 0.32, 0.0])


def _seed_grid(T_end):
    """Seed postures: base joint toward the target azimuth, several elbow /
    wrist families."""
    p = T_end[:3, 3]
    az = float(np.arctan2(-p[2], p[0]))                 # world y-up: azimuth about y, URDF joint1 sign
    seeds = []
    for j1 in (az, 0.0):
        for j2, j4 in ((-0.4, -2.4), (0.3, -2.0), (0.8, -1.5), (1.2, -2.6)):
            for j6 in (1.9, 2.8, 3.6):
                for j7 in (0.785, -2.3):
                    seeds.append(np.array([j1, j2, 0.0, j4, 0.0, j6, j7]))
    return seeds


def config_quality(arm, q, T_end):
    """Smaller is better: distance into the 0.35 rad limit margin, base joint
    away from the target azimuth, wrist near the joint-6 limits."""
    lo, hi = arm.lo, arm.hi
    margin = np.minimum(q - lo, hi - q)
    pen = float(np.sum(np.clip(0.35 - margin, 0.0, None) ** 2)) * 20.0
    p = T_end[:3, 3]
    az = float(np.arctan2(-p[2], p[0]))
    pen += 0.6 * abs(float(np.arctan2(np.sin(q[0] - az), np.cos(q[0] - az))))
    return pen


def solve_end_pose(scene, T_end, q_now, quality=True):
    """Joint solution for a segment's end pose: current configuration first,
    then a grid of seed postures; picks the reachable solution with the best
    configuration quality (plus a small pull toward the current q)."""
    arm = scene.arm
    best = None
    for seed in [np.array(q_now, float), arm.HOME] + _seed_grid(T_end):
        q, ok = arm.ik(T_end, seed, iters=120)
        T = arm.fk(q)
        err = np.linalg.norm(T[:3, 3] - T_end[:3, 3])
        cost = err * 1e3 + (0.0 if ok else 1e3) + 0.4 * np.linalg.norm(q - q_now)
        if quality:
            cost += config_quality(arm, q, T_end)
        if best is None or cost < best[0]:
            best = (cost, q, ok)
    return best[1], best[2]


def plan_family(scene, key_poses, q_now):
    """One joint configuration family for a whole stage: continue the IK
    through the key poses from each seed family and keep the sequence with the
    lowest total (unreachable + joint-limit-margin) cost.  Returns the list of
    joint solutions, one per key pose."""
    arm = scene.arm
    best = None
    for seed in [np.array(q_now, float), arm.HOME] + _seed_grid(key_poses[0]):
        q, qs, cost = np.array(seed, float), [], 0.0
        for T in key_poses:
            q, ok = arm.ik(T, q, iters=150)
            err = np.linalg.norm(arm.fk(q)[:3, 3] - T[:3, 3])
            margin = np.minimum(q - arm.lo, arm.hi - q)
            cost += err * 1e3 + (0.0 if ok else 100.0) + 20.0 * float(np.sum(np.clip(0.3 - margin, 0, None) ** 2))
            qs.append(q)
        cost += 0.02 * np.linalg.norm(qs[0] - q_now)
        if best is None or cost < best[0]:
            best = (cost, qs)
    return best[1], best[0]


def track_frame(scene, T_cmd, q_prev, q_end):
    """One frame of Cartesian tracking with the segment's end configuration
    as null-space posture; falls back to a joint-space step toward it."""
    q, ok = scene.arm.ik(T_cmd, q_prev, iters=30, posture=q_end, w_post=0.12)
    if not ok and q_end is not None:
        d = np.clip(q_end - q_prev, -0.02, 0.02)
        q = scene.arm.clamp(q_prev + d)
    return q, ok


class KinematicScene:
    """Engine-free stand-in: FK-driven arm, static props, for --dry runs."""

    def __init__(self, dt):
        self.dt = dt
        self.props = build_all(verbose=False)
        self.props["cradle"] = cradle_prop(self.props)
        self.plan = planned_placements(self.props)
        self.prop_T = {k: v for k, v in self.plan.items() if not k.startswith("_")}
        self.arm = Fr3Arm(URDF, BASE_T)
        self.q = self.arm.clamp(np.zeros(7))
        self.dropper = dict(meta=self.props["dropper"].meta)
        self.gap = 0.0
        self.cmd_gap = 0.0
        self.est_list = []
        self.e_text = ""
        self.home_pose = T_of(top_down_R("z"), HOME_TCP)
        self.cmd_pose = self.arm.fk(self.q)
        self.is_dry = True
        self.est = None
        import trimesh
        self.link_mesh = {}
        for i in range(8):
            m = trimesh.load(osp.join(REPO, "Assets", "fr3", "meshes", f"link{i}.stl"), force="mesh")
            self.link_mesh[f"fr3_link{i}"] = m
        self.link_mesh[HAND] = trimesh.load(osp.join(REPO, "Assets", "fr3", "meshes", "hand.stl"), force="mesh")
        fm = trimesh.load(osp.join(REPO, "Assets", "fr3", "meshes", "finger_collision.stl"), force="mesh")
        self.link_mesh[FINGER_L] = fm
        self.link_mesh[FINGER_R] = fm

    def link_vertices_local(self):
        """{link: (N,3) vertices in the link frame} for clearance checks."""
        if not hasattr(self, "_lvl"):
            self._lvl = {n: np.asarray(m.vertices, float) for n, m in self.link_mesh.items()}
        return self._lvl

    # --- geometry queries used by the stages
    def tcp_pose(self):
        return self.arm.fk(self.q)

    def prop_pose(self, name):
        return self.prop_T[name]

    def bulb_centre(self):
        return self.plan["_bulb_centre"]

    def new_estimator(self, name, R, e_true=None):
        return YoungEstimator(name, R, e_true=e_true)

    def start_estimation(self, est):
        self.est = est

    def finish_estimation(self):
        self.est = None

    # --- per-frame
    def finger_d(self):
        return 0.5 * self.gap + PAD_T

    def step(self, T_cmd, grip, q_direct=None, q_end=None):
        if grip and grip.get("gap_mm") is not None:
            self.cmd_gap = grip["gap_mm"] * 1e-3
        if grip and grip.get("mode") == "safe":
            self.cmd_gap = 0.012
        self.gap += np.clip(self.cmd_gap - self.gap, -0.08 * self.dt, 0.08 * self.dt)
        if q_direct is not None:
            self.q = np.asarray(q_direct, float)
            return True
        q, ok = track_frame(self, T_cmd, self.q, q_end)
        self.q = q
        return ok


class Recorder:
    def __init__(self, scene, every, dry):
        self.sc, self.every, self.dry = scene, every, dry
        self.rows = []
        self.motion, self.fem, self.maps, self.map_idx = [], {}, {"left": [], "right": []}, []
        self.liquid = []
        self.n = 0
        if dry:
            lp = scene.arm.link_poses(scene.q, scene.finger_d())
            self.body_names = list(lp.keys())
            self.body_names = [n for n in self.body_names if n in scene.link_mesh] + list(scene.prop_T.keys())
            self.rest, self.faces, self.colors, self.kinds = [], [], [], []
            for n in self.body_names:
                if n in scene.link_mesh:
                    m = scene.link_mesh[n]
                    v = np.asarray(m.vertices) @ lp[n][:3, :3].T + lp[n][:3, 3]
                    self.rest.append(v.astype(np.float32)); self.faces.append(np.asarray(m.faces, np.int32))
                    self.colors.append((0.25, 0.25, 0.28) if "finger" in n else (0.92, 0.92, 0.94))
                    self.kinds.append("arm")
                else:
                    p = scene.props[n]
                    T = scene.prop_T[n]
                    v = p.verts @ T[:3, :3].T + T[:3, 3]
                    self.rest.append(v.astype(np.float32)); self.faces.append(p.faces.astype(np.int32))
                    self.colors.append(p.color); self.kinds.append("prop")
            self.link_T0 = lp
        else:
            bodies = list(scene.bodies.values())
            self.body_names = [b.name for b in bodies]
            self.rest = [b.rest_verts.astype(np.float32) for b in bodies]
            self.faces = [b.faces for b in bodies]
            self.colors = [b.color for b in bodies]
            self.kinds = [b.kind for b in bodies]
            has_dropper = scene.dropper is not None   # --no-dropper ablation builds the scene without it
            self.fem_names = ["gel_left", "gel_right"] + (["dropper"] if has_dropper else [])
            self.fem_faces = [scene.pad_asset.surface_faces.astype(np.int32)] * 2 + ([scene.dropper["faces"]] if has_dropper else [])
            for n in self.fem_names:
                self.fem[n] = []

    def record(self, t, stage, seg, grip_info, T_cmd, q, with_map, liquid_pts=None):
        if liquid_pts is not None:
            self.liquid.append(np.asarray(liquid_pts, np.float32))
        sc = self.sc
        if self.dry:
            lp = sc.arm.link_poses(sc.q, sc.finger_d())
            M = []
            for n in self.body_names:
                if n in lp:
                    M.append(lp[n] @ np.linalg.inv(self.link_T0[n]))
                else:
                    M.append(np.eye(4))
            self.motion.append(np.array(M, np.float32))
            gap, fl, fr = sc.gap * 1e3, 0.0, 0.0
            tcp = sc.arm.fk(sc.q)
        else:
            ids = np.array([b.body_id for b in sc.bodies.values()], np.int32)
            C = sc.eng.get_abd_body_transforms(ids)
            self.motion.append(np.array([b.pose_from(c) for c, b in zip(C, sc.bodies.values())], np.float32))
            self.fem["gel_left"].append(sc.gels["left"]["gel"].verts_world().astype(np.float32))
            self.fem["gel_right"].append(sc.gels["right"]["gel"].verts_world().astype(np.float32))
            if "dropper" in self.fem:
                self.fem["dropper"].append(sc.dropper_verts().astype(np.float32))
            gap, fl, fr = grip_info["gap_mm"], grip_info["fn_left"], grip_info["fn_right"]
            tcp = sc.tcp_pose()
            if with_map and sc.gripper.pads:
                for side in ("left", "right"):
                    im = sc.gripper.indentation(side)
                    self.maps[side].append(np.clip(im / 2.0e-3 * 255.0, 0, 255).astype(np.uint8))
                self.map_idx.append(self.n)
        self.rows.append(dict(t=t, stage=stage, seg=seg, gap=gap, fl=fl, fr=fr,
                              etext=sc.e_text, tcp=tcp.astype(np.float32), q=np.asarray(q, np.float32)))
        self.n += 1

    def save(self, path, meta):
        rows = self.rows
        out = dict(
            body_names=np.array(self.body_names), body_kind=np.array(self.kinds),
            body_color=np.array(self.colors, np.float32),
            body_rest=[np.asarray(a, np.float32) for a in self.rest],
            body_faces=[np.asarray(a, np.int32) for a in self.faces],
            motion=np.stack(self.motion) if self.motion else np.zeros((0, 0, 4, 4), np.float32),
            t=np.array([r["t"] for r in rows], np.float32),
            stage=np.array([r["stage"] for r in rows]), segment=np.array([r["seg"] for r in rows]),
            etext=np.array([r["etext"] for r in rows]),
            gap_mm=np.array([r["gap"] for r in rows], np.float32),
            fn_left=np.array([r["fl"] for r in rows], np.float32),
            fn_right=np.array([r["fr"] for r in rows], np.float32),
            tcp=np.stack([r["tcp"] for r in rows]) if rows else np.zeros((0, 4, 4), np.float32),
            q=np.stack([r["q"] for r in rows]) if rows else np.zeros((0, 7), np.float32),
            meta=meta,
        )
        if not self.dry:
            out["fem_names"] = np.array(self.fem_names)
            out["fem_faces"] = [np.asarray(a, np.int32) for a in self.fem_faces]
            out["fem_verts"] = [np.stack(self.fem[n]) for n in self.fem_names]
            out["pad_map_left"] = np.stack(self.maps["left"]) if self.maps["left"] else np.zeros((0, 2, 2), np.uint8)
            out["pad_map_right"] = np.stack(self.maps["right"]) if self.maps["right"] else np.zeros((0, 2, 2), np.uint8)
            out["pad_map_idx"] = np.array(self.map_idx, np.int32)
        if self.liquid:
            out["liquid_pts"] = list(self.liquid)
        save_recording(path, out)
        return path


_FRAME_LOG = bool(os.environ.get("LAB_FRAME_LOG"))


class LipFeedback:
    """During the pour, shift the commanded TCP so the carried beaker's ACTUAL lowest
    rim point (read back from the engine every frame) sits over the waste beaker's
    axis, a little inward along the pour direction. The planned schedule predicts one
    lip point for the full tilt; the real lowest point moves 2-3 cm as the tilt
    progresses and the grasp can hold the beaker off-centre, which put the stream
    34-60 mm from the axis (mouth radius 27 mm) in every liquid run so far."""

    def __init__(self, scene, carried="beaker_mix", target="beaker_waste", inward=0.010, gain=0.02, cap=0.04):
        from lab_scene import abd_world_tf
        self.scene, self.tf = scene, abd_world_tf
        self.b, self.w = scene.bodies[carried], scene.bodies[target]
        rv = self.b.rest_verts
        y0, y1 = rv[:, 1].min(), rv[:, 1].max()
        self.rim_idx = np.where(rv[:, 1] > y0 + 0.85 * (y1 - y0))[0]   # rim ring (rest frame)
        self.inward, self.gain, self.cap = inward, gain, cap
        self.delta = np.zeros(2)
        self.last = None

    def live_verts(self, body):
        C = np.asarray(self.scene.eng.get_abd_body_transforms(np.array([body.body_id], np.int32)))[0]
        W = self.tf(C) @ body.C0inv
        return body.rest_verts @ W[:3, :3].T + W[:3, 3]

    def apply(self, T_plan, active, fade=0.98):
        """Return the commanded TCP pose = planned pose + delta. While `active` (pour-tagged
        segments) delta integrates the rim error; otherwise it fades out smoothly (no jump
        when the correction stops). T_plan must be the PLANNED pose: run v8 lost the beaker
        at the pour -> pouring transition because the hold segment started from the already
        corrected commanded pose and the correction was added on top (a 27 mm jerk)."""
        if active:
            vb = self.live_verts(self.b)
            rim = vb[self.rim_idx]
            lip = rim[np.argmin(rim[:, 1])]
            vw = self.live_verts(self.w)
            axis = np.array([vw[:, 0].mean(), vw[:, 2].mean()])
            d = axis - np.array([vb[:, 0].mean(), vb[:, 2].mean()]); d = d / (np.linalg.norm(d) + 1e-9)
            target = axis - self.inward * d
            err = target - lip[[0, 2]]
            # integral action: delta accumulates until the measured error is 0. (The first
            # version filtered delta TOWARD err, whose fixed point is delta = err = e0/2: run
            # v7 settled at err = +17 mm, exactly half its uncorrected 34 mm offset.) Gain per
            # 5 ms frame; the arm answers a TCP shift within ~50 frames, so 0.02 is well damped.
            self.delta += self.gain * err
            n = np.linalg.norm(self.delta)
            if n > self.cap: self.delta *= self.cap / n
            self.last = dict(lip=lip, axis=axis, err=err, delta=self.delta.copy())
        else:
            self.delta *= fade
        T = np.array(T_plan, float); T[0, 3] += self.delta[0]; T[2, 3] += self.delta[1]
        return T


class DropperPump:
    """Drives the engine liquid pump from the hollow dropper's live geometry: tip ring centre and
    cavity-top ring centre (FEM vertices) define the tube axis; the segment tag picks the mode:
    'aspirate' = suction up the bore, 'hold' = cancel gravity on the column (carrying), 'drip' =
    pulsed squeeze (one slug of particles per pulse), anything else = off (leftovers drain)."""

    G = 9.81

    def __init__(self, scene, suck=0.05, squeeze=0.10, pulse=0.10, period=0.4):   # 10 mm of column per pulse = one drop of ~3 particles
        d = scene.dropper; dm = d["meta"]
        self.scene = scene
        self.v_off, self.tip_idx, self.top_idx = d["v_off"], np.asarray(dm["tip_idx"]), np.asarray(dm["top_idx"])
        self.radius = float(dm["tube_r_in"]) + 0.5e-3
        self.suck, self.squeeze, self.pulse, self.period = suck, squeeze, pulse, period   # conveyor speeds (m/s), pulse timing (s)
        self.last = (0.0, 0.0, 0.0, 0.0)

    def apply(self, tag, t_in_seg):
        v = self.scene.dropper_verts()
        tip, top = v[self.tip_idx].mean(0), v[self.top_idx].mean(0)
        G = self.G
        # (accel, conveyor, follow, track): gravity cancelled + velocity tracking carries the column with
        # the tube; follow stays 0 (a position follow on top doubles the motion and spits the column out)
        if tag == "aspirate": mode = (G, self.suck, 0.0, 0.5)                              # hold + draw in
        elif tag == "hold": mode = (G, 0.0, 0.0, 0.5)                                      # sealed column travels with the tube
        elif tag == "drip": mode = (G, -self.squeeze, 0.0, 0.5) if (t_in_seg % self.period) < self.pulse else (G, 0.0, 0.0, 0.5)
        else: mode = (0.0, 0.0, 0.0, 0.0)
        # bore_radius: the intake only draws in particles under the opening (bore - dhat), never under the
        # glass end face (at 5 cm/s the old intake pushed 15-22 particles into the tube wall)
        self.scene.eng.set_liquid_pump(tip, top, self.radius, *mode, bore_radius=float(self.scene.dropper["meta"]["tube_r_in"]))
        self.last = mode


LIQ_JIT = float(os.environ.get("LAB_LIQ_JITTER", "0.3"))   # seeding jitter in units of the particle spacing (0 = perfect lattice)


def liquid_wall_containers(scene, dx):
    """Axisymmetric cavity descriptors (vessel design frame, revolution axis +y) for the engine
    liquid's grid-level wall boundary condition (Engine.set_liquid_containers)."""
    want = os.environ.get("LAB_LIQ_WALLBC_NAMES", "flask,beaker_mix,beaker_waste,test_tube,test_tube2").split(",")
    names = [n for n in ("flask", "beaker_mix", "beaker_waste", "test_tube", "test_tube2") if n in scene.bodies and n in want]
    if not names:
        return []
    ids = np.array([scene.bodies[n].body_id for n in names], np.int32)
    C = np.asarray(scene.eng.get_abd_body_transforms(ids))        # raw: [:3,:3] = A^T, [:3,3] = p
    out = []
    for n, Ck in zip(names, C):
        A0, p0 = Ck[:3, :3].T, Ck[:3, 3]
        T = np.asarray(scene.prop_T[n], np.float64); RP, tP = T[:3, :3], T[:3, 3]
        Rd = RP.T @ A0; td = RP.T @ (p0 - tP)                    # X_design = Rd xbar + td
        meta = scene.props[n].meta
        if n == "flask":
            t = 2.5e-3
            prof = [(0.003, 0.042 - t), (0.012, 0.042 - t), (0.105, 0.013 - t)]
            y_top, wall = 0.100, t
        elif n.startswith("beaker"):
            _, r_in, y_floor, y_rim = scene.vessel_cavity(n)       # load pose, world (upright vessel)
            yf, yr = y_floor - tP[1], y_rim - tP[1]
            prof = [(yf, r_in), (yr, r_in)]
            y_top, wall = yr - 0.006, 0.003                       # leave the rim / spout free for pouring
        else:                                                     # test tube: hemispherical bottom + cylinder
            ro, ri, L = float(meta["r_out"]), float(meta["r_in"]), float(meta["length"])
            ys = (ro - ri) + np.linspace(0.0, ri, 8)
            prof = [(y, float(np.sqrt(max(ri * ri - (ro - y) ** 2, 0.0)))) for y in ys] + [(L, ri)]
            y_top, wall = L - 0.005, ro - ri
        out.append(dict(name=n, body=int(scene.bodies[n].body_id), Rd=Rd, td=td, profile=np.array(prof, np.float64),
                        y_top=float(y_top), band=float(wall + 1.3 * dx), slip=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="kinematics only, no engine")
    ap.add_argument("--out", default=osp.join(REPO, "out", "lab_tactile"))
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--rec-every", type=int, default=8, help="record every N physics frames")
    ap.add_argument("--map-every", type=int, default=2, help="pad maps every N records")
    ap.add_argument("--stages", default="0,1,2,3,4", help="which stages to run")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--precond", type=int, default=0)
    ap.add_argument("--no-tactile-gpu", action="store_true")
    ap.add_argument("--time-scale", type=float, default=1.0, help="multiply all segment durations")
    ap.add_argument("--liquid", default="", help="vessels to fill with water: comma list of name[:fill], e.g. 'beaker_mix' or "
                                                   "'flask:0.35,test_tube:0.6,beaker_mix:0.4' (fill = fraction of the cavity height; "
                                                   "the flask's fill is a fraction of its 105 mm conical body)")
    ap.add_argument("--liquid-dx", type=float, default=0.005)
    ap.add_argument("--liquid-fill", type=float, default=0.9, help="fill height as a fraction of the cavity (0.9: pours from ~50 deg once the boundary thickening is accounted for)")
    ap.add_argument("--liquid-substep", type=float, default=1e-4)
    ap.add_argument("--liquid-kappa", type=float, default=1e5)
    ap.add_argument("--liquid-ppc", type=float, default=8)
    ap.add_argument("--liquid-device", default="cuda:0")
    ap.add_argument("--liquid-wall-bc", type=int, default=int(os.environ.get("LAB_LIQ_WALLBC", "0")),
                    help="engine liquid: grid-level wall boundary condition on the vessels (1 = on; slip, weightless heavy solid-side nodes)")
    ap.add_argument("--liquid-wall-mass", type=float, default=1e3, help="inertia of the constrained solid-side grid nodes (x rho dx^3)")
    ap.add_argument("--liquid-apic", type=float, default=1.0, help="1: APIC transfers (lively jet; needs the push-apart floor against compaction), 0: PIC (quiet but honey-like); engine liquid: fractions blend the APIC affine term (damps the rest creep)")
    ap.add_argument("--liquid-push", type=float, default=0.9, help="particle push-apart floor in units of the seeding spacing (0 = off)")
    ap.add_argument("--liquid-stick", type=float, default=0.0015, help="engine liquid: particles within this height (m) of the table freeze (spill puddle); 0 = off")
    ap.add_argument("--tube-lip-inward", type=float, default=0.020,
                    help="test-tube pour: target the tube's lowest rim point this far short of the beaker axis (m); the tube empties in one gush near "
                         "horizontal and the stream carries horizontal velocity (0.010 put the landing at the beaker's far rim, 61 %% in)")
    ap.add_argument("--liquid-domain", type=float, nargs=6, default=(0.33, -0.02, -0.36, 0.44, 0.42, 0.72),
                    help="origin xyz + extent xyz of the liquid grid (m)")
    ap.add_argument("--liquid-engine", action="store_true",
                    help="solve the liquid INSIDE StiffGIPC (implicit MPM Lite + IPC monolithic, two-way) instead of the Warp one-way coupler")
    ap.add_argument("--liquid-jmode", type=int, default=3, help="engine liquid J state: 3 hybrid (default), 4 cell density x geometric fill (experimental: no rest creep, but clumps/stalls in impacts), 2 cell density, 0 tracked")
    ap.add_argument("--no-lip-feedback", action="store_true", help="disable the live lip-over-waste correction during the pour")
    ap.add_argument("--no-dropper", action="store_true",
                    help="diagnostic ablation: build the scene without the FEM dropper (stages that do not use it)")
    ap.add_argument("--liquid-mass-floor", type=float, default=0.05,
                    help="engine liquid: grid-node mass floor in units of rho*dx^3 (diagnostic ablations only)")
    ap.add_argument("--liquid-node-pool", type=int, default=24000, help="engine liquid grid-node pool (fixed at finalize; the pour needs ~4x the seed nodes, splashes more)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dt = args.dt
    if args.dry:
        scene = KinematicScene(dt)
    else:
        def _seed_engine_liquid(sc):
            """Particles in every requested vessel cavity, registered with the engine before finalize."""
            dx = args.liquid_dx
            n_sub = max(1, int(round(args.liquid_ppc ** (1.0 / 3.0))))
            hs = dx / n_sub
            m = 0.6 * dx
            rng = np.random.default_rng(0)
            all_pts = []
            for item in args.liquid.split(","):
                name, _, fill = item.strip().partition(":")
                fill = float(fill) if fill else args.liquid_fill
                if name == "flask":
                    # conical flask (lab_assets.build_flask): floor 3 mm above the base, body radius 39.5 mm at
                    # y = 12 mm tapering to 10.5 mm at y = 105 mm (inner wall); the dropper hangs in the neck
                    T = sc.prop_T["flask"]; base = float(T[1, 3]); cx, cz = float(T[0, 3]), float(T[2, 3])   # planned load pose (engine not finalized yet)
                    y_floor = base + 0.003
                    h = y_floor + fill * 0.105
                    def r_in_at(y):
                        return np.where(y < base + 0.012, 0.0395, 0.0395 + (0.0105 - 0.0395) * (y - base - 0.012) / (0.105 - 0.012))
                    ax = np.arange(cx - 0.0395 + m, cx + 0.0395 - m, hs); ay = np.arange(y_floor + m, h, hs); az = np.arange(cz - 0.0395 + m, cz + 0.0395 - m, hs)
                    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
                    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
                    pts = pts + rng.uniform(-LIQ_JIT * hs, LIQ_JIT * hs, pts.shape)
                    pts = pts[np.hypot(pts[:, 0] - cx, pts[:, 2] - cz) < r_in_at(pts[:, 1]) - m]
                    # keep the dropper bore (and a margin around the dipped tube) free
                    Td = sc.prop_T["dropper"]; dm = sc.dropper["meta"]
                    tube_r = dm.get("tube_r", 2.5e-3)
                    near_tube = (np.hypot(pts[:, 0] - Td[0, 3], pts[:, 2] - Td[2, 3]) < tube_r + 1.5e-3) & (pts[:, 1] > Td[1, 3] - 2e-3)
                    pts = pts[~near_tube]
                    print(f"[liquid-engine] flask: floor y {y_floor*1e3:.1f} mm, fill to {h*1e3:.1f} mm (dropper tip at y {Td[1,3]*1e3:.1f} mm) -> {len(pts)} particles", flush=True)
                else:
                    c, r_in, y_floor, y_rim = sc.vessel_cavity(name)
                    h = y_floor + fill * (y_rim - y_floor)
                    m = min(0.6 * dx, max(0.25 * r_in, 1.5e-3))     # narrow vessels (test tube r 6.8 mm) keep a 1.7 mm wall margin, not 3 mm
                    ax = np.arange(c[0] - r_in + m, c[0] + r_in - m, hs)
                    ay = np.arange(y_floor + m, h, hs)
                    az = np.arange(c[1] - r_in + m, c[1] + r_in - m, hs)
                    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
                    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
                    pts = pts + rng.uniform(-LIQ_JIT * hs, LIQ_JIT * hs, pts.shape)
                    pts = pts[np.hypot(pts[:, 0] - c[0], pts[:, 2] - c[1]) < r_in - m]
                    if name.startswith("test_tube"):
                        # hemispherical bottom (lab_assets.build_test_tube: inner sphere of radius r_in centred
                        # r_out above the base): the cylinder clip alone left bottom-layer particles inside the glass
                        T = sc.prop_T[name]; dm = sc.props[name].meta
                        cs = T[:3, 3] + T[:3, :3] @ np.array([0.0, float(dm["r_out"]), 0.0])
                        below = np.minimum(pts[:, 1] - cs[1], 0.0)
                        pts = pts[np.sqrt(np.hypot(pts[:, 0] - cs[0], pts[:, 2] - cs[2]) ** 2 + below ** 2) < r_in - m]
                    if name == "beaker_mix":
                        sc.liquid_seed_info = dict(n=len(pts), r_in=r_in, y_floor=y_floor, y_rim=y_rim, h=h)
                    print(f"[liquid-engine] {name}: cavity r_in {r_in*1e3:.1f} mm floor y {y_floor*1e3:.1f} mm rim y {y_rim*1e3:.1f} mm, "
                          f"fill to {h*1e3:.1f} mm -> {len(pts)} particles", flush=True)
                all_pts.append(pts)
            pts = np.vstack(all_pts)
            o = np.array(args.liquid_domain[:3]); ext = np.array(args.liquid_domain[3:])
            sc.eng.add_liquid(pts, dx, o, o + ext, rho=1000.0, kappa=args.liquid_kappa,
                              particle_volume=hs ** 3, node_pool=args.liquid_node_pool,
                              push_apart=args.liquid_push, j_mode=args.liquid_jmode, stick_h=args.liquid_stick,
                              mass_floor=args.liquid_mass_floor, apic=args.liquid_apic > 0.0, apic_ratio=float(args.liquid_apic))
            print(f"[liquid-engine] total {len(pts)} particles (dx {dx*1e3:.1f} mm, spacing {hs*1e3:.2f} mm), domain {o} + {ext}", flush=True)

        scene = LabScene(dt=dt, precond=args.precond, tactile_gpu=not args.no_tactile_gpu,
                         with_dropper=not args.no_dropper,
                         pre_finalize=_seed_engine_liquid if (args.liquid and args.liquid_engine) else None,
                         extra_cfg=({"energy_rel_tol": float(os.environ.get("LAB_ENERGY_REL_TOL", "0")),
                                     # bound the cost of the rim-crossing frames (thin sheet over the lip: Newton chatters to the cap).
                                     # Only the Newton cap is safe: pcg_tol 1e-4 / ratio 0.05 (run v4) wrecked the arm (TCP error 0.5 m, 90 N pads).
                                     "newton_iter_cap": int(os.environ.get("LAB_NEWTON_CAP", "80")),
                                     "pcg_max_iter_ratio": float(os.environ.get("LAB_PCG_RATIO", "0.3")),
                                     "pcg_tol": float(os.environ.get("LAB_PCG_TOL", "1e-6"))} if (args.liquid and args.liquid_engine) else None))
        scene.home_pose = T_of(top_down_R("z"), HOME_TCP)
        scene.cmd_pose = scene.arm.fk(scene.q)
        scene.is_dry = False
        if args.liquid and args.liquid_engine and args.liquid_wall_bc:
            conts = liquid_wall_containers(scene, args.liquid_dx)
            scene.eng.set_liquid_containers(conts, wall_mass=args.liquid_wall_mass, ground=True)
            print(f"[liquid-engine] wall BC on {[c['name'] for c in conts]}", flush=True)
        scene.est_list, scene.e_text, scene.est = [], "", None
        scene.new_estimator = lambda name, R, e_true=None: YoungEstimator(name, R, e_true=e_true)

        def _start(est):
            scene.est = est
            scene.e_text = est.text()
            for side in ("left", "right"):
                scene.gripper.rest_depth.pop(side, None)   # fresh rest depth at approach

        def _finish():
            if scene.est is not None:
                r = scene.est.fit()
                scene.est_list.append(r)
                scene.e_text = scene.est.text()
                print(f"[E] {scene.e_text}")
            scene.est = None
        scene.start_estimation, scene.finish_estimation = _start, _finish

    liquid, coupler = None, None
    if args.liquid and not args.dry and not args.liquid_engine:
        from stiff_physics.liquid import LiquidMPM, EngineLiquidCoupler
        o = np.array(args.liquid_domain[:3]); ext = np.array(args.liquid_domain[3:])
        liquid = LiquidMPM(o, ext, args.liquid_dx, device=args.liquid_device, gravity=(0, -9.81, 0),
                           kappa=args.liquid_kappa, ppc=args.liquid_ppc, substep_dt=args.liquid_substep,
                           verbose=True, apic=bool(args.liquid_apic))
        liquid.push_apart = float(args.liquid_push)
        liquid.add_halfspace([0, 0.0, 0], [0, 1, 0])                     # table top
        c, r_in, y_floor, y_rim = scene.vessel_cavity(args.liquid)
        h = y_floor + args.liquid_fill * (y_rim - y_floor)
        m = 0.6 * args.liquid_dx    # the boundary mesh is thickened OUTWARD only, the cavity stays exact
        n_seed = liquid.seed_box([c[0] - r_in + m, y_floor + m, c[1] - r_in + m], [c[0] + r_in - m, h, c[1] + r_in - m], jitter=0.3)
        # keep only the cylinder
        xp = liquid.solver.ptc_x.numpy() + liquid.origin
        keep = np.hypot(xp[:, 0] - c[0], xp[:, 2] - c[1]) < r_in - m
        if not keep.all():
            import warp as wp
            for name in ("ptc_x", "ptc_v", "ptc_m", "ptc_k", "ptc_vol0", "ptc_F", "ptc_G", "ptc_dlogJ"):
                a = getattr(liquid.solver, name)
                setattr(liquid.solver, name, wp.array(a.numpy()[keep], dtype=a.dtype, device=liquid.device))
            liquid.solver.n_ptc = int(keep.sum()); liquid.solver.n_ptc_wp.fill_(liquid.solver.n_ptc)
        faces = scene.surface_faces_global()
        coupler = EngineLiquidCoupler(scene.eng, liquid, faces, dt)
        print(f"[liquid] {args.liquid}: cavity r_in {r_in*1e3:.1f} mm floor y {y_floor*1e3:.1f} mm rim y {y_rim*1e3:.1f} mm, "
              f"fill to {h*1e3:.1f} mm -> {liquid.n_particles} particles; grid {liquid.grid_size} dx {args.liquid_dx*1e3:.1f} mm; "
              f"boundary mesh {len(faces)} tris")

    lipfb = LipFeedback(scene) if (args.liquid and args.liquid_engine and not args.dry and not args.no_lip_feedback) else None
    # one live rim feedback per pouring stage: stage 4 (mixing beaker -> waste beaker) and stage 2
    # (test tube -> mixing beaker; the planned mouth position put 54 % of the tube's water on the table)
    lip_fbs = []
    if lipfb is not None:
        lip_fbs.append((lipfb, "4", "beaker"))
        if "test_tube" in scene.bodies:
            lip_fbs.append((LipFeedback(scene, carried="test_tube", target="beaker_mix", inward=args.tube_lip_inward), "2", "tube"))
    pump = DropperPump(scene) if (args.liquid and args.liquid_engine and not args.dry and scene.dropper is not None
                                  and scene.dropper["meta"].get("hollow")) else None
    if pump is not None:
        print(f"[pump] hollow dropper: bore r {scene.dropper['meta']['tube_r_in']*1e3:.1f} mm, tip ring {len(pump.tip_idx)} / top ring {len(pump.top_idx)} vertices", flush=True)
    rec = Recorder(scene, args.rec_every, args.dry)
    stages = all_stages()
    wanted = [int(s) for s in args.stages.split(",")]
    frame, t = 0, 0.0
    n_ik_fail = 0
    t_wall0 = time.time()
    stop = False
    for si, stage in enumerate(stages):
        if si not in wanted or stop:
            continue
        tl = stage.build(scene)
        print(f"\n===== stage {stage.name}: {len(tl.segs)} segments, "
              f"{sum(s.duration for s in tl.segs)*args.time_scale:.1f} s")
        for seg in tl.segs:
            if seg.on_start:
                seg.on_start(scene)
            if seg.grip and not args.dry:
                scene.gripper.command(**seg.grip)
            n_fr = max(1, int(round(seg.duration * args.time_scale / dt)))
            q_end = None
            hint = getattr(seg, "q_hint", None)
            if hint is not None:                              # stage family solution for this end pose
                q_end, ok_end = scene.arm.ik(seg.pose_fn(1.0), hint, iters=150)
            elif seg.q_fn is None:
                q_end, ok_end = solve_end_pose(scene, seg.pose_fn(1.0), scene.q)
            else:
                ok_end = True
            if not ok_end:
                print(f"  [ik] WARN: end pose of '{seg.label}' not reachable")
            if getattr(seg, "joint_space", False):           # approach in joint space
                q0 = np.array(scene.q, float)
                q1 = q_end
                seg.q_fn = (lambda s, q0=q0, q1=q1: q0 + minjerk(s) * (q1 - q0))
                q_end = None
            t_seg = time.time()
            nt_seg = 0 if args.dry else scene.eng.native.get_total_newton_iters()
            ev = set()
            for k in range(n_fr):
                s = (k + 1) / n_fr
                T_plan = seg.pose_fn(s)
                T_cmd = T_plan
                if not args.dry:
                    tag_now = getattr(seg, "tag", None)
                    for fb, prefix, fb_name in lip_fbs:
                        on = (tag_now == "pour" and stage.name.startswith(prefix))
                        T_cmd = fb.apply(T_cmd, active=on)
                        if on and _FRAME_LOG and frame % 50 == 0 and fb.last is not None:
                            L = fb.last
                            print(f"[lip:{fb_name}] frame {frame} lip ({L['lip'][0]:.3f},{L['lip'][1]:.3f},{L['lip'][2]:.3f}) axis ({L['axis'][0]:.3f},{L['axis'][1]:.3f}) err {L['err'][0]*1e3:+.0f},{L['err'][1]*1e3:+.0f} mm delta {fb.delta[0]*1e3:+.0f},{fb.delta[1]*1e3:+.0f} mm", flush=True)
                if pump is not None:
                    pump.apply(getattr(seg, "tag", ""), k * dt)
                    if _FRAME_LOG and frame % 50 == 0 and pump.last != (0.0, 0.0, 0.0, 0.0):
                        print(f"[pump] frame {frame} {seg.label}: acc {pump.last[0]/9.81:+.1f} g conveyor {pump.last[1]*100:+.0f} cm/s, {scene.eng.count_liquid_in_pump()} particles in the bore region", flush=True)
                # holds / dyn segments start from the PLANNED pose; the lip delta lives in lipfb only
                scene.cmd_pose = T_plan
                if args.dry:
                    ok = scene.step(T_cmd, seg.grip, q_direct=seg.q_fn(s) if seg.q_fn else None,
                                    q_end=q_end)
                    ginfo = None
                else:
                    if seg.q_fn is not None:
                        q, ok = np.asarray(seg.q_fn(s), float), True
                    else:
                        q, ok = track_frame(scene, T_cmd, scene.q, q_end)
                    scene.q = q
                    scene.drive_arm(q)
                    ginfo = scene.gripper.update()
                    if _PROF_RANGE is not None and frame in _PROF_RANGE:   # nsys --capture-range=cudaProfilerApi
                        _cuda_drv = ctypes.CDLL("libcuda.so.1")
                        (_cuda_drv.cuProfilerStart if frame == _PROF_RANGE[0] else _cuda_drv.cuProfilerStop)()
                        print(f"[profiler] {'start' if frame == _PROF_RANGE[0] else 'stop'} at frame {frame}", flush=True)
                    _t_step = time.time()
                    scene.eng.step()
                    if _FRAME_LOG and frame % 10 == 0:
                        _nt = scene.eng._engine.get_total_newton_iters(); _pc = scene.eng._engine.get_total_pcg_iters()
                        print(f"[frame] {frame:6d} step {1000*(time.time()-_t_step):7.1f} ms  newton total {_nt}  pcg total {_pc:.0f}  ({seg.label})", flush=True)
                    if coupler is not None:
                        coupler.step()
                    if ginfo["event"]:
                        ev.add(ginfo["event"])
                    if scene.est is not None:
                        dl = float(scene.gripper.indentation("left").max())
                        dr = float(scene.gripper.indentation("right").max())
                        scene.est.add(scene.gripper.gap_m(), ginfo["fn_left"], ginfo["fn_right"], dl, dr)
                if not ok:
                    n_ik_fail += 1
                if frame % args.rec_every == 0:
                    rec.record(t, stage.name, seg.label, ginfo, T_cmd, scene.q,
                               with_map=(rec.n % args.map_every == 0),
                               liquid_pts=(liquid.positions() if liquid is not None else
                                           (scene.eng.get_liquid_particles() if (args.liquid and args.liquid_engine) else None)))
                frame += 1
                t += dt
                if not args.dry and frame % 2400 == 0:      # checkpoint every 12 s of sim
                    rec.save(osp.join(args.out, "lab_demo.npz"),
                             dict(dt=dt, rec_every=args.rec_every, dry=False, estimates=scene.est_list, liquid_dx=args.liquid_dx))
                if args.max_frames and frame >= args.max_frames:
                    stop = True
                    break
            if seg.on_end:
                seg.on_end(scene)
            ms = (time.time() - t_seg) / max(n_fr, 1) * 1e3
            Tn = scene.arm.fk(scene.q) if args.dry else scene.tcp_pose()
            err = np.linalg.norm(Tn[:3, 3] - T_cmd[:3, 3]) * 1e3
            gtxt = "" if args.dry else (f" nt {(scene.eng.native.get_total_newton_iters() - nt_seg) / max(n_fr, 1):4.1f}"
                                        f" gap {ginfo['gap_mm']:5.1f} mm  Fn L/R {ginfo['fn_left']:.2f}/{ginfo['fn_right']:.2f} N"
                                        + (f"  ev={sorted(ev)}" if ev else ""))
            print(f"  [{t:6.1f}s] {seg.label:26s} {n_fr:4d} fr  {ms:6.1f} ms/fr  tcp err {err:5.1f} mm{gtxt}")
            if stop:
                break
        if not args.dry:
            rec.save(osp.join(args.out, "lab_demo.npz"), dict(dt=dt, rec_every=args.rec_every, dry=False,
                                                              estimates=scene.est_list))
    meta = dict(dt=dt, rec_every=args.rec_every, dry=args.dry, ik_fail=n_ik_fail, frames=frame,
                estimates=[] if args.dry else scene.est_list, wall_s=time.time() - t_wall0,
                liquid_dx=args.liquid_dx, liquid=args.liquid, liquid_apic=args.liquid_apic, liquid_push=args.liquid_push,
                liquid_particles=(liquid.n_particles if liquid is not None else
                                  (scene.eng.get_liquid_particle_count() if (args.liquid and args.liquid_engine and not args.dry) else 0)),
                liquid_engine=bool(args.liquid_engine), liquid_jmode=args.liquid_jmode)
    path = rec.save(osp.join(args.out, "lab_demo.npz"), meta)
    print(f"\n[done] {frame} frames ({t:.1f} s sim) in {meta['wall_s']/60:.1f} min, IK fails {n_ik_fail}, -> {path}")
    if not args.dry:
        for r in scene.est_list:
            print("  ", json.dumps(r, default=float))


if __name__ == "__main__":
    main()
