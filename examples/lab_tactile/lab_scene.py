"""Lab-bench scene for the tactile chemistry demo: FR3 arm with VBTS finger
pads, glassware, and a FEM rubber-bulb dropper.

Layout (world, metres, Y-up, table top = ground plane y=0, arm base at the
origin, URDF Z-up mapped to world Y-up):

    flask + dropper   (0.42, 0, +0.28)
    mixing beaker     (0.50, 0,  0.00)
    waste beaker      (0.60, 0, +0.16)
    test-tube rack    (0.40, 0, -0.28)   tube in the centre slot, prop tube at +x
    drip tray + rod   (0.56, 0, -0.14)   rod on two cradles, along x

Load order follows the engine rule: ABD bodies (arm, props) first, FEM last.
"""

from __future__ import annotations

import os
import os.path as osp
import time

import numpy as np

from fr3_kin import Fr3Arm, axis_angle_R
from lab_assets import ASSET_DIR, REPO, build_all
from lab_pad import PAD_SIZE, load_pad_asset

# Arm collision meshes. The original FR3 STL hulls have 100-280 mm edges; one 186 mm edge of
# hand.stl made the edge-edge BVH query serial once the fingers close (2 -> 13 ms per query, most
# of the line-search time while holding). The _sub50 variant bisects every edge longer than 50 mm
# in place (same surface, same mass/inertia; tools/subdivide_long_edges.py): stage 4 -35 %.
URDF = osp.join(REPO, "Assets", "fr3", "fr3_franka_hand_sub50.urdf")
if os.environ.get("LAB_URDF"):   # alternative arm URDF, e.g. LAB_URDF=fr3_franka_hand.urdf (original meshes)
    URDF = os.environ["LAB_URDF"] if osp.isabs(os.environ["LAB_URDF"]) else osp.join(REPO, "Assets", "fr3", os.environ["LAB_URDF"])
# kappa's mass scale is total lumped mass / ALL vertices (massless ABD surface vertices included),
# so the refined arm alone would lower it x0.806 and soften every contact (beaker E* 61.3 -> 54.7
# MPa, grip slip and PCG counts change). Pinned to the value the scene had with the original
# meshes: kappa identical, physics unchanged (verified: Newton counts, slip, E*, finger force).
KAPPA_MEAN_MASS = 1.2292327807099668e-06
ASSETS_ROOT = osp.join(REPO, "Assets")

FINGER_L, FINGER_R, HAND = "fr3_leftfinger", "fr3_rightfinger", "fr3_hand"
PAD_Z_ON_FINGER = 0.044          # pad centre along the finger (TCP is at 0.045)
PAD_T = PAD_SIZE[2]

# planned object placements (local-origin world positions)
PLACE = {
    "flask": np.array([0.42, 0.0, 0.28]),
    "beaker_mix": np.array([0.50, 0.0, 0.00]),
    "beaker_waste": np.array([0.60, 0.0, 0.16]),
    "rack": np.array([0.40, 0.0, -0.28]),
    "tray": np.array([0.56, 0.0, -0.14]),
}
ROD_STAND_H = 0.040              # rod stand (pocket) height on the tray
ROD_STAND_OPEN = 0.012           # pocket opening (rod dia 6 mm -> 3 mm play per side)
GROUND_GAP = 1.0e-3              # free props start this far above the table


def Rx(deg):
    return axis_angle_R([1, 0, 0], np.radians(deg))


def Ry(deg):
    return axis_angle_R([0, 1, 0], np.radians(deg))


def Rz(deg):
    return axis_angle_R([0, 0, 1], np.radians(deg))


def T_of(R=None, p=(0, 0, 0)):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = p
    return T


BASE_T = T_of(Rx(-90.0))         # URDF z-up -> world y-up


def _load_urdf_capture_joint_indices(eng, urdf_path, transform, *args):
    """load_urdf while capturing the importer log; returns
    {joint_name: (joint_constraint_index, "Fixed"|"Revolute")} (the index that
    set_fixed_joint_strength / joint-constraint APIs use)."""
    import os
    import re
    import sys
    import tempfile
    eng.native.set_log_level(1)
    log_path = tempfile.mktemp(prefix="lab_urdf_load_", suffix=".txt")
    saved_fd1 = os.dup(1)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.close(fd)
    try:
        eng.native.load_urdf(urdf_path, transform, *args)
        sys.stdout.flush()
        try:
            os.fsync(1)
        except OSError:
            pass
    finally:
        os.dup2(saved_fd1, 1)
        os.close(saved_fd1)
    try:
        log = open(log_path).read()
    finally:
        try:
            os.unlink(log_path)
        except OSError:
            pass
    out = {}
    for idx, m in enumerate(re.finditer(
            r"\[UrdfSceneImporter\] Joint constraint '([^']+)' \((Fixed|Revolute)\):", log)):
        out[m.group(1)] = (idx, m.group(2))
    return out


def planned_placements(props) -> dict:
    """Load transforms of every prop (and the dropper) from the PLACE table."""
    P = PLACE
    rk = props["rack"].meta
    out = {
        "rack": T_of(p=P["rack"]),
        "tray": T_of(p=P["tray"]),
        "cradle": np.eye(4),
        "flask": T_of(p=P["flask"] + [0, GROUND_GAP, 0]),
        "beaker_mix": T_of(p=P["beaker_mix"] + [0, GROUND_GAP, 0]),
        "beaker_waste": T_of(p=P["beaker_waste"] + [0, GROUND_GAP, 0]),
        "test_tube": T_of(p=P["rack"] + [rk["slot_x"][1], rk["base_top"] + GROUND_GAP, 0]),
        "test_tube2": T_of(p=P["rack"] + [rk["slot_x"][2], rk["base_top"] + GROUND_GAP, 0]),
        "rod": T_of(Rz(90.0), P["tray"] + [0.0, 0.004 + GROUND_GAP + props["rod"].meta["length"] / 2, 0.0]),
    }
    fm, dm = props["flask"].meta, props["dropper"].meta
    r_neck = fm["neck_r_in"]
    y_c = P["flask"][1] + GROUND_GAP + fm["height"] + np.sqrt(max(dm["bulb_r"] ** 2 - r_neck ** 2, 0)) + 1.5e-3
    out["dropper"] = T_of(p=[P["flask"][0], y_c - dm["bulb_cy"], P["flask"][2]])
    out["_bulb_centre"] = np.array([P["flask"][0], y_c, P["flask"][2]])
    return out


def cradle_prop(props):
    """Rod stand on the drip tray: four walls forming a 12 x 12 mm square pocket
    (40 mm tall) in which the glass rod stands upright."""
    import trimesh
    from lab_assets import Prop, box
    c = PLACE["tray"] + [0.0, 0.004, 0.0]
    h, a, t = ROD_STAND_H, ROD_STAND_OPEN / 2, 0.005
    parts = [box((2 * a + 2 * t, h, t), c + [0, h / 2, -(a + t / 2)]),
             box((2 * a + 2 * t, h, t), c + [0, h / 2, +(a + t / 2)]),
             box((t, h, 2 * a), c + [-(a + t / 2), h / 2, 0]),
             box((t, h, 2 * a), c + [+(a + t / 2), h / 2, 0])]
    vs, fs, off = [], [], 0
    for m in parts:
        vs.append(np.asarray(m.vertices)); fs.append(np.asarray(m.faces, np.int64) + off); off += len(m.vertices)
    return Prop("cradle", np.vstack(vs), np.vstack(fs).astype(np.int32), "ABD", 1200.0,
                color=(0.85, 0.85, 0.87))


def abd_world_tf(C: np.ndarray) -> np.ndarray:
    """get_abd_body_transforms() stores the affine block TRANSPOSED
    (sim_engine.cu: ``mat[:3,:3] = A^T``); the physical map of the body's
    reference coordinates is x = R xbar + p with R = C[:3,:3].T.  Verified
    on the pod: without the transpose every single-joint move showed a
    hand orientation error of exactly 2*theta."""
    T = np.array(C, np.float64)
    T[:3, :3] = C[:3, :3].T
    return T


class Body:
    """One ABD body: load record + rest world surface + canonical frame."""

    def __init__(self, rec, name, kind="prop", color=(0.7, 0.7, 0.7)):
        self.rec, self.name, self.kind, self.color = rec, name, kind, color
        self.body_id = int(rec.body_offset)
        self.v_off = int(rec.vertex_offset)
        self.rest_verts = None       # world positions after finalize (static mesh)
        self.faces = None            # local-indexed surface triangles
        self.C0inv = None

    def pose_from(self, C) -> np.ndarray:
        """Rigid motion (world) from a raw get_abd_body_transforms() matrix."""
        return abd_world_tf(C) @ self.C0inv

    def pose(self, eng) -> np.ndarray:
        """Current rigid motion (world) applied to rest_verts."""
        C = eng.get_abd_body_transforms(np.array([self.body_id], np.int32))[0]
        return self.pose_from(C)


class FrankaTactileGripper:
    """Two URDF prismatic finger joints + tactile pads, SDK-style control.

    Gap = pad-face gap in mm (0 = coat surfaces touching).  Each finger joint
    distance d is measured from the load pose (fingers closed), so
    gap = 2*d - 2*pad_t.
    """

    def __init__(self, eng, j_left, j_right, dt, stiffness=25.0):
        self.eng, self.jl, self.jr, self.dt = eng, j_left, j_right, dt
        self.stiffness = stiffness
        self.mode = "pos"
        self.cmd_gap = 0.070
        self.vmax = 0.08
        self.fmax = 20.0
        self.grip_force = 3.0
        self.ki = 1.2e-5
        self._target_d = 0.0          # commanded joint distance (m), both sides
        self.pads = {}                # side -> dict(gel, tf0, gpu)
        self.finger_body = {}         # side -> Body
        self.rest_depth = {}
        self.last = dict(fn_left=0.0, fn_right=0.0)

    # ------------------------------------------------------------ geometry
    def gap_m(self) -> float:
        d = 0.5 * (self.eng.get_prismatic_current_distance(self.jl)
                   + self.eng.get_prismatic_current_distance(self.jr))
        return 2.0 * d - 2.0 * PAD_T

    def gap_mm(self) -> float:
        return self.gap_m() * 1e3

    def pad_tf(self, side) -> np.ndarray:
        b = self.finger_body[side]
        return b.pose(self.eng) @ self.pads[side]["tf0"]

    def pad_forces(self) -> dict:
        """Per-pad (normal_N, local_force_vec) from ONE contact-force fetch."""
        F = np.asarray(self.eng.get_vertex_contact_forces(False, "total"))
        out = {}
        for side, pad in self.pads.items():
            W = self.pad_tf(side)
            Fl = F[pad["gel"].slice] @ W[:3, :3]
            out[side] = (float(np.clip(Fl[:, 2], 0.0, None).sum()), Fl.sum(0))
        return out

    def pad_normal_force(self, side) -> float:
        return self.pad_forces()[side][0]

    def indentation(self, side, smooth_px=1.0):
        """(H,W) indentation map in metres (>=0).  Uses the GPU depth rasteriser
        when one is attached, else the numpy z-buffer (same camera model)."""
        W = self.pad_tf(side)
        g = self.pads[side].get("gpu")
        if g is not None:
            d = g.observe(sensor_tf=W, with_markers=False, smooth_sigma_px=smooth_px,
                          normal_mode="metric")["depth"].cpu().numpy()
        else:
            from stiff_physics.sensors.observation import depth_from_coat_surface, smooth_depth
            gel = self.pads[side]["gel"]
            d = smooth_depth(depth_from_coat_surface(gel.verts_local(W), gel.asset.coat_faces,
                                                     gel.asset.camera), smooth_px)
        if side not in self.rest_depth:
            self.rest_depth[side] = d.copy()
        return np.clip(d - self.rest_depth[side], 0.0, None)

    # ------------------------------------------------------------- commands
    def command(self, mode="pos", gap_mm=None, force_N=None, fmax=None, vmax_mm_s=None):
        self.mode = mode
        if gap_mm is not None:
            self.cmd_gap = max(0.0, gap_mm) * 1e-3
        if force_N is not None:
            self.grip_force = float(force_N)
        if fmax is not None:
            self.fmax = float(fmax)
        if vmax_mm_s is not None:
            self.vmax = vmax_mm_s * 1e-3

    def sync_target(self):
        """Make the commanded joint distance equal the measured one."""
        self._target_d = 0.5 * (self.eng.get_prismatic_current_distance(self.jl)
                                + self.eng.get_prismatic_current_distance(self.jr))

    def update(self) -> dict:
        """Write both finger joints for this frame (call BEFORE engine.step())."""
        pf = self.pad_forces() if self.pads else {}
        fl = pf["left"][0] if pf else 0.0
        fr = pf["right"][0] if pf else 0.0
        self.last = dict(fn_left=fl, fn_right=fr)
        fmeas = max(fl, fr)
        gap = self.gap_m()
        info = dict(mode=self.mode, gap_mm=gap * 1e3, fn_left=fl, fn_right=fr, event="")
        step_lim = self.vmax * self.dt
        if self.mode == "pos":
            err = self.cmd_gap - gap                       # + = open up
            step = float(np.clip(err, -step_lim, step_lim))  # change of gap
            if fmeas > self.fmax and step < 0:             # closing against force cap
                step = 0.0
                info["event"] = "fmax"
            self._target_d += 0.5 * step
        elif self.mode == "safe":
            fn = 0.5 * (fl + fr)
            err = self.grip_force - fn
            step = float(np.clip(self.ki * err, -0.35e-3, 0.12e-3))   # per-frame gap change (closing +)
            if fn < 0.3:
                step = max(step, 0.10e-3)                  # coarse approach until touch
            self._target_d -= 0.5 * step                   # closing = smaller finger distance
        self._target_d = float(np.clip(self._target_d, PAD_T, 0.040))
        for j in (self.jl, self.jr):
            self.eng.native.set_prismatic_force(j, 0.0)
            self.eng.native.set_prismatic_strength(j, self.stiffness)
            self.eng.native.set_prismatic_target(j, float(self._target_d))
        return info


class YoungEstimator:
    """Tactile Young's-modulus estimation from one grasp.

    Series model: after first contact the jaw travel per side splits into gel
    indentation (measured from the pad depth image) and object deformation.
    The object part is fitted with the Hertz law for a curved body against a
    (compliant, but measured) flat:  F = 4/3 * E* * sqrt(R) * d_obj^1.5, i.e.
    F^(2/3) is linear in d_obj with slope (4/3 E* sqrt R)^(2/3).  The slope is
    fitted by least squares over the loading ramp (the intercept absorbs the
    contact-onset ambiguity from the IPC barrier gap dhat).  E = E*(1-nu^2).
    """

    def __init__(self, name, radius_m, nu=0.45, e_true=None):
        self.name, self.R, self.nu, self.e_true = name, radius_m, nu, e_true
        self.samples = []            # (gap_m, F_left, F_right, dgel_left, dgel_right)
        self.result = None

    def add(self, gap_m, fl, fr, dl, dr):
        self.samples.append((gap_m, fl, fr, dl, dr))

    def fit(self, f_min=0.15):
        s = np.array(self.samples, float)
        if len(s) < 8:
            self.result = dict(name=self.name, ok=False, reason="too few samples")
            return self.result
        gap, fl, fr, dl, dr = s.T
        F = 0.5 * (fl + fr)
        on = np.nonzero(F > f_min)[0]
        if len(on) < 6:
            self.result = dict(name=self.name, ok=False, reason="no contact ramp")
            return self.result
        i0 = on[0]
        gap0 = gap[max(i0 - 1, 0)]
        travel = 0.5 * (gap0 - gap)                     # per side, m
        dgel = 0.5 * (dl + dr)
        dobj = travel - dgel
        sel = on
        x, y = dobj[sel], F[sel] ** (2.0 / 3.0)
        if np.ptp(x) < 2e-5:                            # object did not deform measurably
            self.result = dict(name=self.name, ok=True, rigid=True, E=np.inf,
                               dobj_max_mm=float(x.max() * 1e3), dgel_max_mm=float(dgel[sel].max() * 1e3),
                               F_max=float(F.max()), e_true=self.e_true, n=len(sel))
            return self.result
        A = np.vstack([x, np.ones_like(x)]).T
        slope, icpt = np.linalg.lstsq(A, y, rcond=None)[0]
        slope = max(slope, 1e-9)
        E_star = slope ** 1.5 * 3.0 / (4.0 * np.sqrt(self.R))
        E = E_star * (1.0 - self.nu ** 2)
        # the pads resolve ~0.1 mm; anything stiffer than ~10x the gel reads as rigid
        rigid = float(x.max()) < 0.25e-3 or E > 5e6
        self.result = dict(name=self.name, ok=True, rigid=rigid, E=float(E), E_star=float(E_star),
                           dobj_max_mm=float(x.max() * 1e3), dgel_max_mm=float(dgel[sel].max() * 1e3),
                           F_max=float(F.max()), e_true=self.e_true, n=int(len(sel)),
                           slope=float(slope), icpt=float(icpt))
        return self.result

    def text(self) -> str:
        r = self.result
        if r is None:
            return f"{self.name}: estimating..."
        if not r["ok"]:
            return f"{self.name}: n/a ({r['reason']})"
        if r.get("rigid"):
            return (f"{self.name}: rigid (E > 5 MPa, beyond pad resolution)  "
                    f"d_obj {r['dobj_max_mm']:.2f} mm, d_gel {r['dgel_max_mm']:.2f} mm, F {r['F_max']:.1f} N")
        s = f"{self.name}: E ~ {r['E']/1e3:.0f} kPa"
        if r["e_true"]:
            s += f" (true {r['e_true']/1e3:.0f} kPa)"
        s += f"  d_obj {r['dobj_max_mm']:.2f} mm, d_gel {r['dgel_max_mm']:.2f} mm, F {r['F_max']:.1f} N"
        return s


class LabScene:
    def __init__(self, dt=0.005, precond=0, log_level=0, tactile_gpu=True, verbose=True,
                 with_props=True, with_dropper=True, with_gels=True, prop_names=None,
                 joint_ratio=None, extra_cfg=None, pre_finalize=None):
        import os
        if joint_ratio is None:
            joint_ratio = float(os.environ.get("LAB_JOINT_RATIO", "1000"))
        extra_cfg = dict(extra_cfg or {})
        extra_cfg.setdefault("kappa_mean_mass", KAPPA_MEAN_MASS)
        if os.environ.get("LAB_FRIC_ANCHOR") is not None:
            extra_cfg["friction_anchor"] = bool(int(os.environ["LAB_FRIC_ANCHOR"]))
        if os.environ.get("LAB_NEWTON_VTOL") is not None:
            extra_cfg["newton_velocity_tol"] = float(os.environ["LAB_NEWTON_VTOL"])
        if os.environ.get("LAB_PCG_TOL") is not None:
            extra_cfg["pcg_tol"] = float(os.environ["LAB_PCG_TOL"])
        # Solve accuracy (A/B on the A800, 2026-10-03/04, probe_lag.py): the
        # engine defaults (newton_tol 1e-2, pcg_tol 1e-4) let the carried
        # dropper swing 40 deg; 1e-3 leaves a sustained ~5 Hz rocking of
        # 0.3-5 deg (unconverged per-frame residual -> spurious velocity each
        # frame, kicking a lightly damped mode); 1e-4 + pcg 1e-6 converges it
        # (rocking 0.03 deg rms) at ~1.9x the moving-frame cost.  Override with
        # LAB_NEWTON_TOL / LAB_PCG_TOL.
        extra_cfg.setdefault("newton_tol", float(os.environ.get("LAB_NEWTON_TOL", "1e-4")))
        extra_cfg.setdefault("pcg_tol", float(os.environ.get("LAB_PCG_TOL", "1e-6")))
        if os.environ.get("LAB_SEMI") is not None:
            extra_cfg["semi_implicit_enabled"] = bool(int(os.environ["LAB_SEMI"]))
        if os.environ.get("LAB_PRECOND") is not None:
            precond = int(os.environ["LAB_PRECOND"])
        from stiff_physics.engine import Config, Engine
        from stiff_physics.sensors.backend_stiffgipc import GelBody, clear_mesh_cache

        self.dt = dt
        self.verbose = verbose
        self.props = build_all(verbose=False)
        self.pad_asset = load_pad_asset()
        self.arm = Fr3Arm(URDF, BASE_T)
        clear_mesh_cache(ASSETS_ROOT)

        cfg = Config(dt=dt, density=1000.0, young_modulus=1e7, poisson_rate=0.45,
                     friction_rate=0.5, gravity=(0.0, -9.81, 0.0),
                     ground_offset=0.0, ground_normal=(0.0, 1.0, 0.0),
                     joint_strength_ratio=joint_ratio, revolute_driving_strength_ratio=joint_ratio,
                     prismatic_strength_ratio=2000.0, prismatic_driving_strength_ratio=100.0,
                     max_prismatic_step_per_frame=1.5e-3, max_revolute_step_per_frame=0.1,
                     absolute_dhat=5e-4, preconditioner_type=precond,
                     assets_dir=ASSETS_ROOT + "/", soft_motion_rate=1e4, **extra_cfg)
        self.eng = eng = Engine(cfg)
        eng.set_log_level(log_level)
        self.bodies: dict[str, Body] = {}

        # ------------------------------------------------------------ 1. arm
        self.joint_constraint_idx = _load_urdf_capture_joint_indices(eng, URDF, BASE_T, True, False, 1e7, {})
        eng.set_log_level(log_level)
        recs = eng.get_load_records()
        self.arm_bodies = []
        for r in recs:
            if r.body_type != 0:
                continue
            b = Body(r, r.label, "arm", color=(0.92, 0.92, 0.94))
            if r.label in (FINGER_L, FINGER_R):
                b.color = (0.25, 0.25, 0.28)
            self.bodies[r.label] = b
            self.arm_bodies.append(b)
            eng.add_ground_collision_skip(b.body_id)
        self.link_T0 = {n: np.asarray(eng.native.get_urdf_link_transform(n))
                        for n in [f"fr3_link{i}" for i in range(8)] + [HAND, FINGER_L, FINGER_R]}
        n_rev = eng.native.get_num_revolute_joints()
        self.rev_idx = {eng.get_revolute_joint_info(i).name: i for i in range(n_rev)}
        self.pri_idx = {eng.get_prismatic_joint_info(i).name: i
                        for i in range(eng.native.get_num_prismatic_joints())}
        self.arm_joint_ids = [self.rev_idx[j] for j in self.arm.ARM_JOINTS]
        if verbose:
            print(f"[scene] arm: {len(self.arm_bodies)} links, revolute {list(self.rev_idx)}, "
                  f"prismatic {list(self.pri_idx)}")

        # ---------------------------------------------------------- 2. props
        self.prop_T = {}
        self.props["cradle"] = cradle_prop(self.props)
        plan = planned_placements(self.props)
        fixed_names = ("rack", "tray", "cradle")
        free_names = ("flask", "beaker_mix", "beaker_waste", "test_tube", "test_tube2", "rod")
        if prop_names is not None:
            fixed_names = tuple(n for n in fixed_names if n in prop_names)
            free_names = tuple(n for n in free_names if n in prop_names)
        if with_props:
            for name in fixed_names:
                self._add_prop(name, plan[name], fixed=True)
            # the rod stand sits ON the tray floor (coincident faces): two Fixed
            # bodies at zero distance would blow the barrier up -> never test them
            if "cradle" in self.bodies and "tray" in self.bodies:
                eng.add_collision_exclusion(self.bodies["cradle"].body_id, self.bodies["tray"].body_id)
            for name in free_names:
                self._add_prop(name, plan[name])

        # ------------------------------------------------------- 3. FEM bodies
        # dropper: resting in the flask neck
        dp = self.props["dropper"]
        dm = dp.meta
        T_drop = plan["dropper"]
        y_c = plan["_bulb_centre"][1]
        self.prop_T["dropper"] = T_drop
        self.dropper = None
        if not with_dropper:
            dp = None
        if dp is not None:
            self._add_dropper(eng, dp, dm, T_drop, y_c, verbose)
        self.gels = {}
        if with_gels:
            self._add_gels(eng, GelBody)

        if pre_finalize is not None:      # e.g. register the engine liquid (must precede finalize)
            pre_finalize(self)
        t0 = time.time()
        eng.finalize()
        if verbose:
            print(f"[scene] finalize {time.time()-t0:.1f}s: {eng.vertex_count} verts, "
                  f"{eng.abd_body_count} ABD, {eng.fem_body_count} FEM")
        self._after_finalize(eng, dt, tactile_gpu, verbose)

    def _add_dropper(self, eng, dp, dm, T_drop, y_c, verbose):
        dv = dp.verts @ T_drop[:3, :3].T + T_drop[:3, 3]
        eng.load_mesh_from_data(dv, dp.tets, verts_per_face=4, dimensions=3, body_type="FEM",
                                transform=np.eye(4), young_modulus=dm["E_bulb"], boundary_type="Free")
        recs = eng.get_load_records()
        rec = recs[-1]
        rec_idx = len(recs) - 1          # global body id (ABD first, then FEM) == load-record index
        self.dropper = dict(rec=rec, v_off=int(rec.vertex_offset), n=len(dp.verts), faces=dp.faces,
                            body_id=rec_idx, color=dp.color,
                            bulb_mask=(dp.verts[:, 1] > dm["tube_len"] + 2e-3), meta=dm)
        import os
        ty = dp.tet_young.astype(np.float64)
        if os.environ.get("LAB_TUBE_E_SCALE"):      # diagnostic: soften the dropper's glass tube (PCG conditioning study)
            ty = np.where(ty > dm["E_bulb"] * 10, ty * float(os.environ["LAB_TUBE_E_SCALE"]), ty)
            print(f"[scene] dropper tube Young scaled by {os.environ['LAB_TUBE_E_SCALE']}: E_tube {ty.max():.3g}", flush=True)
        eng.native.set_per_tet_young_for_body(rec_idx, ty)
        eng.set_soft_body_density(rec_idx, dp.density)
        eng.set_body_friction(rec_idx, 0.8)
        if verbose:
            print(f"[scene] dropper FEM body {rec_idx}: {len(dp.verts)} v / {len(dp.tets)} t, "
                  f"bulb centre y={y_c*1e3:.1f} mm, E_bulb={dm['E_bulb']:.0e}, E_tube={dm['E_tube']:.0e}")

    def _add_gels(self, eng, GelBody):
        # gel pads pinned to the finger inner faces (pad local z = finger +y,
        # pad local x = finger +z, pad local y = finger +x; coat at y=-t)
        R_pf = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)     # columns: x_pad=ez, y_pad=ex, z_pad=ey
        T_pf = T_of(R_pf, [0.0, -PAD_T, PAD_Z_ON_FINGER])
        for side, link in (("left", FINGER_L), ("right", FINGER_R)):
            W = self.link_T0[link] @ T_pf
            gel = GelBody(self.pad_asset, eng, world_tf=W)
            fb = self.bodies[link]
            gel.add_pinned(fb.body_id, fb.v_off)
            self.gels[side] = dict(gel=gel, tf0=W, link=link)
        # exclusions: pads never collide with the hand/fingers or each other
        hand_ids = [self.bodies[n].body_id for n in (HAND, FINGER_L, FINGER_R)]
        for side in self.gels:
            g = self.gels[side]["gel"].body_offset
            for hid in hand_ids:
                eng.add_collision_exclusion(g, hid)
        eng.add_collision_exclusion(self.gels["left"]["gel"].body_offset,
                                    self.gels["right"]["gel"].body_offset)

    def _after_finalize(self, eng, dt, tactile_gpu, verbose):
        # ----------------------------------------------------- post-finalize
        self._post_finalize()
        self.gripper = FrankaTactileGripper(eng, self.pri_idx["fr3_finger_joint1"],
                                            self.pri_idx["fr3_finger_joint2"], dt)
        for side in self.gels:
            self.gripper.finger_body[side] = self.bodies[self.gels[side]["link"]]
        if tactile_gpu:
            from stiff_physics.sensors.gpu_observation import GpuTactileRenderer
            for side in self.gels:
                self.gels[side]["gpu"] = GpuTactileRenderer(self.pad_asset, eng, self.gels[side]["gel"])
        self.gripper.pads = {s: dict(gel=self.gels[s]["gel"], tf0=self.gels[s]["tf0"],
                                     gpu=self.gels[s].get("gpu")) for s in self.gels}
        self.gripper.sync_target()
        self.q = np.array([self.eng.get_revolute_current_angles_abs()[i] for i in self.arm_joint_ids])
        if verbose:
            print(f"[scene] load-pose joint angles {np.round(self.q, 3)}")
            if self.gels:
                self.check_pad_placement()

    # ------------------------------------------------------------------ util
    def _add_prop(self, name, T, fixed=False):
        p = self.props[name]
        wv = p.verts @ T[:3, :3].T + T[:3, 3]
        self.eng.load_mesh_from_data(wv, p.faces, verts_per_face=3, dimensions=3, body_type="ABD",
                                     transform=np.eye(4), young_modulus=1e8,
                                     boundary_type="Fixed" if fixed else "Free")
        rec = self.eng.get_load_records()[-1]
        b = Body(rec, name, "fixed" if fixed else "prop", color=p.color)
        if fixed:
            self.eng.add_ground_collision_skip(b.body_id)
        else:
            self.eng.set_abd_body_density(b.body_id, p.density)
        self.eng.set_body_friction(b.body_id, 0.5)
        self.bodies[name] = b
        self.prop_T[name] = T
        return b

    def _post_finalize(self):
        eng = self.eng
        ids = np.array([b.body_id for b in self.bodies.values()], np.int32)
        C0 = eng.get_abd_body_transforms(ids)
        for b, c in zip(self.bodies.values(), C0):
            b.C0inv = np.linalg.inv(abd_world_tf(c))
        for b in self.bodies.values():
            bv = eng.get_abd_body(b.body_id)
            b.rest_verts = np.asarray(bv.get_vertices(), np.float64).copy()
            b.faces = np.asarray(bv.get_surface_faces(local_indices=True), np.int32).copy()
        # stiffen the URDF's fixed joints (hand mount etc.) so the tool does not sag
        n_fixed = 0
        for jname, (jidx, jtype) in self.joint_constraint_idx.items():
            if jtype == "Fixed":
                eng.native.set_fixed_joint_strength(jidx, 1000.0)
                n_fixed += 1
        if self.verbose:
            print(f"[scene] fixed-joint strength bumped on {n_fixed} joints: "
                  f"{[n for n, (_, t) in self.joint_constraint_idx.items() if t == 'Fixed']}")

    def check_pad_placement(self):
        """Sanity: gel stick faces must sit on the finger inner faces."""
        for side, g in self.gels.items():
            v = g["gel"].verts_world()
            W = g["tf0"]
            loc = (v - W[:3, 3]) @ W[:3, :3]
            print(f"[scene] pad {side}: local z span {loc[:,2].min()*1e3:.2f}..{loc[:,2].max()*1e3:.2f} mm "
                  f"(expect 0..{PAD_T*1e3:.1f}), centre {v.mean(0).round(4)}")

    def link_vertices_local(self):
        """{link: (N,3) vertices in the link frame} from the engine's load-pose
        surfaces (for kinematic clearance checks)."""
        if not hasattr(self, "_lvl"):
            self._lvl = {}
            for n, T0 in self.link_T0.items():
                if n in self.bodies and self.bodies[n].rest_verts is not None:
                    v = self.bodies[n].rest_verts
                    self._lvl[n] = (v - T0[:3, 3]) @ T0[:3, :3]
        return self._lvl

    # -------------------------------------------------------------- queries
    def tcp_pose(self) -> np.ndarray:
        """Current TCP world pose from the hand body's rigid motion."""
        hb = self.bodies[HAND]
        return hb.pose(self.eng) @ self.link_T0[HAND] @ T_of(p=[0, 0, 0.1034])

    def prop_pose(self, name) -> np.ndarray:
        return self.bodies[name].pose(self.eng) @ self.prop_T[name]

    def dropper_verts(self) -> np.ndarray:
        d = self.dropper
        return np.asarray(self.eng.get_vertices())[d["v_off"]:d["v_off"] + d["n"]]

    def bulb_centre(self) -> np.ndarray:
        v = self.dropper_verts()
        return v[self.dropper["bulb_mask"]].mean(0)

    def dropper_tip(self) -> np.ndarray:
        """Centre of the tip ring, on the tube axis. (A single lowest vertex sat on the outer
        wall, 4.5 mm off axis, so "centre tip over beaker/neck" aimed 4.5 mm beside the axis.)"""
        return self.dropper_verts()[self.dropper["meta"]["tip_idx"]].mean(0)

    # -------------------------------------------------------------- liquid
    def surface_faces_global(self, names=None, with_dropper=True):
        """Surface triangles of the props (and the dropper) in GLOBAL engine
        vertex indices (input order == engine order with preconditioner 0), for
        the liquid's boundary mesh."""
        F = []
        for n, b in self.bodies.items():
            if b.kind == "arm" or (names is not None and n not in names):
                continue
            F.append(np.asarray(b.faces, np.int64) + int(b.rec.vertex_offset))
        if with_dropper and self.dropper is not None:
            F.append(np.asarray(self.dropper["faces"], np.int64) + int(self.dropper["v_off"]))
        return np.vstack(F) if F else np.zeros((0, 3), np.int64)

    def vessel_cavity(self, name):
        """(centre_xz, inner_radius, floor_y, rim_y) of an upright cylindrical
        vessel, estimated from its mesh at the load pose."""
        p = self.props[name]; T = self.prop_T[name]
        v = p.verts @ T[:3, :3].T + T[:3, 3]
        cx, cz = float(T[0, 3]), float(T[2, 3])
        y0, y1 = float(v[:, 1].min()), float(v[:, 1].max())
        r = np.hypot(v[:, 0] - cx, v[:, 2] - cz)
        # wall vertices (any height above the base region) have radius >= inner radius;
        # GLB vessels may only have rim/base rings, so do not rely on mid-height vertices
        upper = v[:, 1] > y0 + 0.2 * (y1 - y0)
        r_in = float(r[upper].min()) if upper.any() else 0.8 * float(r.max())
        low = (r < 0.5 * r_in) & (v[:, 1] < y0 + 0.5 * (y1 - y0))
        floor_y = float(v[low, 1].max()) if low.any() else y0 + 2.0e-3
        return np.array([cx, cz]), r_in, floor_y, y1

    # -------------------------------------------------------------- driving
    def drive_arm(self, q):
        for jid, v in zip(self.arm_joint_ids, q):
            self.eng.set_revolute_target(jid, float(v))

    def arm_q(self) -> np.ndarray:
        a = self.eng.get_revolute_current_angles_abs()
        return np.array([a[i] for i in self.arm_joint_ids])
