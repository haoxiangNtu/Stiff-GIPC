"""Shared scaffolding for the RobotGym reproductions (examples/robotgym/*):
FR3 arms with the standard Franka hand, finger control, rigid props, cloth / FEM bodies and a
recorder that writes the lab_tactile recording schema, so render_lab_video.py (pyrender),
caustica-work/render_mitsuba.py (path tracing) and export_lab_usd.py work unchanged.

World: metres, Y up, ground plane at y = 0. URDF z-up is mapped with Rx(-90).
"""
from __future__ import annotations

import os
import os.path as osp
import re
import sys
import tempfile

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
LAB = osp.join(REPO, "examples", "lab_tactile")
if LAB not in sys.path:
    sys.path.insert(0, LAB)
from fr3_kin import Fr3Arm, axis_angle_R  # noqa: E402
from lab_rec import save_recording  # noqa: E402

ASSETS = osp.join(REPO, "Assets")
URDF = osp.join(ASSETS, "fr3", "fr3_franka_hand_sub50.urdf")
# kappa's mass scale with the refined (sub50) arm meshes, pinned like examples/lab_tactile/lab_scene.py
KAPPA_MEAN_MASS = 1.2292327807099668e-06
ARM_LINKS = [f"fr3_link{i}" for i in range(8)] + ["fr3_hand", "fr3_leftfinger", "fr3_rightfinger"]
FINGER_L, FINGER_R, HAND = "fr3_leftfinger", "fr3_rightfinger", "fr3_hand"
PAD_Z_ON_FINGER = 0.044   # tactile pad centre along the finger (examples/lab_tactile/lab_scene.py; TCP at 0.045)


# ----------------------------------------------------------------------------- transforms
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


def R_cols(x, y, z):
    """Rotation whose columns are the given (unit) axes."""
    return np.stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)], 1)


def arm_base_T(pos, yaw_deg):
    """FR3 base on the floor at ``pos`` (world, y up), rotated about world +y by yaw_deg
    (yaw 0 = URDF +x forward = world +x)."""
    return T_of(Ry(yaw_deg) @ Rx(-90.0), pos)


def abd_world_tf(C: np.ndarray) -> np.ndarray:
    """get_abd_body_transforms() stores the affine block transposed; the physical map is
    x = R xbar + p with R = C[:3,:3].T (verified in the lab demo)."""
    T = np.array(C, np.float64)
    T[:3, :3] = C[:3, :3].T
    return T


def load_obj(path):
    """vertices (N,3), triangles (M,3) of a plain .obj (v / f lines; f may carry /vt/vn)"""
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            F += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return np.array(V, float), np.array(F, np.int64)


def arm_clearance(arm, tl, V_static, F_static, links=None, dt=0.05, sample=0.004, finger_of=None, verbose=True):
    """Kinematic clearance (no engine): the arm's link collision meshes posed by FK along timeline tl vs a fixed triangle mesh.
    Returns {segment label: (min distance m, link)}; the static surface is sampled every `sample` m (distance error <= ~sample/2).
    finger_of(seg) -> per-finger joint distance (default: the segment's finger target, else the last one)."""
    import trimesh
    from scipy.spatial import cKDTree
    links = links or [f"fr3_link{i}" for i in range(2, 8)] + ["fr3_hand", "fr3_leftfinger", "fr3_rightfinger"]
    mesh_of = {l: (osp.join(ASSETS, "fr3", "meshes", "finger_collision.stl") if "finger" in l else
                   osp.join(ASSETS, "fr3", "meshes_sub50", ("hand" if l == "fr3_hand" else l.replace("fr3_", "")) + ".stl")) for l in links}
    LV = {l: np.asarray(trimesh.load(p_).vertices) for l, p_ in mesh_of.items()}
    st = trimesh.Trimesh(np.asarray(V_static), np.asarray(F_static), process=False)
    n = int(max(1, st.area / (sample * sample)))
    P, _ = trimesh.sample.sample_surface_even(st, n) if n < 4_000_000 else trimesh.sample.sample_surface(st, n)
    tree = cKDTree(np.vstack([P, st.vertices]))
    q = arm.kin.ik(tl.segs[0].T_to, arm.kin.HOME, iters=300)[0]
    fd = 0.0
    worst = {}
    for t in np.arange(0.0, tl.total + 1e-9, dt):
        seg, s_ = tl.at(t)
        q = arm.kin.ik(seg.pose(s_), q, iters=60)[0]
        if seg.finger is not None:
            fd = seg.finger
        LP = arm.kin.link_poses(q, fd)
        for l in links:
            W = LV[l] @ LP[l][:3, :3].T + LP[l][:3, 3]
            d = float(tree.query(W)[0].min())
            if seg.label not in worst or d < worst[seg.label][0]:
                worst[seg.label] = (d, l, round(float(t), 2))
    if verbose:
        for lab, (d, l, t) in worst.items():
            flag = "  <-- CLOSE" if d < 0.01 else ""
            print(f"[clearance] {arm.name:5s} {lab[:38]:38s} min {d*1e3:6.1f} mm ({l} at {t:.2f} s){flag}")
    return worst


def minjerk(s):
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


# ----------------------------------------------------------------------------- engine helpers
def _load_urdf_capture(eng, urdf_path, transform, *args):
    """load_urdf while capturing the importer log; returns {joint_name: (constraint_idx, type)}
    (constraint indices are global over all URDFs loaded so far, in log order)."""
    eng.native.set_log_level(1)
    log_path = tempfile.mktemp(prefix="rg_urdf_load_", suffix=".txt")
    saved = os.dup(1)
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
        os.dup2(saved, 1)
        os.close(saved)
    try:
        log = open(log_path).read()
    finally:
        try:
            os.unlink(log_path)
        except OSError:
            pass
    return log


class Body:
    """One ABD body: load record + rest world surface + canonical frame (lab_scene.Body)."""

    def __init__(self, rec, name, kind="prop", color=(0.7, 0.7, 0.7)):
        self.rec, self.name, self.kind, self.color = rec, name, kind, color
        self.body_id = int(rec.body_offset)
        self.v_off = int(rec.vertex_offset)
        self.rest_verts = None
        self.faces = None
        self.C0inv = None

    def pose_from(self, C) -> np.ndarray:
        return abd_world_tf(C) @ self.C0inv


class Arm:
    """One FR3 in the engine: kinematics (fr3_kin), joint ids, finger control."""

    def __init__(self, name, base_T, stiffness_finger=25.0):
        self.name = name
        self.base_T = base_T
        self.kin = Fr3Arm(URDF, base_T)
        self.rev_ids = None      # engine revolute indices of fr3_joint1..7
        self.pri_ids = None      # engine prismatic indices of the two finger joints
        self.bodies: dict[str, Body] = {}
        self.q = self.kin.HOME.copy()
        self.finger_d = 0.0      # commanded finger joint distance (0 = closed at load)
        self.finger_goal = 0.0
        self.finger_v = 0.08     # m/s of gap change
        self.stiff = stiffness_finger
        self.pad_t = 0.0         # gel pad thickness on each finger (tactile scenes): the commanded half-gap is the pad face's

    def drive(self, eng, q):
        self.q = np.asarray(q, float)
        for jid, v in zip(self.rev_ids, self.q):
            eng.set_revolute_target(jid, float(v))

    def fingers(self, eng, dt):
        step = self.finger_v * dt * 0.5
        self.finger_d += float(np.clip(self.finger_goal - self.finger_d, -step, step))
        d = min(self.finger_d + self.pad_t, 0.040)   # joint limit: an open hand with pads opens 2 pad_t less
        for j in self.pri_ids:
            eng.native.set_prismatic_force(j, 0.0)
            eng.native.set_prismatic_strength(j, self.stiff)
            eng.native.set_prismatic_target(j, float(d))

    def gap_m(self, eng):
        return 2.0 * 0.5 * sum(eng.get_prismatic_current_distance(j) for j in self.pri_ids) - 2.0 * self.pad_t

    def tcp(self):
        return self.kin.fk(self.q)


class Scene:
    """Engine + arms + props + cloth/FEM bodies + recorder."""

    def __init__(self, cfg_kwargs, log_level=0, tactile=False, tactile_gpu=True):
        from stiff_physics.engine import Config, Engine
        base = dict(density=1000.0, young_modulus=1e7, poisson_rate=0.45, friction_rate=0.5,
                    gravity=(0.0, -9.81, 0.0), ground_offset=0.0, ground_normal=(0.0, 1.0, 0.0),
                    joint_strength_ratio=1000.0, revolute_driving_strength_ratio=1000.0,
                    prismatic_strength_ratio=2000.0, prismatic_driving_strength_ratio=100.0,
                    max_prismatic_step_per_frame=1.5e-3, max_revolute_step_per_frame=0.1,
                    absolute_dhat=5e-4, preconditioner_type=0, kappa_mean_mass=KAPPA_MEAN_MASS,
                    newton_tol=1e-3, pcg_tol=1e-5, assets_dir=ASSETS + "/")
        base.update(cfg_kwargs)
        self.dt = float(base["dt"])
        self.cfg = Config(**base)
        self.eng = Engine(self.cfg)
        self.eng.set_log_level(log_level)
        self.log_level = log_level
        self.arms: list[Arm] = []
        self.bodies: dict[str, Body] = {}     # every ABD body, in load order
        self.fem: dict[str, dict] = {}        # name -> dict(body_id, v_off, n, faces)
        self.fixed_joint_ids = []
        self._n_rev = 0
        self._n_pri = 0
        # tactile fingertips: the lab demo's VBTS gel pads (examples/lab_tactile), one per finger of every arm
        self.tactile, self.tactile_gpu = bool(tactile), bool(tactile_gpu)
        self.pads: list[dict] = []
        if self.tactile:
            from lab_pad import PAD_SIZE, load_pad_asset
            from stiff_physics.sensors.backend_stiffgipc import clear_mesh_cache
            self.pad_asset = load_pad_asset()
            self.pad_t = float(PAD_SIZE[2])
            clear_mesh_cache(ASSETS)

    # ----------------------------------------------------------------- loading
    def add_arm(self, name, pos, yaw_deg, q0=None):
        arm = Arm(name, arm_base_T(pos, yaw_deg))
        if q0 is not None:
            arm.q = np.asarray(q0, float)
        init = {j: float(v) for j, v in zip(arm.kin.ARM_JOINTS, arm.q)}
        n_rec0 = len(self.eng.get_load_records())
        log = _load_urdf_capture(self.eng, URDF, arm.base_T, True, False, 1e7, init)
        self.eng.set_log_level(self.log_level)
        # joint-constraint indices (Fixed and Revolute share one index space) are global in log order
        k = 0
        for m in re.finditer(r"\[UrdfSceneImporter\] Joint constraint '([^']+)' \((Fixed|Revolute)\):", log):
            if m.group(2) == "Fixed":
                self.fixed_joint_ids.append(self._joint_constraint_base + k)
            k += 1
        self._joint_constraint_base += k
        recs = self.eng.get_load_records()[n_rec0:]
        for r in recs:
            if r.body_type != 0:
                continue
            b = Body(r, f"{name}/{r.label}", f"arm:{name}", color=(0.92, 0.92, 0.94))
            if r.label in (FINGER_L, FINGER_R):
                b.color = (0.25, 0.25, 0.28)
            self.bodies[b.name] = b
            arm.bodies[r.label] = b
            self.eng.add_ground_collision_skip(b.body_id)
        n_rev = self.eng.native.get_num_revolute_joints()
        n_pri = self.eng.native.get_num_prismatic_joints()
        names_rev = [self.eng.get_revolute_joint_info(i).name for i in range(self._n_rev, n_rev)]
        names_pri = [self.eng.get_prismatic_joint_info(i).name for i in range(self._n_pri, n_pri)]
        arm.rev_ids = [self._n_rev + names_rev.index(j) for j in arm.kin.ARM_JOINTS]
        arm.pri_ids = [self._n_pri + names_pri.index(j) for j in ("fr3_finger_joint1", "fr3_finger_joint2")]
        self._n_rev, self._n_pri = n_rev, n_pri
        self.arms.append(arm)
        return arm

    _joint_constraint_base = 0

    def add_tactile_pads(self):
        """Gel pads (FEM) hard-pinned to the inner face of every finger of every arm (as lab_scene._add_gels). Call after
        the last ABD body and before the other FEM bodies. The pads are recorded like any FEM body (rendered on the
        fingertips); their contact forces and indentation images are read by tactile_read()."""
        if not self.tactile:
            return
        from stiff_physics.sensors.backend_stiffgipc import GelBody
        R_pf = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)     # pad x = finger z, pad y = finger x, pad z = finger y
        T_pf = T_of(R_pf, [0.0, -self.pad_t, PAD_Z_ON_FINGER])
        for arm in self.arms:
            lp = arm.kin.link_poses(arm.q, 0.0)                         # load pose: fingers closed
            for side, link in (("left", FINGER_L), ("right", FINGER_R)):
                W = lp[link] @ T_pf
                gel = GelBody(self.pad_asset, self.eng, world_tf=W)
                fb = arm.bodies[link]
                gel.add_pinned(fb.body_id, fb.v_off)
                rec = self.eng.get_load_records()[-1]
                name = f"pad_{arm.name}_{side}"
                self.fem[name] = dict(rec=rec, body_id=int(rec.body_offset), v_off=int(rec.vertex_offset),
                                      color=(0.16, 0.16, 0.18), faces=None, n=None)
                self.pads.append(dict(name=name, arm=arm, side=side, link=link, gel=gel, tf0=W, gpu=None, rest=None))
            arm.pad_t = self.pad_t
        for p in self.pads:       # pads never collide with their own hand / fingers, nor with the other pad of that hand,
            for ln in (HAND, FINGER_L, FINGER_R):
                self.eng.add_collision_exclusion(p["gel"].body_offset, p["arm"].bodies[ln].body_id)
            # and without self-contact: the gel is 2.5 mm thick, so with a large absolute dhat (bed sheet 2 mm) its own faces
            # near the edges sit inside the barrier (Newton at the cap). NOT add_collision_exclusion(g, g): the skip-matrix
            # diagonal means "isolated", and the pad then collides with nothing (the sheet passed through it).
            if hasattr(self.eng.native, "set_body_self_collision"):
                self.eng.native.set_body_self_collision(int(p["gel"].body_offset), False)
        for i in range(len(self.pads)):
            for j in range(i + 1, len(self.pads)):
                if self.pads[i]["arm"] is self.pads[j]["arm"]:
                    self.eng.add_collision_exclusion(self.pads[i]["gel"].body_offset, self.pads[j]["gel"].body_offset)
        print(f"[scene] tactile pads: {len(self.pads)} gels ({self.pad_asset.n_verts} verts each), pad {self.pad_t*1e3:.1f} mm thick")

    def pad_tf(self, p, C=None):
        """Current pad frame (pad z = into the gel, coat at z = 0 facing the object)."""
        b = p["arm"].bodies[p["link"]]
        if C is None:
            C = self.eng.get_abd_body_transforms(np.array([b.body_id], np.int32))[0]
        return b.pose_from(C) @ p["tf0"]

    def tactile_read(self, maps=True, smooth_px=1.0):
        """Per pad: normal force (N, contact force along the pad normal) and, optionally, the indentation image
        (metres, >= 0, relative to the pad's first reading; GPU depth rasteriser when available)."""
        if not self.pads:
            return [], []
        F = np.asarray(self.eng.get_vertex_contact_forces(False, "total"))
        ids = np.array([p["arm"].bodies[p["link"]].body_id for p in self.pads], np.int32)
        C = self.eng.get_abd_body_transforms(ids)
        fn, ims = [], []
        for p, c in zip(self.pads, C):
            W = self.pad_tf(p, c)
            Fl = F[p["gel"].slice] @ W[:3, :3]
            fn.append(float(np.clip(Fl[:, 2], 0.0, None).sum()))
            if not maps:
                continue
            if p["gpu"] is not None:
                d = p["gpu"].observe(sensor_tf=W, with_markers=False, smooth_sigma_px=smooth_px, normal_mode="metric")["depth"].cpu().numpy()
            else:
                from stiff_physics.sensors.observation import depth_from_coat_surface, smooth_depth
                d = smooth_depth(depth_from_coat_surface(p["gel"].verts_local(W), p["gel"].asset.coat_faces, p["gel"].asset.camera), smooth_px)
            if p["rest"] is None:
                p["rest"] = d.copy()
            ims.append(np.clip(d - p["rest"], 0.0, None))
        return fn, ims

    def add_box(self, name, centre, size, fixed=True, color=(0.6, 0.6, 0.62), density=1000.0, mu=0.5, cell=0.05):
        """Axis-aligned box as an ABD body. The faces are tessellated with ~``cell``-sized triangles: one
        long edge per face would make the BVH edge queries serial against a cloth (the 186 mm hand edge in
        the lab demo cost 7x; a 1 m slab edge stalled the first sheet run)."""
        c, s = np.asarray(centre, float), 0.5 * np.asarray(size, float)
        verts, faces, index = [], [], {}
        def vid(p):
            key = tuple(np.round(p, 9))
            if key not in index:
                index[key] = len(verts); verts.append(np.array(p, float))
            return index[key]
        for axis in range(3):
            u, v = [(1, 2), (0, 2), (0, 1)][axis]
            nu, nv = max(1, int(np.ceil(2 * s[u] / cell))), max(1, int(np.ceil(2 * s[v] / cell)))
            for sign in (-1.0, 1.0):
                for i in range(nu):
                    for j in range(nv):
                        p = np.zeros((2, 2, 3))
                        for di in range(2):
                            for dj in range(2):
                                q = np.zeros(3); q[axis] = sign * s[axis]
                                q[u] = -s[u] + 2 * s[u] * (i + di) / nu; q[v] = -s[v] + 2 * s[v] * (j + dj) / nv
                                p[di, dj] = c + q
                        a, b, cc, d = vid(p[0, 0]), vid(p[1, 0]), vid(p[1, 1]), vid(p[0, 1])
                        # outward winding: normal along +axis*sign
                        n = np.cross(p[1, 0] - p[0, 0], p[0, 1] - p[0, 0])
                        if n[axis] * sign > 0:
                            faces += [(a, b, cc), (a, cc, d)]
                        else:
                            faces += [(a, cc, b), (a, d, cc)]
        v = np.array(verts); f = np.array(faces, np.int32)
        self.eng.load_mesh_from_data(v, f, verts_per_face=3, dimensions=3, body_type="ABD",
                                     transform=np.eye(4), young_modulus=1e8,
                                     boundary_type="Fixed" if fixed else "Free")
        rec = self.eng.get_load_records()[-1]
        b = Body(rec, name, "fixed" if fixed else "prop", color=color)
        if fixed:
            self.eng.add_ground_collision_skip(b.body_id)
        else:
            self.eng.set_abd_body_density(b.body_id, density)
        self.eng.set_body_friction(b.body_id, mu)
        self.bodies[name] = b
        return b

    def add_static_mesh(self, name, verts, faces, color=(0.6, 0.6, 0.62), mu=0.5, kind="fixed"):
        """A fixed ABD body from triangles (an asset's own surface, e.g. the FBX bed or kitchen; no volume needed for a
        fixed body). Merge parts that touch or overlap into ONE call: two fixed bodies with coincident / intersecting faces
        start inside each other's barrier (E = -nan, see add_basin)."""
        v = np.asarray(verts, float); f = np.asarray(faces, np.int32)
        self.eng.load_mesh_from_data(v, f, verts_per_face=3, dimensions=3, body_type="ABD", transform=np.eye(4),
                                     young_modulus=1e8, boundary_type="Fixed")
        rec = self.eng.get_load_records()[-1]
        b = Body(rec, name, kind, color=color)
        self.eng.add_ground_collision_skip(b.body_id)
        self.eng.set_body_friction(b.body_id, mu)
        self.bodies[name] = b
        return b

    def add_cloth(self, name, obj_path, transform, young=1e4, density=None, color=(0.86, 0.89, 0.95)):
        self.eng.load_mesh(obj_path, dimensions=2, body_type="FEM", transform=transform,
                           young_modulus=young, boundary_type="Free", density=density)
        rec = self.eng.get_load_records()[-1]
        self.fem[name] = dict(rec=rec, body_id=int(rec.body_offset), v_off=int(rec.vertex_offset),
                              color=color, faces=None, n=None)
        return self.fem[name]

    def add_fem(self, name, verts, tets_or_path, transform=None, young=1e5, density=None, color=(0.95, 0.85, 0.3)):
        """Tet FEM body from a .msh path (verts=None) or arrays."""
        if isinstance(tets_or_path, str):
            self.eng.load_mesh(tets_or_path, dimensions=3, body_type="FEM",
                               transform=np.eye(4) if transform is None else transform,
                               young_modulus=young, boundary_type="Free", density=density)
        else:
            self.eng.load_mesh_from_data(np.asarray(verts, float), np.asarray(tets_or_path, np.int32),
                                         verts_per_face=4, dimensions=3, body_type="FEM",
                                         transform=np.eye(4) if transform is None else transform,
                                         young_modulus=young, boundary_type="Free")
            if density is not None:
                self.eng.set_soft_body_density(len(self.eng.get_load_records()) - 1, float(density))
        rec = self.eng.get_load_records()[-1]
        self.fem[name] = dict(rec=rec, body_id=int(rec.body_offset), v_off=int(rec.vertex_offset),
                              color=color, faces=None, n=None)
        return self.fem[name]

    # ----------------------------------------------------------------- finalize
    def finalize(self):
        eng = self.eng
        eng.finalize()
        ids = np.array([b.body_id for b in self.bodies.values()], np.int32)
        C0 = eng.get_abd_body_transforms(ids)
        for b, c in zip(self.bodies.values(), C0):
            b.C0inv = np.linalg.inv(abd_world_tf(c))
            bv = eng.get_abd_body(b.body_id)
            b.rest_verts = np.asarray(bv.get_vertices(), np.float64).copy()
            b.faces = np.asarray(bv.get_surface_faces(local_indices=True), np.int32).copy()
        for name, f in self.fem.items():
            bv = eng.get_fem_body(f["body_id"])
            f["faces"] = np.asarray(bv.get_surface_faces(local_indices=True), np.int32).copy()
            f["n"] = int(bv.vertex_count)
        for jidx in self.fixed_joint_ids:
            eng.native.set_fixed_joint_strength(jidx, 1000.0)
        if self.pads and self.tactile_gpu:
            try:   # the GPU depth rasteriser needs torch; without it the numpy z-buffer gives the same camera model
                from stiff_physics.sensors.gpu_observation import GpuTactileRenderer
                for p in self.pads:
                    p["gpu"] = GpuTactileRenderer(self.pad_asset, eng, p["gel"])
            except Exception as exc:
                for p in self.pads:
                    p["gpu"] = None
                print(f"[scene] tactile: GPU renderer unavailable ({type(exc).__name__}: {exc}); CPU depth images")
        if self.pads:
            self.tactile_read(maps=True)       # rest depth of every pad (fingers open, nothing touching)

    def fem_verts(self, name):
        f = self.fem[name]
        return np.asarray(self.eng.get_fem_body(f["body_id"]).get_vertices(), np.float64)

    # ----------------------------------------------------------------- stepping
    def step(self):
        for a in self.arms:
            a.fingers(self.eng, self.dt)
        self.eng.step()


class Recorder:
    """lab_tactile recording schema (lab_rec.save_recording)."""

    def __init__(self, scene: Scene):
        self.sc = scene
        self.names = [b.name for b in scene.bodies.values()]
        self.kinds = [b.kind for b in scene.bodies.values()]
        self.colors = [b.color for b in scene.bodies.values()]
        self.rest = [b.rest_verts for b in scene.bodies.values()]
        self.faces = [b.faces for b in scene.bodies.values()]
        self.fem_names = list(scene.fem.keys())
        self.fem_faces = [scene.fem[n]["faces"] for n in self.fem_names]
        self.fem_colors = [scene.fem[n]["color"] for n in self.fem_names]
        self.fem = {n: [] for n in self.fem_names}
        self.motion, self.rows, self.liquid = [], [], []
        self.q_arms, self.gap_arms, self.q_meas = [], [], []
        self.tac_names = [p["name"] for p in scene.pads]
        self.tac_fn, self.tac_maps, self.tac_idx = [], [], []
        self.map_every = 1          # indentation images every N records (forces every record)

    def record(self, t, stage, seg, liquid_pts=None, extra_text=""):
        sc = self.sc
        ids = np.array([b.body_id for b in sc.bodies.values()], np.int32)
        C = sc.eng.get_abd_body_transforms(ids)
        self.motion.append(np.array([b.pose_from(c) for c, b in zip(C, sc.bodies.values())], np.float32))
        for n in self.fem_names:
            self.fem[n].append(sc.fem_verts(n).astype(np.float32))
        if liquid_pts is not None:
            self.liquid.append(np.asarray(liquid_pts, np.float32))
        self.q_arms.append(np.array([a.q for a in sc.arms], np.float32))
        self.gap_arms.append(np.array([a.gap_m(sc.eng) * 1e3 for a in sc.arms], np.float32))
        ang = np.asarray(sc.eng.get_revolute_current_angles_abs(), float)
        self.q_meas.append(np.array([[ang[j] for j in a.rev_ids] for a in sc.arms], np.float32))
        if sc.pads:
            with_map = (len(self.rows) % self.map_every) == 0
            fn, ims = sc.tactile_read(maps=with_map)
            self.tac_fn.append(np.array(fn, np.float32))
            if with_map:
                self.tac_maps.append(np.stack([np.clip(im / 2.0e-3 * 65535.0, 0, 65535).astype(np.uint16) for im in ims]))
                self.tac_idx.append(len(self.rows))
        a0 = sc.arms[0] if sc.arms else None
        self.rows.append(dict(t=t, stage=stage, seg=seg, etext=extra_text,
                              gap=(a0.gap_m(sc.eng) * 1e3 if a0 else 0.0),
                              tcp=(a0.tcp() if a0 else np.eye(4)).astype(np.float32),
                              q=np.asarray(a0.q if a0 else np.zeros(7), np.float32)))

    def save(self, path, meta):
        rows = self.rows
        out = dict(
            body_names=np.array(self.names), body_kind=np.array(self.kinds),
            body_color=np.array(self.colors, np.float32),
            body_rest=[np.asarray(a, np.float32) for a in self.rest],
            body_faces=[np.asarray(a, np.int32) for a in self.faces],
            motion=np.stack(self.motion),
            t=np.array([r["t"] for r in rows], np.float32),
            stage=np.array([r["stage"] for r in rows]), segment=np.array([r["seg"] for r in rows]),
            etext=np.array([r["etext"] for r in rows]),
            gap_mm=np.array([r["gap"] for r in rows], np.float32),
            fn_left=np.zeros(len(rows), np.float32), fn_right=np.zeros(len(rows), np.float32),
            tcp=np.stack([r["tcp"] for r in rows]), q=np.stack([r["q"] for r in rows]),
            meta=dict(meta, fem_colors=[list(map(float, c)) for c in self.fem_colors],
                      arms=[dict(name=a.name, base_T=np.asarray(a.base_T, float).tolist()) for a in self.sc.arms]),
            q_arms=np.stack(self.q_arms) if self.q_arms else np.zeros((len(rows), 0, 7), np.float32),
            gap_arms=np.stack(self.gap_arms) if self.gap_arms else np.zeros((len(rows), 0), np.float32),
            q_meas=np.stack(self.q_meas) if self.q_meas else np.zeros((len(rows), 0, 7), np.float32),
            fem_names=np.array(self.fem_names),
            fem_faces=[np.asarray(a, np.int32) for a in self.fem_faces],
            fem_verts=[np.stack(self.fem[n]) for n in self.fem_names],
            pad_map_left=np.zeros((0, 2, 2), np.uint8), pad_map_right=np.zeros((0, 2, 2), np.uint8),
            pad_map_idx=np.zeros(0, np.int32),
        )
        if self.tac_names:
            # all pads: tac_fn (records, pads) N, tac_maps (maps, pads, H, W) uint16 = indentation / 2 mm * 65535 at the records
            # tac_map_idx; the first arm's pads also fill the lab_tactile fields (render_lab_video.py dashboard)
            fn = np.stack(self.tac_fn); maps = np.stack(self.tac_maps); idx = np.array(self.tac_idx, np.int32)
            out.update(tac_names=np.array(self.tac_names), tac_fn=fn, tac_maps=maps, tac_map_idx=idx)
            il = self.tac_names.index(f"pad_{self.sc.arms[0].name}_left"); ir = self.tac_names.index(f"pad_{self.sc.arms[0].name}_right")
            m8 = (maps.astype(np.uint32) * 255 // 65535).astype(np.uint8)     # lab schema: uint8 = indentation / 2 mm * 255
            out.update(fn_left=fn[:, il].copy(), fn_right=fn[:, ir].copy(), pad_map_left=m8[:, il].copy(),
                       pad_map_right=m8[:, ir].copy(), pad_map_idx=idx)
            out["meta"] = dict(out["meta"], tactile=dict(pads=self.tac_names, pad_thickness=self.sc.pad_t,
                                                       map_scale_m=2.0e-3, pixel_mm=0.079375))
        if self.liquid:
            out["liquid_pts"] = list(self.liquid)
        save_recording(path, out)
        return path


# ----------------------------------------------------------------------------- trajectories
class Segment:
    def __init__(self, label, dur, T_from, T_to, finger=None):
        self.label, self.dur, self.T_from, self.T_to, self.finger = label, float(dur), T_from, T_to, finger

    def pose(self, s):
        """Cartesian min-jerk interpolation of position; rotation from the two frames (slerp via axis-angle)."""
        u = minjerk(s)
        p = (1 - u) * self.T_from[:3, 3] + u * self.T_to[:3, 3]
        R0, R1 = self.T_from[:3, :3], self.T_to[:3, :3]
        dR = R1 @ R0.T
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        if ang < 1e-9:
            R = R0
        else:
            ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * np.sin(ang))
            R = axis_angle_R(ax, u * ang) @ R0
        return T_of(R, p)


class Timeline:
    """Per-arm list of segments starting from pose T0; add(label, dur, T_to=None, finger=None)
    (T_to None = hold the previous pose, e.g. while the fingers move)."""

    def __init__(self, T0):
        self.segs: list[Segment] = [Segment("start", 0.0, T0, T0)]

    def add(self, label, dur, T_to=None, finger=None):
        T_from = self.segs[-1].T_to if self.segs else None
        self.segs.append(Segment(label, dur, T_from, T_to if T_to is not None else T_from, finger))
        return self

    @property
    def total(self):
        return sum(s.dur for s in self.segs)

    def shift(self, labels, d):
        """Move the target poses of the named segments by d (world xyz) at run time (closed-loop re-aiming), keeping the
        chain continuous: each following segment starts where the shifted one now ends."""
        for j, sg in enumerate(self.segs):
            if sg.label in labels:
                T = sg.T_to.copy(); T[:3, 3] += d; sg.T_to = T
                if j + 1 < len(self.segs):
                    self.segs[j + 1].T_from = T

    def at(self, t):
        """(segment, local progress 0..1)."""
        acc = 0.0
        for s in self.segs:
            if t < acc + s.dur or s is self.segs[-1]:
                return s, min(1.0, (t - acc) / max(s.dur, 1e-9))
            acc += s.dur
        return self.segs[-1], 1.0
