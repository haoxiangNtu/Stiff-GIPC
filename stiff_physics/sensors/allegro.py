"""Allegro hand kinematics, enough to pose a grasp and draw the hand.

Taccel's Figure 1 is a hand, not two floating pads. To show that honestly the
grasp has to be DEFINED by the hand -- pick joint angles, run forward kinematics,
see where the fingertips land, and put the object in the pinch -- rather than
placing pads by hand and drawing a hand near them afterwards.

What this gives:
  * `AllegroURDF`  : links (visual STL + origin), joints (origin, axis, limits)
  * `.fk(q)`       : every link's world transform
  * `.ik(...)`     : joint angles that put one fingertip at a target pose
  * `oppose(...)`  : the two fingers' angles that make their pads face each other
                     across a given gap -- the precision grasp

Frames worth stating once, because they are what everything else depends on:
  * the tac_fabr entries mount every gel at pos=0 rot=0, so the GEL LOCAL FRAME
    IS THE `*_tip` LINK FRAME. After `SensorAsset.reframed(R, t)` the gel lives
    in a rotated frame X, so the world transform of the reframed gel is
    `T_tip @ inv(X)` -- not `T_tip`.
  * in the tip-link frame the sensing face is the plane x = +16.71 mm with the
    object arriving from +x, and the face centre is (16.71, 0, 5.12) mm.
"""

from __future__ import annotations

import os.path as osp
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# sensing face, in the *_tip link frame (mm -> m below)
PAD_FACE_X = 16.71e-3
PAD_CENTRE = np.array([16.71e-3, 0.0, 5.12e-3])
PAD_NORMAL = np.array([1.0, 0.0, 0.0])          # points at the object

FINGER_CHAINS = {                                # tip link -> its actuated joints
    "index": ("link_3.0_tip", ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"]),
    "middle": ("link_7.0_tip", ["joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0"]),
    "ring": ("link_11.0_tip", ["joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"]),
    "thumb": ("link_15.0_tip", ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"]),
}


def rpy_to_R(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle_to_R(axis, th) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray                     # 4x4
    axis: np.ndarray
    lower: float = -np.pi
    upper: float = np.pi


@dataclass
class Link:
    name: str
    mesh: str = ""
    mesh_origin: np.ndarray = field(default_factory=lambda: np.eye(4))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))


class AllegroURDF:
    """Minimal URDF reader + FK/IK. Revolute and fixed joints only, which is all
    the Allegro description uses."""

    def __init__(self, urdf_path: str):
        self.path = urdf_path
        self.dir = osp.dirname(urdf_path)
        root = ET.parse(urdf_path).getroot()

        self.links: dict[str, Link] = {}
        for l in root.findall("link"):
            lk = Link(l.get("name"))
            v = l.find("visual")
            if v is not None:
                m = v.find("geometry/mesh")
                if m is not None:
                    lk.mesh = m.get("filename")
                    if m.get("scale"):
                        lk.scale = np.array([float(x) for x in m.get("scale").split()])
                o = v.find("origin")
                if o is not None:
                    T = np.eye(4)
                    if o.get("xyz"):
                        T[:3, 3] = [float(x) for x in o.get("xyz").split()]
                    if o.get("rpy"):
                        T[:3, :3] = rpy_to_R([float(x) for x in o.get("rpy").split()])
                    lk.mesh_origin = T
            self.links[lk.name] = lk

        self.joints: dict[str, Joint] = {}
        for j in root.findall("joint"):
            T = np.eye(4)
            o = j.find("origin")
            if o is not None:
                if o.get("xyz"):
                    T[:3, 3] = [float(x) for x in o.get("xyz").split()]
                if o.get("rpy"):
                    T[:3, :3] = rpy_to_R([float(x) for x in o.get("rpy").split()])
            ax = j.find("axis")
            lim = j.find("limit")
            self.joints[j.get("name")] = Joint(
                name=j.get("name"), type=j.get("type"),
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                origin=T,
                axis=np.array([float(x) for x in ax.get("xyz").split()]) if ax is not None
                else np.array([0.0, 0.0, 1.0]),
                lower=float(lim.get("lower")) if lim is not None and lim.get("lower")
                else -np.pi,
                upper=float(lim.get("upper")) if lim is not None and lim.get("upper")
                else np.pi)

        self.child_of = {j.child: j for j in self.joints.values()}
        roots = [n for n in self.links if n not in self.child_of]
        self.root = roots[0]

    # ------------------------------------------------------------------ FK
    def fk(self, q: dict[str, float] | None = None,
           base: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """World transform of every link."""
        q = q or {}
        out = {self.root: np.eye(4) if base is None else np.asarray(base, float)}
        pending = [n for n in self.links if n != self.root]
        # the description is shallow; a few sweeps resolve every parent
        for _ in range(len(self.links)):
            left = []
            for n in pending:
                j = self.child_of[n]
                if j.parent not in out:
                    left.append(n)
                    continue
                T = j.origin.copy()
                if j.type in ("revolute", "continuous"):
                    R = np.eye(4)
                    R[:3, :3] = axis_angle_to_R(j.axis, float(q.get(j.name, 0.0)))
                    T = T @ R
                out[n] = out[j.parent] @ T
            pending = left
            if not pending:
                break
        return out

    def limits(self, joints) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([self.joints[j].lower for j in joints]),
                np.array([self.joints[j].upper for j in joints]))

    # ------------------------------------------------------------------ IK
    def ik(self, finger: str, target: np.ndarray, q0=None, base=None,
           w_rot: float = 0.03, iters: int = 400):
        """Joint angles putting `finger`'s tip link at `target` (4x4).

        Position is in metres and rotation is dimensionless, so the rotation
        residual is scaled by `w_rot` (metres per radian-ish) or the solve is
        dominated by orientation and never closes the last 0.1 mm.
        """
        from scipy.optimize import least_squares

        tip, joints = FINGER_CHAINS[finger]
        lo, hi = self.limits(joints)
        x0 = np.clip(np.zeros(len(joints)) if q0 is None else np.asarray(q0, float), lo, hi)

        def res(x):
            T = self.fk(dict(zip(joints, x)), base=base)[tip]
            e_p = T[:3, 3] - target[:3, 3]
            e_R = T[:3, :3] @ target[:3, :3].T
            ang = np.array([e_R[2, 1] - e_R[1, 2], e_R[0, 2] - e_R[2, 0],
                            e_R[1, 0] - e_R[0, 1]]) * 0.5
            return np.concatenate([e_p, w_rot * ang])

        s = least_squares(res, x0, bounds=(lo, hi), max_nfev=iters)
        return dict(zip(joints, s.x)), float(np.linalg.norm(s.fun[:3]))

    # ------------------------------------------------- the grasp itself
    def oppose(self, fa: str = "index", fb: str = "thumb", gap: float = 12.1e-3,
               q0: dict | None = None, base=None, restarts: int = 24, seed: int = 0,
               gap_tol: float = 1.0e-3, lat_tol: float = 6.0e-3):
        """Angles making the two pads face each other `gap` apart, centres aligned.

        Returns (q, info). `gap` is measured between the two sensing FACES, so
        pass tile thickness + 2x the clearance.

        The residual weights are not free: position enters in metres (1e-2 scale)
        while the antiparallel term is dimensionless (up to 2), so weighting it
        like a length lets the solver trade 14 degrees of pad tilt for a
        fraction of a millimetre. A tilted pad against a FIXED tile does not
        self-align -- it contacts on one edge -- so orientation is weighted to
        dominate, and the problem is multi-started because the thumb chain has
        local minima it will otherwise sit in.
        """
        from scipy.optimize import least_squares

        ja = FINGER_CHAINS[fa][1]
        jb = FINGER_CHAINS[fb][1]
        joints = ja + jb
        lo, hi = self.limits(joints)
        x0 = np.array([(q0 or {}).get(j, 0.5 * (self.joints[j].lower + self.joints[j].upper))
                       for j in joints])
        x0 = np.clip(x0, lo, hi)

        def poses(x):
            T = self.fk(dict(zip(joints, x)), base=base)
            return T[FINGER_CHAINS[fa][0]], T[FINGER_CHAINS[fb][0]]

        def res(x):
            Ta, Tb = poses(x)
            ca = Ta[:3, :3] @ PAD_CENTRE + Ta[:3, 3]
            cb = Tb[:3, :3] @ PAD_CENTRE + Tb[:3, 3]
            na = Ta[:3, :3] @ PAD_NORMAL
            nb = Tb[:3, :3] @ PAD_NORMAL
            ya = Ta[:3, :3] @ np.array([0.0, 1.0, 0.0])
            yb = Tb[:3, :3] @ np.array([0.0, 1.0, 0.0])
            # scalar gap and scalar lateral distance, NOT the raw 3-vector
            # (cb - ca) - na*gap: with the vector form the solver can trade gap
            # against lateral inside one residual block and settles 37 mm off to
            # the side. Separating them, and weighting the gap x3, is what makes
            # every restart converge onto the actual pinch.
            dc = cb - ca
            along = float(np.dot(dc, na))
            return np.concatenate([
                [(along - gap) * 3.0],         # the gap: must be the tile thickness
                [np.linalg.norm(dc - along * na)],   # lateral offset, scalar
                0.02 * (na + nb),              # pads as antiparallel as the hand allows
                0.01 * (ya - yb),              # not twisted about the normal
            ])

        # Pick by a PHYSICAL criterion, not by the least-squares cost. The two
        # goals trade against each other (see the class docstring), and whichever
        # weighting is chosen, argmin(cost) lands on solutions that are numerically
        # good and physically useless -- e.g. pads perfectly antiparallel but 182 mm
        # apart, or a 2 mm gap with the thumb 37 mm off to the side. So: solve many
        # times, keep only the ones that actually form the pinch, and among those
        # take the flattest.
        rng = np.random.default_rng(seed)
        cands = []
        for k in range(max(1, restarts) + 1):
            xk = x0 if k == 0 else lo + rng.random(len(joints)) * (hi - lo)
            t = least_squares(res, xk, bounds=(lo, hi), max_nfev=3000)
            Ta, Tb = poses(t.x)
            ca = Ta[:3, :3] @ PAD_CENTRE + Ta[:3, 3]
            cb = Tb[:3, :3] @ PAD_CENTRE + Tb[:3, 3]
            na = Ta[:3, :3] @ PAD_NORMAL
            nb = Tb[:3, :3] @ PAD_NORMAL
            dc = cb - ca
            al = float(np.dot(dc, na))
            latv = float(np.linalg.norm(dc - al * na))
            tilt = float(np.degrees(np.arccos(np.clip(-np.dot(na, nb), -1, 1))))
            cands.append((abs(al - gap) < gap_tol and latv < lat_tol,
                          tilt, latv, al, t.x.copy(), t.cost))
        ok = [c for c in cands if c[0]]
        if not ok:                       # nothing formed the pinch: fall back to cost
            ok = sorted(cands, key=lambda c: c[5])[:1]
            print(f"[oppose] WARNING: no restart met gap+-{gap_tol*1e3:.1f} mm / "
                  f"lateral<{lat_tol*1e3:.1f} mm; falling back to lowest cost")
        best_c = min(ok, key=lambda c: c[1])
        s = type("R", (), {"x": best_c[4], "cost": best_c[5]})()
        q = dict(zip(joints, s.x))
        Ta, Tb = poses(s.x)
        ca = Ta[:3, :3] @ PAD_CENTRE + Ta[:3, 3]
        cb = Tb[:3, :3] @ PAD_CENTRE + Tb[:3, 3]
        na = Ta[:3, :3] @ PAD_NORMAL
        nb = Tb[:3, :3] @ PAD_NORMAL
        n_feasible = sum(1 for c in cands if c[0])
        info = dict(
            gap_mm=float(np.dot(cb - ca, na) * 1e3),
            lateral_mm=float(np.linalg.norm((cb - ca) - np.dot(cb - ca, na) * na) * 1e3),
            antiparallel_deg=float(np.degrees(np.arccos(np.clip(-np.dot(na, nb), -1, 1)))),
            centre=0.5 * (ca + cb), normal=na, up=Ta[:3, :3] @ np.array([0.0, 1.0, 0.0]),
            pose_a=Ta, pose_b=Tb, cost=float(s.cost), n_feasible=n_feasible,
            n_restarts=int(restarts) + 1)
        return q, info

    # ------------------------------------------------------------- meshes
    def link_mesh(self, name: str):
        """(verts (N,3) m, faces (F,3)) of a link's visual mesh, in link frame."""
        from .stl_io import read_stl

        lk = self.links[name]
        if not lk.mesh:
            return None
        p = lk.mesh
        if not osp.isabs(p):
            p = osp.join(self.dir, p.replace("package://allegro_hand_description/", ""))
        if not osp.exists(p):
            return None
        v, f = read_stl(p)
        v = v * lk.scale
        return v @ lk.mesh_origin[:3, :3].T + lk.mesh_origin[:3, 3], f


def gel_world_from_tip(T_tip: np.ndarray, reframe_R: np.ndarray,
                       reframe_t: np.ndarray) -> np.ndarray:
    """World transform for a gel that was `reframed(reframe_R, reframe_t)`.

    The fabrication mounts the gel at the tip link with zero offset, so the
    un-reframed gel's world transform is exactly T_tip. Reframing moved the mesh
    by X: p' = R p + t, so the reframed body sits at T_tip @ inv(X).
    """
    X = np.eye(4)
    X[:3, :3] = reframe_R
    X[:3, 3] = reframe_t
    return T_tip @ np.linalg.inv(X)
