"""Joint-driven parallel gripper with a commercial-SDK-style control API.

The earlier gripper demo drove Fixed carriers kinematically (teleport). This
module replaces that with the engine's REAL articulation: every moving part is
a free ABD body on a prismatic joint, so position control, speed control,
force limits and tactile-in-the-loop grasping (SAFE mode) are all closed
through the physics rather than scripted.

Rig (all bodies loaded by ``build()``, joints before finalize)::

        anchor  (Fixed)                 the gripper's mount point
          |  vertical prismatic (lift)
        carriage (free)                 z-bar carrying both jaws
          |  horizontal prismatic x2 (closing, one per jaw)
        jawL / jawR (free)              stitch-carriers for the FEM pads
          ~  stitch springs
        gel pads (FEM)                  the tactile membranes

Control vocabulary (mirrors commercial gripper SDKs, vendor-neutral):

    POSITION  set_position(opening_mm, vmax, fmax): servo the opening; the
              command is rate-limited to vmax and STOPS ADVANCING while the
              measured drive force exceeds fmax (force-capped position move).
    SPEED     set_speed(v_mm_s, fmax): closing speed command, +v closes.
    SAFE      grip-force regulation from the tactile pads: an integral
              controller drives the pad normal force to ``grip_force`` and a
              slip monitor raises the setpoint when the grasp slips.

    get_gripper_status() -> {position, velocity, force, temperature}
    open_gripper() / close_gripper() / mode(...) context manager /
    set_control_param(...) / calibrate() / release() / set_led_color(...)

Call ``update()`` once per frame BEFORE engine.step() — an unconfigured
prismatic joint runs a default driving state, so every joint is written every
frame (hard-won rule).
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum

import numpy as np


def _box(size, centre):
    sx, sy, sz = size
    v = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2)
                  for z in (-sz / 2, sz / 2)], float) + np.asarray(centre, float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
                 np.int32)
    c = v.mean(0)
    for i, t in enumerate(f):
        p = v[t]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p.mean(0) - c) < 0:
            f[i] = t[::-1]
    return v, f


class ControlMode(Enum):
    POSITION = 1
    SPEED = 2
    SAFE = 3


class ParallelGripper:
    """Two-jaw parallel gripper articulated through engine prismatic joints.

    Openings are in mm of pad-face gap; 0 = faces touching. Build phase must
    run before ``engine.finalize()``; pads are stitched by the caller to
    ``jaw_recs`` and registered with ``attach_pads``.
    """

    ControlMode = ControlMode

    def __init__(self, engine, grasp_centre, opening0_mm: float,
                 pad_thickness: float = 3e-3, dt: float = 0.005,
                 carrier_back_off: float = 0.05, max_lift_mm: float = 50.0):
        self.eng = engine
        self.centre = np.asarray(grasp_centre, float)
        self.op0 = opening0_mm * 1e-3           # build-time opening (m)
        self.pad_t = pad_thickness
        self.dt = dt
        self.back = carrier_back_off
        self.max_lift = max_lift_mm * 1e-3
        self.sides = {"left": -1, "right": +1}

        # ------- control state -------
        self.mode_ = ControlMode.POSITION
        self.cmd_opening = self.op0             # commanded opening (m)
        self.cmd_speed = 0.0                    # closing speed (m/s, + closes)
        self.vmax = 80e-3                       # m/s
        self.fmax = 27.0                        # N
        self.grip_force = 5.0                   # SAFE setpoint per pad (N)
        self.kp, self.ki = 0.0, 1.2e-5          # SAFE: m per (N·frame)
        self.stiffness = 25.0                   # joint servo strength
        self.lift_target = 0.0                  # m
        self.lift_stiffness = 40.0
        self.lift_vmax = 80e-3                  # m/s, independent of the jaws
        self.led = (0, 0, 0)
        self._targets = {n: 0.0 for n in self.sides}   # per-jaw close distance
        self._lift_t = 0.0
        self._int_err = 0.0
        self._pads = {}
        self._feedback = None
        self._slip_accum = 0.0
        self._events = []
        self._prev_opening = None

    # ------------------------------------------------------------ build phase
    def build(self):
        """Load anchor + carriage + jaws and add the three joints.
        Returns {"left": rec, "right": rec} of the jaw carrier bodies."""
        e = self.eng
        cy = self.centre[1]
        self.jaw_z0 = {n: s * (self.op0 / 2 + self.pad_t + self.back)
                       for n, s in self.sides.items()}

        av, af = _box((0.02, 0.008, 0.02), self.centre + [0.0, 0.10, 0.0])
        e.load_mesh_from_data(av, af, verts_per_face=3, dimensions=3,
                              body_type="ABD", transform=np.eye(4),
                              young_modulus=1e7, boundary_type="Fixed")
        self.anchor = e.get_load_records()[-1]

        cv, cf = _box((0.016, 0.008, 0.09), self.centre + [0.0, 0.065, 0.0])
        e.load_mesh_from_data(cv, cf, verts_per_face=3, dimensions=3,
                              body_type="ABD", transform=np.eye(4),
                              young_modulus=1e7)
        self.carriage = e.get_load_records()[-1]

        self.jaw_recs, self._jaw_v0 = {}, {}
        for n, s in self.sides.items():
            c = self.centre + np.array([0.0, 0.0, self.jaw_z0[n]])
            v, f = _box((0.012, 0.012, 0.012), c)
            e.load_mesh_from_data(v, f, verts_per_face=3, dimensions=3,
                                  body_type="ABD", transform=np.eye(4),
                                  young_modulus=1e7)
            self.jaw_recs[n] = e.get_load_records()[-1]
            self._jaw_v0[n] = v[0].copy()

        a_i, c_i = self.anchor.body_offset, self.carriage.body_offset
        self.j_lift = e.add_prismatic_joint(
            a_i, c_i, world_center=(self.centre + [0.0, 0.065, 0.0]).tolist(),
            world_axis=[0.0, 1.0, 0.0], lower_limit=0.0,
            upper_limit=self.max_lift)
        self.j_close = {}
        for n, s in self.sides.items():
            c = self.centre + np.array([0.0, 0.0, self.jaw_z0[n]])
            self.j_close[n] = e.add_prismatic_joint(
                c_i, self.jaw_recs[n].body_offset, world_center=c.tolist(),
                world_axis=[0.0, 0.0, -float(s)], lower_limit=0.0,
                upper_limit=self.op0 / 2 + 1e-3)
        return self.jaw_recs

    def jaw_anchor(self, name):
        """(global anchor vertex id, its build-time world position) for
        stitching a pad to jaw `name`."""
        r = self.jaw_recs[name]
        return r.vertex_offset, self._jaw_v0[name]

    def post_finalize(self):
        """Cache canonical rest transforms (call right after finalize)."""
        ids = [self.carriage.body_offset] + \
              [self.jaw_recs[n].body_offset for n in self.sides]
        self._ids = np.array(ids, np.int32)
        C0 = self.eng.get_abd_body_transforms(self._ids).copy()
        self._C0inv = np.stack([np.linalg.inv(c) for c in C0])

    def attach_pads(self, pads: dict, pad_tf0: dict):
        """pads: {"left": GelBody, ...}; pad_tf0: build-time pad world poses."""
        self._pads = pads
        self._pad_tf0 = {n: np.asarray(t) for n, t in pad_tf0.items()}

    def bind_feedback(self, fn):
        """fn() -> dict with per-pad normal force 'fn_left'/'fn_right' (N) and
        'slip_mm' (grasp slip increment since last call). SAFE mode input."""
        self._feedback = fn

    # ---------------------------------------------------------- kinematics
    def rigid_motion(self, k: int) -> np.ndarray:
        """Canonical-frame rigid motion of body k in [carriage, jawL, jawR].
        (get_abd_body_transforms returns the CANONICAL frame, not the load
        pose — compose, never read it as a pose.)"""
        Cn = self.eng.get_abd_body_transforms(self._ids)
        return Cn[k] @ self._C0inv[k]

    def pad_tf(self, name: str) -> np.ndarray:
        """Current world pose of pad `name` (rides its jaw)."""
        k = 1 + list(self.sides).index(name)
        return self.rigid_motion(k) @ self._pad_tf0[name]

    def opening_mm(self) -> float:
        d = [self.eng.get_prismatic_current_distance(self.j_close[n])
             for n in self.sides]
        return (self.op0 - sum(d)) * 1e3

    def lift_mm(self) -> float:
        return self.eng.get_prismatic_current_distance(self.j_lift) * 1e3

    # ---------------------------------------------------------- control API
    def set_position(self, opening_mm: float, vmax: float = 80.0,
                     fmax: float = 27.0):
        """Non-blocking position command (opening in mm; vmax mm/s; fmax N)."""
        self.mode_ = ControlMode.POSITION
        self.cmd_opening = np.clip(opening_mm, 0.0, self.op0 * 1e3) * 1e-3
        self.vmax = vmax * 1e-3
        self.fmax = fmax

    def set_speed(self, v_mm_s: float, fmax: float = 27.0):
        """Speed command: +v closes, -v opens, 0 stops."""
        self.mode_ = ControlMode.SPEED
        self.cmd_speed = v_mm_s * 1e-3
        self.fmax = fmax

    def open_gripper(self):
        self.set_position(self.op0 * 1e3)

    def close_gripper(self):
        self.set_position(0.0)

    def set_grip_force(self, newtons: float):
        """SAFE-mode per-pad normal-force setpoint."""
        self.grip_force = float(newtons)

    def set_control_param(self, stiffness: float | None = None,
                          kp: float | None = None, ki: float | None = None,
                          kd: float | None = None):
        if stiffness is not None:
            self.stiffness = stiffness
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        # kd accepted for signature parity; the integral law doesn't use it

    @contextmanager
    def mode(self, m: ControlMode):
        prev = self.mode_
        self.mode_ = m
        try:
            yield self
        finally:
            self.mode_ = prev

    def enable_mode(self, m: ControlMode):
        self.mode_ = m

    def disable_mode(self):
        self.mode_ = ControlMode.POSITION

    def set_lift(self, lift_mm: float):
        """Vertical carriage target (sim extra: the wrist the real gripper
        gets from its robot arm)."""
        self.lift_target = np.clip(lift_mm * 1e-3, 0.0, self.max_lift)

    def set_led_color(self, r: int, g: int, b: int):
        self.led = (int(r) & 255, int(g) & 255, int(b) & 255)

    def calibrate(self):
        """Reset command state to the current measured opening."""
        self.cmd_opening = self.opening_mm() * 1e-3
        self._int_err = 0.0
        self._slip_accum = 0.0

    def release(self):
        self._pads = {}
        self._feedback = None

    def _measured_force(self, fb: dict | None = None) -> float:
        """Grip force in N. With tactile pads bound this is the pad normal
        force (the gripper's own sensors — exact). Without them, estimate
        from servo lag: the engine applies K_eff ≈ strength/dt² per metre of
        (target − d) lag, verified on the A800 probe (186 N at 0.141 mm lag,
        strength 25, dt 5 ms). NOTE get_prismatic_drive_force() itself uses
        the raw strength as K and under-reads by ~1/dt² — engine-fix candidate."""
        if fb is None and self._feedback is not None:
            fb = self._feedback()
        if fb and ("fn_left" in fb or "fn_right" in fb):
            return max(fb.get("fn_left", 0.0), fb.get("fn_right", 0.0))
        k_eff = self.stiffness / self.dt ** 2
        return max(k_eff * max(0.0, self._targets[n]
                               - self.eng.get_prismatic_current_distance(
                                   self.j_close[n]))
                   for n in self.sides)

    def get_gripper_status(self) -> dict:
        op = self.opening_mm()
        vel = 0.0 if self._prev_opening is None else \
            (op - self._prev_opening) / self.dt
        return dict(position=op, velocity=vel, force=self._measured_force(),
                    temperature=25.0)

    # ------------------------------------------------------------- per-frame
    def update(self) -> dict:
        """Write every joint's drive state for this frame (call before
        engine.step()). Returns the controller's telemetry."""
        e = self.eng
        op_now = self.opening_mm() * 1e-3
        fb = self._feedback() if self._feedback else {}
        f_meas = self._measured_force(fb)
        info = dict(mode=self.mode_.name, opening_mm=op_now * 1e3,
                    drive_N=f_meas, setpoint_N=self.grip_force,
                    slip_mm=fb.get("slip_mm", 0.0), event="")

        if self.mode_ is ControlMode.POSITION:
            err = self.cmd_opening - op_now                  # + means open up
            step = np.clip(-err, -self.vmax * self.dt, self.vmax * self.dt)
            if f_meas > self.fmax and step > 0:              # force-capped
                step = 0.0
                info["event"] = "fmax"
            for n in self.sides:
                self._targets[n] = np.clip(self._targets[n] + step / 2,
                                           0.0, self.op0 / 2)
        elif self.mode_ is ControlMode.SPEED:
            step = np.clip(self.cmd_speed, -self.vmax, self.vmax) * self.dt
            if f_meas > self.fmax and step > 0:
                step = 0.0
                info["event"] = "fmax"
            for n in self.sides:
                self._targets[n] = np.clip(self._targets[n] + step / 2,
                                           0.0, self.op0 / 2)
        elif self.mode_ is ControlMode.SAFE:
            fn = 0.5 * (fb.get("fn_left", 0.0) + fb.get("fn_right", 0.0))
            slip = fb.get("slip_mm", 0.0)
            self._slip_accum += max(slip, 0.0)
            if self._slip_accum > 0.25:                      # slip event
                self.grip_force = min(self.grip_force * 1.6, self.fmax)
                self._slip_accum = 0.0
                info["event"] = f"slip->regrip {self.grip_force:.1f}N"
                self._events.append((len(self._events), info["event"]))
            err = self.grip_force - fn                       # N
            self._int_err = np.clip(self._int_err + err, -400.0, 400.0)
            step = np.clip(self.kp * err + self.ki * err,
                           -0.35e-3, 0.12e-3)                # m, per frame
            if fn < 0.3:                                     # not touching yet
                step = max(step, 0.10e-3)                    # coarse approach
            for n in self.sides:
                self._targets[n] = np.clip(self._targets[n] + step,
                                           0.0, self.op0 / 2)
            info["fn_N"] = fn
            info["setpoint_N"] = self.grip_force

        # lift axis: rate-limited toward its target (own speed limit)
        derr = np.clip(self.lift_target - self._lift_t,
                       -self.lift_vmax * self.dt, self.lift_vmax * self.dt)
        self._lift_t += derr

        for n in self.sides:
            j = self.j_close[n]
            e.native.set_prismatic_force(j, 0.0)
            e.native.set_prismatic_strength(j, self.stiffness)
            e.native.set_prismatic_target(j, float(self._targets[n]))
        e.native.set_prismatic_force(self.j_lift, 0.0)
        e.native.set_prismatic_strength(self.j_lift, self.lift_stiffness)
        e.native.set_prismatic_target(self.j_lift, float(self._lift_t))

        self._prev_opening = op_now * 1e3
        return info
