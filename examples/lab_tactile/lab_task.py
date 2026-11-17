"""Task sequencer for the lab demo: stages -> segments -> per-frame TCP pose
and gripper commands.

Stage geometry is computed lazily from the live scene when the stage starts,
and the alignment-critical segments are *relative* moves whose target is
measured at segment start from the held object's actual pose (tip position,
mouth position, rim point).  That is what makes insertions and pours robust
to the object having shifted or tilted in the soft tactile pads.

In ``--dry`` (kinematics only) runs objects do not follow the gripper, so
every measured quantity falls back to its planned value.
"""

from __future__ import annotations

import os
import numpy as np

from fr3_kin import axis_angle_R, rot_log

EY = np.array([0.0, 1.0, 0.0])
GRIP_CARRY = 6.0          # N per pad while carrying (the estimate uses a softer ramp first)


def minjerk(s):
    s = float(np.clip(s, 0.0, 1.0))
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def slerp_R(R0, R1, s):
    w = rot_log(R0.T @ R1)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return R0.copy()
    return R0 @ axis_angle_R(w / th, th * s)


def lerp_T(T0, T1, s):
    u = minjerk(s)
    T = np.eye(4)
    T[:3, :3] = slerp_R(T0[:3, :3], T1[:3, :3], u)
    T[:3, 3] = (1 - u) * T0[:3, 3] + u * T1[:3, 3]
    return T


def top_down_R(close_axis="z", flip=False):
    """Tool z = down, tool y (finger closing axis) = world x or z."""
    z = -EY
    y = np.array([0.0, 0.0, 1.0]) if close_axis == "z" else np.array([1.0, 0.0, 0.0])
    if flip:
        y = -y
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def T_of(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def tilt_R(R, deg, axis="x"):
    """Rotate the tool frame about its own x (or y) axis."""
    a = [1, 0, 0] if axis == "x" else [0, 1, 0]
    return R @ axis_angle_R(a, np.radians(deg))


def best_top_down(scene, close_axis, pos, extra_R=None):
    """Top-down grasp rotation (closing along world x or z) choosing the yaw
    flip (fingers swapped) that the arm reaches with the smaller joint motion."""
    best = None
    for flip in (False, True):
        R = top_down_R(close_axis, flip)
        if extra_R is not None:
            R = R @ extra_R
        for seed in (scene.q, scene.arm.HOME):
            q, ok = scene.arm.ik(T_of(R, pos), seed, iters=120)
            cost = (np.linalg.norm(q - scene.q) + (0.0 if ok else 10.0))
            if best is None or cost < best[0]:
                best = (cost, R, ok)
    if not best[2]:
        print(f"[task] WARN: top-down grasp at {np.round(pos, 3)} not reachable")
    return best[1]


def side_grasp_R(obj_xz, target_xz, base_xz=(0.0, 0.0), pitch_deg=0.0, flip=False):
    """Side grasp for pouring: tool z = horizontal approach (hand on the
    robot's side of the object), tool y (finger closing axis) = horizontal
    line from the object to the pour target, tool x = up.  Pouring is then a
    roll about tool z that tips the vessel's mouth along +/-y toward the
    target, and the hand's wide dimension dips only on the far side.
    Returns (R, s) with s = +1/-1 the sign that maps +y onto the target."""
    d = np.array([target_xz[0] - obj_xz[0], 0.0, target_xz[1] - obj_xz[1]], float)
    d /= np.linalg.norm(d)
    z0 = np.array([d[2], 0.0, -d[0]])              # horizontal, perpendicular to d
    o = np.array([obj_xz[0] - base_xz[0], 0.0, obj_xz[1] - base_xz[1]])
    s = 1.0 if float(z0 @ o) > 0 else -1.0         # hand between base and object
    z = s * z0
    y = s * d
    x = np.cross(y, z)                             # = up
    R = np.stack([x, y, z], axis=1)
    if pitch_deg:                                  # tilt the approach axis downward (about tool y)
        R = R @ axis_angle_R([0, 1, 0], np.radians(-pitch_deg))
    if flip:                                       # fingers swapped (180 deg about tool z)
        R = R @ axis_angle_R([0, 0, 1], np.pi)
        s = -s
    return R, s, z0


def roll_for_tilt(pitch_deg, tilt_deg):
    """Wrist roll (about the pitched tool z) that tilts the vessel's vertical
    axis by tilt_deg.  With alpha = 90 - pitch the angle between the vessel
    axis and the roll axis:  cos(tilt) = cos^2(a) + sin^2(a) cos(roll).
    Returns None when the pitch cannot reach that tilt (needs 2*alpha >= tilt)."""
    a = np.radians(90.0 - pitch_deg)
    c = (np.cos(np.radians(tilt_deg)) - np.cos(a) ** 2) / max(np.sin(a) ** 2, 1e-9)
    if c < -1.0 + 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pour_R(R, sgn, roll_deg):
    """Tip the vessel toward the pour target by rolling the tool about its own
    (pitched) z axis.  Rolling about the pitched axis keeps the hand's 200 mm
    width from swinging fully vertical (a horizontal pour axis dropped the far
    end of the hand beside the beaker); the roll angle comes from
    roll_for_tilt so the vessel still reaches the requested tilt."""
    return R @ axis_angle_R([0, 0, 1], np.radians(sgn * roll_deg))


def choose_side_variant(scene, obj_xz, target_xz, make_key_poses, tilt_deg,
                        pitches=(0.0, 15.0, 25.0, 35.0), avoid=(), clearance=0.04):
    """Pick (pitch, flip) for a side grasp by the stage family-planning cost
    (reachability + joint-limit margins along all key poses, under the
    engine's clamped joint limits).  ``avoid``: prop names the arm must keep ``clearance`` (m, collision-mesh
    vertex distance) away from at every key pose -- variants tie on the plan cost, and a 5 mm shift of the
    carried beaker flipped the stage-4 choice to a pose whose link6 knocked the flask over (full5).
    Returns (R, s, family_q)."""
    from run_lab_demo import plan_family as _pf
    lvl = scene.link_vertices_local()
    links = [n for n in ("fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_link7", "fr3_link6", "fr3_link5")
             if n in lvl]

    def min_height(q):
        lp = scene.arm.link_poses(q)
        return min(float((lvl[n] @ lp[n][:3, :3].T + lp[n][:3, 3])[:, 1].min()) for n in links)

    obst = None
    if avoid:
        from scipy.spatial import cKDTree
        pts = []
        for n in avoid:
            if n in scene.props:
                T = scene.prop_pose(n)
                pts.append(np.asarray(scene.props[n].verts) @ T[:3, :3].T + T[:3, 3])
        if pts:
            obst = cKDTree(np.vstack(pts))
    arm_links = [n for n in ("fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7", "fr3_hand", "fr3_leftfinger", "fr3_rightfinger")
                 if n in lvl]

    def min_clear(q):
        if obst is None:
            return np.inf
        lp = scene.arm.link_poses(q)
        return min(float(obst.query(lvl[n] @ lp[n][:3, :3].T + lp[n][:3, 3])[0].min()) for n in arm_links)

    best = None
    for pitch in pitches:
        roll = roll_for_tilt(pitch, tilt_deg)
        if roll is None:                                  # this pitch cannot tilt the vessel enough
            continue
        for flip in (False, True):
            R, sg, z0 = side_grasp_R(obj_xz, target_xz, pitch_deg=pitch, flip=flip)
            qs, cost = _pf(scene, make_key_poses(R, sg, roll), scene.q)
            h = min(min_height(q) for q in qs)
            cost += max(0.0, 0.035 - h) * 500.0          # keep the wrist >= 35 mm above the table
            d = min(min_clear(q) for q in qs)
            cost += max(0.0, clearance - d) * 500.0      # keep the arm clear of the other props
            if os.environ.get("LAB_PLAN_DEBUG"):
                print(f"[task]   variant pitch {pitch:.0f} flip {flip}: cost {cost:.3f}, min height {h*1e3:.0f} mm, clearance {d*1e3:.0f} mm")
            if best is None or cost < best[0]:
                best = (cost, R, sg, roll, qs, pitch, flip, h, d)
    print(f"[task] side grasp: pitch {best[5]:.0f} deg flip {best[6]}, roll {best[3]:.0f} deg for "
          f"{tilt_deg:.0f} deg tilt (plan cost {best[0]:.2f}, min robot height {best[7]*1e3:.0f} mm, "
          f"min clearance to {list(avoid)} {best[8]*1e3:.0f} mm)")
    return best[1], best[2], best[3], best[4]


def roll_R(R, deg):
    """Rotate the tool frame about its own z axis (wrist roll)."""
    return R @ axis_angle_R([0, 0, 1], np.radians(deg))


def pick_tilt_sign(scene, R, p_pivot_full, r_tool, deg, axis="y"):
    """+1 or -1: the tilt direction whose full-tilt pose the arm reaches with
    the smaller joint motion (both are checked at the mid and full angle)."""
    best = None
    for sgn in (1.0, -1.0):
        cost, ok_all = 0.0, True
        q = np.array(scene.q, float)
        for frac in (0.5, 1.0):
            Rt = tilt_R(R, sgn * deg * frac, axis)
            T = T_of(Rt, p_pivot_full - Rt @ r_tool)
            q2, ok = scene.arm.ik(T, q, iters=120)
            cost += np.linalg.norm(q2 - q) + (0.0 if ok else 10.0)
            ok_all &= ok
            q = q2
        if best is None or cost < best[0]:
            best = (cost, sgn, ok_all)
    if not best[2]:
        print(f"[task] WARN: pour tilt {deg} deg about tool {axis} not fully reachable")
    return best[1]


def plan_family(scene, key_poses):
    from run_lab_demo import plan_family as _pf
    qs, cost = _pf(scene, key_poses, scene.q)
    return qs


def tcp_now(scene) -> np.ndarray:
    return scene.cmd_pose if scene.is_dry else scene.tcp_pose()


def to_tool(scene, p_world) -> np.ndarray:
    """World point -> current tool (TCP) coordinates."""
    T = tcp_now(scene)
    return T[:3, :3].T @ (np.asarray(p_world, float) - T[:3, 3])


class Segment:
    def __init__(self, label, duration, pose_fn, grip=None, on_start=None, on_end=None, tag="",
                 q_fn=None):
        self.label, self.duration, self.pose_fn = label, duration, pose_fn
        self.grip, self.on_start, self.on_end, self.tag = grip, on_start, on_end, tag
        self.q_fn = q_fn                     # optional joint-space trajectory s -> q(7,)
        self.q_hint = None                   # stage-family joint solution for the end pose
        self.joint_space = False


class Timeline:
    def __init__(self, stage):
        self.stage = stage
        self.segs: list[Segment] = []

    def add(self, label, dur, pose_fn, grip=None, on_start=None, on_end=None, tag="", q_fn=None,
            q_hint=None):
        self.segs.append(Segment(label, dur, pose_fn, grip, on_start, on_end, tag, q_fn))
        self.segs[-1].q_hint = q_hint
        return self

    def move(self, label, dur, T_to, grip=None, joint_space=False, **kw):
        """Absolute move from the commanded pose at segment start to T_to.
        joint_space=True: the runner interpolates joints to a well-conditioned
        IK solution of T_to instead of tracking the straight Cartesian line
        (used for the free-space approach at the start of each stage, so the
        arm enters every stage in a good configuration family)."""
        st = {}
        T_to = np.asarray(T_to)
        user_start = kw.pop("on_start", None)

        def start(scene):
            st["T0"] = scene.cmd_pose.copy()
            if user_start:
                user_start(scene)

        self.add(label, dur, lambda s: lerp_T(st["T0"], T_to, s), grip, on_start=start, **kw)
        self.segs[-1].joint_space = joint_space
        return self

    def hint(self, q):
        """Attach a stage-family joint solution to the last added segment."""
        self.segs[-1].q_hint = None if q is None else np.array(q, float)
        return self

    def move_rel(self, label, dur, delta_fn, grip=None, **kw):
        """Translate by delta_fn(scene) (world, evaluated at segment start)."""
        st = {}
        user_start = kw.pop("on_start", None)

        def start(scene):
            st["T0"] = scene.cmd_pose.copy()
            st["d"] = np.asarray(delta_fn(scene), float)
            if user_start:
                user_start(scene)

        def fn(s):
            T = st["T0"].copy()
            T[:3, 3] = st["T0"][:3, 3] + minjerk(s) * st["d"]
            return T

        self.add(label, dur, fn, grip, on_start=start, **kw)
        return self

    def dyn(self, label, dur, make_fn, grip=None, **kw):
        """Pose function built at segment start: make_fn(scene) -> fn(s)."""
        st = {}
        user_start = kw.pop("on_start", None)

        def start(scene):
            st["fn"] = make_fn(scene)
            if user_start:
                user_start(scene)

        self.add(label, dur, lambda s: st["fn"](s), grip, on_start=start, **kw)
        return self

    def hold(self, label, dur, grip=None, **kw):
        st = {}
        user_start = kw.pop("on_start", None)

        def start(scene):
            st["T"] = scene.cmd_pose.copy()
            if user_start:
                user_start(scene)

        self.add(label, dur, lambda s: st["T"], grip, on_start=start, **kw)
        return self


# ------------------------------------------------------------ small helpers

def _above(T, h):
    T2 = T.copy()
    T2[:3, 3] = T[:3, 3] + np.array([0, h, 0])
    return T2


def centre_over(xz, getter):
    """delta that puts getter(scene) (a world point) over the xz target."""
    def f(sc):
        if sc.is_dry:
            return np.zeros(3)
        p = getter(sc)
        return np.array([xz[0] - p[0], 0.0, xz[1] - p[2]])
    return f


def descend_until(getter, y_target, planned_drop):
    """delta that brings getter(scene)[1] to y_target (dry: fixed planned drop)."""
    def f(sc):
        if sc.is_dry:
            return np.array([0.0, -planned_drop, 0.0])
        return np.array([0.0, y_target - getter(sc)[1], 0.0])
    return f


def up(h):
    return lambda sc: np.array([0.0, h, 0.0])


def est_hooks(est):
    return dict(on_start=lambda sc: sc.start_estimation(est), on_end=lambda sc: sc.finish_estimation())


# ---------------------------------------------------------------- stages

class Stage:
    name = "stage"

    def build(self, scene) -> Timeline:
        raise NotImplementedError


class PreSettle(Stage):
    """Arm from the load pose to home (joint space), fingers open, objects settle."""
    name = "0 准备 / settle"

    def build(self, scene):
        tl = Timeline(self)
        q0 = np.array(scene.q, float)
        q1, ok = scene.arm.ik(scene.home_pose, scene.arm.HOME, iters=80)
        if not ok:
            print("[task] WARN: home pose IK did not converge")

        def q_fn(s):
            return q0 + minjerk(s) * (q1 - q0)

        tl.add("to home (joint space)", 3.0, lambda s: scene.arm.fk(q_fn(s)),
               grip=dict(mode="pos", gap_mm=60, vmax_mm_s=60), q_fn=q_fn)
        tl.hold("settle", 0.8)
        return tl


class DropperStage(Stage):
    name = "1 滴管取液 / dropper"

    def build(self, scene):
        tl = Timeline(self)
        bc = np.array(scene.bulb_centre(), float)
        seat_y = bc[1]                                   # bulb height when resting on the neck
        p_g = bc + np.array([0.0, 0.007, 0.0])
        R = best_top_down(scene, "x", p_g)
        Tg = T_of(R, p_g)
        dm = scene.dropper["meta"]
        L_tip = dm["bulb_cy"]                            # tip is this far below the bulb centre
        est = scene.new_estimator("dropper bulb", dm["bulb_r"], e_true=dm["E_bulb"])
        Tf = scene.prop_pose("flask")
        flask_xz = Tf[:3, 3][[0, 2]]
        rim_flask = Tf[1, 3] + scene.props["flask"].meta["height"]
        beaker = scene.prop_pose("beaker_mix")[:3, 3]
        rim_y = beaker[1] + scene.props["beaker_mix"].meta["height"]
        lift = 0.14
        T_over_beaker = T_of(R, [beaker[0], rim_y + 0.07 + L_tip + 0.007, beaker[2]])

        def tip(sc):
            return sc.dropper_tip()

        def bulb(sc):
            return sc.bulb_centre()

        def seat_delta(sc):                              # bulb over the neck axis, 3 mm above its seat
            if sc.is_dry:
                return np.array([0.0, -(lift - 0.003 - (bc[1] + lift - L_tip - (rim_flask - 0.03))), 0.0])
            b = bulb(sc)
            return np.array([flask_xz[0] - b[0], seat_y + 0.003 - b[1], flask_xz[1] - b[2]])

        tip0_above = bc[1] + lift - L_tip                # planned tip height above the flask
        tl.move("approach flask", 2.0, _above(Tg, 0.13), grip=dict(mode="pos", gap_mm=40), joint_space=True)
        tl.move("descend to bulb", 1.6, Tg)
        tl.hold("grasp bulb (tactile)", 2.2, grip=dict(mode="safe", force_N=2.5), **est_hooks(est))
        tl.hold("secure grip", 0.8, grip=dict(mode="safe", force_N=GRIP_CARRY))
        tl.hold("aspirate", 2.5, grip=dict(mode="safe", force_N=GRIP_CARRY), tag="aspirate")   # engine liquid: pump draws water up the bore (~25 particles = 8 drops)
        tl.move("lift dropper", 2.6, _above(Tg, lift), tag="hold")                              # "hold": pump cancels gravity on the column
        tl.move("to mixing beaker", 3.2, T_over_beaker, tag="hold")
        tl.hold("settle", 1.2, tag="hold")
        tl.move_rel("centre tip over beaker", 1.0, centre_over(beaker[[0, 2]], tip), tag="hold")
        tl.move_rel("lower over beaker", 1.4, descend_until(tip, rim_y + 0.025, 0.045), tag="hold")
        tl.hold("squeeze bulb (drip)", 2.4, grip=dict(mode="safe", force_N=9.0), tag="drip")   # pulsed squeeze -> drops
        tl.hold("release squeeze", 1.0, grip=dict(mode="safe", force_N=GRIP_CARRY))
        tl.move_rel("raise", 1.4, up(0.06))
        tl.move("back to flask", 3.2, _above(Tg, lift))
        tl.hold("settle", 1.5)
        tl.move_rel("centre tip over neck", 1.2, centre_over(flask_xz, tip))
        tl.move_rel("enter neck", 2.0, descend_until(tip, rim_flask - 0.03, tip0_above - (rim_flask - 0.03)))
        tl.move_rel("seat bulb on neck", 2.4, seat_delta)
        tl.hold("release dropper", 1.0, grip=dict(mode="pos", gap_mm=40))
        tl.move_rel("retreat", 1.2, up(0.12))
        return tl


class TestTubeStage(Stage):
    name = "2 试管倾倒 / test tube"
    TILT = 100.0                                  # vessel tilt from vertical (mouth well below horizontal)

    def build(self, scene):
        tl = Timeline(self)
        Tt = scene.prop_pose("test_tube")
        L, r = scene.props["test_tube"].meta["length"], scene.props["test_tube"].meta["r_out"]
        bottom0 = Tt[:3, 3].copy()
        top = bottom0 + Tt[:3, :3] @ np.array([0, L, 0])
        beaker = scene.prop_pose("beaker_mix")[:3, 3]
        rim_y = beaker[1] + scene.props["beaker_mix"].meta["height"]
        p_g = top + np.array([0, -0.012, 0])          # pads 12 mm below the mouth
        TH = self.TILT
        H0, H1 = 0.13, 0.045                          # mouth height above the rim: upright -> poured
        lift, back = 0.13, 0.08

        def key_poses(R, sg, roll):
            z_t = R[:, 2]
            rm = R.T @ (top - p_g)
            Rf = pour_R(R, sg, roll)
            pm0 = np.array([beaker[0], rim_y + H0, beaker[2]])
            pm1 = np.array([beaker[0], rim_y + H1, beaker[2]])
            Tg_ = T_of(R, p_g)
            T_side_ = T_of(R, p_g - back * z_t)
            return [T_of(R, p_g - back * z_t + np.array([0, 0.05, 0])), T_side_, Tg_, _above(Tg_, lift),
                    T_of(R, pm0 - R @ rm), T_of(Rf, pm1 - Rf @ rm), T_of(R, pm0 - R @ rm),
                    _above(Tg_, lift), Tg_, T_side_, _above(T_side_, 0.10)]

        R, sgn, ROLL, fam = choose_side_variant(scene, top[[0, 2]], beaker[[0, 2]], key_poses, TH)
        z_t = R[:, 2]
        Tg = T_of(R, p_g)
        est = scene.new_estimator("test tube (glass)", r, e_true=None)
        slot_xz = scene.prop_T["test_tube"][:3, 3][[0, 2]]

        def bottom(sc):
            return sc.prop_pose("test_tube")[:3, 3]

        def mouth_w(sc):
            T = sc.prop_pose("test_tube")
            return T[:3, 3] + T[:3, :3] @ np.array([0, L, 0])

        rm_planned = R.T @ (top - p_g)                # mouth in tool coords (planned)

        def mouth_tool(sc):
            return rm_planned if sc.is_dry else to_tool(sc, mouth_w(sc))

        def p_mouth(u):
            return np.array([beaker[0], rim_y + H0 + (H1 - H0) * u, beaker[2]])

        def pour(forward):
            def make(sc):
                rm = mouth_tool(sc)

                def fn(s):
                    u = minjerk(s) if forward else 1.0 - minjerk(s)
                    Rt = pour_R(R, sgn, ROLL * u)
                    return T_of(Rt, p_mouth(u) - Rt @ rm)
                return fn
            return make

        T_pour0 = T_of(R, p_mouth(0.0) - R @ rm_planned)
        T_pre = T_of(R, p_g - back * z_t + np.array([0, 0.05, 0]))
        T_side = T_of(R, p_g - back * z_t)
        q_pre, q_side, q_g, q_lift, q_p0, q_full, q_p0b, q_liftb, q_gb, q_sideb, q_ret = fam
        tl.move("approach (side)", 2.4, T_pre, grip=dict(mode="pos", gap_mm=30), joint_space=True).hint(q_pre)
        tl.move("level with tube", 1.2, T_side).hint(q_side)
        tl.move("close in", 1.5, Tg).hint(q_g)
        tl.hold("grasp tube (tactile)", 2.0, grip=dict(mode="safe", force_N=3.0), **est_hooks(est))
        tl.hold("secure grip", 0.6, grip=dict(mode="safe", force_N=GRIP_CARRY))
        tl.move("lift tube", 2.0, _above(Tg, lift)).hint(q_lift)
        tl.move("to beaker", 3.0, T_pour0).hint(q_p0)
        tl.hold("settle", 0.8)
        tl.move_rel("align mouth", 0.8,
                    lambda sc: np.zeros(3) if sc.is_dry else p_mouth(0.0) - mouth_w(sc))
        tl.dyn("pour", 5.0, pour(True), tag="pour").hint(q_full)        # 20 deg/s: gentle stream
        tl.hold("pouring", 3.0, tag="pour")        # PIC water drains slower (onset ~69 deg): give it time
        tl.dyn("upright", 2.6, pour(False)).hint(q_p0b)
        tl.move("back to rack", 3.0, _above(Tg, lift)).hint(q_liftb)
        tl.hold("settle", 1.0)
        tl.move_rel("centre over slot", 1.0, centre_over(slot_xz, bottom))
        tl.move_rel("insert into rack", 2.4, descend_until(bottom, bottom0[1] + 0.003, lift - 0.003)).hint(q_gb)
        tl.hold("release tube", 1.0, grip=dict(mode="pos", gap_mm=30))
        tl.move_rel("back off", 1.2, lambda sc: -back * z_t).hint(q_sideb)
        tl.move_rel("retreat", 1.0, up(0.10)).hint(q_ret)
        return tl


class RodStage(Stage):
    name = "3 玻璃棒搅拌 / stir"

    def build(self, scene):
        tl = Timeline(self)
        L, r = scene.props["rod"].meta["length"], scene.props["rod"].meta["radius"]

        def ends(sc):
            T = sc.prop_pose("rod")
            return [T[:3, 3] + T[:3, :3] @ np.array([sg * L / 2, 0, 0]) for sg in (-1, 1)]

        def rod_tip(sc):
            return min(ends(sc), key=lambda e: e[1])

        e0 = ends(scene)
        top = max(e0, key=lambda e: e[1])
        bottom0 = min(e0, key=lambda e: e[1])
        p_g = top + np.array([0, -0.012, 0])
        R = best_top_down(scene, "z", p_g)
        Tg = T_of(R, p_g)
        est = scene.new_estimator("glass rod", r, e_true=None)
        stand_xz = scene.prop_T["rod"][:3, 3][[0, 2]]
        beaker = scene.prop_pose("beaker_mix")[:3, 3]
        dip_y = beaker[1] + 0.016                         # rod tip height while stirring
        hover = 0.08
        T_above_beaker = T_of(R, [beaker[0], dip_y + (L - 0.012) + hover, beaker[2]])
        lift = 0.09

        def stir(sc):
            T0 = sc.cmd_pose.copy()

            def fn(s):
                ang = 2 * np.pi * 3.0 * s
                rad = 0.008 * min(1.0, s * 6) * min(1.0, (1 - s) * 6)
                T = T0.copy()
                T[:3, 3] = T0[:3, 3] + np.array([rad * np.cos(ang), 0, rad * np.sin(ang)])
                return T
            return fn

        tl.move("approach rod stand", 2.0, _above(Tg, 0.10), grip=dict(mode="pos", gap_mm=22), joint_space=True)
        tl.move("descend to rod", 1.5, Tg)
        tl.hold("grasp rod (tactile)", 2.0, grip=dict(mode="safe", force_N=2.5), **est_hooks(est))
        tl.hold("secure grip", 0.6, grip=dict(mode="safe", force_N=GRIP_CARRY))
        tl.move("lift rod", 2.0, _above(Tg, lift))
        tl.move("over beaker", 3.0, T_above_beaker)
        tl.hold("settle", 0.8)
        tl.move_rel("centre rod over beaker", 1.0, centre_over(beaker[[0, 2]], rod_tip))
        tl.move_rel("dip", 1.6, descend_until(rod_tip, dip_y, hover))
        tl.dyn("stir", 4.0, stir, tag="stir")
        tl.move_rel("raise", 1.5, up(0.09))
        tl.move("back over stand", 3.0, _above(Tg, lift))
        tl.hold("settle", 1.0)
        tl.move_rel("centre over stand", 1.0, centre_over(stand_xz, rod_tip))
        tl.move_rel("insert into stand", 2.2, descend_until(rod_tip, bottom0[1] + 0.003, lift - 0.003))
        tl.hold("release rod", 1.0, grip=dict(mode="pos", gap_mm=22))
        tl.move_rel("retreat", 1.0, up(0.10))
        return tl


class BeakerStage(Stage):
    name = "4 摇匀 + 5 倒入废液杯 / shake & pour"
    TILT = 100.0

    def build(self, scene):
        tl = Timeline(self)
        Tb = scene.prop_pose("beaker_mix")
        bc = Tb[:3, 3].copy()
        H, rad = scene.props["beaker_mix"].meta["height"], scene.props["beaker_mix"].meta["outer_r"]
        waste = scene.prop_pose("beaker_waste")[:3, 3]
        waste_rim = waste[1] + scene.props["beaker_waste"].meta["height"]
        p_g = bc + np.array([0, H - 0.016, 0])        # pads 16 mm below the rim
        est = scene.new_estimator("beaker (glass)", rad, e_true=None)
        TH = self.TILT
        # pour-lip height above the waste rim: start high enough that the beaker's
        # far bottom edge clears the waste rim, then keep the lip low (H1) once the
        # tilt makes the lip the lowest point.  Water spills from ~33 deg (half
        # full), so the lip must already be low and over the target by then; the
        # first liquid run (lip 60-97 mm high, 33 deg/s) threw 89 % onto the table.
        H1 = 0.020
        H0 = H + 0.03                                 # planning value; exact table below
        w_r_out = scene.props["beaker_waste"].meta["outer_r"]
        # the stream leaves the lip away from the beaker centre: put the lip a little
        # before the waste axis along the mix->waste direction
        d_xz = waste[[0, 2]] - bc[[0, 2]]; d_xz = d_xz / np.linalg.norm(d_xz)
        lip_xz = waste[[0, 2]] - 0.020 * d_xz
        ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        rho = rad * 0.92
        back, lift = 0.08, 0.09
        loc_rim = np.stack([rho * np.cos(ang), np.full_like(ang, H), rho * np.sin(ang)], 1)

        def key_poses(R, sg, roll):
            z_t = R[:, 2]
            Rf = pour_R(R, sg, roll)
            pts = (loc_rim + bc - p_g) @ R
            low = pts[np.argmin((pts @ Rf.T)[:, 1])]
            pr0 = np.array([lip_xz[0], waste_rim + H0, lip_xz[1]])
            pr1 = np.array([lip_xz[0], waste_rim + H1, lip_xz[1]])
            Tg_ = T_of(R, p_g)
            T_side_ = T_of(R, p_g - back * z_t)
            return [T_of(R, p_g - back * z_t + np.array([0, 0.05, 0])), T_side_, Tg_, _above(Tg_, lift),
                    T_of(R, pr0 - R @ low), T_of(Rf, pr1 - Rf @ low), T_of(R, pr0 - R @ low),
                    _above(Tg_, 0.10), Tg_, T_side_, _above(T_side_, 0.12), scene.home_pose]

        R, sgn, ROLL, fam = choose_side_variant(scene, bc[[0, 2]], waste[[0, 2]], key_poses, TH,
                                                avoid=("flask", "rack", "tray", "test_tube", "test_tube2"))
        z_t = R[:, 2]
        Tg = T_of(R, p_g)
        Rfull = pour_R(R, sgn, ROLL)

        # exact lip-height table: for each tilt fraction u, how far the carried
        # beaker body would dip below its lip over the waste beaker's footprint
        # (verts in tool coords at the grasp, rotated by the pour roll)
        verts_tool = (scene.props["beaker_mix"].verts + bc - p_g) @ R
        # The gripper rides with the beaker. The pour roll turns the closing axis toward vertical, so one
        # finger hangs below the beaker body: in every recorded run it grazed the waste rim mid-pour
        # (0.3-1.3 mm) and once shoved the waste beaker 4.7 mm. Put the fingers and the hand into the same
        # clearance table (pads touching the beaker wall: finger joint distance = outer_r + pad thickness).
        n_beaker = len(verts_tool)
        try:
            from lab_scene import PAD_T
            lp = scene.arm.link_poses(fam[2], finger_d=rad + PAD_T)
            lvl = scene.link_vertices_local()
            grip = [lvl[n] @ lp[n][:3, :3].T + lp[n][:3, 3] for n in ("fr3_leftfinger", "fr3_rightfinger", "fr3_hand") if n in lvl]
            if grip:
                verts_tool = np.concatenate([verts_tool, (np.concatenate(grip) - p_g) @ R])
        except Exception as e:                       # never block the stage on the extra margin
            print(f"[task] gripper clearance in the pour table skipped: {type(e).__name__}: {e}")
        _rim0 = (loc_rim + bc - p_g) @ R
        low0 = _rim0[np.argmin((_rim0 @ Rfull.T)[:, 1])]
        _us = np.linspace(0.0, 1.0, 101); _tab = []
        for _u in _us:
            Rt_ = pour_R(R, sgn, ROLL * _u)
            W = (verts_tool - low0) @ Rt_.T
            horiz = np.hypot(W[:, 0] + lip_xz[0] - waste[0], W[:, 2] + lip_xz[1] - waste[2])
            over = horiz < w_r_out + 0.004
            depth = float(W[over, 1].min()) if over.any() else 0.0
            _tab.append(max(H1, -depth + 0.012))
        _tab = np.array(_tab)
        if os.environ.get("LAB_PLAN_DEBUG"):
            _tb = []
            for _u in _us:
                W = (verts_tool[:n_beaker] - low0) @ pour_R(R, sgn, ROLL * _u).T
                horiz = np.hypot(W[:, 0] + lip_xz[0] - waste[0], W[:, 2] + lip_xz[1] - waste[2])
                over = horiz < w_r_out + 0.004
                _tb.append(max(H1, -(float(W[over, 1].min()) if over.any() else 0.0) + 0.012))
            print("[task] pour lip height over the waste rim (mm), beaker only -> with gripper, u = 0 .. 1 step 0.1: "
                  + " ".join(f"{a*1e3:.0f}->{b*1e3:.0f}" for a, b in zip(np.array(_tb)[::10], _tab[::10])))
        H0 = float(_tab[0])

        def lip_height(theta_deg):
            return float(np.interp(theta_deg / TH, _us, _tab))

        def rim_tool(sc):
            """The rim point (tool coords) that ends up lowest at full tilt."""
            if sc.is_dry:
                loc = np.stack([rho * np.cos(ang), np.full_like(ang, H), rho * np.sin(ang)], 1)
                pw = loc + bc
                pts = (pw - p_g) @ R
            else:
                T = sc.prop_pose("beaker_mix")
                loc = np.stack([rho * np.cos(ang), np.full_like(ang, H), rho * np.sin(ang)], 1)
                pw = loc @ T[:3, :3].T + T[:3, 3]
                Tt = tcp_now(sc)
                pts = (pw - Tt[:3, 3]) @ Tt[:3, :3]
            return pts[np.argmin((pts @ Rfull.T)[:, 1])]

        def p_rim(u):
            return np.array([lip_xz[0], waste_rim + lip_height(TH * u), lip_xz[1]])

        def pour(forward, dz=0.0):
            def make(sc):
                low = rim_tool(sc)

                def fn(s):
                    u = minjerk(s) if forward else 1.0 - minjerk(s)
                    Rt = pour_R(R, sgn, ROLL * u)
                    return T_of(Rt, p_rim(u) + np.array([0.0, dz, 0.0]) - Rt @ low)
                return fn
            return make

        def align(sc):
            if sc.is_dry:
                return np.zeros(3)
            Tt = tcp_now(sc)
            return p_rim(0.0) - (Tt[:3, 3] + Tt[:3, :3] @ rim_tool(sc))

        def shake(sc):
            T0 = sc.cmd_pose.copy()

            def fn(s):
                T = T0.copy()
                env = min(1.0, s * 5) * min(1.0, (1 - s) * 5)
                T[:3, 3] = T0[:3, 3] + 0.012 * env * np.sin(2 * np.pi * 6.6 * s) * R[:, 1]
                return T
            return fn

        def base(sc):
            return sc.prop_pose("beaker_mix")[:3, 3]

        class _Dry:
            is_dry = True
        low0 = rim_tool(_Dry())
        T_pour0 = T_of(R, p_rim(0.0) - R @ low0)
        T_pre = T_of(R, p_g - back * z_t + np.array([0, 0.05, 0]))
        T_side = T_of(R, p_g - back * z_t)
        q_pre, q_side, q_g, q_lift, q_p0, q_full, q_p0b, q_back, q_gb, q_sideb, q_ret, q_home = fam
        tl.move("approach (side)", 2.4, T_pre, grip=dict(mode="pos", gap_mm=72), joint_space=True).hint(q_pre)
        tl.move("level with beaker", 1.2, T_side).hint(q_side)
        tl.move("close in", 1.6, Tg).hint(q_g)
        tl.hold("grasp beaker (tactile)", 2.0, grip=dict(mode="safe", force_N=5.0), **est_hooks(est))
        tl.hold("secure grip", 0.6, grip=dict(mode="safe", force_N=7.0))
        tl.move("lift beaker", 2.0, _above(Tg, lift)).hint(q_lift)
        tl.dyn("shake", 2.4, shake, tag="shake").hint(q_lift)
        tl.move("to waste beaker", 3.2, T_pour0).hint(q_p0)
        tl.hold("settle", 0.8)
        tl.move_rel("align rim", 0.8, align).hint(q_p0)
        tl.dyn("pour", 5.0, pour(True), tag="pour").hint(q_full)        # 20 deg/s: gentle stream
        tl.hold("pouring", 3.0, tag="pour")        # PIC water drains slower (onset ~69 deg): give it time
        # Turning back upright along the pour path passed within millimetres of the waste rim, and
        # the lip feedback's horizontal correction relaxes at the same moment: the run with the
        # 35 deg grasp variant clipped the waste beaker (15 mm shove, 43 deg spin). The beaker is
        # empty here, so lift it straight up first (tagged 'pour' so the lip correction is held,
        # not relaxed, while still low) and run the upright path the same height higher.
        UPRIGHT_LIFT = float(os.environ.get("LAB_UPRIGHT_LIFT", "0.03"))
        if UPRIGHT_LIFT > 0:
            tl.move_rel("lift clear", 1.0, up(UPRIGHT_LIFT), tag="pour").hint(q_full)
        tl.dyn("upright", 2.6, pour(False, dz=UPRIGHT_LIFT)).hint(q_p0b)
        tl.move("back", 3.2, _above(Tg, 0.10)).hint(q_back)
        tl.hold("settle", 0.8)
        tl.move_rel("centre over spot", 1.0, centre_over(bc[[0, 2]], base))
        tl.move_rel("set down", 2.0, descend_until(base, bc[1] + 0.002, 0.10 - 0.002)).hint(q_gb)
        tl.hold("release beaker", 1.0, grip=dict(mode="pos", gap_mm=72))
        tl.move_rel("back off", 1.2, lambda sc: -back * z_t).hint(q_sideb)
        tl.move_rel("retreat", 1.2, up(0.12)).hint(q_ret)
        tl.move("home", 2.4, scene.home_pose).hint(q_home)
        return tl


def all_stages():
    return [PreSettle(), DropperStage(), TestTubeStage(), RodStage(), BeakerStage()]
