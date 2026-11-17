"""Nuka-Physics 'G1 shirt folding' with StiffGIPC: a Unitree G1 (29 DoF, Dex3 hands, fixed pelvis) stands at a table and folds a
flat T-shirt (single-layer IPC cloth) in half with a bimanual pinch-lift-lay motion, then folds it again sideways.

Plan (z-up robot frame, pelvis at z 0.80): table top z 0.90, the shirt hem overhangs the near table edge by ~5 cm and the sleeves
overhang the sides; each Dex3 hand pinches a hem corner from the side (index above, thumb below the overhang), lifts, carries the hem
over to the collar and lays it down (fold 1), then both hands pinch the left edge and fold it onto the right half (fold 2).
Arm joints follow IK solutions of tools/urdf_kin.py (7-joint chains, waist pitch scripted) through the engine's position drives;
fingers close through low-strength position drives (contact wins, IPC keeps the cloth between the pads).

    PYTHONPATH=. python examples/nuka/g1_shirt_fold.py --dry-run            # kinematic plan only, no engine
    PYTHONPATH=. python examples/nuka/g1_shirt_fold.py --out out/nuka/fold1
"""
import argparse, itertools, json, math, os, os.path as osp, sys, time
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "tools"))
from urdf_kin import UrdfKin  # noqa: E402

ASSETS = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "Assets"))
URDF = osp.join(ASSETS, "nuka", "g1", "g1_hull.urdf")
R_CONV = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])   # z-up -> engine y-up
PELVIS_Z = 0.80
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")


def minjerk(s):
    s = float(np.clip(s, 0.0, 1.0)); return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def zup_to_yup_T(T):
    """4x4 in the z-up robot/world frame -> engine y-up frame (y_up = R_CONV z_up)."""
    C = np.eye(4); C[:3, :3] = R_CONV
    return C @ T @ np.linalg.inv(C)


# ----------------------------------------------------------------------------- hand geometry
def hull_pts(link):
    import trimesh
    return np.asarray(trimesh.load(osp.join(ASSETS, "nuka", "g1", "meshes", f"{link}_hull.stl"), force="mesh").vertices)


def pinch_pose(kin, side):
    """Grid-search the Dex3 joint values (within limits) that bring the thumb_2 and index_1 hulls closest; returns (joint dict,
    pinch point in the PALM frame, gap). middle joints follow the index."""
    best = None
    lim = {k: kin.joints[f"{side}_hand_{k}_joint"] for k in HAND_JOINTS}
    # candidates strictly INSIDE the joint limits: a drive target at a limit fights the limit barrier and the hand never closes
    cand = {k: np.linspace(lim[k]["lower"] + 0.1 * (lim[k]["upper"] - lim[k]["lower"]), lim[k]["upper"] - 0.1 * (lim[k]["upper"] - lim[k]["lower"]), 5)
            for k in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1")}
    P_t = hull_pts(f"{side}_hand_thumb_2_link"); P_i = hull_pts(f"{side}_hand_index_1_link")
    for t0, t1, t2, i0, i1 in itertools.product(cand["thumb_0"], cand["thumb_1"], cand["thumb_2"], cand["index_0"], cand["index_1"]):
        q = {f"{side}_hand_thumb_0_joint": t0, f"{side}_hand_thumb_1_joint": t1, f"{side}_hand_thumb_2_joint": t2,
             f"{side}_hand_index_0_joint": i0, f"{side}_hand_index_1_joint": i1}
        T = kin.fk(q)
        Tpi = np.linalg.inv(T[f"{side}_hand_palm_link"])
        A = (Tpi @ T[f"{side}_hand_thumb_2_link"] @ np.c_[P_t, np.ones(len(P_t))].T).T[:, :3]
        B = (Tpi @ T[f"{side}_hand_index_1_link"] @ np.c_[P_i, np.ones(len(P_i))].T).T[:, :3]
        d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2); i, j = np.unravel_index(np.argmin(d), d.shape)
        if best is None or d[i, j] < best[0]:
            best = (d[i, j], dict(q), 0.5 * (A[i] + B[j]), B.mean(0) - A.mean(0))     # axis: thumb hull centroid -> index hull centroid
    gap, q, p_pinch, axis = best
    q[f"{side}_hand_middle_0_joint"] = q[f"{side}_hand_index_0_joint"]; q[f"{side}_hand_middle_1_joint"] = q[f"{side}_hand_index_1_joint"]
    return q, p_pinch, axis / max(1e-9, np.linalg.norm(axis)), gap


def pinch_geometry(kin, side, q):
    """(pinch point in the PALM frame, thumb->index axis, gap) for a given hand joint dict."""
    P_t = hull_pts(f"{side}_hand_thumb_2_link"); P_i = hull_pts(f"{side}_hand_index_1_link")
    T = kin.fk(q); Tpi = np.linalg.inv(T[f"{side}_hand_palm_link"])
    A = (Tpi @ T[f"{side}_hand_thumb_2_link"] @ np.c_[P_t, np.ones(len(P_t))].T).T[:, :3]
    B = (Tpi @ T[f"{side}_hand_index_1_link"] @ np.c_[P_i, np.ones(len(P_i))].T).T[:, :3]
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2); i, j = np.unravel_index(np.argmin(d), d.shape)
    axis = B.mean(0) - A.mean(0)
    return 0.5 * (A[i] + B[j]), axis / max(1e-9, np.linalg.norm(axis)), d[i, j]


def constrained_pinch(kin, side, axis_ref, p_ref):
    """Full grid search of the hand closure (inside the limits) scored by gap + orientation deviation of the thumb->index axis
    from a reference (the mirrored left-hand axis) + pinch-point deviation, so the right hand closes tightly AND its palm
    orientation stays reachable (symmetric with the left arm)."""
    lim = {k: kin.joints[f"{side}_hand_{k}_joint"] for k in HAND_JOINTS}
    cand = {k: np.linspace(lim[k]["lower"] + 0.1 * (lim[k]["upper"] - lim[k]["lower"]), lim[k]["upper"] - 0.1 * (lim[k]["upper"] - lim[k]["lower"]), 6)
            for k in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1")}
    best = None
    for t0, t1, t2, i0, i1 in itertools.product(cand["thumb_0"], cand["thumb_1"], cand["thumb_2"], cand["index_0"], cand["index_1"]):
        q = {f"{side}_hand_thumb_0_joint": t0, f"{side}_hand_thumb_1_joint": t1, f"{side}_hand_thumb_2_joint": t2,
             f"{side}_hand_index_0_joint": i0, f"{side}_hand_index_1_joint": i1, f"{side}_hand_middle_0_joint": i0, f"{side}_hand_middle_1_joint": i1}
        p, axis, gap = pinch_geometry(kin, side, q)
        score = gap + 0.03 * (1.0 - float(np.dot(axis, axis_ref))) + 0.3 * float(np.linalg.norm(p - p_ref))
        if best is None or score < best[0]:
            best = (score, q, p, axis, gap)
    return best[1], best[2], best[3], best[4]


def mirrored_pinch(kin, side_from, q_from, side_to):
    """Right-hand closed pose = the left one with mirrored joint signs (the Dex3 right-hand limits are the negated left ones),
    clamped into the right limits; keeps the two palm orientations symmetric (the independent grid search gave the right hand a
    different closure whose orientation the right arm could not reach)."""
    def clamp(m, v):
        j = kin.joints[m]; lo = j["lower"] + 0.1 * (j["upper"] - j["lower"]); hi = j["upper"] - 0.1 * (j["upper"] - j["lower"])
        return float(min(max(v, lo), hi))
    base = {n.replace(f"{side_from}_", f"{side_to}_", 1): -v for n, v in q_from.items()}
    names = [f"{side_to}_hand_{k}_joint" for k in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1")]
    best = None
    for d in itertools.product((-0.3, 0.0, 0.3), repeat=5):            # local refinement around the mirrored closure
        q_to = dict(base)
        for n, dv in zip(names, d):
            q_to[n] = clamp(n, base[n] + dv)
        q_to[f"{side_to}_hand_middle_0_joint"] = q_to[f"{side_to}_hand_index_0_joint"]; q_to[f"{side_to}_hand_middle_1_joint"] = q_to[f"{side_to}_hand_index_1_joint"]
        p, axis, gap = pinch_geometry(kin, side_to, q_to)
        if best is None or gap < best[3]:
            best = (q_to, p, axis, gap)
    return best


def palm_target(p_pinch_world, R_palm, p_pinch_local):
    T = np.eye(4); T[:3, :3] = R_palm; T[:3, 3] = p_pinch_world - R_palm @ p_pinch_local
    return T


# ----------------------------------------------------------------------------- plan
class Plan:
    """Waypoints: list of (t_end, label, {side: (p_pinch_world, R_palm)}, waist_pitch, finger_state {side: 0 open..1 closed})."""

    def __init__(self, args):
        self.kin = UrdfKin(URDF)
        self.sides = ("left", "right")
        self.fingers = getattr(args, "fingers", "down")
        self.pinch = {"left": pinch_pose(self.kin, "left")}                 # q, p_local, axis_local (thumb->index), gap
        qL, pL, aL, gL = self.pinch["left"]
        self.pinch["right"] = constrained_pinch(self.kin, "right", aL * np.array([1.0, -1.0, 1.0]), pL * np.array([1.0, -1.0, 1.0]))
        zT = args.table_z; self.zT = zT
        L = args.shirt_len; W = args.shirt_w
        x_edge = args.table_edge_x - 0.006
        hem_x = x_edge; L_flat = L - args.prebend                   # the flat part spans [x_edge, x_edge + L_flat]
        self.shirt_x0, self.shirt_len, self.shirt_w = hem_x, L, W
        z_hem = zT + 0.004
        self.R = {}
        if args.grip == "flap":
            # hem flap hanging at the table edge (mesh pre-bent), pinch axis thumb -> index = -x (thumb behind the flap).
            # fingers DOWN (-z) wedge the cloth above the pinch point between the palm and the thumb (the palm spans the flap plane):
            # use UP (+z, the hem enters between the fingertips from above, palm below the hem) or SIDE (fingers point inward along
            # -/+y from outside the shirt's side edge, the palm stays outside the cloth, any pinch height)
            pdir = np.array([-1.0, 0.0, 0.0])
            fdirs = {"down": {"left": (0.0, 0.0, -1.0), "right": (0.0, 0.0, -1.0)}, "up": {"left": (0.0, 0.0, 1.0), "right": (0.0, 0.0, 1.0)},
                     "side": {"left": (0.0, -1.0, 0.0), "right": (0.0, 1.0, 0.0)}}[args.fingers]
        else:
            # short, stiff overhang stays ~horizontal: fingers FORWARD (+x) over the hem from the robot side, pinch axis = +z
            # (index above the hem, thumb below it in front of the table edge) — the bed-sheet fold recipe
            pdir = np.array([0.0, 0.0, 1.0]); fdirs = {"left": (1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0)}
        for s in self.sides:
            fdir = np.array(fdirs[s])
            axis_local = self.pinch[s][2]
            x_l = np.array([1.0, 0.0, 0.0]); a_l = axis_local - np.dot(axis_local, x_l) * x_l; a_l /= np.linalg.norm(a_l)
            x_w = fdir; a_w = pdir - np.dot(pdir, x_w) * x_w; a_w /= np.linalg.norm(a_w)
            B_l = np.stack([x_l, a_l, np.cross(x_l, a_l)], 1); B_w = np.stack([x_w, a_w, np.cross(x_w, a_w)], 1)
            self.R[s] = B_w @ B_l.T
        yl, yr = W / 2 - args.pinch_inset, -(W / 2 - args.pinch_inset)
        def at(x, y, z):
            return np.array([x, y, z])
        if args.grip == "flap":
            x_flap = args.table_edge_x - 0.006 + args.pinch_dx
            z_grip = zT - args.table_thick - args.grip_below
            xr_extra = args.dx_right_extra                    # optional extra offset for the right hand (0 with the mirrored closure)
            def pair(dx=0.0, dz=0.0, x=None, z=None, dy=0.0):
                xx = x_flap + dx if x is None else x; zz = z_grip + dz if z is None else z
                return {"left": (at(xx, yl + dy, zz), self.R["left"]), "right": (at(xx + (xr_extra if x is None else 0.0), yr - dy, zz), self.R["right"])}
            c = {}
            if args.fingers == "side":
                # from outside the shirt's side edges: hover out and low, approach level with the grip, then slide inward
                c["hover"] = pair(0.0, -args.side_hover_dz, dy=args.side_out + args.side_hover_dy); c["approach"] = pair(0.0, 0.0, dy=args.side_out)
                c["pinch"] = pair(0.0, 0.0)
            else:
                # hover low and a little in front, approach = same x as the grip but below the flap bottom, then a pure vertical rise
                c["hover"] = pair(-0.06, -0.24); c["approach"] = pair(0.0, -0.09); c["pinch"] = pair(0.0, 0.0)
            c["lift"] = pair(-0.03, z=zT + 0.14)
            if args.fingers == "side":       # pull the pinched corners out toward the robot first: the closed thumb sits under the slab edge
                c["pull"] = pair(-args.pull, 0.0); c["lift"] = pair(-args.pull, z=zT + 0.14)
            if args.arc:
                # carry the pinched hem on a circle about the fold line (table edge, top of the slab) whose radius is the pinch
                # point's distance from it: farther waypoints drag the whole shirt (the flap is only that long below the pinch)
                xf_, zf_ = x_edge, zT + 0.004; r_ = zf_ - z_grip
                def arc(phi_deg, zmin=None, slack=0.0, dx=0.0):
                    rr = r_ - slack
                    ph = math.radians(phi_deg); xx = xf_ + dx - rr * math.sin(ph); zz = zf_ - rr * math.cos(ph)
                    return pair(x=xx, z=max(zz, zmin) if zmin is not None else zz)
                # the hanging cloth between the fold line and the pinch is exactly r long, so on the full-radius arc it is taut and
                # the lift / over motion drags the whole shirt ~3.5 cm toward the robot (fd/fe runs): carry the hem on r - slack
                # (the pull keeps the full radius: the thumbs must clear the slab edge first)
                sl = args.arc_slack
                # the drag happens only as the hands reach the 110 deg lift waypoint (flap nearly horizontal toward the robot, its
                # tension pulls the table half; at 180 deg it pulls straight up): --arc-slack-lift slackens just that waypoint
                c["pull"] = arc(args.arc_pull); c["lift"] = arc(110.0, slack=sl + args.arc_slack_lift)
                c["over"] = arc(180.0, slack=sl); c["mid"] = arc(235.0, slack=sl)
                c["lay"] = arc(args.arc_lay, zmin=zT + args.lay_dz); c["up"] = pair(x=xf_ + 0.8 * r_, z=zT + 0.15); c["home"] = pair(-0.07, 0.14)
                if args.fingers == "side":   # after opening, slide the hands out sideways (the thumb is under the folded flap), then up
                    xl, zl = c["lay"]["left"][0][0], c["lay"]["left"][0][2]
                    c["slide"] = pair(x=xl, z=zl, dy=args.side_out); c["up"] = pair(x=xl - 0.02, z=zT + 0.15, dy=args.side_out)
                self.arc_r = r_
            x_mid = x_edge + 0.5 * L_flat; x_lay = x_edge + L_flat - 0.03
            c["mid"] = pair(x=x_mid, z=zT + 0.20); c["lay"] = pair(x=x_lay, z=zT + 0.05); c["up"] = pair(x=x_lay - 0.06, z=zT + 0.16); c["home"] = pair(-0.07, 0.14)
        else:
            hem_x = args.table_edge_x - args.overhang; self.shirt_x0 = hem_x
            px = hem_x + 0.4 * args.overhang; z_p = zT + 0.004 - args.droop
            back = 0.07
            def pair(dx=0.0, dz=0.0, x=None, z=None):
                xx = px + dx if x is None else x; zz = z_p + dz if z is None else z
                return {"left": (at(xx, yl, zz), self.R["left"]), "right": (at(xx, yr, zz), self.R["right"])}
            c = {}
            c["hover"] = pair(-back, 0.08); c["approach"] = pair(-back, 0.0); c["pinch"] = pair(0.0, 0.0)
            c["lift"] = pair(0.02, 0.14)
            x_mid = hem_x + 0.5 * L; x_lay = hem_x + L - 0.03
            c["mid"] = pair(x=x_mid, z=zT + 0.20); c["lay"] = pair(x=x_lay, z=zT + 0.05); c["up"] = pair(x=x_lay - 0.06, z=zT + 0.16); c["home"] = pair(-back, 0.12)
        w_far = args.waist_far
        # (t_end, label, targets, waist pitch, finger closure, IK rotation weight)
        closed, opened = {"left": 1.0, "right": 1.0}, {"left": 0.0, "right": 0.0}
        self.way = [
            (1.0, "settle", c["hover"], 0.0, opened, 0.4),
            (2.5, "approach", c["approach"], 0.0, opened, 0.4),
            (4.0, "reach", c["pinch"], 0.0, opened, 0.4),
            (5.5, "pinch", c["pinch"], 0.0, closed, 0.4)]
        dt_pull = 0.0
        if args.arc:
            wa = args.arc_waist; k = args.slow     # k stretches the fold-over segments (lower accelerations / inertial drag)
            if args.flip_at is not None:
                # FLIP: carry the hem just past the vertical over the fold line, open there and let the flap fall over by itself
                # (holding the two corners down to the lay drags the sagging hem middle into a heap)
                # --fold-shift: with the flap vertical, translate the hands +dx (drags the table half back toward its start) and
                # flip about the shifted fold line
                dxs = args.fold_shift
                c["flip"] = arc(args.flip_at, slack=args.arc_slack, dx=dxs); xl, zl = c["flip"]["left"][0][0], c["flip"]["left"][0][2]
                if dxs:
                    c["shift"] = arc(180.0, slack=args.arc_slack, dx=dxs)
                c["fslide"] = pair(x=xl, z=zl, dy=args.side_out); c["fup"] = pair(x=xl - 0.03, z=zl + 0.06, dy=args.side_out)
                # --over-roll / --flip-roll: roll the hands about the fold axis (world y; the side-grasp fingers point along it) so the
                # cloth leaves the fingers closer to the fold line: with the pinch orientation held through the arc it leaves them
                # pointing UP and loops above the hands (fm: ~7 cm of slack in the loop pulls the crease ~5 cm into the table half)
                def rolled(pr, bdeg):
                    if not bdeg:
                        return pr
                    b = math.radians(bdeg); Ry = np.array([[math.cos(b), 0.0, math.sin(b)], [0.0, 1.0, 0.0], [-math.sin(b), 0.0, math.cos(b)]])
                    return {sd: (pr[sd][0], Ry @ pr[sd][1]) for sd in pr}
                c["over"] = rolled(c["over"], args.over_roll)
                if dxs:
                    c["shift"] = rolled(c["shift"], args.over_roll)
                c["flip"] = rolled(c["flip"], args.flip_roll); c["fslide"] = rolled(c["fslide"], args.flip_roll)
                rw = 0.5 if (args.over_roll or args.flip_roll) else 0.1
                self.way += [
                    (5.5 + 1.0 * k, "pull", c["pull"], 0.0, closed, 0.4),
                    (5.5 + 2.2 * k, "lift", c["lift"], 0.0, closed, 0.15),
                    (5.5 + 3.4 * k, "over", c["over"], 0.5 * wa, closed, rw)]
                if dxs:
                    self.way.append((self.way[-1][0] + 0.6 * k, "shift", c["shift"], 0.5 * wa, closed, rw))
                self.way.append((self.way[-1][0] + 0.8 * k, "flip", c["flip"], 0.5 * wa, closed, rw))
                t_f = self.way[-1][0]
                self.way += [
                    (t_f + 0.4, "release", c["flip"], 0.5 * wa, opened, rw),
                    (t_f + 1.2, "slide", c["fslide"], 0.5 * wa, opened, 0.3),
                    (t_f + 2.4, "retreat", c["fup"], 0.25 * wa, opened, 0.1),
                    (t_f + 4.4, "home", c["home"], 0.0, opened, 0.3),
                    (t_f + 6.4, "settle2", c["home"], 0.0, opened, 0.3)]
                self.total = self.way[-1][0]
            else:
              self.way += [
                (5.5 + 1.0 * k, "pull", c["pull"], 0.0, closed, 0.4),
                (5.5 + 2.2 * k, "lift", c["lift"], 0.0, closed, 0.15),
                (5.5 + 3.4 * k, "over", c["over"], 0.5 * wa, closed, 0.1),
                (5.5 + 4.6 * k, "carry", c["mid"], wa, closed, 0.1),
                (5.5 + 5.8 * k, "lay", c["lay"], wa, closed, 0.1),
                (5.5 + 5.8 * k + 1.5, "release", c["lay"], wa, opened, 0.1)]
              if "slide" in c:
                  self.way.append((self.way[-1][0] + 1.0, "slide", c["slide"], wa, opened, 0.3))
              t_r = self.way[-1][0]
              self.way += [
                  (t_r + 1.5, "retreat", c["up"], 0.5 * wa, opened, 0.1),
                  (t_r + 3.5, "home", c["home"], 0.0, opened, 0.3)]
            self.total = self.way[-1][0]
        elif "pull" in c:
            self.way.append((6.5, "pull", c["pull"], 0.0, closed, 0.4)); dt_pull = 1.0
        if not args.arc:
          self.way += [
            (7.0 + dt_pull, "lift", c["lift"], 0.0, closed, 0.15),
            (9.5 + dt_pull, "carry", c["mid"], w_far * 0.6, closed, 0.1),
            (12.0 + dt_pull, "lay", c["lay"], w_far, closed, 0.1),
            (13.5 + dt_pull, "release", c["lay"], w_far, opened, 0.1),
            (15.0 + dt_pull, "retreat", c["up"], w_far * 0.5, opened, 0.1),
            (17.0 + dt_pull, "home", c["home"], 0.0, opened, 0.3),
          ]
        self.total = self.way[-1][0]
        if getattr(args, "sleeves", False):
            sw = self.sleeve_waypoints(args, zT, at)
            T_s = sw[-1][0]
            self.way = sw + [(t + T_s, lab, cart, w, f, rw) for (t, lab, cart, w, f, rw) in self.way]
            self.total = self.way[-1][0]; self.t_sleeves = T_s
        # IK for every waypoint (warm-started), in the z-up pelvis frame (pelvis at the origin)
        self.q_way = []
        self.ik_err = []
        # multi-start IK: the previous waypoint's solution, the zero pose and an 'arms forward' pose; keep the best per arm
        seeds_base = [{}, {f"{s}_shoulder_pitch_joint": -0.8 for s in self.sides} | {f"{s}_elbow_joint": 1.2 for s in self.sides}]
        q_prev = {}
        for t_end, label, cart, waist, fing, rot_w in self.way:
            best_q = dict(q_prev); best_q["waist_pitch_joint"] = waist
            errs = {}
            for s in self.sides:
                p_w, R_w = cart[s]
                T_t = palm_target(p_w - np.array([0.0, 0.0, PELVIS_Z]), R_w, self.pinch[s][1])
                chain = self.kin.chain(f"{s}_hand_palm_link", "torso_link")
                best = None
                for seed in [dict(q_prev)] + [dict(sd) for sd in seeds_base]:
                    seed["waist_pitch_joint"] = waist
                    for k, v in best_q.items():          # keep the other arm / waist as already solved
                        if k not in chain:
                            seed[k] = v
                    qs, ep, er = self.kin.ik(f"{s}_hand_palm_link", T_t, seed, chain, iters=300, rot_w=rot_w, lam=0.03, step=0.9)
                    score = ep + 0.1 * er
                    if best is None or score < best[0]:
                        best = (score, qs, ep, er)
                for k in chain:
                    best_q[k] = best[1][k]
                errs[s] = (best[2], best[3])
            self.q_way.append(best_q); q_prev = best_q; self.ik_err.append((label, errs))

    def sleeve_waypoints(self, args, zT, at):
        """Side folds first (as in Nuka's demo): each sleeve hangs over the narrow table's side edge (mesh pre-bent at |y| = ys);
        both hands at once (the outward pulls balance) pinch their sleeve flap from the front -- fingers forward (+x) tilted
        30 deg down, thumb inside under the slab, index outside -- pull it out, carry it on an arc about the side edge and let it
        fall over at --sleeve-flip deg, then go up, out and down beside the table into the hem fold's start pose."""
        ys, zc = args.sleeve_ys, zT + 0.004
        xp, zp, yin = args.sleeve_x, args.sleeve_z, args.sleeve_in
        r_s = math.hypot(yin, zc - zp)
        fdir = np.array([math.cos(math.radians(30)), 0.0, -math.sin(math.radians(30))])
        R_sl = {}
        for s, sg in (("left", 1.0), ("right", -1.0)):
            axis_local = self.pinch[s][2]
            x_l = np.array([1.0, 0.0, 0.0]); a_l = axis_local - np.dot(axis_local, x_l) * x_l; a_l /= np.linalg.norm(a_l)
            pdir = np.array([0.0, sg, 0.0]); a_w = pdir - np.dot(pdir, fdir) * fdir; a_w /= np.linalg.norm(a_w)
            B_l = np.stack([x_l, a_l, np.cross(x_l, a_l)], 1); B_w = np.stack([fdir, a_w, np.cross(fdir, a_w)], 1)
            R_sl[s] = B_w @ B_l.T
        self.R_sl = R_sl
        def sp(x, yabs, z, R=None):
            return {s: (at(x, sg * yabs, z), (R_sl[s] if R is None else R[s])) for s, sg in (("left", 1.0), ("right", -1.0))}
        psi0 = math.degrees(math.atan2(-yin, zc - zp))         # pinch point angle on the arc (inside the flap plane: < 0)
        def arc(psi):
            # roll the hand with the arc about the side edge (+x, ~ the finger axis: a wrist roll), --sleeve-roll fraction of the
            # arc angle: the thumb stays on the fold-line side instead of swinging down into the table top (fixed orientation:
            # thumb / palm inside the slab from the flip on)
            ph = math.radians(psi); out = sp(xp, ys + r_s * math.sin(ph), zc - r_s * math.cos(ph))
            b = math.radians(args.sleeve_roll * (psi - psi0))
            for s_, sg in (("left", 1.0), ("right", -1.0)):
                c_, s2 = math.cos(sg * b), math.sin(sg * b)
                Rx = np.array([[1.0, 0.0, 0.0], [0.0, c_, -s2], [0.0, s2, c_]])
                out[s_] = (out[s_][0], Rx @ out[s_][1])
            return out
        fo = args.sleeve_open; half = {"left": fo, "right": fo}; closed = {"left": 1.0, "right": 1.0}; opened = {"left": 0.0, "right": 0.0}
        if getattr(args, "sleeve_index_first", False):
            # (thumb, index) closure pair: the index is already closed on the approach, outside the flap, and the open thumb is
            # inside; at s_pinch only the thumb closes and presses the flap against the index. The half-closed approach brings the
            # index up from below the flap edge, so it ends up inside the flap and the sleeve rests on it instead of being pinched.
            half = {sd: np.array([args.sleeve_thumb_open, 1.0]) for sd in ("left", "right")}
        carry = closed
        if getattr(args, "sleeve_index_first", False) and getattr(args, "sleeve_thumb_close", 1.0) != 1.0:
            carry = {sd: np.array([args.sleeve_thumb_close, 1.0]) for sd in ("left", "right")}   # thumb over-closed: firmer pinch
        x_front = args.table_edge_x + 0.02
        way = [
            (1.5, "s_pre", sp(args.table_edge_x, ys + 0.06, zp - 0.06), 0.0, half, 0.3),
            (2.5, "s_hover", sp(x_front, ys - yin, zp), 0.0, half, 0.4),
            (4.0, "s_reach", sp(xp, ys - yin, zp), 0.0, half, 0.4),
            (5.0, "s_pinch", sp(xp, ys - yin, zp), 0.0, carry, 0.4),
            (5.8, "s_pull", arc(35.0), 0.0, carry, 0.4),
            (6.8, "s_lift", arc(110.0), 0.0, carry, 0.2),
            (7.8, "s_over", arc(180.0), 0.0, carry, 0.2),
            (8.8, "s_flip", arc(args.sleeve_flip), 0.0, carry, args.sleeve_flip_rotw)]
        k_s = getattr(args, "sleeve_slow", 1.0)
        if k_s != 1.0:              # stretch the carried segments (pinch end 5.0 s -> flip 8.8 s), shift everything after
            way = [w if w[0] <= 5.0 else (5.0 + (w[0] - 5.0) * k_s,) + tuple(w[1:]) for w in way]
        yf, zf_ = way[-1][2]["left"][0][1], way[-1][2]["left"][0][2]
        R_f = {s_: way[-1][2][s_][1] for s_ in self.sides}        # keep the (rolled) flip orientation until the joint-space swap
        rel = {"left": args.sleeve_release_open, "right": args.sleeve_release_open}   # a fully opening thumb swings into the table
        if getattr(args, "sleeve_release_index", False):
            # open only the index (above the cloth at the flip); the thumb under the laid sleeve stays put and is then slid out
            tc = args.sleeve_thumb_close if getattr(args, "sleeve_index_first", False) else 1.0
            rel = {sd: np.array([tc, args.sleeve_release_open]) for sd in ("left", "right")}
        tail = [
            (9.2, "s_release", arc(args.sleeve_flip), 0.0, rel, 0.2),
            (10.0, "s_up", sp(xp, yf, zf_ + 0.06, R=R_f), 0.0, rel, 0.2),
            (11.0, "s_out", sp(xp - 0.03, ys + 0.09, zf_ + 0.06, R=R_f), 0.0, opened, 0.2),
            # hand-off to the hem grasp: the two orientations sit on different arm branches (shoulder yaw ~1.6 rad, wrist roll
            # ~2.2 rad apart), a Cartesian path with orientation slerp jumps branches -> interpolate in JOINT space ("j_" label)
            # outside and above the table, then go down beside it in Cartesian space
            (12.8, "j_swap", sp(args.table_edge_x - 0.006, ys + 0.06, zT, R=self.R), 0.0, opened, 0.4),
            (13.8, "s_down", sp(args.table_edge_x - 0.006, ys + 0.05, zT - 0.06, R=self.R), 0.0, opened, 0.4)]
        lay_in, lay_dz, so = getattr(args, "sleeve_lay_in", 0.0), getattr(args, "sleeve_lay_dz", 0.0), getattr(args, "sleeve_slide_out", 0.0)
        if lay_in > 0.0 or lay_dz > 0.0 or so > 0.0:
            # lay the sleeve down before letting go: from the flip pose move the hand inward / down (flip orientation kept, no
            # further wrist roll -- a later flip angle sends the arm IK onto another branch), release there, slide the hand back
            # (-x) at that height so the thumb comes out from under the sleeve instead of lifting it, then up / out as before
            yl, zl = yf - lay_in, zf_ - lay_dz; t1 = 8.8; tail = []
            if lay_in > 0.0 or lay_dz > 0.0:
                t1 += 0.4; tail.append((t1, "s_lay", sp(xp, yl, zl, R=R_f), 0.0, carry, args.sleeve_flip_rotw))
            t1 += 0.4; tail.append((t1, "s_release", sp(xp, yl, zl, R=R_f), 0.0, rel, 0.2))
            pz = getattr(args, "sleeve_press_dz", 0.0)
            if pz > 0.0:     # lower the opened hand onto the laid sleeve (the thumb under its end comes down towards the body)
                zl -= pz; t1 += 0.4; tail.append((t1, "s_press", sp(xp, yl, zl, R=R_f), 0.0, rel, 0.2))
            if so > 0.0:
                t1 += 0.6; tail.append((t1, "s_slide", sp(xp - so, yl, zl, R=R_f), 0.0, rel, 0.2))
            tail += [(t1 + 0.8, "s_up", sp(xp, yl, zl + 0.06, R=R_f), 0.0, rel, 0.2),
                     (t1 + 1.8, "s_out", sp(xp - 0.03, ys + 0.09, zl + 0.06, R=R_f), 0.0, opened, 0.2),
                     (t1 + 3.6, "j_swap", sp(args.table_edge_x - 0.006, ys + 0.06, zT, R=self.R), 0.0, opened, 0.4),
                     (t1 + 4.6, "s_down", sp(args.table_edge_x - 0.006, ys + 0.05, zT - 0.06, R=self.R), 0.0, opened, 0.4)]
        if k_s != 1.0:
            tail = [(w[0] + 3.8 * (k_s - 1.0),) + tuple(w[1:]) for w in tail]
        way += tail
        return way

    def build_dense(self, step=0.04):
        """Cartesian min-jerk interpolation between waypoints (palm pinch point; fixed orientation), IK solved every `step`
        seconds warm-started from the previous sample. Joint-space interpolation between far-apart IK solutions swings the
        hands off the straight path (the thumb crossed the hanging flap during the rise) — this keeps the hands on the line."""
        self.dense_t, self.dense_q = [], []
        q = dict(self.q_way[0])
        # achieved start pose (the hover target may be unreachable): start the first segment from the FK of q_way[0]
        Tfk = self.kin.fk(q)
        start = {sd: (Tfk[f"{sd}_hand_palm_link"][:3, 3] + np.array([0.0, 0.0, PELVIS_Z]) + Tfk[f"{sd}_hand_palm_link"][:3, :3] @ self.pinch[sd][1]) for sd in self.sides}
        t_prev = 0.0; w_prev = self.way[0][3]; self.dense_report = []
        R_prev = {sd: self.way[0][2][sd][1] for sd in self.sides}
        def slerp(Ra, Rb, a):
            if np.allclose(Ra, Rb):
                return Rb
            from scipy.spatial.transform import Rotation as Rot
            rv = Rot.from_matrix(Rb @ Ra.T).as_rotvec()
            return Rot.from_rotvec(a * rv).as_matrix() @ Ra
        for iw, (t_end, label, cart, waist, fing, rot_w) in enumerate(self.way):
            n = max(1, int(round((t_end - t_prev) / step))); worst = (0.0, 0.0, 0.0)
            if label.startswith("j_"):           # joint-space segment between the two waypoint IK solutions
                qa, qb = dict(q), self.q_way[iw]
                for i in range(1, n + 1):
                    t = t_prev + (t_end - t_prev) * i / n; a = minjerk(i / n)
                    q_prev = dict(q)
                    q = {k: (1 - a) * qa.get(k, 0.0) + a * qb.get(k, 0.0) for k in set(qa) | set(qb)}
                    q["waist_pitch_joint"] = (1 - a) * w_prev + a * waist
                    jump = max(abs(q[k] - q_prev.get(k, 0.0)) for k in q if not k.endswith("_hand_thumb_0_joint"))
                    worst = max(worst, (jump, 0.0, t))
                    self.dense_t.append(t); self.dense_q.append(dict(q))
                self.dense_report.append((label, worst))
                start = {sd: cart[sd][0] for sd in self.sides}; R_prev = {sd: cart[sd][1] for sd in self.sides}
                t_prev, w_prev = t_end, waist
                continue
            for i in range(1, n + 1):
                t = t_prev + (t_end - t_prev) * i / n
                a = minjerk((t - t_prev) / max(1e-6, t_end - t_prev))
                q["waist_pitch_joint"] = (1 - a) * w_prev + a * waist
                for sd in self.sides:
                    p_t = (1 - a) * start[sd] + a * cart[sd][0]; R_w = slerp(R_prev[sd], cart[sd][1], a)
                    T_t = palm_target(p_t - np.array([0.0, 0.0, PELVIS_Z]), R_w, self.pinch[sd][1])
                    chain = self.kin.chain(f"{sd}_hand_palm_link", "torso_link")
                    q_prev = dict(q)
                    q1, ep, er = self.kin.ik(f"{sd}_hand_palm_link", T_t, dict(q_prev), chain, iters=(150 if label.startswith("s_") else 40), rot_w=rot_w, lam=0.03, step=0.9)
                    jump = max(abs(q1[k] - q_prev.get(k, 0.0)) for k in chain)
                    if ep > (0.001 if label.startswith("s_") else 0.004) or jump > 0.25:     # sleeve segments: retry above 1 mm (others unchanged)
                        # warm start lost the branch: retry from the joint-space blend of the two waypoint solutions (same IK
                        # branch as the waypoints), keep the result closest to the previous sample among the accurate ones
                        qa = self.q_way[iw - 1] if iw > 0 else self.q_way[0]; qb = self.q_way[iw]; cands = [(ep, jump, q1)]
                        for seed in ({k: (1 - a) * qa.get(k, 0.0) + a * qb.get(k, 0.0) for k in set(qa) | set(qb)}, dict(qb), dict(qa)):
                            sd_q = dict(q_prev); sd_q.update({k: seed[k] for k in chain if k in seed})
                            q2, ep2, er2 = self.kin.ik(f"{sd}_hand_palm_link", T_t, sd_q, chain, iters=300, rot_w=rot_w, lam=0.03, step=0.9)
                            cands.append((ep2, max(abs(q2[k] - q_prev.get(k, 0.0)) for k in chain), q2))
                        if label.startswith("s_"):    # sleeve segments: the most accurate retry that stays on the branch
                            good = [c for c in cands if c[1] < 0.25] or cands
                            ep, jump, q1 = min(good, key=lambda c: (c[0], c[1]))
                        else:
                            good = [c for c in cands if c[0] < 0.004] or cands
                            ep, jump, q1 = min(good, key=lambda c: (c[1], c[0]))
                    q = q1; worst = max(worst, (jump, ep, t))
                self.dense_t.append(t); self.dense_q.append(dict(q))
            self.dense_report.append((label, worst))
            start = {sd: cart[sd][0] for sd in self.sides}; R_prev = {sd: cart[sd][1] for sd in self.sides}
            t_prev, w_prev = t_end, waist
        self.dense_t = np.array(self.dense_t)

    def joint_targets(self, t):
        """Joint targets at time t: linear interpolation of the dense Cartesian-path IK samples; finger state per side; label."""
        if not hasattr(self, "dense_t"):
            self.build_dense()
        label = self.way[-1][1]; f_prev = self.way[0][4]; t_prev = 0.0; f = dict(self.way[-1][4])
        for (t_end, lab, cart, waist, fing, _rw) in self.way:
            if t <= t_end:
                a = minjerk((t - t_prev) / max(1e-6, t_end - t_prev)); label = lab
                f = {sd: (1 - a) * f_prev[sd] + a * fing[sd] for sd in self.sides}
                break
            t_prev, f_prev = t_end, fing
        i = int(np.searchsorted(self.dense_t, t))
        if i <= 0:
            return dict(self.dense_q[0]), f, label
        if i >= len(self.dense_t):
            return dict(self.dense_q[-1]), f, label
        t0, t1 = self.dense_t[i - 1], self.dense_t[i]; a = (t - t0) / max(1e-9, t1 - t0)
        q0, q1 = self.dense_q[i - 1], self.dense_q[i]
        return {k: (1 - a) * q0.get(k, 0.0) + a * v for k, v in q1.items()}, f, label

    def finger_targets(self, side, f):
        if np.ndim(f):                        # (thumb, index) closure pair (--sleeve-index-first)
            out = {k: (float(f[0]) if "_thumb_" in k else float(f[1])) * v for k, v in self.pinch[side][0].items()}
        else:
            out = {k: f * v for k, v in self.pinch[side][0].items()}
        if self.fingers == "side":            # thumb-index pinch only: a closing middle finger sweeps through the flap and pushes it past the thumb
            for k in out:
                if "_middle_" in k:
                    out[k] = 0.0
        return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/nuka/fold_dry")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--table-z", type=float, default=0.90)
    ap.add_argument("--table-edge-x", type=float, default=0.27, help="near edge of the table (z-up x)")
    ap.add_argument("--grip", default="front", choices=["front", "flap"])
    ap.add_argument("--prebend", type=float, default=0.15, help="flap grip: hem flap length that starts hanging (mesh shirt_bent15.obj)")
    ap.add_argument("--overhang", type=float, default=0.03, help="front grip: hem overhang beyond the table edge (flat mesh shirt_flat.obj)")
    ap.add_argument("--droop", type=float, default=0.004, help="front grip: expected hem droop at the pinch point")
    ap.add_argument("--pinch-dx", type=float, default=0.05, help="flap grip: pinch point behind the flap plane (the open thumb must start behind the flap; the index sweeps the cloth into it)")
    ap.add_argument("--dx-right-extra", type=float, default=0.0, help="flap grip: extra pinch offset for the right hand")
    ap.add_argument("--shirt-len", type=float, default=0.30)
    ap.add_argument("--shirt-w", type=float, default=0.28)
    ap.add_argument("--pinch-inset", type=float, default=0.03, help="pinch point inset from the shirt side edge")
    ap.add_argument("--pinch-depth", type=float, default=0.015, help="(unused with the flap grip)")
    ap.add_argument("--table-thick", type=float, default=0.04)
    ap.add_argument("--grip-below", type=float, default=0.055, help="grip point below the table slab bottom (thumb top must clear the slab)")
    ap.add_argument("--waist-far", type=float, default=0.5, help="waist pitch (rad) at the far lay-down point")
    ap.add_argument("--finger-strength", type=float, default=0.05)
    ap.add_argument("--cloth-young", type=float, default=2e5)
    ap.add_argument("--bend-young", type=float, default=3e5)
    ap.add_argument("--cloth-density", type=float, default=200.0)
    ap.add_argument("--mu", type=float, default=0.6)
    ap.add_argument("--dhat", type=float, default=2e-3)
    ap.add_argument("--newton-cap", type=int, default=50)
    ap.add_argument("--ptype", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=None)
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--max-rev-step", type=float, default=100.0,
                    help="joint slew cap (rad per step); goes through the Config: set_max_revolute_step_per_frame() before finalize() is a no-op "
                         "and finalize() applies the Config value (0.1 in robotgym.common.Scene)")
    ap.add_argument("--finger-test", type=float, default=None, help="diagnostic: close both hands from this time over 0.5 s, ignoring the plan")
    ap.add_argument("--no-cloth", action="store_true", help="diagnostic: no shirt")
    ap.add_argument("--fingers", default="down", choices=["down", "up", "side"], help="flap grip: finger direction at the pinch")
    ap.add_argument("--side-out", type=float, default=0.06, help="side approach: start this far outside the pinch point (m)")
    ap.add_argument("--side-hover-dy", type=float, default=0.03, help="side approach: hover this much further out than the approach (m)")
    ap.add_argument("--side-hover-dz", type=float, default=0.04, help="side approach: hover this much lower than the grip (m)")
    ap.add_argument("--pull", type=float, default=0.04, help="side grasp: pull the pinched corners this far toward the robot before lifting (m)")
    ap.add_argument("--hand-mu", type=float, default=None, help="robot link friction (pair mu = geometric mean with the cloth's)")
    ap.add_argument("--arc-slack", type=float, default=0.0, help="arc: carry the hem on a radius this much shorter than the hanging length "
                    "after the pull (slack in the flap; 0 = taut, drags the shirt)")
    ap.add_argument("--arc-slack-lift", type=float, default=0.0, help="arc: extra slack at the 110 deg lift waypoint only (where the taut "
                    "flap pulls the table half toward the robot)")
    ap.add_argument("--shirt-mesh", default="", help="shirt OBJ under Assets/nuka (overrides the prebend / sleeve name), e.g. shirt_bent17s7_sl141.obj (0.07 m sleeves)")
    ap.add_argument("--sleeves", action="store_true", help="fold both sleeves in first (needs the pre-bent sleeve mesh shirt_bentNN_slYYY.obj, "
                    "--table-half-w ~ys and a thin slab)")
    ap.add_argument("--sleeve-ys", type=float, default=0.141, help="sleeve fold line |y| (mesh bend; table half-width just outside it)")
    ap.add_argument("--sleeve-x", type=float, default=0.355, help="sleeve pinch x (world; sleeves span ~0.334-0.433)")
    ap.add_argument("--sleeve-z", type=float, default=0.868, help="sleeve pinch height (flap hangs to ~0.865)")
    ap.add_argument("--sleeve-in", type=float, default=0.009, help="pinch point inside the hanging sleeve plane (the thumb starts behind it)")
    ap.add_argument("--sleeve-open", type=float, default=0.5, help="hand closure while approaching the sleeve (fully open, the thumb hits the slab)")
    ap.add_argument("--sleeve-flip", type=float, default=240.0, help="release angle on the arc about the table's side edge")
    ap.add_argument("--sleeve-roll", type=float, default=0.0, help="sleeve arc: hand roll about the side edge as a fraction of the arc angle (1 = turns like a page)")
    ap.add_argument("--sleeve-release-open", type=float, default=0.5, help="hand closure at the sleeve release (0 = fully open)")
    ap.add_argument("--sleeve-flip-rotw", type=float, default=0.2, help="IK orientation weight on the sleeve flip segment")
    ap.add_argument("--sleeve-index-first", action="store_true", help="sleeves: approach with the index already closed (outside the flap) "
                    "and the thumb open (--sleeve-thumb-open); at the pinch only the thumb closes, pressing the flap against the index")
    ap.add_argument("--sleeve-thumb-open", type=float, default=0.0, help="--sleeve-index-first: thumb closure while approaching")
    ap.add_argument("--sleeve-finger-strength", type=float, default=None, help="finger drive strength from the sleeve pinch to the flip "
                    "(default: --finger-strength throughout)")
    ap.add_argument("--sleeve-slow", type=float, default=1.0, help="time stretch of the sleeve pull / lift / over / flip segments")
    ap.add_argument("--sleeve-thumb-close", type=float, default=1.0, help="--sleeve-index-first: thumb closure from the pinch to the flip "
                    "(> 1 over-closes the thumb against the index: firmer pinch; the index limits leave no room to over-close it)")
    ap.add_argument("--sleeve-release-index", action="store_true", help="sleeve release: open only the index (to --sleeve-release-open); "
                    "the thumb stays closed under the laid sleeve")
    ap.add_argument("--sleeve-slide-out", type=float, default=0.0, help="after the sleeve release, slide the hand this far back (-x) at the "
                    "release height before going up (pulls the thumb out from under the sleeve instead of lifting it)")
    ap.add_argument("--sleeve-lay-in", type=float, default=0.0, help="after the sleeve flip, move the hand this far inward (m) before the release")
    ap.add_argument("--sleeve-lay-dz", type=float, default=0.0, help="after the sleeve flip, move the hand this far down (m) before the release")
    ap.add_argument("--sleeve-press-dz", type=float, default=0.0, help="after the sleeve release, lower the opened hand this far (m) before sliding out")
    ap.add_argument("--table-half-w", type=float, default=0.19, help="table half-width (sleeves: just outside --sleeve-ys)")
    ap.add_argument("--over-roll", type=float, default=0.0, help="flip: roll the hands this many degrees about the fold axis at the over / "
                    "shift waypoints (reachable on the G1: 0 to -45)")
    ap.add_argument("--flip-roll", type=float, default=0.0, help="flip: hand roll (deg) at the flip / release waypoints (reachable: 0 to -60)")
    ap.add_argument("--fold-shift", type=float, default=0.0, help="flip: translate the hands this far (+x) with the flap vertical, then "
                    "flip about the shifted fold line")
    ap.add_argument("--flip-at", type=float, default=None, help="arc: open the hands at this angle about the fold line (deg, e.g. 215) and let the flap fall over")
    ap.add_argument("--slow", type=float, default=1.0, help="arc: time stretch of the pull / lift / over / carry / lay segments")
    ap.add_argument("--table-mu", type=float, default=None, help="table friction (pairs use the geometric mean with the other body's mu)")
    ap.add_argument("--table-legs", action="store_true", help="four fixed legs under the slab (looks; well clear of the hands)")
    ap.add_argument("--arc", action="store_true", help="flap grip: carry the hem on a circle about the fold line (radius = pinch depth)")
    ap.add_argument("--arc-pull", type=float, default=35.0, help="arc: first waypoint after the pinch (deg about the fold line)")
    ap.add_argument("--arc-waist", type=float, default=0.2, help="arc: waist pitch while folding over (rad)")
    ap.add_argument("--arc-lay", type=float, default=268.0, help="arc: lay-down angle about the fold line (deg; 270 = flat)")
    ap.add_argument("--lay-dz", type=float, default=0.035, help="arc: lowest pinch-point height above the table top at the lay-down (m)")
    ap.add_argument("--diag-from", type=float, default=None, help="diagnostic: from this time on, print finger angles and the clearance of every "
                                                                        "hand link to the cloth / table slab every 0.1 s")
    args = ap.parse_args()

    plan = Plan(args)
    print(f"[plan] pinch gaps: left {plan.pinch['left'][3]*1e3:.1f} mm, right {plan.pinch['right'][3]*1e3:.1f} mm; pinch point (palm frame) left {np.round(plan.pinch['left'][1], 3).tolist()} right {np.round(plan.pinch['right'][1], 3).tolist()}")
    for (label, errs), q in zip(plan.ik_err, plan.q_way):
        print(f"[plan] {label:8s} IK err left {errs['left'][0]*1e3:5.1f} mm / {math.degrees(errs['left'][1]):4.1f} deg, right {errs['right'][0]*1e3:5.1f} mm / {math.degrees(errs['right'][1]):4.1f} deg, waist {q.get('waist_pitch_joint', 0):.2f}")
    if args.dry_run:
        # table / hand hull clearance at the pinch waypoint
        wi = {w[1]: i for i, w in enumerate(plan.way)}          # by label: a prepended sleeve phase shifts the indices
        for tag, iw, closed in (("reach(open)", wi["reach"], False), ("pinch(closed)", wi["pinch"], True)):
            q = dict(plan.q_way[iw])
            if closed:
                for s in plan.sides:
                    q.update(plan.finger_targets(s, 1.0))
            T = plan.kin.fk(q)
            for s in plan.sides:
                for l in ("thumb_2", "index_1", "palm"):
                    P = hull_pts(f"{s}_hand_{l}_link"); Pw = (T[f"{s}_hand_{l}_link"] @ np.c_[P, np.ones(len(P))].T).T[:, :3] + np.array([0, 0, PELVIS_Z])
                    print(f"[plan] {tag:13s} {s:5s} {l:8s} hull x [{Pw[:,0].min():.3f},{Pw[:,0].max():.3f}] y [{Pw[:,1].min():.3f},{Pw[:,1].max():.3f}] z [{Pw[:,2].min():.3f},{Pw[:,2].max():.3f}]")
        if args.grip == "flap":
            xf = args.table_edge_x - 0.006; z_lo, z_hi = args.table_z + 0.004 - args.prebend, args.table_z; y_hw = args.shirt_w / 2
            for tag, iw, closed in (("approach(open)", wi["approach"], False), ("reach(open)", wi["reach"], False), ("pinch(closed)", wi["pinch"], True)):
                q = dict(plan.q_way[iw])
                if closed:
                    for sd in plan.sides:
                        q.update(plan.finger_targets(sd, 1.0))
                T = plan.kin.fk(q); bad = []; near = []
                for sd in plan.sides:
                    for l in ("palm", "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"):
                        P = hull_pts(f"{sd}_hand_{l}_link"); Pw = (T[f"{sd}_hand_{l}_link"] @ np.c_[P, np.ones(len(P))].T).T[:, :3] + np.array([0, 0, PELVIS_Z])
                        inside = (np.abs(Pw[:, 1]) < y_hw) & (Pw[:, 2] > z_lo) & (Pw[:, 2] < z_hi)
                        if inside.any():
                            dxp = Pw[inside, 0] - xf
                            if dxp.min() < 0 < dxp.max(): bad.append(f"{sd[0].upper()}.{l}[{dxp.min()*1e3:+.0f},{dxp.max()*1e3:+.0f}]")
                            else: near.append(f"{sd[0].upper()}.{l}[{dxp.min()*1e3:+.0f},{dxp.max()*1e3:+.0f}]")
                    W = hull_pts(f"{sd}_wrist_yaw_link"); Ww = (T[f"{sd}_wrist_yaw_link"] @ np.c_[W, np.ones(len(W))].T).T[:, :3] + np.array([0, 0, PELVIS_Z])
                    inside = (np.abs(Ww[:, 1]) < y_hw) & (Ww[:, 2] > z_lo) & (Ww[:, 2] < z_hi)
                    if inside.any() and (Ww[inside, 0] - xf).min() < 0 < (Ww[inside, 0] - xf).max(): bad.append(f"{sd[0].upper()}.wrist")
                print(f"[plan] {tag:14s} links CROSSING the flap: {bad if bad else 'none'} | in front of / behind it: {near}")
        if args.sleeves:
            # sleeve gates: hand links vs the slab (penetration, clearance under it) and vs the hanging sleeve plane |y| = ys inside
            # the sleeve region (no link may straddle it before the pinch; at the pinch only the thumb / index pads)
            ys = args.sleeve_ys; yt = args.table_half_w; zb, zt = args.table_z - args.table_thick, args.table_z
            idx = {w[1]: i for i, w in enumerate(plan.way)}
            for tag in ("s_hover", "s_reach", "s_pinch", "s_pull", "s_lift", "s_over", "s_flip", "s_release", "s_up"):
                q = dict(plan.q_way[idx[tag]])
                f = plan.way[idx[tag]][4]
                for sd in plan.sides:
                    q.update(plan.finger_targets(sd, f[sd]))
                T = plan.kin.fk(q); pen = []; cross = []; clear = 9.0; above = 9.0
                for sd, sg in (("left", 1.0), ("right", -1.0)):
                    for l in ("palm", "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"):
                        P = hull_pts(f"{sd}_hand_{l}_link"); Pw = (T[f"{sd}_hand_{l}_link"] @ np.c_[P, np.ones(len(P))].T).T[:, :3] + np.array([0, 0, PELVIS_Z])
                        ins = (Pw[:, 0] > args.table_edge_x) & (np.abs(Pw[:, 1]) <= yt) & (Pw[:, 2] > zb) & (Pw[:, 2] < zt)
                        if ins.any(): pen.append(f"{sd[0].upper()}.{l}({ins.sum()})")
                        under = (Pw[:, 0] > args.table_edge_x) & (np.abs(Pw[:, 1]) <= yt) & (Pw[:, 2] <= zb)
                        if under.any(): clear = min(clear, (zb - Pw[under, 2]).min())
                        over = (Pw[:, 0] > args.table_edge_x) & (np.abs(Pw[:, 1]) <= yt) & (Pw[:, 2] >= zt)
                        if over.any(): above = min(above, (Pw[over, 2] - zt).min())
                        reg = (Pw[:, 0] > 0.334) & (Pw[:, 0] < 0.433) & (Pw[:, 2] > 0.862) & (Pw[:, 2] < zt)
                        if reg.any():
                            yy = sg * Pw[reg, 1] - ys
                            if yy.min() < 0 < yy.max(): cross.append(f"{sd[0].upper()}.{l}[{yy.min()*1e3:+.0f},{yy.max()*1e3:+.0f}]")
                print(f"[plan] sleeve {tag:9s} slab-penetrating {pen or 'none'} | under-slab clearance {clear*1e3:.1f} mm | above-top clearance {above*1e3:.1f} mm | links CROSSING the sleeve plane {cross or 'none'}")
            plan.build_dense()
            t0, t1 = plan.way[idx["s_pre"]][0], plan.way[idx["s_down"]][0]
            worst_pen = (0, None)
            for t in plan.dense_t:
                if not (t0 <= t <= t1): continue
                q, fs, _lab = plan.joint_targets(t)                 # arm samples + the scheduled finger closure at t
                for sd in plan.sides:
                    q.update(plan.finger_targets(sd, fs[sd]))
                T = plan.kin.fk(q); npen = 0
                for sd in plan.sides:
                    for l in ("palm", "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"):
                        P = hull_pts(f"{sd}_hand_{l}_link"); Pw = (T[f"{sd}_hand_{l}_link"] @ np.c_[P, np.ones(len(P))].T).T[:, :3] + np.array([0, 0, PELVIS_Z])
                        npen += int(((Pw[:, 0] > args.table_edge_x) & (np.abs(Pw[:, 1]) <= yt) & (Pw[:, 2] > zb) & (Pw[:, 2] < zt)).sum())
                if npen > worst_pen[0]: worst_pen = (npen, t)
            print(f"[plan] sleeve hand-off sweep {t0:.1f}-{t1:.1f} s: max hull points inside the slab {worst_pen[0]} (t {worst_pen[1]})")
        print(f"[plan] table edge x {args.table_edge_x}, top z {args.table_z}, slab bottom z {args.table_z - args.table_thick:.3f}; flap plane x {args.table_edge_x - 0.006:.3f}, grip at x {args.table_edge_x - 0.006 + args.pinch_dx:.3f} z {args.table_z - args.table_thick - args.grip_below:.3f}, flap bottom z {args.table_z + 0.004 - args.prebend:.3f}")
        plan.build_dense()
        for label, (jump, ep, tj) in plan.dense_report:
            print(f"[plan] dense {label:8s} max joint step {jump:.3f} rad (t {tj:.2f}), pos err {ep*1e3:.1f} mm")
        for t in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.5, 6.2, 7.0, 8.0, 9.5, 11.0, 12.0, 13.5, 15.0, 17.0):
            qj, f, label = plan.joint_targets(t); q = dict(qj)
            for sd in plan.sides:
                q.update(plan.finger_targets(sd, f[sd]))
            T = plan.kin.fk(q); out = []
            for sd in plan.sides:
                for l in ("thumb_2", "index_1", "palm"):
                    P = hull_pts(f"{sd}_hand_{l}_link"); Pw = (T[f"{sd}_hand_{l}_link"] @ np.c_[P, np.ones(len(P))].T).T[:, :3] + np.array([0, 0, PELVIS_Z])
                    out.append(f"{sd[0].upper()}.{l[:5]} x[{Pw[:,0].min():.2f},{Pw[:,0].max():.2f}] z[{Pw[:,2].min():.2f},{Pw[:,2].max():.2f}]")
                W = hull_pts(f"{sd}_wrist_yaw_link"); Ww = (T[f"{sd}_wrist_yaw_link"] @ np.c_[W, np.ones(len(W))].T).T[:, :3] + np.array([0, 0, PELVIS_Z])
                out.append(f"{sd[0].upper()}.wrist x[{Ww[:,0].min():.2f},{Ww[:,0].max():.2f}] z[{Ww[:,2].min():.2f},{Ww[:,2].max():.2f}]")
            print(f"[trace t={t:4.1f} {label:8s}] " + " | ".join(out))
        return

    from common import Body, Recorder, Scene, _load_urdf_capture  # noqa: E402
    os.makedirs(args.out, exist_ok=True)
    t_ik = time.time(); plan.build_dense(); print(f"[plan] dense Cartesian path: {len(plan.dense_t)} IK samples in {time.time()-t_ik:.0f} s", flush=True)
    for label, (jump, ep, tj) in plan.dense_report:
        print(f"[plan] dense {label:8s} max joint step {jump:.3f} rad (t {tj:.2f}), pos err {ep*1e3:.1f} mm", flush=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=args.ptype, newton_iter_cap=args.newton_cap, cloth_thickness=1e-3,
                    cloth_young_modulus=args.cloth_young, bend_young_modulus=args.bend_young, cloth_density=args.cloth_density,
                    strain_rate=100.0, absolute_dhat=args.dhat, friction_rate=args.mu, max_revolute_step_per_frame=args.max_rev_step),
               log_level=args.log_level)
    eng = sc.eng
    # G1, fixed pelvis, loaded at the first waypoint pose
    q0 = plan.q_way[0]
    T = np.eye(4); T[:3, :3] = R_CONV; T[:3, 3] = R_CONV @ np.array([0.0, 0.0, PELVIS_Z])
    init = {k: float(v) for k, v in q0.items() if abs(v) > 1e-9}
    n_rec0 = len(eng.get_load_records())
    log = _load_urdf_capture(eng, URDF, T, True, False, 1e7, init)
    eng.set_log_level(args.log_level)
    bad = [l for l in log.splitlines() if "Failed" in l]
    if bad:
        print("[warn] URDF import:", bad[:3], flush=True)
    robot = []
    for r in eng.get_load_records()[n_rec0:]:
        if r.body_type != 0:
            continue
        b = Body(r, f"g1/{r.label}", "prop", color=(0.82, 0.82, 0.84)); sc.bodies[b.name] = b; robot.append(b)
        eng.add_ground_collision_skip(b.body_id); eng.set_body_friction(b.body_id, args.hand_mu if args.hand_mu is not None else args.mu)
    n_rev = eng.native.get_num_revolute_joints()
    rev = {eng.get_revolute_joint_info(i).name: i for i in range(n_rev)}
    hand_ids = {s: [rev[f"{s}_hand_{k}_joint"] for k in HAND_JOINTS] for s in plan.sides}
    for s in plan.sides:
        for j in hand_ids[s]:
            eng.set_revolute_strength(int(j), args.finger_strength)
    eng.set_max_revolute_step_per_frame(100.0)
    # table (fixed box) and the shirt
    tx0, tx1 = args.table_edge_x, args.table_edge_x + 0.60; ty = args.table_half_w
    # per-pair friction is the geometric mean of the two bodies' mu: a grippy table holds the light shirt without making the fingers sticky
    table_mu = args.table_mu if args.table_mu is not None else args.mu
    table = sc.add_box("table", (0.5 * (tx0 + tx1), args.table_z - args.table_thick / 2, 0.0), (tx1 - tx0, args.table_thick, 2 * ty), fixed=True, cell=0.05, mu=table_mu, color=(0.55, 0.45, 0.35))
    eng.add_ground_collision_skip(table.body_id)
    if args.table_legs:   # four legs well behind the near edge (the hands and the hanging flap work under the first ~5 cm)
        for i, (lx, lz) in enumerate(((tx0 + 0.18, ty - 0.03), (tx0 + 0.18, -ty + 0.03), (tx1 - 0.04, ty - 0.03), (tx1 - 0.04, -ty + 0.03))):
            h = args.table_z - args.table_thick - 0.002
            leg = sc.add_box(f"table_leg{i}", (lx, 0.5 * h + 0.001, lz), (0.035, h, 0.035), fixed=True, cell=0.05, mu=args.mu, color=(0.55, 0.45, 0.35))
            eng.add_ground_collision_skip(leg.body_id)
    x_edge = args.table_edge_x - 0.006
    Ts = np.eye(4); Ts[1, 3] = args.table_z + 0.004
    if args.grip == "flap":
        Ts[0, 3] = x_edge - args.prebend; mesh = f"shirt_bent{int(round(args.prebend*100)):02d}.obj"
        if args.sleeves:
            mesh = f"shirt_bent{int(round(args.prebend*100)):02d}_sl{int(round(args.sleeve_ys*1000))}.obj"
        if args.shirt_mesh:
            mesh = args.shirt_mesh      # flap hangs at x = prebend, flat part along +x
    else:
        Ts[0, 3] = args.table_edge_x - args.overhang; mesh = "shirt_flat.obj"
        x_edge = Ts[0, 3]
    if not args.no_cloth:
        cloth = sc.add_cloth("shirt", osp.join(ASSETS, "nuka", mesh), Ts, young=args.cloth_young, color=(0.62, 0.78, 0.95))
        eng.set_body_friction(len(eng.get_load_records()) - 1, args.mu)
    else:
        cloth = {}
    sc.finalize()
    for s in plan.sides:                     # drive settings again after finalize (the slew cap above is the one that matters)
        for j in hand_ids[s]:
            eng.set_revolute_strength(int(j), args.finger_strength)
    eng.set_max_revolute_step_per_frame(args.max_rev_step)
    rec = Recorder(sc)
    S0 = sc.fem_verts("shirt") if not args.no_cloth else np.zeros((1, 3))
    x_half = x_edge + 0.5 * (args.shirt_len - (args.prebend if args.grip == "flap" else 0.0))
    hem_mask = (S0[:, 0] < x_half)                                              # near half of the shirt
    edge_mask = (S0[:, 1] < S0[:, 1].min() + 0.006) if args.grip == "flap" else (S0[:, 0] < S0[:, 0].min() + 0.006)   # the hem edge
    hand_links = [b for b in robot if any(k in b.rec.label for k in ("hand_palm", "thumb_2", "index_1", "wrist_yaw"))]
    def hands_bbox():
        out = []
        for b in hand_links:
            V = np.asarray(eng.get_abd_body(b.body_id).get_vertices(), float)
            out.append(f"{b.rec.label.replace('_link','').replace('left_hand_','L.').replace('right_hand_','R.').replace('left_','L.').replace('right_','R.')} x[{V[:,0].min():.2f},{V[:,0].max():.2f}] y[{V[:,1].min():.2f},{V[:,1].max():.2f}]")
        return " | ".join(out)
    finger_links = [b for b in robot if "_hand_" in b.rec.label]
    tbl_c = np.array([0.5 * (tx0 + tx1), args.table_z - args.table_thick / 2, 0.0]); tbl_h = 0.5 * np.array([tx1 - tx0, args.table_thick, 2 * ty])
    def clearance():
        from scipy.spatial import cKDTree
        Sc = sc.fem_verts("shirt") if not args.no_cloth else None; tr = cKDTree(Sc) if Sc is not None else None; out = []
        for b in finger_links:
            V = np.asarray(eng.get_abd_body(b.body_id).get_vertices(), float)
            q = np.abs(V - tbl_c) - tbl_h; dbox = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)   # box SDF
            dcl = tr.query(V)[0].min() if tr is not None else np.inf
            out.append(f"{b.rec.label.replace('_hand_','.').replace('_link','')}: cloth {dcl*1e3:5.1f} table {dbox.min()*1e3:6.1f}")
        return " | ".join(out)
    print(f"[scene] G1 {len(robot)} links / {n_rev} joints, table top {args.table_z} m from x {tx0:.2f}, shirt {cloth.get('n', '?')} verts ({mesh}), grip {args.grip}, shirt x from {Ts[0, 3]:.3f}", flush=True)
    n_steps = int(round((args.max_time or plan.total) / args.dt)); rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    meta = dict(example="g1_shirt_fold", dt=args.dt, fps=args.fps, fem_materials={"shirt": "cloth"}, fem_colors=[[0.62, 0.78, 0.95]],
                robot=dict(urdf="nuka/g1/g1_hull.urdf", prefix="g1/", T=T.tolist(), init=init))
    t0 = time.time(); newton0 = eng.native.get_total_newton_iters(); last_label = ""; k_last = 0
    s_grip = ("s_pinch", "s_pull", "s_lift", "s_over", "s_flip"); strength_now = args.finger_strength
    ffmt = lambda f: f"{f:.2f}" if np.ndim(f) == 0 else "/".join(f"{x:.2f}" for x in f)
    for k in range(n_steps + 1):
        t = k * args.dt
        qj, fstate, label = plan.joint_targets(t)
        if args.sleeve_finger_strength is not None:
            want = args.sleeve_finger_strength if label in s_grip else args.finger_strength
            if want != strength_now:
                for s in plan.sides:
                    for j in hand_ids[s]:
                        eng.set_revolute_strength(int(j), want)
                print(f"[{t:5.2f}s] finger strength {strength_now} -> {want} ({label})", flush=True); strength_now = want
        if args.finger_test is not None:
            fstate = {sd: minjerk((t - args.finger_test) / 0.5) for sd in plan.sides}
        for name, v in qj.items():
            if name in rev:
                eng.set_revolute_target(rev[name], float(v))
        for s in plan.sides:
            for name, v in plan.finger_targets(s, fstate[s]).items():
                eng.set_revolute_target(rev[name], float(v))
        diag = (args.finger_test is not None or (args.diag_from is not None and t >= args.diag_from - 1e-9)) and k % int(round(0.1 / args.dt)) == 0
        if label != last_label or k % int(round(0.5 / args.dt)) == 0 or diag:
            S = sc.fem_verts("shirt") if not args.no_cloth else S0; nit = eng.native.get_total_newton_iters()
            hem_mean_z = S[hem_mask, 1].mean(); folded = float(np.mean(S[hem_mask, 0] > x_half))
            E = S[edge_mask]
            print(f"[{t:5.2f}s] {label:8s} hem z mean {hem_mean_z*1e3:6.1f} mm, hem half on far half {folded*100:5.1f} %, shirt x [{S[:,0].min():.3f},{S[:,0].max():.3f}] | "
                  f"hem EDGE mean x {E[:,0].mean():.3f} y {E[:,1].mean():.3f} (y min {E[:,1].min():.3f}) | newton/step {(nit-newton0)/max(1,k-k_last):4.1f} | {(time.time()-t0)/60:4.1f} min", flush=True)
            if diag and args.diag_from is not None:
                print("[clear] " + clearance(), flush=True)
            if k % int(round(1.0 / args.dt)) == 0 or diag:
                print("[hands] " + hands_bbox(), flush=True)
                qm = np.asarray(eng.get_revolute_current_angles_abs(), float)
                for sd in plan.sides:
                    ft = plan.finger_targets(sd, fstate[sd])
                    print(f"[fingers] {sd} f={ffmt(fstate[sd])}: " + " ".join(f"{n.replace(sd+'_hand_','').replace('_joint','')}: {ft[n]:+.2f}/{qm[rev[n]]:+.2f}" for n in ft)
                          + f" | drive tau index_0 {eng.get_revolute_drive_torque(rev[f'{sd}_hand_index_0_joint']):+.3f} N m", flush=True)
                for sd in plan.sides:
                    chain = plan.kin.chain(f"{sd}_hand_palm_link", "torso_link")
                    tgt = [qj.get(n, 0.0) for n in chain]; meas = [qm[rev[n]] for n in chain]
                    Tfk = plan.kin.fk({n: v for n, v in qj.items()})[f"{sd}_hand_palm_link"]
                    p_fk = Tfk[:3, 3] + np.array([0.0, 0.0, PELVIS_Z])            # z-up
                    pb = [b for b in robot if b.rec.label == f"{sd}_hand_palm_link"][0]
                    C = eng.get_abd_body_transforms(np.array([pb.body_id], np.int32))[0]
                    from common import abd_world_tf
                    Mp = abd_world_tf(C) @ pb.C0inv
                    p_load_palm = (T @ np.append(plan.kin.fk(init)[f"{sd}_hand_palm_link"][:3, 3], 1.0))[:3]          # palm origin at load, engine frame
                    p_eng = R_CONV.T @ (Mp @ np.append(p_load_palm, 1.0))[:3]                                            # now, z-up
                    print(f"[joints] {sd}: " + " ".join(f"{n.replace(sd+'_','').replace('_joint','')}: {tv:+.2f}/{mv:+.2f}" for n, tv, mv in zip(chain, tgt, meas))
                          + f" | palm origin FK {np.round(p_fk, 3).tolist()} engine {np.round(p_eng, 3).tolist()}", flush=True)
            newton0 = nit; last_label = label; k_last = k
        if k % rec_every == 0:
            rec.record(t, 0, label)
            if k > 0 and k % int(round(3.0 / args.dt)) == 0:
                rec.save(osp.join(args.out, "recording.npz"), dict(meta, partial=True))
        sc.step()
    S = sc.fem_verts("shirt") if not args.no_cloth else S0
    meta["wall_min"] = (time.time() - t0) / 60
    meta["fold_fraction"] = float(np.mean(S[hem_mask, 0] > x_half))
    rec.save(osp.join(args.out, "recording.npz"), meta)
    print(f"[done] {n_steps} steps in {meta['wall_min']:.1f} min; hem half on the far half {meta['fold_fraction']*100:.1f} %, shirt x-extent {(S[:,0].max()-S[:,0].min())*1e3:.0f} mm -> {osp.join(args.out, 'recording.npz')}", flush=True)


if __name__ == "__main__":
    main()
