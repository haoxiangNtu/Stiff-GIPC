#!/usr/bin/env python3
"""FD manipulation TELEOP sandbox.

Fixed robot standing at a desk (table + 4 drawers) with 3 toys on the tabletop.
A polyscope UI drives:
  * left / right arm joints (7 each) via sliders
  * a WuJi-style GRASP synergy slider per hand (0 open .. 1 closed) that drives
    every finger joint of that hand together
  * each of the 4 drawers (prismatic joint) open/closed
Press "Run / stop" to advance physics; drag any slider to command targets.

Robot body-box proxies stay excluded from collision; only the hands + toys +
table + drawers collide.

Run:  STIFF_SKIP_CCD_SANITY=1 python examples/fd_manip_scene.py
"""
import sys, os, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _m in [m for m in list(sys.modules)
           if m == "stiff_physics" or m.startswith("stiff_physics.") or m == "pystiffgipc"]:
    del sys.modules[_m]
sys.meta_path[:] = [f for f in sys.meta_path
                    if "stiff_physics" not in type(f).__name__.lower()
                    and "stiff_physics" not in getattr(type(f), "__module__", "").lower()]
from stiff_physics import Engine, Config
from stiff_physics.robot import Robot

# fd_hands_col_lite.urdf = full-res <visual>, decimated <collision> (~6k verts/hand
# vs ~114k) — see FD-light/decimate_hands.py.  FD_URDF_FULL=1 restores full-res
# collision for A/B timing.
_URDF_DIR = "/home/ps/Downloads/FD-light/fd-urdf-full/FD-URDF"
# lite2 = tiered decimation (grasp faces high-res, mechanism links minimal) —
# fixes the shattered-fingers look of v1. FD_URDF_FULL=1 -> full-res collision;
# FD_URDF_V1=1 -> old flat-budget lite.
# default = fd_hands_col_wt.urdf: WATERTIGHT tiered-decimated hand collision
# (pymeshfix-repaired, see FD-light/watertight_hands.py). The raw Fudan hand STLs
# are all open shells -> wrong ABD mass/volume + ugly collision, hence the repair.
# default = fd_hands_col.urdf: FULL-PRECISION hand meshes (nice visual, NO box
# proxies on the hand). The engine's BVHSkip#3 already drops every link that's
# ground-skipped + excluded-from-all-pairs from the BVH BUILD — so with the
# fingertip-only whitelist, only the 5 fingertips (+ toys/drawers) enter the BVH,
# yet every link keeps its full mesh for rendering. (Body links are boxes here;
# whole-robot full-res needs the collision-buffer-from-active-faces patch.)
#   FD_URDF_TIPS=1 -> tips-only box proxies (lightest RAM)
#   FD_URDF_WT=1   -> watertight decimated hands
# FD_SKIN=1: SIMULATE a coarse proxy but RENDER the full-res visual mesh, its
# per-body transform recovered by Kabsch from the proxy each frame (fullres_skin).
# In skin mode the sim robot defaults to the lightest proxy (tips) unless the user
# forces FD_URDF — full precision is a render concern now, not a sim one.
SKIN = os.environ.get("FD_SKIN") == "1"
ROBOT = os.path.join(_URDF_DIR,
                     os.environ.get("FD_URDF")     if os.environ.get("FD_URDF")
                     else "fd_hands_col_tips.urdf"  if os.environ.get("FD_URDF_TIPS") == "1"
                     else "fd_hands_col_tips.urdf"  if SKIN
                     else "fd_hands_col_lite.urdf"  if os.environ.get("FD_URDF_V1") == "1"
                     else "fd_hands_col_lite2.urdf" if os.environ.get("FD_URDF_V2") == "1"
                     else "fd_hands_col_wt.urdf"    if os.environ.get("FD_URDF_WT") == "1"
                     else "fd_hands_col.urdf")
FULL_URDF = os.path.join(_URDF_DIR, "fd.urdf")   # source of full-res visual meshes
DESK  = "/home/ps/Downloads/geo/desk"
TOYS  = "/home/ps/Downloads/geo"

# ---- placement (from the confirmed layout pass) ----------------------------
ROBOT_TY, ROBOT_YAW, ROBOT_POS = 0.91, 90.0, (0.95, 0.0, 1.30)
DESK_YAW, DESK_POS = 0.0, (0.0, 0.0, 0.0)
TABLE_TOP_Y = 1.052
TOY_SPOTS = [(0.85, 0.40), (1.10, 0.38)]
TOY_FILES = ["Toy00.obj", "Toy02.obj"]
# Test scene: one drawer by default (top one — best hand height / reach).
# Re-enable all four with: FD_DRAWERS=drawer00,drawer01,drawer02,drawer03
DRAWERS = os.environ.get("FD_DRAWERS", "drawer00").split(",")

# ---- grasp synergy tuning --------------------------------------------------
FINGER_CLOSE_DEG = 55.0     # revolute finger flexion at grasp=1
FINGER_SLIDE     = 0.0      # prismatic finger travel (m) at grasp=1 (0 = off)
# per-hand flexion sign (left/right axes are mirrored). Curl INWARD by default;
# flip a hand's sign via FD_GRASP_SIGN_L / FD_GRASP_SIGN_R if it everts.
FINGER_SIGN_L    = float(os.environ.get("FD_GRASP_SIGN_L", "-1"))
FINGER_SIGN_R    = float(os.environ.get("FD_GRASP_SIGN_R", "-1"))
DRAWER_MAX       = 0.34     # max drawer opening (m)
DRAWER_AXIS      = np.array([0.0, 0.0, 1.0])   # drawers pull toward robot (+Z)


def load_obj(path):
    V, F = [], []
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            # FAN-triangulate n-gons. The old [1:4] slice DROPPED one triangle
            # per quad (Toy00.obj has 44 quads) -> the "closed" surface had 44
            # holes -> the divergence-theorem second moment integrated NEGATIVE
            # (trMxx=-0.0159) -> negative rotational kinetic curvature -> the
            # optimizer actively CHOSE flips = the 16 m/s toy kick.
            idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(V, np.float64), np.asarray(F, np.int32)


def yaw(deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    m = np.eye(4); m[0, 0] = c; m[0, 2] = s; m[2, 0] = -s; m[2, 2] = c
    return m


# Collision whitelist mode (see below): the table is excluded from collision, so
# the toys' support surface becomes the GROUND plane raised to tabletop height —
# only the toys ever see the ground.  Legacy mode keeps the floor at 0 and the
# real table supporting the toys.
WHITELIST = os.environ.get("FD_COLL_WHITELIST", "1") == "1"
# FD_FLOOR=1: no table; lower the physics ground to the REAL floor (y=0) and rest
# the FEM toy(s) on it — a plain "toy on the ground" demo (pairs with FD_SKIN).
# Overrides the WHITELIST tabletop-height ground trick.
FLOOR = os.environ.get("FD_FLOOR", "0") == "1"
GROUND_Y = 0.0 if FLOOR else (TABLE_TOP_Y if WHITELIST else 0.0)
# Scene composition knobs:
#   FD_NO_DESK=1  -> robot only (no table, no drawers, no toys)
#   FD_FLOOR=1    -> robot + toy(s) resting on the real floor (no table)
#   FD_TABLE=0    -> drop the table+drawers (keep toys; they rest on the raised
#                    ground plane at tabletop height)
#   FD_NTOYS=N    -> number of toys (0/1/2; default 2, or 1 in floor mode)
#   FD_NEWTON_CAP -> Newton iteration cap (default 150)
NO_DESK    = os.environ.get("FD_NO_DESK", "0") == "1" and not FLOOR
LOAD_TABLE = (not NO_DESK) and (not FLOOR) and os.environ.get("FD_TABLE", "1") == "1"
NTOYS      = 0 if NO_DESK else int(os.environ.get("FD_NTOYS", "1" if FLOOR else "2"))

eng = Engine(Config(gravity=(0.0, -9.8, 0.0), dt=float(os.environ.get("FD_DT", "0.01")),
                    ground_normal=(0, 1, 0), ground_offset=GROUND_Y,
                    newton_iter_cap=int(os.environ.get("FD_NEWTON_CAP", "150")),
                    # [v082 port] experiment knobs for the ABD-kick root-cause hunt
                    friction_rate=float(os.environ.get("FD_FRICTION", "0.4")),
                    absolute_dhat=float(os.environ.get("FD_ABS_DHAT", "0")),
                    # NOTE: pcg_tol 1e-6 was tested against the motor-drive toy
                    # kicks — kicks were bit-identical, so they are NOT PCG
                    # noise. Keep engine default; env-switchable for experiments.
                    pcg_tol=float(os.environ.get("FD_PCG_TOL", "1e-4")),
                    # default 0.002 m/frame would need 170 frames to open the
                    # drawer fully; 0.005 -> ~68 frames. (0.01 worked but its
                    # larger per-frame residual let solver noise kick the toys.)
                    max_prismatic_step_per_frame=0.005,
                    joint_strength_ratio=float(os.environ.get("FD_JSR", "100")),
                    prismatic_strength_ratio=float(os.environ.get("FD_PSR", "100")),
                    # global damping: kills the ringing of motor-dragged bodies
                    # (drawer) that caused 40-110-iteration Newton spike frames;
                    # also steadies teleop.
                    velocity_damping=float(os.environ.get("FD_DAMP", "0.3"))))

# --- robot (fixed base, motorised revolute joints, facing -Z) ---------------
Rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
Trobot = np.eye(4)
Trobot[:3, :3] = yaw(ROBOT_YAW)[:3, :3] @ Rx
Trobot[:3, 3] = (ROBOT_POS[0], ROBOT_TY, ROBOT_POS[2])
eng.native.load_urdf(os.path.abspath(ROBOT), Trobot, True, True, 1e7, {})  # root_fixed, motor

# --- re-add the finger four-bar loop-closure pins that URDF (a tree) dropped -
# on export.  Without them the connecting/slider bars float free. FD_HAND_LOOPS=0
# to disable. Geometry-inferred (see examples/hand_loops.py).
if os.environ.get("FD_HAND_LOOPS", "1") == "1":
    sys.path.insert(0, os.path.join(ROOT, "examples"))
    from hand_loops import add_loop_closures
    _robot_recs = {r.label: r for r in eng.get_load_records()}
    add_loop_closures(eng, os.path.abspath(ROBOT), _URDF_DIR, _robot_recs,
                      strength=float(os.environ.get("FD_LOOP_JS", "2000")))

# --- desk: fixed table + 4 free drawers -------------------------------------
Tdesk = yaw(DESK_YAW); Tdesk[:3, 3] = DESK_POS
table_id = None
drawer_body = {}
drawer_ctr = {}
for name in (["table"] + DRAWERS if LOAD_TABLE else []):
    v, f = load_obj(os.path.join(DESK, name + ".obj"))
    bt = "Fixed" if name == "table" else "Free"
    eng.load_mesh_from_data(v, f, verts_per_face=3, dimensions=3, body_type="ABD",
                            transform=Tdesk, young_modulus=1e8, boundary_type=bt)
    bid = eng.get_load_records()[-1].body_offset
    if name == "table":
        table_id = bid
    else:
        drawer_body[name] = bid
        # Anchor the rail at the VOLUME CENTROID, not the bbox centre: the
        # open-top drawer's COM sits low, and a drive force offset from the
        # COM pumps a pitch torque into the softly-locked rotational DOFs
        # every frame (the Newton-spike source).
        try:
            import trimesh as _tm
            _mm = _tm.Trimesh(vertices=v, faces=f, process=False)
            c = _mm.center_mass if _mm.is_watertight else (v.min(0) + v.max(0)) / 2.0
        except Exception:
            c = (v.min(0) + v.max(0)) / 2.0
        drawer_ctr[name] = (Tdesk @ np.append(c, 1.0))[:3]
        # Enclosed-shell volume x density 1e3 gives an absurd 22 kg drawer;
        # 150 kg/m^3 -> ~3.3 kg (realistic), so the drive spring doesn't ring.
        eng.native.set_abd_body_density(bid, 150.0)

# prismatic joint: table <-> each drawer, slide along +Z (toward robot).
# FD_DRAWER_RAIL=joint  -> exclude table<->drawer collision (rail = joint only)
# default (contact)     -> keep real sliding contact in the cavity
RAIL_JOINT_ONLY = os.environ.get("FD_DRAWER_RAIL", "contact") == "joint"
drawer_joint = {}
for name, bid in drawer_body.items():
    j = eng.native.add_prismatic_joint(table_id, bid, np.asarray(drawer_ctr[name], float),
                                       DRAWER_AXIS, 0.0, DRAWER_MAX, name)
    drawer_joint[name] = j
    if RAIL_JOINT_ONLY:
        eng.add_collision_exclusion(table_id, bid)
    eng.add_ground_collision_skip(bid)
if table_id is not None:
    eng.add_ground_collision_skip(table_id)      # static-vs-ground pairs are wasted

# --- toys on the tabletop ----------------------------------------------------
# Toys are TETRAHEDRAL FEM SOFT bodies: the .obj surface is tet-meshed once by
# pytetwild (fTetWild) -> ~1k tets, cached as <name>.tet.npz (see
# geo/tetify_toys.py), then loaded body_type="FEM".  A soft tet body distributes
# mass over real tet volume and absorbs contact energy into deformation, so a
# grasp actually conforms — unlike the old ABD affine-solid toys (12-DOF rigid,
# which got launched by the resting-contact barrier since a rigid block has no
# internal DOF to absorb the impulse).
# ENGINE ORDER CONSTRAINT: all ABD bodies (robot/table/drawers) must load BEFORE
# any FEM body — they do, toys are last.  Escape hatch: FD_TOY_ABD=1 restores
# the old rigid ABD toys for A/B.
sys.path.insert(0, TOYS)
from tetify_toys import tetify

TOY_ABD   = os.environ.get("FD_TOY_ABD", "0") == "1"
TOY_YOUNG = float(os.environ.get("FD_TOY_YOUNG", "5e6"))   # FEM softness (Pa)
if FLOOR:
    # rest the toy on the real floor, in front of the robot (robot at x~0.95,
    # z~1.30 facing -Z; smaller z is "in front").
    TOY_SPOTS = [(0.95, 0.80), (0.72, 0.80)]
# FD_TOY_SPOT="x,z" (one or more, ';'-separated) overrides the toy world (x,z)
# spots — e.g. put the toy within an arm's reach of the hand. The hands rest near
# world (R:1.21, L:0.68, z:1.30); a spot ~0.3 m in front (smaller z) is graspable.
_spot_env = os.environ.get("FD_TOY_SPOT")
if _spot_env:
    TOY_SPOTS = [tuple(float(v) for v in s.split(",")) for s in _spot_env.split(";")]
world_top = GROUND_Y if FLOOR else (TABLE_TOP_Y + DESK_POS[1])
toy_bodies = []   # list of dicts: {name, body_offset, vtx_off, vtx_cnt, is_fem}
for (tx, tz), fn in list(zip(TOY_SPOTS, TOY_FILES))[:NTOYS]:
    if TOY_ABD:
        v, f = load_obj(os.path.join(TOYS, fn))
        verts, cells, vpf = v, f, 3
        btype = "ABD"
    else:
        verts, cells = tetify(os.path.join(TOYS, fn))   # cached tet mesh
        vpf = 4
        btype = "FEM"
    lo, hi = verts.min(0), verts.max(0); ctr = (lo + hi) / 2
    T = np.eye(4)
    # Spawn clearance MUST exceed dHat (~3.3mm for this scene bbox): a body born
    # inside the barrier band gets ejected at 10+ m/s on frame 0. Keep 10mm.
    T[0, 3] = tx - ctr[0]; T[1, 3] = world_top - lo[1] + 0.010; T[2, 3] = tz - ctr[2]
    eng.load_mesh_from_data(verts, cells, verts_per_face=vpf, dimensions=3,
                            body_type=btype, transform=T,
                            young_modulus=(1e8 if TOY_ABD else TOY_YOUNG),
                            boundary_type="Free")
    rec = eng.get_load_records()[-1]
    if TOY_ABD:
        # solid enclosed-volume mass at 5e3 -> ~350-400 g (see git history).
        eng.native.set_abd_body_density(rec.body_offset, 5e3)
    toy_bodies.append(dict(name=fn, body_offset=rec.body_offset,
                           vtx_off=rec.vertex_offset, vtx_cnt=rec.vertex_count,
                           is_fem=not TOY_ABD))
_toy_desc = ", ".join("{}:{}v".format(t["name"], t["vtx_cnt"]) for t in toy_bodies)
print(f"[teleop] toys: {'ABD rigid' if TOY_ABD else 'FEM soft'} "
      f"x{len(toy_bodies)} ({_toy_desc})")

# --- COLLISION WHITELIST -----------------------------------------------------
# The hands carry 99% of the collision cost, and most of the 150k+ collision
# pairs are same-hand finger<->finger self-proximity plus contact between links
# that never touch anything (slider / connecting / abpart mechanism parts).
# Strict whitelist: the ONLY collisions kept are
#     fingers <-> toys,  fingers <-> drawers,  toys <-> drawers,  toys <-> ground
# "fingers" = the distal + proximal finger links (the grasp surfaces); palm
# base_link, arms, torso, table and all internal mechanism links collide with
# nothing.  Everything else is excluded pairwise; only toys see the ground.
# FD_COLL_WHITELIST=0 restores the old proxy-only exclusion for A/B.
import xml.etree.ElementTree as ET
recs = eng.get_load_records()
# COLLISION body id != load-record body_offset for FEM bodies: the exclusion
# matrix indexes ABD in [0, n_abd) and FEM in [n_abd, n_abd+n_fem) (load_mesh.h),
# but rec.body_offset for a FEM body is its FEM-LOCAL index (0,1,..).  So the
# global collision id is body_offset for ABD, and n_abd+body_offset for FEM.
n_abd = eng.native.get_abd_body_count()
n_fem = eng.native.get_fem_body_count()
all_ids = list(range(n_abd + n_fem))

FINGER_PFX = ("if_", "mf_", "rf_", "lf_", "th_")
def _is_finger(lbl):
    # collision surface = ONLY the fingertip (distal) links (the grasp contact).
    return any(lbl.startswith(p) for p in FINGER_PFX) and "distal" in lbl

finger_ids = {r.body_offset for r in recs if _is_finger(r.label)}   # ABD -> direct
drawer_ids = set(drawer_body.values())                              # ABD -> direct
# FEM toys: global id = n_abd + fem_local_offset;  ABD toys (FD_TOY_ABD=1):
# body_offset is already the global ABD id -> use it directly.
toy_ids    = {(n_abd + t["body_offset"]) if t["is_fem"] else t["body_offset"]
              for t in toy_bodies}

if os.environ.get("FD_COLL_WHITELIST", "1") == "1":
    active = finger_ids | toy_ids | drawer_ids
    # allowed unordered pairs
    allow = set()
    for a in finger_ids:
        for b in toy_ids | drawer_ids: allow.add(frozenset((a, b)))
    for a in toy_ids:
        for b in drawer_ids: allow.add(frozenset((a, b)))
    # exclude every pair not explicitly allowed
    n_excl = 0
    for i in range(len(all_ids)):
        a = all_ids[i]
        for j in range(i + 1, len(all_ids)):
            b = all_ids[j]
            if frozenset((a, b)) not in allow:
                eng.add_collision_exclusion(a, b); n_excl += 1
    # only toys collide with the ground
    for b in all_ids:
        if b not in toy_ids:
            eng.add_ground_collision_skip(b)
    print(f"[teleop] collision whitelist: fingers={len(finger_ids)} toys={len(toy_ids)} "
          f"drawers={len(drawer_ids)} | allowed_pairs={len(allow)} excluded={n_excl}")
    print(f"[teleop] BODYID all_ids range=[{min(all_ids)},{max(all_ids)}] n={len(all_ids)} "
          f"| table_id={table_id} drawers={sorted(drawer_ids)} toys={sorted(toy_ids)} "
          f"| finger_ids(sample)={sorted(finger_ids)[:6]}")
    robot_body_ids = [r.body_offset for r in recs if r.body_offset not in active]
else:
    # legacy: exclude only the URDF box proxies
    body_links = set()
    for link in ET.parse(ROBOT).getroot().findall("link"):
        for col in link.findall("collision"):
            m = col.find("geometry/mesh")
            if m is not None and "meshes_proxy" in (m.get("filename") or ""):
                body_links.add(link.get("name"))
    robot_body_ids = [r.body_offset for r in recs if r.label in body_links]
    for b in robot_body_ids:
        eng.add_ground_collision_skip(b)
        for k in all_ids:
            if k != b:
                eng.add_collision_exclusion(b, k)

eng.finalize()

# Joint-rail mode has no table contact to rest on -> disable drawer gravity so
# it can't sag through the cabinet. Contact mode keeps gravity (rests on rails).
if RAIL_JOINT_ONLY:
    for _n, _bid in drawer_body.items():
        try:
            eng.native.set_body_apply_gravity(_bid, False)
        except Exception as e:                   # signature drift safety net
            print(f"[warn] set_body_apply_gravity({_n}) failed: {e}")

# --- joint bookkeeping ------------------------------------------------------
rob = Robot(eng)
rj = rob.revolute_joints
pj = rob.prismatic_joints
L_ARM = [j for j in rj if j.name.startswith("arm_l")]
R_ARM = [j for j in rj if j.name.startswith("arm_r")]
FINGER_PREF = ("if_", "mf_", "rf_", "lf_", "th_")
def finger_rev(side):   # side '_l' or '_r'
    return [j for j in rj if j.name.startswith(FINGER_PREF) and j.name.endswith(side)]
def finger_pris(side):
    return [j for j in pj if j.name.startswith(FINGER_PREF) and j.name.endswith(side)]
L_FING_R, R_FING_R = finger_rev("_l"), finger_rev("_r")
L_FING_P, R_FING_P = finger_pris("_l"), finger_pris("_r")
print(f"[teleop] L_arm={len(L_ARM)} R_arm={len(R_ARM)} "
      f"L_fingers={len(L_FING_R)}+{len(L_FING_P)}p R_fingers={len(R_FING_R)}+{len(R_FING_P)}p "
      f"drawers={len(drawer_joint)}")

def apply_grasp(side_rev, side_pris, s, sign=None):
    if sign is None:   # infer hand from the joint names
        sign = FINGER_SIGN_L if (side_rev and side_rev[0].name.endswith("_l")) else FINGER_SIGN_R
    ang = np.radians(FINGER_CLOSE_DEG) * s * sign
    for j in side_rev:
        eng.native.set_revolute_target(j.index, ang)
    for j in side_pris:
        eng.native.set_prismatic_target(j.index, FINGER_SLIDE * s)

V = np.asarray(eng.get_vertices()); F = np.asarray(eng.get_surface_faces())
print(f"[teleop] verts={len(V)} faces={len(F)} bodies={len(recs)}")

# --- collision-load diagnostic ----------------------------------------------
if os.environ.get("FD_COLINFO"):
    proxy = set(robot_body_ids)
    print("\n=== COLLISION LOAD DIAGNOSTIC ===")
    print(f"bodies={len(recs)}  total verts={sum(r.vertex_count for r in recs)}  "
          f"faces={len(F)}")
    rr = sorted(recs, key=lambda r: -r.vertex_count)
    print("-- heaviest bodies (vtx) [C=collides, G=excl-from-ground, P=proxy-excluded] --")
    for r in rr[:14]:
        tags = ("P" if r.body_offset in proxy else "C")
        print(f"   {r.vertex_count:7d}v  body={r.body_offset:3d}  {tags}  {r.label}")
    coll_verts = sum(r.vertex_count for r in recs if r.body_offset not in proxy)
    print(f"-- colliding verts (non-proxy) = {coll_verts} of {sum(r.vertex_count for r in recs)} --")
    hand = sum(r.vertex_count for r in recs
               if r.body_offset not in proxy and ("_link_l" in r.label or "_link_r" in r.label)
               and not r.label.startswith(("arm_",)))
    base = sum(r.vertex_count for r in recs if r.label in ("base_link_l", "base_link_r"))
    tbl = sum(r.vertex_count for r in recs if r.body_offset == table_id)
    drw = sum(r.vertex_count for r in recs if r.body_offset in drawer_body.values())
    toy = sum(t["vtx_cnt"] for t in toy_bodies)
    print(f"-- by role: hands(incl base)={hand}  of-which base_link_l/r={base}  "
          f"table={tbl}  drawers={drw}  toys={toy} --")
    def toyY():
        vv = np.asarray(eng.get_vertices())
        return [float(vv[t["vtx_off"]:t["vtx_off"] + t["vtx_cnt"], 1].mean()) for t in toy_bodies]
    y0 = toyY()
    p0 = eng.native.get_total_collision_pairs()
    for _ in range(80):
        eng.step()
    p1 = eng.native.get_total_collision_pairs()
    y1 = toyY()
    print(f"-- toy centroid Y: start={['%.3f'%v for v in y0]} -> after80={['%.3f'%v for v in y1]} "
          f"(tabletop={TABLE_TOP_Y:.3f}, ground_off={GROUND_Y:.3f}) --")
    print(f"-- collision-pairs counter: {p0:.0f} -> {p1:.0f} (delta over 80 steps = {p1-p0:.0f}) --")
    sys.exit(0)

# --- headless drive probe: does driving still KICK the toys? -----------------
# FD_PROBE=drawer (default) ramps drawer00 0->DRAWER_MAX over FD_PROBE_FRAMES and
# logs, per frame, the peak toy vertex speed + Newton iters. The whole point of
# making toys FEM soft bodies is to see whether the motor-drive kick survives.
# Deterministic, no GUI. Prints peak speed + writes a CSV.
if os.environ.get("FD_PROBE"):
    mode = os.environ.get("FD_PROBE", "drawer")
    nfr = int(os.environ.get("FD_PROBE_FRAMES", "120"))
    hold = 20
    dj = drawer_joint[sorted(drawer_joint)[0]] if drawer_joint else None
    tv0 = np.asarray(eng.get_vertices())
    rng = [(t["vtx_off"], t["vtx_off"] + t["vtx_cnt"], t["name"]) for t in toy_bodies]
    rows, peak, peak_fr = [], 0.0, -1
    _niters = getattr(eng.native, "get_total_newton_iters", lambda: 0)
    _vvel = getattr(eng.native, "get_vertex_velocities", None)
    prev_iters = _niters()
    for fr in range(nfr + hold):
        if mode == "drawer" and dj is not None:
            tgt = DRAWER_MAX * min(1.0, fr / nfr)
            eng.native.set_prismatic_target(dj, tgt)
        elif mode == "grasp":
            s = min(1.0, fr / nfr)
            apply_grasp(L_FING_R, L_FING_P, s); apply_grasp(R_FING_R, R_FING_P, s)
        eng.step()
        cur = np.asarray(eng.get_vertices())
        if _vvel is not None:
            vv = np.asarray(_vvel())
        else:   # fallback: finite-difference positions
            vv = (cur - tv0) / 0.01; tv0 = cur
        it = _niters(); d_it = it - prev_iters; prev_iters = it
        spd = [float(np.linalg.norm(vv[a:b], axis=1).max()) for a, b, _ in rng]
        smax = max(spd)
        # [det-trace v2] chirality invariant det(A), API-free: least-squares
        # affine fit A = argmin |A X0 - X| from the toy's own vertices
        # (centered), X0 = frame-0 shape. det(+1)=rotation, det(<0)=reflection.
        if fr == 0:
            _det_ref = []
            for a, b, _n in rng:
                X0 = cur[a:b] - cur[a:b].mean(0)
                _det_ref.append((X0, np.linalg.pinv(X0)))
        _dets = []
        for (X0, X0p), (a, b, _n) in zip(_det_ref, rng):
            Xc = cur[a:b] - cur[a:b].mean(0)
            A_fit = (X0p @ Xc)          # (3,3): X0 @ A_fit ~= Xc
            _dets.append(float(np.linalg.det(A_fit)))
        if smax > peak: peak, peak_fr = smax, fr
        rows.append((fr, d_it, *spd))
        if smax > 0.5 or d_it > 15 or (_dets and min(_dets) < 0.5):
            print(f"  [probe] fr={fr:3d} iters={d_it:3d} "
                  + " ".join(f"{n.split('.')[0]}={s:6.3f}m/s" for (_,_,n), s in zip(rng, spd))
                  + ("  det=" + ",".join(f"{d:+.3f}" for d in _dets) if _dets else ""))
    csv = f"/tmp/fd_probe_{mode}.csv"
    with open(csv, "w") as fp:
        fp.write("frame,newton_iters," + ",".join(n for _,_,n in rng) + "\n")
        for r in rows: fp.write(",".join(map(str, r)) + "\n")
    print(f"[probe] mode={mode} PEAK toy speed = {peak:.3f} m/s at frame {peak_fr} "
          f"| toys={'FEM' if not TOY_ABD else 'ABD'} young={TOY_YOUNG:.0e} | csv={csv}")
    sys.exit(0)

# --- optional offscreen grasp/pose test -------------------------------------
if os.environ.get("OUT"):
    import polyscope as ps
    g = float(os.environ.get("GRASP", "0"))
    apply_grasp(L_FING_R, L_FING_P, g); apply_grasp(R_FING_R, R_FING_P, g)
    for _ in range(int(os.environ.get("NSTEP", "0"))):
        eng.step()
    ps.set_program_name("fd"); ps.init(); ps.set_up_dir("y_up"); ps.set_ground_plane_mode("shadow_only")
    Vt = np.asarray(eng.get_vertices())
    ps.register_surface_mesh("scene", Vt, F, color=(0.72, 0.77, 0.86))
    ps.set_window_size(1100, 850)
    zoom = os.environ.get("ZOOM")   # label of a body to zoom on (e.g. base_link_r)
    if zoom:
        r = next(r for r in recs if r.label == zoom)
        t = Vt[r.vertex_offset:r.vertex_offset + r.vertex_count].mean(0)
        ps.look_at((t[0] + 0.35, t[1] + 0.12, t[2] + 0.45), tuple(t))
    else:
        c = Vt.mean(0)
        ps.look_at((c[0] + 2.4, c[1] + 1.3, c[2] + 2.8), (c[0], 0.9, c[2]))
    ps.screenshot(os.environ["OUT"], transparent_bg=False); print("saved", os.environ["OUT"])
    sys.exit(0)

# --- collision-AABB viz (proxy for "show BVH"): the engine build lacks the
#     get_edge_bvh_aabbs() debug getter, so we draw the per-body bounding box of
#     every body that PARTICIPATES in collision. After the whitelist excludes all
#     robot self-collision, only the finger links + toys + drawers get a box —
#     which is exactly what verifies the exclusion. --------------------------
_BOX_EDGES = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
                       [0,4],[1,5],[2,6],[3,7]], dtype=np.int64)

# vertex ranges [lo,hi) of every colliding body (fingers, drawers, toys)
_coll_ranges = []
for _r in recs:
    if _is_finger(_r.label) or (_r.body_offset in drawer_body.values()):
        _coll_ranges.append((_r.vertex_offset, _r.vertex_offset + _r.vertex_count))
for _t in toy_bodies:
    _coll_ranges.append((_t["vtx_off"], _t["vtx_off"] + _t["vtx_cnt"]))

def _collision_aabbs(Vall):
    boxes = []
    for lo, hi in _coll_ranges:
        sub = Vall[lo:hi]
        if len(sub): boxes.append(np.concatenate([sub.min(0), sub.max(0)]))
    return np.asarray(boxes) if boxes else np.zeros((0, 6))

def _aabbs_to_wire(aabbs):
    n = aabbs.shape[0]
    if n == 0: return np.zeros((0, 3)), np.zeros((0, 2), np.int64)
    lx, ly, lz, ux, uy, uz = [aabbs[:, i] for i in range(6)]
    C = np.empty((n, 8, 3))
    C[:, 0] = np.stack([lx, ly, lz], 1); C[:, 1] = np.stack([ux, ly, lz], 1)
    C[:, 2] = np.stack([ux, uy, lz], 1); C[:, 3] = np.stack([lx, uy, lz], 1)
    C[:, 4] = np.stack([lx, ly, uz], 1); C[:, 5] = np.stack([ux, ly, uz], 1)
    C[:, 6] = np.stack([ux, uy, uz], 1); C[:, 7] = np.stack([lx, uy, uz], 1)
    edges = (_BOX_EDGES[None] + (np.arange(n) * 8)[:, None, None]).reshape(-1, 2)
    return C.reshape(-1, 3), edges

# --- GUI --------------------------------------------------------------------
import polyscope as ps
import polyscope.imgui as psim
ps.set_program_name("FD manip teleop"); ps.init(); ps.set_up_dir("y_up")
ps.set_ground_plane_mode("shadow_only")

# Full-res visual skin over the coarse sim proxy (see fullres_skin.py). The robot
# renders at full precision (Kabsch-driven per-body transforms) while the engine
# only advances the coarse proxy; the "scene" mesh then draws non-robot faces only.
skin = None
if SKIN:
    from fullres_skin import FullResSkin
    skin = FullResSkin(ps, eng, recs, FULL_URDF, _URDF_DIR, V)
    _robot_mask = skin.robot_vertex_mask(len(V))
    _keep = ~_robot_mask[F].any(1)          # faces with NO robot vertex
    F_scene = F[_keep]
    skin.update(V)                          # place skin at rest
    mesh = ps.register_surface_mesh("scene", V, F_scene,
                                    color=(0.72, 0.77, 0.86)) if len(F_scene) else None
else:
    mesh = ps.register_surface_mesh("scene", V, F, color=(0.72, 0.77, 0.86))

st = {"run": False,
      "arm_l": [0.0] * 7, "arm_r": [0.0] * 7,
      "grasp_l": 0.0, "grasp_r": 0.0,
      "show_aabb": False, "aabb_net": None,
      "drawer": {n: 0.0 for n in drawer_joint},
      # per-joint hand control (every finger revolute + prismatic joint)
      "hl_rev": [0.0] * len(L_FING_R), "hr_rev": [0.0] * len(R_FING_R),
      "hl_pris": [0.0] * len(L_FING_P), "hr_pris": [0.0] * len(R_FING_P)}

_FINGER_ORDER = ["th", "if", "mf", "rf", "lf"]   # thumb, index, middle, ring, little
def _finger_key(nm):
    return next((p for p in _FINGER_ORDER if nm.startswith(p + "_")), "other")

def hand_block(label, rev_joints, pris_joints, rkey, pkey):
    """Every finger joint of one hand, grouped by finger, as its own slider."""
    if not psim.TreeNode(label):
        return
    # index each joint's slot in the flat state list
    rev_idx = {id(j): i for i, j in enumerate(rev_joints)}
    pris_idx = {id(j): i for i, j in enumerate(pris_joints)}
    for fk in _FINGER_ORDER:
        rj = [j for j in rev_joints if _finger_key(j.name) == fk]
        pj = [j for j in pris_joints if _finger_key(j.name) == fk]
        if not (rj or pj):
            continue
        # unique ImGui id per hand ("##"+rkey): same visible label, distinct id,
        # else the left/right "if finger" nodes collide and one stops responding.
        if psim.TreeNode(fk + " finger##" + rkey):
            for j in rj:
                i = rev_idx[id(j)]
                lo, hi = j.lower_limit_deg, j.upper_limit_deg
                if hi <= lo: lo, hi = -114.0, 114.0
                ch, st[rkey][i] = psim.SliderFloat(j.name, st[rkey][i], lo, hi)
                if ch: eng.native.set_revolute_target(j.index, np.radians(st[rkey][i]))
            for j in pj:
                i = pris_idx[id(j)]
                ch, st[pkey][i] = psim.SliderFloat(j.name + " (m)", st[pkey][i], -0.03, 0.03)
                if ch: eng.native.set_prismatic_target(j.index, st[pkey][i])
            psim.TreePop()
    psim.TreePop()

def _refresh_aabb(Vall):
    nodes, edges = _aabbs_to_wire(_collision_aabbs(Vall))
    st["aabb_net"] = ps.register_curve_network("collision_AABB", nodes, edges,
                                               color=(0.95, 0.35, 0.15), radius=0.0015)

def arm_block(label, joints, key):
    if psim.TreeNode(label):
        for i, j in enumerate(joints):
            lo, hi = j.lower_limit_deg, j.upper_limit_deg
            if hi <= lo: lo, hi = -90.0, 90.0
            ch, st[key][i] = psim.SliderFloat(f"{j.name}", st[key][i], lo, hi)
            if ch:
                eng.native.set_revolute_target(j.index, np.radians(st[key][i]))
        psim.TreePop()

def callback():
    if psim.Button("Run / stop"): st["run"] = not st["run"]
    psim.SameLine(); psim.TextUnformatted("RUNNING" if st["run"] else "paused")
    ch_b, st["show_aabb"] = psim.Checkbox("show collision AABB (BVH boxes)", st["show_aabb"])
    if ch_b:
        if st["show_aabb"]:
            _refresh_aabb(np.asarray(eng.get_vertices()))
        elif st["aabb_net"] is not None:
            ps.remove_curve_network("collision_AABB"); st["aabb_net"] = None
    psim.TextUnformatted(f"colliding bodies (drawn) = {len(_coll_ranges)}  "
                         f"(robot self-collision fully excluded)")
    psim.Separator()
    arm_block("Left arm", L_ARM, "arm_l")
    arm_block("Right arm", R_ARM, "arm_r")
    if psim.TreeNode("Grasp synergy (0 open .. 1 closed)"):
        ch, st["grasp_l"] = psim.SliderFloat("left hand", st["grasp_l"], 0.0, 1.0)
        if ch: apply_grasp(L_FING_R, L_FING_P, st["grasp_l"])
        ch, st["grasp_r"] = psim.SliderFloat("right hand", st["grasp_r"], 0.0, 1.0)
        if ch: apply_grasp(R_FING_R, R_FING_P, st["grasp_r"])
        psim.TreePop()
    # every individual finger joint (per-finger sub-trees)
    hand_block("Left hand joints (per-finger)",  L_FING_R, L_FING_P, "hl_rev", "hl_pris")
    hand_block("Right hand joints (per-finger)", R_FING_R, R_FING_P, "hr_rev", "hr_pris")
    if psim.TreeNode("Drawers (m open)"):
        for n in sorted(drawer_joint):
            ch, st["drawer"][n] = psim.SliderFloat(n, st["drawer"][n], 0.0, DRAWER_MAX)
            if ch: eng.native.set_prismatic_target(drawer_joint[n], st["drawer"][n])
        psim.TreePop()
    if st["run"]:
        eng.step()
        Vnow = np.asarray(eng.get_vertices())
        if skin is not None:
            skin.update(Vnow)         # full-res visual follows the coarse proxy
        if mesh is not None:
            mesh.update_vertex_positions(Vnow)
        if st["show_aabb"]:            # boxes track the moving bodies
            _refresh_aabb(Vnow)

ps.set_user_callback(callback)
ps.show()
