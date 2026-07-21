"""Multi-env Franka-grasps-duck — faithful replica of the uipc 512-env duck video.

Each env = an FR3 Franka (DEFAULT: fr3_franka_hand_uipc.urdf — uipc-exact OBB coarse
arms + convex hand/fingers) + a rubber duck (FEM soft, assets/duck/duck_tet.npz) tiled
in a grid. The arm is DYNAMICS-driven: each frame's recorded joint angles are pushed as
PD set_joint_target (set_revolute_target / set_prismatic_target), NOT kinematic teleport.
All envs replay the same Newton IK grasp trajectory (/tmp/franka_ik_traj.npz: q (980,9)
= 7 arm joints + 2 finger joints, plus franka_base & duck_pos) — arm descends, fingers
close on the duck, lift. GRASP_URDF picks the arm: fr3uipc (default, OBB coarse) /
fr3convex (convex fingers) / panda (coarse panda) / fr3 (full .dae->STL, finest).

Env vars:
  GRASP_N=4          number of envs (franka+duck pairs)
  GRASP_SPACING=1.2  grid spacing [m] (franka reach ~0.8m, keep envs apart)
  GRASP_DUCK=FEM     FEM (soft, deforms when grasped) or ABD (rigid)
  GRASP_FRAME_START / GRASP_FRAME_END  trajectory window
  GRASP_PHASE=0      per-env trajectory offset (heterogeneous difficulty; 0=identical)
  GRASP_MARGIN=8     triplet_internal_margin (FEM contact buffer)
  GRASP_HEADLESS=1   1=headless, 0=polyscope GUI
"""
import os, sys, time, math
# CCD line-search sanity re-check storms on the closing grasp (1M+ msgs); skip it.
os.environ.setdefault("STIFF_SKIP_CCD_SANITY", "1")
import numpy as np
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from stiff_physics import Engine, Config
from stiff_physics.robot import Robot

# METIS CACHE TRAP: load_mesh_from_data writes tmp_mesh_<N>.msh reusing N across runs,
# and the metis partitioner caches the vertex permutation by that name. A duck of a
# different vertex count loaded at the same N inherits the WRONG cached permutation, so
# get_vertices un-permutes incorrectly -> scrambled vertex positions (looks like the
# duck penetrates the table and the grasp breaks). Clear the per-run tmp_mesh_* caches
# at startup so each mesh gets a fresh, correct partition.
import glob as _glob
for _c in _glob.glob(_ROOT + "/Assets/sorted_mesh/tmp_mesh_*"):
    try: os.remove(_c)
    except OSError: pass

# Use the SAME franka as the recording: rbs's FR3 (fr3_joint*), converted for
# StiffGIPC (finger.dae collision -> STL). GRASP_URDF=panda for the Panda variant.
_FR3 = _ROOT + "/Assets/fr3/fr3_franka_hand.urdf"
_FR3_CVX = _ROOT + "/Assets/fr3/fr3_franka_hand_convex.urdf"  # convex-hull fingers (matches uipc)
_FR3_UIPC = _ROOT + "/Assets/fr3/fr3_franka_hand_uipc.urdf"  # uipc EXACT colliders: OBB arms + convex hand/fingers
_PANDA = _ROOT + "/Assets/sim_data/urdf/franka_panda/panda_arm_hand_coarse.urdf"
_SEL = os.environ.get("GRASP_URDF", "fr3uipc")   # DEFAULT: OBB coarse arms + convex hand/fingers (uipc-exact)
URDF = {"panda": _PANDA, "fr3convex": _FR3_CVX, "fr3uipc": _FR3_UIPC}.get(_SEL, _FR3)
JPREFIX = "panda" if URDF == _PANDA else "fr3"
# uipc's EXACT duck tetmesh (270v/889tet, dumped from rbs build_duck_tetmesh). Now the
# DEFAULT — it grasps +149mm matching uipc once the METIS CACHE is cleared (see below).
# (Earlier this mesh "penetrated the table / wouldn't grasp" — that was 100% a STALE
# metis cache: load_mesh_from_data writes tmp_mesh_<N>.msh reusing index N across runs,
# so a 270v duck inherited the 274v duck's cached vertex permutation -> get_vertices
# un-permuted with the wrong perm -> scrambled positions -> false penetration + broken
# contact. NOT a mesh-quality problem.)
DUCK_TET = os.environ.get("GRASP_DUCK_MESH", _ROOT + "/Assets/duck/duck_tet_uipc.npz")
TRAJ = os.environ.get("GRASP_TRAJ", os.path.join(_ROOT, "Assets", "trajectories", "franka_ik_traj.npz"))
if not os.path.exists(TRAJ): TRAJ = "/tmp/franka_ik_traj.npz"  # dev fallback


def gpu_mb():
    try:
        import subprocess
        return int(subprocess.check_output(
            ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]).decode().split("\n")[0])
    except Exception:
        return -1


def tf(pos):
    T = np.eye(4); T[:3, 3] = pos; return T


def box_mesh(center, half):
    """Axis-aligned box surface (8 verts, 12 tris) for a rigid table."""
    cx, cy, cz = center; hx, hy, hz = half
    v = np.array([[cx-hx, cy-hy, cz-hz],[cx+hx, cy-hy, cz-hz],[cx+hx, cy+hy, cz-hz],[cx-hx, cy+hy, cz-hz],
                  [cx-hx, cy-hy, cz+hz],[cx+hx, cy-hy, cz+hz],[cx+hx, cy+hy, cz+hz],[cx-hx, cy+hy, cz+hz]], float)
    f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                  [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], np.int32)
    return v, f


# rbs/Newton franka URDF is Z-up; StiffGIPC world is Y-up. Rotate -90 deg about X
# (z->y) like the fold-shirt example's make_arm_tf, so the arm stands up in Y and
# gravity (-Y) is correct.
_RY = Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix()
def arm_tf(pos):
    T = np.eye(4); T[:3, :3] = _RY; T[:3, 3] = _RY @ np.asarray(pos); return T


def main():
    N        = int(os.environ.get("GRASP_N", "4"))
    spacing  = float(os.environ.get("GRASP_SPACING", "1.2"))
    ducktype = os.environ.get("GRASP_DUCK", "FEM").upper()
    phase    = int(os.environ.get("GRASP_PHASE", "0"))
    margin   = float(os.environ.get("GRASP_MARGIN", "8"))
    headless = int(os.environ.get("GRASP_HEADLESS", "1"))

    d = np.load(TRAJ)
    q = d["q"].astype(np.float64)          # (T, 9): 7 arm + 2 finger
    franka_base = d["franka_base"].astype(np.float64)
    duck_pos    = d["duck_pos"].astype(np.float64)
    tbl_pos     = d["table_pos"].astype(np.float64)
    tbl_half    = d["table_half"].astype(np.float64)
    L = q.shape[0]
    f0 = int(os.environ.get("GRASP_FRAME_START", "0"))
    f1 = min(int(os.environ.get("GRASP_FRAME_END", str(L))), L)
    print(f"[grasp] N={N} duck={ducktype} traj={L}f base={franka_base.round(2)} "
          f"duck={duck_pos.round(2)} phase={phase}", flush=True)

    dk = np.load(DUCK_TET)
    dverts, dcells = dk["verts"].astype(np.float64), dk["cells"].astype(np.int32)
    # IDENTITY orientation — faithful to rbs/uipc. The rbs scene builder
    # (softbody_franka_grasp.build_duck_tetmesh) reads the Newton rubber-duck USD
    # SurfaceMesh points DIRECTLY and applies NO rotation ("Identity orientation,
    # matching Newton's placement, rot=quat_identity"); the USD is already Z-up
    # (UsdGeom.GetStageUpAxis==Z), same as our Z-up world. Our duck_tet.npz is the
    # raw USD points (bbox 62x80x55mm, x=width y=front-back z=height), so we load it
    # as-is. (An earlier R_x(90) here was WRONG — it presented the duck's wide face
    # to the jaws and the grasp slipped.) NO scaling: scale=1, same as rbs.
    dscale = float(os.environ.get("GRASP_DUCK_SCALE", "1.0"))
    dverts = dverts * dscale
    print(f"[grasp] duck (identity, scale={dscale}) bbox {(dverts.max(0)-dverts.min(0)).round(3)}", flush=True)

    # The recording is Z-up (duck height = z). StiffGIPC gravity is configurable;
    # set it to -Z so we replay in the RECORDING's native frame — no rotation, no
    # re-placement, just drive the recorded joint angles q. (This was the missing
    # piece: default gravity is -Y, which laid the z-up scene on its side.)
    # Z-up gravity + Z-up ground at the table-top height (table_pos.z + table_half.z
    # = 0.1+0.1 = 0.2) so the duck (z=0.23) rests on it until grasped, mirroring the
    # rbs table. Replay the recorded q in the recording's native Z-up frame.
    SUBSTEP = int(os.environ.get("GRASP_SUBSTEP", "1"))   # engine substeps per recorded frame
    BASE_DT = float(os.environ.get("GRASP_DT", "0.016667"))   # match uipc's 1/60
    cfg = Config(dt=BASE_DT / SUBSTEP, gravity=(0.0, 0.0, -9.8),
                 ground_normal=(0.0, 0.0, 1.0), ground_offset=0.15,
                 poisson_rate=0.45, friction_rate=float(os.environ.get("GRASP_FRICTION", "0.5")),
                 relative_dhat=1e-3,
                 newton_tol=5e-2, newton_iter_cap=int(os.environ.get("GRASP_NITER","50")),
                 # MAS OFF by default: this contact-light grasp runs faster without it
                 # (N=16: 1300 vs 639 env-steps/s) AND skips metis entirely, so the
                 # metis-cache trap can't occur. GRASP_PRECOND=1 re-enables MAS.
                 preconditioner_type=int(os.environ.get("GRASP_PRECOND","0")),
                 semi_implicit_enabled=bool(int(os.environ.get("GRASP_SEMI","0"))),
                 revolute_driving_strength_ratio=100.0,
                 prismatic_strength_ratio=float(os.environ.get("GRASP_GRIP", "2000.0")),
                 assets_dir=_ROOT + "/Assets/")
    cfg._cfg.absolute_dhat = float(os.environ.get("GRASP_DHAT", "0.0019"))
    # These pre-allocate WORST-CASE contact/linear-system buffers (per-env, x buff_scale).
    # Smaller => more envs fit, but too small => contact/triplet OVERFLOW -> crash.
    cfg._cfg.collision_detection_buff_scale = float(os.environ.get("GRASP_CBUF", "1.0"))
    cfg._cfg.linear_system_buff_scale = float(os.environ.get("GRASP_LBUF", "1.5"))
    cfg._cfg.triplet_internal_margin = margin
    eng = Engine(cfg)

    side = int(math.ceil(math.sqrt(N)))
    ANAMES = [f"{JPREFIX}_joint{i+1}" for i in range(7)]
    # load at traj-start arm pose, fingers OPEN (0.04) so the duck fits between them
    init_angles = {ANAMES[i]: float(q[f0, i]) for i in range(7)}
    init_angles[f"{JPREFIX}_finger_joint1"] = 0.04
    init_angles[f"{JPREFIX}_finger_joint2"] = 0.04

    # FAITHFUL rbs layout — exact recorded positions, NO hacks: franka @ franka_base,
    # rigid table @ table_pos, duck @ duck_pos (on the table). Z-up gravity. Then
    # replay the recorded joint trajectory q as-is. (Earlier "gripper doesn't reach"
    # was a STALE get_urdf_link_transform = load-time FK; we now track the gripper
    # from the real finger vertices instead.)
    table_v, table_f = box_mesh(tbl_pos, tbl_half)
    # REST THE DUCK ON THE TABLE (no initial penetration), matching uipc EXACTLY.
    # uipc places its duck so the bottom sits +2.4mm above the table top (duck bottom
    # 0.3024, table top 0.3000) — a clean gap > d_hat so the barrier starts un-violated.
    # Our duck_tet.npz is a different tetrahedralization whose bottom is ~34mm below its
    # centroid, so translating it straight to duck_pos.z=0.23 sinks its lowest vertex to
    # z=0.196 — 3.6mm INTO the table top (0.200); with abs_dhat=1.9mm < 3.6mm the IPC
    # barrier starts violated (v0.6.2 sanity check, STIFF_SKIP_CCD_SANITY=0, flags
    # table<->duck intersections every frame) and the grasp is corrupted. Shift the duck
    # up so its bottom rests +2.4mm above the table top, reproducing uipc's clearance.
    table_top = float(tbl_pos[2] + tbl_half[2])
    duck_zmin = float(dverts[:, 2].min())
    duck_place = duck_pos.copy()
    duck_place[2] = table_top + 0.0024 - duck_zmin   # uipc's +2.4mm clearance
    print(f"[grasp] LAYOUT franka_base={franka_base.round(3)} table_top={table_top:.3f} "
          f"duck_pos={duck_pos.round(3)} -> rest_z={duck_place[2]:.3f} "
          f"(was {duck_pos[2]:.3f}, +{(duck_place[2]-duck_pos[2])*1000:.1f}mm to clear table)", flush=True)

    # FAITHFUL: uipc's 300kPa. (Earlier this needed softening to ~30kPa to grip, but
    # that was the metis-cache scramble, not a real mesh problem — with a fresh cache
    # the 300kPa duck lifts +149mm matching uipc.)
    duck_young = float(os.environ.get("GRASP_YOUNG", "300000"))
    t0 = time.perf_counter()
    # IMPORTANT: StiffGIPC requires ALL ABD bodies loaded BEFORE any FEM body
    # ("ABD mesh shouldn't be loaded after FEM mesh"). So load every env's franka+table
    # (ABD) first, then every env's duck (FEM) — NOT interleaved per env.
    offs = [np.array([(e % side) * spacing, (e // side) * spacing, 0.0]) for e in range(N)]
    for e in range(N):
        robot_body_start = eng.abd_body_count
        eng.native.load_urdf(URDF, tf(franka_base + offs[e]), True, False, 1e7, init_angles)
        for body_id in range(robot_body_start, eng.abd_body_count):
            eng.add_ground_collision_skip(body_id)
        eng.load_mesh_from_data(table_v, table_f, 3, 3, 0, tf(offs[e]), 1e9, 1)   # rigid Fixed table
        eng.add_ground_collision_skip(eng.abd_body_count - 1)
    for e in range(N):
        eng.load_mesh_from_data(dverts, dcells, 4, 3,
                                0 if ducktype == "ABD" else 1,
                                tf(duck_place + offs[e]),
                                1e8 if ducktype == "ABD" else duck_young, 0)
    print(f"[grasp] loaded {N} (franka+table+duck), {eng.native.get_abd_body_count()} ABD bodies", flush=True)

    eng.finalize()
    if N == 1:
        _dz = eng.get_vertices()[-dverts.shape[0]:, 2]
        print(f"[grasp] duck after load: zmin={_dz.min():.4f} table_top={table_top:.3f} -> "
              f"{'PENETRATES %.1fmm'%((table_top-_dz.min())*1000) if _dz.min()<table_top else 'rests +%.1fmm above'%((_dz.min()-table_top)*1000)}", flush=True)
    # env-0 franka loaded first; its 2 fingers are the last (2*nfv) verts of the franka
    # block (table=8 verts + duck=nduck come after env-0 only for N=1). Finger vert count
    # depends on the collider: raw STL=318/finger, convex hull (fr3convex)=68/finger.
    nfv = 68 if URDF in (_FR3_CVX, _FR3_UIPC) else 318
    nduck = dverts.shape[0]
    total = len(eng.get_vertices())
    franka0_end = total - (8 + nduck) if N == 1 else total
    fv0, fv1 = max(0, franka0_end - 2 * nfv), franka0_end
    robot = Robot(eng)
    nr, npz = len(robot.revolute_joints), len(robot.prismatic_joints)
    rpe, ppe = nr // N, npz // N   # joints per env (7 rev, 2 pri for panda)
    print(f"[grasp] {nr} revolute ({rpe}/env) + {npz} prismatic ({ppe}/env)  "
          f"load={time.perf_counter()-t0:.1f}s  GPU={gpu_mb()}MB", flush=True)

    # CLAMP (default OFF = faithful): replay the recorded finger target (~0.02, 40mm
    # gap) exactly. uipc holds the duck at 0.02 and so do we now — the gripper keeps a
    # visible ~40mm gap (NOT "slamming shut"/crushing). CLAMP=1 was only needed to
    # compensate for the duck-table penetration bug; with the duck resting on the table
    # the faithful 0.02 grip holds the duck through the full +130mm lift.
    CLAMP = int(os.environ.get("GRASP_CLAMP", "0"))
    CLOSE = float(os.environ.get("GRASP_CLOSE", "0.0"))   # clamp target (0=full close)
    lift_dj2 = [0.0]  # shoulder (joint2) delta for the post-grasp lift phase
    def apply(fr):
        for e in range(N):
            qi = q[(fr + e * phase) % L].copy()
            qi[1] += lift_dj2[0]   # raise the arm by tilting the shoulder
            for i in range(min(rpe, 7)):
                robot.set_revolute_position(e * rpe + i, float(qi[i]), degree=False)
            for j in range(min(ppe, 2)):
                fj = float(qi[7 + j])
                if CLAMP and fj < 0.038: fj = CLOSE   # clamp to a firm-but-not-crushing target
                robot.set_prismatic_position(e * ppe + j, fj, millimeters=False)

    # WARMUP: uipc ramps 120 frames from the all-zeros rest pose to traj[0] before
    # the recorded trajectory (its headless run = 120 + 980 = 1100 frames). Replicate
    # it so the arm eases into the start pose instead of snapping to traj[0] at frame 0.
    WARMUP = int(os.environ.get("GRASP_WARMUP", "120"))   # match uipc's 120-frame ramp
    def apply_warmup(k):
        a = (k + 1) / max(WARMUP, 1)            # 0 -> 1 ramp
        for e in range(N):
            for i in range(min(rpe, 7)):
                robot.set_revolute_position(e * rpe + i, float(a * q[0, i]), degree=False)
            for j in range(min(ppe, 2)):
                robot.set_prismatic_position(e * ppe + j, 0.04, millimeters=False)  # fingers open

    LIFT = int(os.environ.get("GRASP_LIFT", "0"))   # post-grasp lift frames (0=off; faithful replay)
    nduck = dverts.shape[0]
    if headless:
        ms = []; z0 = None; gz_min = 1e9; lowest_g = np.zeros(3); duck_zmax = -1e9
        # phase 0: warmup ramp; phase 1: replay grasp; phase 2: lift
        seq = [("warmup", w) for w in range(WARMUP)] + \
              [("grasp", fr) for fr in range(f0, f1)] + \
              [("lift", f1 - 1)] * LIFT
        for k, (ph, fr) in enumerate(seq):
            if ph == "warmup":
                apply_warmup(fr)
                t = time.perf_counter()
                for _ in range(SUBSTEP): eng.step()
                ms.append((time.perf_counter()-t)*1000.0)
                continue
            if ph == "lift":
                lift_dj2[0] = -0.6 * (k - WARMUP - (f1 - f0)) / max(LIFT, 1)  # ramp shoulder up
            apply(fr)
            t = time.perf_counter()
            for _ in range(SUBSTEP): eng.step()
            ms.append((time.perf_counter()-t)*1000.0)
            # env-0 duck height (verts loaded as franka0,duck0,... -> env0 duck is
            # the FEM block right after franka0; track its z = grasp success signal)
            V = eng.get_vertices()
            dz = float(V[-nduck:, 2].mean()) if N == 1 else None        # duck height
            g = V[fv0:fv1].mean(0) if N == 1 else None                  # gripper (finger) centroid
            if z0 is None: z0 = dz                                      # duck start height
            if N == 1:
                duck_zmax = max(duck_zmax, dz)
                if g[2] < gz_min: gz_min = g[2]; lowest_g = g.copy()
            if k % 60 == 0 or k == len(seq) - 1:
                gs = f" duck_z={dz:+.3f} gripper=[{g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f}]" if g is not None else ""
                print(f"[grasp] {ph} k={k:4d} step={ms[-1]:6.0f}ms{gs}", flush=True)
            if int(os.environ.get("GRASP_DIAG", "0")) and N == 1 and fr in (240,300,360,400,419,440,460,480,500,540,720):
                from scipy.spatial.distance import cdist as _cd
                duck = V[-nduck:]; lf = V[fv0:fv0+(fv1-fv0)//2]; rf = V[fv0+(fv1-fv0)//2:fv1]
                fg = _cd(lf, rf).min()*1000
                ax = np.array([0.707,-0.707,0.0]); pr = duck@ax; dw = (pr.max()-pr.min())*1000
                md = min(_cd(lf,duck).min(), _cd(rf,duck).min())*1000
                gc = (lf.mean(0)+rf.mean(0))/2; dcn = duck.mean(0)
                off = float(np.hypot(gc[0]-dcn[0], gc[1]-dcn[1])*1000)
                print(f"   [DIAG] fr{fr:4d} fcmd={float(q[fr,7]):.3f} fgap={fg:5.1f}mm "
                      f"duckW_axis={dw:5.1f}mm finger->duck={md:4.1f}mm gripVSduck_xy={off:5.1f}mm "
                      f"duckZ={dcn[2]:+.3f} gripZ={gc[2]:+.3f}", flush=True)
        if N == 1:
            peak_lift = duck_zmax - z0   # uipc's own metric: PEAK lift over the trajectory
            print(f"[grasp] LOWEST gripper = {lowest_g.round(3)}  duck_z peak = {duck_zmax:+.3f} "
                  f"peak_lift = {peak_lift:+.3f}m "
                  f"({'GRASP LIFTED (uipc bar: >0.03)' if peak_lift > 0.03 else 'duck did NOT lift'})", flush=True)
        if z0 is not None:
            zf = float(eng.get_vertices()[-nduck:, 2].mean())
            print(f"[grasp] env0 duck z: start={z0:+.3f} -> end={zf:+.3f}  lift={zf-z0:+.3f}m "
                  f"({'GRASPED+LIFTED' if zf-z0 > 0.05 else 'no lift'})", flush=True)
        mm = float(np.mean(ms))
        print(f"\n[grasp-RESULT] N={N} {ducktype}: mean {mm:.1f}ms/step ({1000.0/mm:.2f} fps, "
              f"{N*1000.0/mm:.0f} env-steps/s) peak GPU={gpu_mb()}MB", flush=True)
        return

    import polyscope as ps, polyscope.imgui as psim
    v = eng.get_vertices(); fa = eng.get_surface_faces()
    ps.init(); ps.set_up_dir("z_up"); ps.set_ground_plane_mode("shadow_only")
    st = dict(idx=f0, run=False, mesh=ps.register_surface_mesh("scene", v, fa, color=(0.85,0.8,0.3)))
    def cb():
        if psim.Button("Start/Pause"): st['run'] = not st['run']
        psim.SameLine()
        if psim.Button("Reset"): st['idx'] = f0; st['run'] = False
        psim.Text(f"frame {st['idx']}/{L}  N={N} {ducktype}  GPU {gpu_mb()}MB")
        if st['run'] and st['idx'] < f1:
            apply(st['idx'])
            t=time.perf_counter(); eng.step(); dtms=(time.perf_counter()-t)*1000.0
            psim.Text(f"step {dtms:.0f} ms")
            st['mesh'].update_vertex_positions(eng.get_vertices())
            st['idx'] += 1
    ps.set_user_callback(cb); ps.show()


if __name__ == "__main__":
    main()
