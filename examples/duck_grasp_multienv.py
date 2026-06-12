"""Multi-env Franka-grasps-duck — faithful replica of the uipc 512-env duck video.

Each env = a Franka panda (panda_arm_hand_coarse.urdf) + a rubber duck (FEM soft,
assets/duck/duck_tet.npz) tiled in a grid. All envs replay the same Newton IK grasp
trajectory (/tmp/franka_ik_traj.npz: q (980,9) = 7 arm joints + 2 finger joints,
plus franka_base & duck_pos) — arm descends, fingers close on the duck, lift.

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

URDF = _ROOT + "/Assets/sim_data/urdf/franka_panda/panda_arm_hand_coarse.urdf"
DUCK_TET = _ROOT + "/assets/duck/duck_tet.npz"
TRAJ = "/tmp/franka_ik_traj.npz"


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

    # The recording is Z-up (duck height = z). StiffGIPC gravity is configurable;
    # set it to -Z so we replay in the RECORDING's native frame — no rotation, no
    # re-placement, just drive the recorded joint angles q. (This was the missing
    # piece: default gravity is -Y, which laid the z-up scene on its side.)
    # Z-up gravity + Z-up ground at the table-top height (table_pos.z + table_half.z
    # = 0.1+0.1 = 0.2) so the duck (z=0.23) rests on it until grasped, mirroring the
    # rbs table. Replay the recorded q in the recording's native Z-up frame.
    cfg = Config(dt=0.02, gravity=(0.0, 0.0, -9.8),
                 ground_normal=(0.0, 0.0, 1.0), ground_offset=0.15,
                 poisson_rate=0.45, friction_rate=0.4, relative_dhat=1e-3,
                 newton_tol=5e-2, newton_iter_cap=50, preconditioner_type=1,
                 revolute_driving_strength_ratio=100.0,
                 prismatic_strength_ratio=2000.0, assets_dir=_ROOT + "/Assets/")
    cfg._cfg.absolute_dhat = 0.0019
    cfg._cfg.collision_detection_buff_scale = 1.0
    cfg._cfg.linear_system_buff_scale = 1.5
    cfg._cfg.triplet_internal_margin = margin
    eng = Engine(cfg)

    side = int(math.ceil(math.sqrt(N)))
    ANAMES = [f"panda_joint{i+1}" for i in range(7)]
    # load at traj-start arm pose, fingers OPEN (0.04) so the duck fits between them
    init_angles = {ANAMES[i]: float(q[f0, i]) for i in range(7)}
    init_angles["panda_finger_joint1"] = 0.04
    init_angles["panda_finger_joint2"] = 0.04

    # FAITHFUL rbs layout — exact recorded positions, NO hacks: franka @ franka_base,
    # rigid table @ table_pos, duck @ duck_pos (on the table). Z-up gravity. Then
    # replay the recorded joint trajectory q as-is. (Earlier "gripper doesn't reach"
    # was a STALE get_urdf_link_transform = load-time FK; we now track the gripper
    # from the real finger vertices instead.)
    table_v, table_f = box_mesh(tbl_pos, tbl_half)
    print(f"[grasp] LAYOUT franka_base={franka_base.round(3)} table_top={float(tbl_pos[2]+tbl_half[2]):.3f} "
          f"duck={duck_pos.round(3)}", flush=True)

    t0 = time.perf_counter()
    for e in range(N):
        r, c = divmod(e, side)
        off = np.array([c * spacing, r * spacing, 0.0])   # grid offset (Z-up: XY plane)
        eng.native.load_urdf(URDF, tf(franka_base + off), True, False, 1e7, init_angles)
        eng.load_mesh_from_data(table_v, table_f, 3, 3, 0, tf(off), 1e9, 1)   # rigid Fixed table
        eng.load_mesh_from_data(dverts, dcells, 4, 3,
                                0 if ducktype == "ABD" else 1,
                                tf(duck_pos + off),
                                1e8 if ducktype == "ABD" else 3e5, 0)
    # env-0 franka loaded first -> its finger verts (bodies 8,9) are [303,939)
    fv0, fv1 = 303, 939
    print(f"[grasp] loaded {N} (franka+table+duck), {eng.native.get_abd_body_count()} ABD bodies", flush=True)

    eng.finalize()
    robot = Robot(eng)
    nr, npz = len(robot.revolute_joints), len(robot.prismatic_joints)
    rpe, ppe = nr // N, npz // N   # joints per env (7 rev, 2 pri for panda)
    print(f"[grasp] {nr} revolute ({rpe}/env) + {npz} prismatic ({ppe}/env)  "
          f"load={time.perf_counter()-t0:.1f}s  GPU={gpu_mb()}MB", flush=True)

    lift_dj2 = [0.0]  # shoulder (joint2) delta for the post-grasp lift phase
    def apply(fr):
        for e in range(N):
            qi = q[(fr + e * phase) % L].copy()
            qi[1] += lift_dj2[0]   # raise the arm by tilting the shoulder
            for i in range(min(rpe, 7)):
                robot.set_revolute_position(e * rpe + i, float(qi[i]), degree=False)
            for j in range(min(ppe, 2)):
                robot.set_prismatic_position(e * ppe + j, float(qi[7 + j]), millimeters=False)

    LIFT = int(os.environ.get("GRASP_LIFT", "0"))   # post-grasp lift frames (0=off; faithful replay)
    nduck = dverts.shape[0]
    if headless:
        ms = []; z0 = None; gz_min = 1e9; lowest_g = np.zeros(3); duck_zmax = -1e9
        # phase 1: replay the recorded grasp; phase 2: lift (ramp shoulder, fingers held)
        seq = [("grasp", fr) for fr in range(f0, f1)] + \
              [("lift", f1 - 1)] * LIFT
        for k, (ph, fr) in enumerate(seq):
            if ph == "lift":
                lift_dj2[0] = -0.6 * (k - (f1 - f0)) / max(LIFT, 1)  # ramp shoulder up
            apply(fr)
            t = time.perf_counter(); eng.step(); ms.append((time.perf_counter()-t)*1000.0)
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
        if N == 1:
            print(f"[grasp] LOWEST gripper = {lowest_g.round(3)}  duck_z max during replay = {duck_zmax:+.3f} "
                  f"({'LIFTED off table' if duck_zmax > 0.28 else 'never lifted'})", flush=True)
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
