"""Nuka-Physics 'Go2 cloth drape' with StiffGIPC: a square cloth is released above a standing Unitree Go2 and drapes over it.

Reference (Nuka go2_cloth_drape.py): VBD cloth (particle mass 0.012, friction 1.8) dropped on a PD-standing Go2, 800 steps at
240 Hz. Here: IPC triangle cloth (1 mm, E 3e5, bending 3e4, 200 kg/m^3) on a Go2 (unitree_ros go2_description, per-link convex
hull collision meshes, fixed base, joints held at the standing pose), friction 0.6, dhat 2 mm.

    PYTHONPATH=. python examples/nuka/go2_cloth_drape.py --out out/nuka/drape1
"""
import argparse, os, os.path as osp, sys, time
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
from common import ASSETS, Body, Recorder, Scene, _load_urdf_capture  # noqa: E402

STAND = {"hip": 0.0, "thigh": 0.8, "calf": -1.5}      # Go2 standing pose (URDF order FL, FR, RL, RR x hip, thigh, calf)
BASE_Y = 0.376                                          # base link height that puts the hull feet ~5 mm above the ground


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--duration", type=float, default=3.5)
    ap.add_argument("--cloth", default="triMesh/cloth_60x60.obj", help="1 x 1 m grid (3721 verts), scaled by --cloth-size")
    ap.add_argument("--cloth-size", type=float, default=0.9)
    ap.add_argument("--cloth-young", type=float, default=3e5)
    ap.add_argument("--bend-young", type=float, default=3e4)
    ap.add_argument("--cloth-density", type=float, default=200.0)
    ap.add_argument("--drop-height", type=float, default=0.60, help="cloth plane height at t = 0")
    ap.add_argument("--cloth-dx", type=float, default=0.0, help="cloth centre offset along x (robot length axis)")
    ap.add_argument("--mu", type=float, default=0.6)
    ap.add_argument("--dhat", type=float, default=2e-3)
    ap.add_argument("--newton-cap", type=int, default=40)
    ap.add_argument("--ptype", type=int, default=1)
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=args.ptype, newton_iter_cap=args.newton_cap, cloth_thickness=1e-3,
                    cloth_young_modulus=args.cloth_young, bend_young_modulus=args.bend_young, cloth_density=args.cloth_density,
                    strain_rate=100.0, absolute_dhat=args.dhat), log_level=args.log_level)
    # Go2 (URDF is z-up): rotate to y-up, fixed base, joints initialised at the standing pose
    T = np.eye(4); T[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float); T[1, 3] = BASE_Y
    init = {f"{leg}_{j}_joint": v for leg in ("FL", "FR", "RL", "RR") for j, v in STAND.items()}
    n_rec0 = len(sc.eng.get_load_records())
    log = _load_urdf_capture(sc.eng, osp.join(ASSETS, "nuka", "go2", "go2_hull.urdf"), T, True, False, 1e7, init)
    sc.eng.set_log_level(args.log_level)
    bad = [l for l in log.splitlines() if "Failed" in l]
    if bad: print("[warn] URDF import:", bad[:3])
    robot = []
    for r in sc.eng.get_load_records()[n_rec0:]:
        if r.body_type != 0: continue
        b = Body(r, f"go2/{r.label}", "fixed", color=(0.80, 0.80, 0.82))   # NOT "arm*": the lab renderer would draw the FR3 instead; sc.bodies[b.name] = b; robot.append(b)
        sc.eng.add_ground_collision_skip(b.body_id); sc.eng.set_body_friction(b.body_id, args.mu)
    n_rev = sc.eng.native.get_num_revolute_joints()
    rev = {sc.eng.get_revolute_joint_info(i).name: i for i in range(n_rev)}
    targets = {rev[n]: v for n, v in init.items() if n in rev}
    # cloth: flat square above the robot
    Tc = np.eye(4); Tc[:3, :3] *= args.cloth_size; Tc[0, 3] = args.cloth_dx; Tc[1, 3] = args.drop_height
    cloth = sc.add_cloth("cloth", osp.join(ASSETS, args.cloth), Tc, young=args.cloth_young)
    sc.eng.set_body_friction(len(sc.eng.get_load_records()) - 1, args.mu)
    sc.finalize()
    for jid, v in targets.items(): sc.eng.set_revolute_target(jid, float(v))
    lo = np.full(3, 9.0); hi = np.full(3, -9.0)
    for b in robot:
        V = np.asarray(sc.eng.get_abd_body(b.body_id).get_vertices(), float); lo = np.minimum(lo, V.min(0)); hi = np.maximum(hi, V.max(0))
    rec = Recorder(sc)
    print(f"[scene] Go2: {len(robot)} links, {n_rev} revolute joints held at the standing pose, bbox x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] z[{lo[2]:.3f},{hi[2]:.3f}]; "
          f"cloth {args.cloth_size:.2f} m square, {cloth['n'] if cloth.get('n') else '?'} verts at y {args.drop_height:.2f}", flush=True)
    n_steps = int(round(args.duration / args.dt)); rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    meta = dict(example="go2_cloth_drape", dt=args.dt, fps=args.fps, cloth_size=args.cloth_size, cloth_young=args.cloth_young, bend_young=args.bend_young,
                cloth_density=args.cloth_density, drop_height=args.drop_height, mu=args.mu, fem_materials={"cloth": "cloth"}, fem_colors=[[0.55, 0.70, 0.90]])
    t0 = time.time(); t_last = t0
    for k in range(n_steps + 1):
        tt = k * args.dt
        for jid, v in targets.items(): sc.eng.set_revolute_target(jid, float(v))
        if k % max(1, int(round(0.25 / args.dt))) == 0:
            C = sc.fem_verts("cloth"); now = time.time()
            print(f"[{tt:5.2f}s] cloth y [{C[:,1].min()*1e3:6.1f}, {C[:,1].max()*1e3:6.1f}] mm, on the ground {int(np.sum(C[:,1] < 0.01))} verts | "
                  f"{(now-t0)/60:4.1f} min, {(now-t_last)/max(1,int(round(0.25/args.dt))):.3f} s/step", flush=True)
            t_last = now
        if k % rec_every == 0:
            rec.record(tt, 0, "drape")
        sc.step()
    meta["wall_min"] = (time.time() - t0) / 60
    rec.save(osp.join(args.out, "recording.npz"), meta)
    print(f"[done] {n_steps} steps in {meta['wall_min']:.1f} min -> {osp.join(args.out, 'recording.npz')}")


if __name__ == "__main__":
    main()
