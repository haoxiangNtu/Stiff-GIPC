"""Scene-rest probe: build the lab scene (engine), seed water in beaker_mix
exactly like run_lab_demo, then step engine + liquid for T s with the arm
idle.  Reports clumping and containment.  Env: LIQ_MESH=all|beaker (boundary
mesh = all props+dropper, or only the mixing beaker), LIQ_T (3 s)."""
import os, sys, time, numpy as np, warp as wp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)
wp.config.quiet = True
from scipy.spatial import cKDTree
from lab_scene import LabScene
from stiff_physics.liquid import LiquidMPM, EngineLiquidCoupler
dt = 0.005; dx = 0.005
sc = LabScene(dt=dt, precond=0, tactile_gpu=False, log_level=0, verbose=False)
o = np.array([0.33, -0.02, -0.16]); ext = np.array([0.44, 0.34, 0.48])
KAPPA = float(os.environ.get("LIQ_KAPPA", "1e5")); SUB = float(os.environ.get("LIQ_SUBSTEP", "1e-4"))
liq = LiquidMPM(o, ext, dx, device="cuda:0", gravity=(0, -9.81, 0), kappa=KAPPA, ppc=8, substep_dt=SUB, verbose=True, apic=bool(int(os.environ.get("LIQ_APIC", "1"))))
liq.add_halfspace([0, 0, 0], [0, 1, 0])
liq.push_apart = float(os.environ.get("LIQ_PUSH", "0"))
c, r_in, y_floor, y_rim = sc.vessel_cavity("beaker_mix"); h = y_floor + 0.9 * (y_rim - y_floor); m = 0.6 * dx
liq.seed_box([c[0] - r_in + m, y_floor + m, c[1] - r_in + m], [c[0] + r_in - m, h, c[1] + r_in - m], jitter=0.3)
xp = liq.solver.ptc_x.numpy() + liq.origin; keep = np.hypot(xp[:, 0] - c[0], xp[:, 2] - c[1]) < r_in - m
for name in ("ptc_x", "ptc_v", "ptc_m", "ptc_k", "ptc_vol0", "ptc_F", "ptc_G", "ptc_dlogJ"):
    a = getattr(liq.solver, name); setattr(liq.solver, name, wp.array(a.numpy()[keep], dtype=a.dtype, device=liq.device))
liq.solver.n_ptc = int(keep.sum()); liq.solver.n_ptc_wp.fill_(liq.solver.n_ptc)
mode = os.environ.get("LIQ_MESH", "all")
faces = sc.surface_faces_global() if mode == "all" else sc.surface_faces_global(names=["beaker_mix"], with_dropper=False)
# orientation check of the engine faces for the beaker
X = np.asarray(sc.eng.get_vertices(), np.float64); fb = sc.surface_faces_global(names=["beaker_mix"], with_dropper=False)
print(f"engine beaker faces: {len(fb)}, signed volume {LiquidMPM.check_orientation(X, fb)*1e6:+.2f} cm3 (asset: +17.93)")
coupler = EngineLiquidCoupler(sc.eng, liq, faces, dt)
print(f"apic={liq.solver.enable_apic} push={liq.push_apart} kappa={KAPPA:.0e} substep={SUB:.0e} mesh={mode} tris={len(faces)} particles={liq.n_particles} thicken={liq.thicken*1e3:.0f}mm band_ext={liq.band_ext*1e3:.0f}mm interior={liq.interior_type} flip={liq.solver.flip_ratio} proj={liq.particle_projection}")
T = float(os.environ.get("LIQ_T", "3")); t0 = time.time()
for k in range(int(T / dt)):
    sc.drive_arm(sc.q); sc.gripper.update(); sc.eng.step(); coupler.step()
    if k % 100 == 99:
        p = liq.positions(); d, _ = cKDTree(p).query(p, k=2); nn = d[:, 1]
        inside = (np.hypot(p[:, 0] - c[0], p[:, 2] - c[1]) < r_in + 0.003) & (p[:, 1] > 0) & (p[:, 1] < y_rim + 0.005)
        F = liq.solver.ptc_F.numpy(); J = F[:, 0, 0] * F[:, 1, 1] * F[:, 2, 2]
        print(f"t={(k+1)*dt:4.2f}s inside {inside.mean()*100:5.1f}%  clumped {(nn<1e-3).mean()*100:5.1f}%  NN {np.median(nn)*1e3:.2f} mm  proj {liq.projected_last()}  y {p[:,1].min()*1e3:.1f}-{p[:,1].max()*1e3:.1f} mm  J 10/50/90% {np.percentile(J,10):.3f}/{np.median(J):.3f}/{np.percentile(J,90):.3f}  vel rms {np.sqrt((liq.velocities()**2).sum(1).mean()):.4f} m/s", flush=True)
print("wall %.0f s" % (time.time() - t0))
