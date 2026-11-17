"""Pour test: the open box from test_liquid_cup.py is filled, settled, then
rotated 100 deg about a horizontal axis through its rim over 2 s and held.
Reports the fraction of water that left the cup (expected: most of it)."""
import os, sys, time, numpy as np, warp as wp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
wp.config.quiet = True
from stiff_physics.liquid import LiquidMPM, BC_STICKY, BC_SLIP, BC_SEPARABLE
import trimesh

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
dx = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
interior = os.environ.get("LIQ_INTERIOR", "separable")
t_wall = 0.004
parts = [trimesh.creation.box(extents=[0.088, t_wall, 0.088]).apply_translation([0, t_wall / 2, 0])]
for sx in (-1, 1):
    parts.append(trimesh.creation.box(extents=[t_wall, 0.104, 0.088]).apply_translation([sx * (0.040 + t_wall / 2), 0.052, 0]))
    parts.append(trimesh.creation.box(extents=[0.088, 0.104, t_wall]).apply_translation([0, 0.052, sx * (0.040 + t_wall / 2)]))
cup = trimesh.util.concatenate(parts)
origin = np.array([-0.20, -0.02, -0.20]); extent = np.array([0.40, 0.30, 0.40])
liq = LiquidMPM(origin, extent, dx, device=dev, gravity=(0, -9.81, 0), kappa=1e5, ppc=8, substep_dt=1e-4, flip_ratio=float(os.environ.get("LIQ_FLIP", "0.95")), verbose=True)
liq.interior_type = {"separable": BC_SEPARABLE, "slip": BC_SLIP, "sticky": BC_STICKY}[interior]
liq.band_ext = float(os.environ.get("LIQ_BAND_EXT", "0")) * dx
if os.environ.get("LIQ_NOPROJ"): liq.particle_projection = False
liq.add_halfspace([0, 0.0, 0], [0, 1, 0])          # table under the cup
SM = float(os.environ.get("LIQ_SEED_MARGIN_MM", "5")) * 1e-3
n = liq.seed_box([-0.040 + SM, 0.004 + SM, -0.040 + SM], [0.040 - SM, 0.062, 0.040 - SM], jitter=0.5)
V0 = cup.vertices.copy(); F = cup.faces
liq.set_surface_mesh(V0, F, thicken=(float(os.environ["LIQ_THICK_MM"]) * 1e-3 if os.environ.get("LIQ_THICK_MM") else None))
print(f"flip={liq.solver.flip_ratio} band_ext={liq.band_ext*1e3:.1f}mm thick={liq.thicken*1e3:.1f}mm proj={liq.particle_projection} interior={interior} dx={dx} particles={n} grid={liq.grid_size}")
dt = 0.005
# lift the cup 0.12 m first (so poured water falls onto the table), then tilt about the +x rim edge
def pose(tt):
    lift = min(tt, 1.0) / 1.0 * 0.12
    ang = np.radians(100.0) * np.clip((tt - 1.5) / 2.0, 0.0, 1.0)
    c, s_ = np.cos(ang), np.sin(ang)
    R = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])        # rotate about x: +z rim edge goes down
    pivot = np.array([0.0, 0.104 + lift, 0.044])
    return (V0 + [0, lift, 0] - pivot) @ R.T + pivot
def inside_frac(V):
    # cup frame: invert the current pose for the water points
    p = liq.positions(); c = V.mean(0)
    # local axes from the current cup: use the pose rotation by comparing face normals is overkill; use the fact that the
    # cup's local y axis is the rotated [0,1,0]
    tt = liq.sim_time; ang = np.radians(100.0) * np.clip((tt - 1.5) / 2.0, 0.0, 1.0); cs, sn = np.cos(ang), np.sin(ang)
    R = np.array([[1, 0, 0], [0, cs, -sn], [0, sn, cs]])
    lift = min(tt, 1.0) / 1.0 * 0.12; pivot = np.array([0.0, 0.104 + lift, 0.044])
    q = (p - pivot) @ R + pivot - [0, lift, 0]          # back to the rest cup frame
    ins = (np.abs(q[:, 0]) < 0.040) & (np.abs(q[:, 2]) < 0.040) & (q[:, 1] > 0.004) & (q[:, 1] < 0.104)
    return ins.mean(), p
t0 = time.time()
for k in range(int(5.0 / dt)):
    V = pose((k + 1) * dt)
    liq.update_surface(V, dt); liq.step(dt)
    if k % 50 == 49:
        f, p = inside_frac(V)
        print(f"t={liq.sim_time:4.2f}s  tilt {np.degrees(np.radians(100.0) * np.clip((liq.sim_time - 1.5) / 2.0, 0, 1)):5.1f} deg  inside cup {f*100:5.1f}%  "
              f"on table {(p[:,1] < 0.01).mean()*100:5.1f}%  y max {p[:,1].max():.3f}  bc {liq.n_bc} proj {liq.projected_last()}", flush=True)
print("wall %.1f s" % (time.time() - t0))
