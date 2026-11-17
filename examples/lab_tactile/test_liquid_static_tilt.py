"""Diagnostic: a STATIC open box tilted by ALPHA deg about the x axis through
its +z rim edge, with water seeded as a horizontal slab whose level is above
the low rim.  Correct physics: the excess water pours over the low rim within
~1 s.  Configs via env: LIQ_MODE = int (interior-only separable, thickened) |
ext (exterior band 1 dx + interior slip, no thickening) | none (no node BCs,
particle projection only)."""
import os, sys, time, numpy as np, warp as wp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
wp.config.quiet = True
from stiff_physics.liquid import LiquidMPM, BC_SEPARABLE, BC_SLIP
import trimesh
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; dx = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
ALPHA = float(os.environ.get("LIQ_ALPHA", "45")); MODE = os.environ.get("LIQ_MODE", "int")
t_wall = 0.004
parts = [trimesh.creation.box(extents=[0.088, t_wall, 0.088]).apply_translation([0, t_wall / 2, 0])]
for sx in (-1, 1):
    parts.append(trimesh.creation.box(extents=[t_wall, 0.104, 0.088]).apply_translation([sx * (0.040 + t_wall / 2), 0.052, 0]))
    parts.append(trimesh.creation.box(extents=[0.088, 0.104, t_wall]).apply_translation([0, 0.052, sx * (0.040 + t_wall / 2)]))
cup = trimesh.util.concatenate(parts)
LIFT = 0.12; a = np.radians(ALPHA); c, s_ = np.cos(a), np.sin(a)
R = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]); pivot = np.array([0.0, 0.104 + LIFT, 0.044])
V = (cup.vertices + [0, LIFT, 0] - pivot) @ R.T + pivot               # +z rim edge is the low lip (at pivot height)
origin = np.array([-0.20, -0.02, -0.20]); extent = np.array([0.40, 0.34, 0.40])
FLIP = float(os.environ.get("LIQ_FLIP", "0.95"))
liq = LiquidMPM(origin, extent, dx, device=dev, gravity=(0, -9.81, 0), kappa=1e5, ppc=8, substep_dt=1e-4, verbose=False, flip_ratio=FLIP)
liq.add_halfspace([0, 0.0, 0], [0, 1, 0])
TH = os.environ.get("LIQ_THICK_MM")
if MODE == "int":
    liq.interior_type = BC_SEPARABLE; liq.band_ext = 0.0; thick = None
elif MODE == "ext":
    liq.interior_type = BC_SLIP; liq.band_ext = float(os.environ.get("LIQ_BAND_EXT", "1.0")) * dx; thick = 0.0
elif MODE == "both":
    liq.interior_type = {"separable": BC_SEPARABLE, "slip": BC_SLIP}[os.environ.get("LIQ_INTERIOR", "separable")]; liq.band_ext = float(os.environ.get("LIQ_BAND_EXT", "1.0")) * dx; thick = None
else:
    liq.interior_type = BC_SEPARABLE; liq.band_ext = 0.0; thick = None
# water: horizontal slab inside the tilted box, top 15 mm ABOVE the lip height
lip_y = pivot[1]
cand, vol0 = liq._lattice(V.min(0) + 0.004, V.max(0) - 0.004)
cm = trimesh.Trimesh(V, cup.faces, process=False)
# inside the cavity: transform candidates to the box frame and test the inner box
q = (cand - pivot) @ R + pivot - [0, LIFT, 0]
SM = float(os.environ.get("LIQ_SEED_MARGIN_MM", "4")) * 1e-3
inner = (np.abs(q[:, 0]) < 0.040 - SM) & (np.abs(q[:, 2]) < 0.040 - SM) & (q[:, 1] > 0.004 + SM) & (q[:, 1] < 0.104)
sel = inner & (cand[:, 1] < lip_y + 0.015)
pts = cand[sel]; n = liq._seed(pts, vol0, (0, 0, 0))
if TH is not None: thick = float(TH) * 1e-3
liq.set_surface_mesh(V, cup.faces, thicken=thick)
if os.environ.get("LIQ_NOPROJ"): liq.particle_projection = False
if MODE == "none":
    liq.band = 0.0; liq.band_ext = 0.0      # mesh query reach 0 -> no node BCs; projection net only
if MODE == "nomesh":
    liq.mesh = None; liq.band = 0.0; liq.band_ext = 0.0; liq.particle_projection = False   # water block on the domain floor only
above0 = (liq.positions()[:, 1] > lip_y).mean()
print(f"flip={FLIP} mode={MODE} alpha={ALPHA} dx={dx*1e3:.0f}mm particles={n} thicken={liq.thicken*1e3:.1f}mm  initially above the lip: {above0*100:.1f}%")
dt = 0.005; t0 = time.time(); snaps = {0.0: liq.positions()}
T_END = float(os.environ.get("LIQ_TEND", "2.0"))
for k in range(int(T_END / dt)):
    liq.update_surface(V, dt); liq.step(dt)
    if round(liq.sim_time, 3) in (0.1, 0.2, 0.5, 1.0, 2.0): snaps[round(liq.sim_time, 3)] = liq.positions()
    if k % 40 == 39:
        p = liq.positions(); qq = (p - pivot) @ R + pivot - [0, LIFT, 0]
        inside = (np.abs(qq[:, 0]) < 0.040) & (np.abs(qq[:, 2]) < 0.040) & (qq[:, 1] > 0.0) & (qq[:, 1] < 0.104)
        from scipy.spatial import cKDTree
        dd, _ = cKDTree(p).query(p, k=2); nn = dd[:, 1]
        print(f"t={liq.sim_time:4.2f}s  inside {inside.mean()*100:5.1f}%  above lip {(p[:,1] > lip_y).mean()*100:5.1f}%  on table {(p[:,1] < 0.01).mean()*100:5.1f}%  bc {liq.n_bc} proj {liq.projected_last()}  clumped {(nn<1e-3).mean()*100:5.1f}%  NN {np.median(nn)*1e3:.2f}mm", flush=True)
print("wall %.0f s" % (time.time() - t0))
np.savez(os.environ.get("LIQ_SNAP", f"out/lag/static_{MODE}_snaps.npz"), V=V, F=cup.faces, pivot=pivot, R=R, lift=LIFT, lip_y=lip_y, **{f"t{k}": v for k, v in snaps.items()})
