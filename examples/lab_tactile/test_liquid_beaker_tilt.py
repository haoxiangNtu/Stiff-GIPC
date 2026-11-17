"""Beaker tilt test with the real lab beaker mesh: fill to half, lift, then
tilt slowly (0 -> 100 deg over 8 s) about a horizontal axis through the lip.
Reports the fraction still inside vs tilt; physically the water must start
leaving at ~33 deg (half full) and be gone well before 90 deg."""
import os, sys, time, numpy as np, warp as wp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)
wp.config.quiet = True
from stiff_physics.liquid import LiquidMPM, BC_SEPARABLE, BC_SLIP, BC_STICKY
from lab_assets import build_all

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
dx = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
interior = os.environ.get("LIQ_INTERIOR", "separable")
thick = os.environ.get("LIQ_THICKEN")               # m, default = dx - 1 mm
props = build_all(verbose=False); p = props["beaker_mix"]
V0 = np.asarray(p.verts, np.float64).copy(); F = np.asarray(p.faces, np.int64)
H = float(V0[:, 1].max()); r = np.hypot(V0[:, 0], V0[:, 2]); r_in = float(r[V0[:, 1] > 0.2 * H].min())
origin = np.array([-0.20, -0.02, -0.20]); extent = np.array([0.40, 0.30, 0.40])
liq = LiquidMPM(origin, extent, dx, device=dev, gravity=(0, -9.81, 0), kappa=float(os.environ.get("LIQ_KAPPA", "1e5")), ppc=8, substep_dt=float(os.environ.get("LIQ_SUBSTEP", "1e-4")), flip_ratio=float(os.environ.get("LIQ_FLIP", "0.95")), verbose=False, apic=bool(int(os.environ.get("LIQ_APIC", "1"))))
liq.push_apart = float(os.environ.get("LIQ_PUSH", "0"))
liq.interior_type = {"separable": BC_SEPARABLE, "slip": BC_SLIP, "sticky": BC_STICKY}[interior]
liq.band_ext = float(os.environ.get("LIQ_BAND_EXT", "0")) * dx
if os.environ.get("LIQ_NOPROJ"): liq.particle_projection = False
liq.add_halfspace([0, 0.0, 0], [0, 1, 0])
m = float(os.environ.get("LIQ_SEED_MARGIN_MM", str(0.6 * dx * 1e3))) * 1e-3
FILL = float(os.environ.get("LIQ_FILL", "0.5"))
n = liq.seed_box([-r_in + m, 0.004 + m, -r_in + m], [r_in - m, FILL * H, r_in - m], jitter=0.3)
xp = liq.solver.ptc_x.numpy() + liq.origin; keep = np.hypot(xp[:, 0], xp[:, 2]) < r_in - m
for name in ("ptc_x", "ptc_v", "ptc_m", "ptc_k", "ptc_vol0", "ptc_F", "ptc_G", "ptc_dlogJ"):
    a = getattr(liq.solver, name); setattr(liq.solver, name, wp.array(a.numpy()[keep], dtype=a.dtype, device=liq.device))
liq.solver.n_ptc = int(keep.sum()); liq.solver.n_ptc_wp.fill_(liq.solver.n_ptc)
liq.set_surface_mesh(V0, F, thicken=(float(thick) * 1e-3 if thick else None))
print(f"apic={liq.solver.enable_apic} push={liq.push_apart} fill={FILL} margin={m*1e3:.1f}mm flip={liq.solver.flip_ratio} band_ext={liq.band_ext*1e3:.1f}mm proj={liq.particle_projection} interior={interior} thicken={liq.thicken*1e3:.1f}mm dx={dx*1e3:.0f}mm particles={liq.n_particles} beaker H={H*1e3:.0f} r_in={r_in*1e3:.1f}")
dt = 0.005; LIFT = 0.12
def pose(tt):
    lift = min(tt, 1.0) * LIFT
    ang = np.radians(100.0) * np.clip((tt - 1.5) / 8.0, 0.0, 1.0)      # 12.5 deg/s
    c, s_ = np.cos(ang), np.sin(ang)
    R = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
    pivot = np.array([0.0, H + lift, r_in])                            # lip on the +z side
    return (V0 + [0, lift, 0] - pivot) @ R.T + pivot, R, pivot, lift
t0 = time.time(); first_out = None; snaps = {}
land = np.full(liq.n_particles, np.nan); crossed = np.zeros(liq.n_particles, bool); pts_prev = liq.positions()
for k in range(int(11.0 / dt)):
    tt = (k + 1) * dt
    V, R, pivot, lift = pose(tt)
    liq.update_surface(V, dt); liq.step(dt)
    _ang = np.degrees(np.radians(100.0) * np.clip((tt - 1.5) / 8.0, 0, 1))
    for a_s in (0.0, 20.0, 40.0, 60.0, 80.0):
        key = f"a{int(a_s)}"
        if key not in snaps and _ang >= a_s and tt > 1.4: snaps[key] = dict(p=liq.positions(), V=V.copy(), R=R.copy(), pivot=pivot.copy(), lift=lift, ang=_ang)
    pts_now = liq.positions()
    if k > 0:
        yc = pivot[1] - 0.03
        c = (~crossed) & (pts_prev[:, 1] >= yc) & (pts_now[:, 1] < yc)
        land[c] = pts_now[c][:, 2] - pivot[2]; crossed[c] = True
    pts_prev = pts_now
    if k % 40 == 39:
        pts = liq.positions(); q = (pts - pivot) @ R + pivot - [0, lift, 0]
        inside = (np.hypot(q[:, 0], q[:, 2]) < r_in + 0.003) & (q[:, 1] > 0.0) & (q[:, 1] < H + 0.005)
        ang = np.degrees(np.radians(100.0) * np.clip((tt - 1.5) / 8.0, 0, 1))
        if first_out is None and inside.mean() < 0.98: first_out = ang
        print(f"t={tt:5.2f}s tilt {ang:5.1f} deg  inside {inside.mean()*100:5.1f}%  on table {(pts[:,1] < 0.01).mean()*100:5.1f}%  proj {liq.projected_last()}", flush=True)
print(f"first 2% out at tilt {first_out} deg (physical ~33 deg for half full); wall {time.time()-t0:.0f}s")
if os.environ.get("LIQ_SNAP"):
    np.savez(os.environ["LIQ_SNAP"], F=F, **{f"{k}_{kk}": vv for k, d in snaps.items() for kk, vv in d.items()})
