"""Bench (2026-11-12): the kinematic dropper pump on a RIGID tube with a PRESCRIBED cavity volume.

Isolates the pump of the Rokae dropper replay from the hand: the glass stem of that scene (bore r 3.2 mm, outer 4.6 mm,
200 mm) is moved kinematically through the task's phases above two open boxes, and the bulb's cavity volume follows a
script. The control law is the scene's AirChamberPump (target volume from the trapped air, conveyor speed
= error / (bore area x tau), tip-in-air and pendant-drop rules). Liquid and solver settings come from the launcher


  squeezed in air -> dip -> release (intake) -> hold -> lift -> move (+ slow squeeze, optional wobble) -> drip squeeze -> wait
  -> [return + release + squeeze under water]. The tube is a heavy body teleported every step (this build ignores
  set_abd_body_velocities and the per-body gravity switch for a free body), so its walls JUMP relative to the particles:
  keep the sideways speed low (--t-move; 0.6 mm jumps made a pendant particle cost 50-300 Newton iterations per frame).
  Judge the pump on the phases where the tube stands still; transport is the full replay's job.

    PYTHONPATH=. python examples/liquid/bench_pump_20261112.py --out out/pump_bench/p1
"""
import argparse, json, os, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "lab_tactile"))
sys.path.insert(0, HERE)
from stiff_physics.engine import Engine, Config  # noqa: E402
import lab_assets as LA  # noqa: E402
from m2_box import open_box  # noqa: E402

R_OUT, R_IN, STEM_L, SECTIONS = 0.0046, 0.0032, 0.200, 20
G = 9.81


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


class AirChamberPump:
    """rokae_dropper's AirChamberPump.apply() with the cavity volume passed in (keep the two in step)."""

    def __init__(self, eng, vp, tau=0.05, vmax=0.15, track=1.0, v_drop=0.0, gain_area=None):
        self.eng, self.vp, self.tau, self.vmax, self.track, self.v_drop = eng, vp, tau, vmax, track, v_drop
        self.area = np.pi * R_IN ** 2
        self.gain_area = self.area if gain_area is None else gain_area      # area the conveyor speed is computed with
        self.V_air_ref = None

    def apply(self, liq, tip, top, V):
        ax = top - tip
        L = float(np.linalg.norm(ax))
        ax = ax / L
        rel = liq - tip
        s = rel @ ax
        rad = np.linalg.norm(rel - np.outer(s, ax), axis=1)
        in_bore = (s > 0) & (s < L) & (rad < R_IN)
        around = (s > -8e-3) & (s < 10e-3) & (rad < R_OUT + 4e-3) & ~in_bore
        V_col = float(in_bore.sum()) * self.vp
        submerged = int(around.sum()) >= 6
        V_bore = self.area * L
        if self.V_air_ref is None:
            self.V_air_ref = V + V_bore - V_col
        target = V + V_bore - self.V_air_ref
        event = ""
        if target < -0.05 * self.vp:
            self.V_air_ref = V + V_bore
            target = 0.0
            event = "bubble"
        target = min(max(target, 0.0), V_bore)
        if target > V_col and not submerged:
            self.V_air_ref = V + V_bore - V_col
            target = V_col
            event = event or "air-in"
        err = target - V_col
        if self.v_drop > 0 and not submerged and -self.v_drop < err < 0:
            err = 0.0
            event = event or "pendant"
        conveyor = float(np.clip(err / self.gain_area / self.tau, -self.vmax, self.vmax))
        self.eng.set_liquid_pump(tip, top, R_IN + 0.5e-3, G, conveyor, 0.0, self.track, bore_radius=R_IN)
        return dict(V_col=V_col, n_col=int(in_bore.sum()), target=target, conveyor=conveyor, submerged=submerged, event=event,
                    s_lo=float(s[in_bore].min()) if in_bore.any() else 0.0, s_hi=float(s[in_bore].max()) if in_bore.any() else 0.0,
                    in_bore=in_bore)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/pump_bench/p")
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--dhat", type=float, default=1e-3)
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--pcg-tol", type=float, default=1e-4)
    ap.add_argument("--newton-cap", type=int, default=300)
    ap.add_argument("--tilt", type=float, default=9.0, help="tube tilt from the vertical (deg)")
    ap.add_argument("--dip", type=float, default=0.008, help="tip depth under the source surface (m)")
    ap.add_argument("--dv-in", type=float, default=1.0, help="cavity release under water (mL)")
    ap.add_argument("--t-release", type=float, default=1.0)
    ap.add_argument("--dv-carry", type=float, default=0.10, help="slow cavity shrink during lift + move (mL)")
    ap.add_argument("--wobble", type=float, default=0.0, help="cavity fluctuation amplitude during lift + move (mL, 2 Hz)")
    ap.add_argument("--dv-drip", type=float, default=0.34, help="cavity shrink of the drip squeeze (mL)")
    ap.add_argument("--t-drip", type=float, default=1.5)
    ap.add_argument("--drift", type=float, default=0.0, help="lateral tip drift under water during the release (m)")
    ap.add_argument("--pump-drop-ul", type=float, default=47.0)
    ap.add_argument("--pump-vmax", type=float, default=0.15, help="conveyor speed limit (m/s)")
    ap.add_argument("--pump-gain-r", type=float, default=0.0, help="radius of the area the conveyor speed is computed with (m); 0 = the bore")
    ap.add_argument("--drive", default="teleport", choices=("velocity", "teleport"))
    ap.add_argument("--t-move", type=float, default=4.0, help="duration of the sideways moves (s)")
    ap.add_argument("--dv-squeeze2", type=float, default=0.30, help="--ret: squeeze under water after the second intake (mL)")
    ap.add_argument("--air", type=float, default=0.015, help="start height of the tip above the source surface (m)")
    ap.add_argument("--ret", action="store_true", help="go back to the source box and release the bulb under water")
    ap.add_argument("--stop-after", default="", help="stop after this phase (e.g. hold)")
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dx, dt = args.dx, args.dt
    hs = dx / 2
    vp = hs ** 3
    cfg = Config(dt=dt, density=1000.0, young_modulus=1e7, poisson_rate=0.45, friction_rate=0.5, gravity=(0.0, -9.81, 0.0),
                 ground_offset=0.0, ground_normal=(0.0, 1.0, 0.0), absolute_dhat=args.dhat, preconditioner_type=0,
                 soft_motion_rate=1e4, newton_tol=args.newton_tol, pcg_tol=args.pcg_tol, newton_iter_cap=args.newton_cap,
                 pcg_max_iter_ratio=0.3, energy_rel_tol=0.0)
    eng = Engine(cfg)
    eng.set_log_level(args.log_level)
    floor_y, inner, wall = 0.006, 0.05, 0.005
    cs, cr = (0.20, 0.20), (0.32, 0.20)
    top_s, top_r = floor_y + 0.038, floor_y + 0.015
    for c, hgt in ((cs, 0.055), (cr, 0.04)):
        V, F = open_box(c, inner, wall, floor_y, hgt, 0.005)
        eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    # tube: tip at the local origin, axis +y, tilted about z (the top leans towards -x, away from the receiving box)
    ys = np.linspace(0.0, STEM_L, 13)
    mt = LA.revolve(np.array([(R_OUT, y) for y in ys] + [(R_IN, y) for y in ys[::-1]]), SECTIONS)
    tv, tf = np.asarray(mt.vertices, float), np.asarray(mt.faces, np.int32)
    th = np.radians(args.tilt)
    Rt = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    tip0 = np.array([cs[0], top_s + args.air, cs[1]])            # starts in the air, bulb already squeezed
    axis = Rt @ np.array([0.0, 1.0, 0.0])
    wv0 = tv @ Rt.T + tip0
    eng.load_mesh_from_data(wv0, tf, verts_per_face=3, dimensions=3, body_type="ABD", transform=np.eye(4), young_modulus=1e8, boundary_type="Free")
    body = int(eng.get_load_records()[-1].body_offset)
    eng.set_abd_body_density(body, 1.0e7)
    # water: jittered lattice in both boxes, the tube's volume (bore included) kept free
    rng = np.random.default_rng(0)
    m = 0.6 * dx
    pts, lab = [], []
    for k, (c, top) in enumerate(((cs, top_s), (cr, top_r))):
        h2 = inner / 2
        g = np.meshgrid(np.arange(c[0] - h2 + m, c[0] + h2 - m, hs), np.arange(floor_y + m, top, hs), np.arange(c[1] - h2 + m, c[1] + h2 - m, hs), indexing="ij")
        p = np.stack([x.ravel() for x in g], 1)
        p = p + rng.uniform(-0.31 * hs, 0.31 * hs, p.shape)
        loc = (p - tip0) @ Rt                       # tube frame
        keep = ~((np.hypot(loc[:, 0], loc[:, 2]) < R_OUT + 1.5e-3) & (loc[:, 1] > -3e-3) & (loc[:, 1] < STEM_L + 5e-3))
        pts.append(p[keep])
        lab.append(np.full(int(keep.sum()), k, np.int8))
    pts, lab = np.vstack(pts), np.concatenate(lab)
    eng.add_liquid(pts, dx, np.array([0.10, -0.005, 0.10]), np.array([0.42, 0.42, 0.30]), rho=1000.0, kappa=1e5, particle_volume=vp,
                   node_pool=90000, push_apart=0.9, j_mode=0, stick_h=0.0, mass_floor=0.05, apic=True, apic_ratio=1.0)
    eng.finalize()
    eng._engine.set_body_apply_gravity(body, False)
    C0 = np.asarray(eng.get_abd_body_transforms(np.array([body], np.int32)))[0]
    W0 = np.array(C0, np.float64)
    W0[:3, :3] = C0[:3, :3].T

    def teleport(offset):
        T = np.eye(4)
        T[:3, 3] = offset
        W = T @ W0
        C = np.array(W)
        C[:3, :3] = W[:3, :3].T
        eng.teleport_abd_bodies(np.array([body], np.int32), C[None])

    ids = np.array([body], np.int32)

    def drive(off_target):
        """returns the tube's offset at the start of the step (what the pump sees, as in the replay)"""
        if args.drive == "teleport":
            teleport(off_target)
            return off_target
        Cn = np.asarray(eng.get_abd_body_transforms(ids))[0]
        p = np.asarray(Cn[:3, 3] - C0[:3, 3], float)
        Vm = np.zeros((4, 4))
        Vm[:3, 3] = (off_target - p) / dt
        eng.set_abd_body_velocities(ids, Vm[None])
        return p

    y_in = -(args.air + args.dip)                  # tip offset when dipped
    lift = 0.176 - tip0[1]                         # carry height of the tip above the table (replay: 178 mm)
    drop_y = 0.094 - tip0[1]                       # drip height (replay: 94 mm)
    span = cr[0] - cs[0]
    V_rest = 12.6e-6
    ml = 1e-6
    phases = [("dip", 0.8), ("settle", 0.3), ("release", args.t_release), ("hold", 0.5), ("lift", 1.0), ("move", args.t_move), ("hold2", 0.5),
              ("drip", args.t_drip), ("wait", 1.0)]
    if args.ret:
        phases += [("back", args.t_move), ("lower", 1.0), ("release2", 1.0), ("hold3", 0.5), ("squeeze2", 1.0), ("end", 0.5)]
    pump = AirChamberPump(eng, vp, vmax=args.pump_vmax, v_drop=args.pump_drop_ul * 1e-9,
                          gain_area=np.pi * args.pump_gain_r ** 2 if args.pump_gain_r > 0 else None)
    print(f"[bench] {len(pts)} particles ({(lab == 0).sum()} source, {(lab == 1).sum()} receiving), particle {vp*1e9:.1f} ul; tube tilt {args.tilt} deg, "
          f"tip {args.dip*1e3:.0f} mm under the source surface; release {args.dv_in} mL = {args.dv_in*ml/vp:.0f} particles, drip {args.dv_drip} mL = "
          f"{args.dv_drip*ml/vp:.0f} particles, carry {args.dv_carry} mL, wobble {args.wobble} mL", flush=True)
    rows, ever = [], np.zeros(len(pts), bool)
    off_prev = np.zeros(3)
    nt0 = eng._engine.get_total_newton_iters()
    t = 0.0
    step = 0
    wall0 = time.time()
    marks = {}
    for name, dur in phases:
        t_ph = 0.0
        nst = int(round(dur / dt))
        for i in range(nst):
            u = (i + 1) / nst
            off = np.array([0.0, y_in, 0.0])
            V = V_rest - args.dv_in * ml
            if name == "dip":
                off = np.array([0.0, y_in * smooth(u), 0.0])
            elif name == "release":
                V = V_rest - args.dv_in * ml * (1.0 - smooth(u))
                off = np.array([args.drift * u, y_in, 0.0])
            elif name == "hold":
                V = V_rest
                off = np.array([args.drift, y_in, 0.0])
            elif name in ("lift", "move"):
                uc = (u if name == "lift" else 1.0 + args.t_move * u) / (1.0 + args.t_move)   # progress over lift + move
                V = V_rest - args.dv_carry * ml * uc + args.wobble * ml * np.sin(2 * np.pi * 2.0 * (t + dt))
                if name == "lift":
                    off = np.array([args.drift * (1 - u), y_in + (lift - y_in) * smooth(u), 0.0])
                else:
                    off = np.array([span * smooth(u), lift + (drop_y - lift) * smooth(u), 0.0])
            elif name in ("hold2", "drip", "wait"):
                V = V_rest - args.dv_carry * ml - (args.dv_drip * ml * smooth(u) if name == "drip" else (args.dv_drip * ml if name == "wait" else 0.0))
                off = np.array([span, drop_y, 0.0])
            elif name == "back":
                V = V_rest - args.dv_carry * ml - args.dv_drip * ml
                off = np.array([span * (1 - smooth(u)), drop_y + (lift - drop_y) * smooth(u), 0.0])
            elif name == "lower":
                V = V_rest - args.dv_carry * ml - args.dv_drip * ml
                off = np.array([0.0, lift + (y_in - lift) * smooth(u), 0.0])
            elif name in ("release2", "hold3"):
                V = V_rest - (args.dv_carry + args.dv_drip) * ml * (1.0 - (smooth(u) if name == "release2" else 1.0))
            elif name in ("squeeze2", "end"):
                V = V_rest - args.dv_squeeze2 * ml * (smooth(u) if name == "squeeze2" else 1.0)
            p_now = drive(off)
            terr = float(np.linalg.norm(p_now - off_prev))                 # where the last step left the tube vs its target
            off_prev = off
            tip = tip0 + p_now
            top = tip + axis * STEM_L
            liq = np.asarray(eng.get_liquid_particles())
            st = pump.apply(liq, tip, top, V)
            ever |= st.pop("in_bore")
            ts = time.time()
            eng.step()
            wall_s = time.time() - ts
            nt = eng._engine.get_total_newton_iters()
            it = nt - nt0
            nt0 = nt
            t += dt
            step += 1
            es = eng.get_liquid_pump_state()
            row = dict(step=step, t=t, phase=name, V=V, newton=int(it), wall=wall_s, tip_y=float(tip[1]), track_err=terr,
                       ring=float(es["V_air"]), feed=float(es["p_gauge"]), leave=float(es["V_liq"]), c_slab=float(es["V0"]), **{k: v for k, v in st.items()})
            rows.append(row)
            if step % 20 == 0 or it >= 50:
                print(f"t {t:5.2f} {name:8s} V {V*1e6:6.3f} | bore {st['n_col']:3d} target {st['target']/vp:5.1f} conveyor {st['conveyor']*100:6.2f} cm/s "
                      f"sub {int(st['submerged'])} {st['event']:7s} | column s {st['s_lo']*1e3:5.1f}..{st['s_hi']*1e3:5.1f} mm | ring {row['ring']:.0f} feed {row['feed']:.0f} "
                      f"| newton {it:3d} {wall_s*1e3:5.0f} ms | drive err {row['track_err']*1e3:.3f} mm", flush=True)
        marks[name] = step
        if args.stop_after == name:
            break
    x = np.asarray(eng.get_liquid_particles())

    def where(x):
        ax = axis
        rel = x - (tip0 + off)                                              # the last commanded pose
        s = rel @ ax
        rad = np.linalg.norm(rel - np.outer(s, ax), axis=1)
        out = np.full(len(x), 4, np.int8)                                   # 4 = elsewhere (airborne / on a rim)
        out[x[:, 1] < 0.012] = 3                                            # 3 = table
        for k, c in ((0, cs), (1, cr)):
            inb = (np.abs(x[:, 0] - c[0]) < inner / 2 + 1e-3) & (np.abs(x[:, 2] - c[1]) < inner / 2 + 1e-3) & (x[:, 1] > floor_y - 1e-3) & (x[:, 1] < 0.065)
            out[inb] = k                                                    # 0 = source box, 1 = receiving box
        out[(s > 0) & (s < STEM_L) & (rad < R_IN)] = 2                      # 2 = bore
        return out

    w = where(x)
    src = lab == 0
    its = np.array([r["newton"] for r in rows])
    summ = dict(
        particles=int(len(pts)), drawn_ever=int((ever & src).sum()),
        end_source_origin=dict(source_box=int((src & (w == 0)).sum()), receiving_box=int((src & (w == 1)).sum()), bore=int((src & (w == 2)).sum()),
                               table=int((src & (w == 3)).sum()), elsewhere=int((src & (w == 4)).sum())),
        receiving_origin_moved=int(((lab == 1) & (w != 1)).sum()),
        newton_mean=float(its.mean()), newton_max=int(its.max()), frames_ge50=int((its >= 50).sum()), frames_cap=int((its >= args.newton_cap).sum()),
        wall_s=time.time() - wall0, marks=marks)
    for name in ("settle", "release", "hold", "hold2", "wait", "lower", "hold3", "end"):
        if name in marks:
            r = rows[marks[name] - 1]
            summ["at_" + name] = dict(bore=r["n_col"], target=r["target"] / vp, s_lo_mm=r["s_lo"] * 1e3, s_hi_mm=r["s_hi"] * 1e3)
    json.dump(dict(args=vars(args), summary=summ, rows=rows), open(os.path.join(args.out, "log.json"), "w"))
    np.savez_compressed(os.path.join(args.out, "final.npz"), x=x, lab=lab, where=w, ever=ever)
    print("[bench] summary " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
