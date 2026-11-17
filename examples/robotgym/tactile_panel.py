#!/usr/bin/env python3
"""Tactile panel for a robotgym recording made with --tactile: per frame, the indentation image of every fingertip gel
(heat map, or shaded like a GelSight camera image), the tangential (shear) displacement of the gel surface as arrows,
its peak indentation, peak shear and normal force, and the normal-force history with a time cursor. The shear field is
computed from the recorded gel vertices (pad frame fitted to the pinned back face). Writes PNG frames (480 x 720 by
default) that ffmpeg puts next to the path-traced video:

    python examples/robotgym/tactile_panel.py recording.npz --out panel_dir [--every 1]
    ffmpeg -i pathtraced.mp4 -framerate 25 -i panel_dir/p%05d.png -filter_complex "[0:v][1:v]hstack" tactile.mp4
"""
import argparse
import os
import os.path as osp

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

PIXEL_MM = 0.079375
SCALE_M = 2.0e-3            # tac_maps: uint16 = indentation / 2 mm * 65535 (uint8 / 255 in early recordings)


def _cjk_font():
    for f in font_manager.findSystemFonts():
        if "NotoSansCJK" in f and ("Regular" in f or "Medium" in f):
            return font_manager.FontProperties(fname=f)
    return None


def gelsight_shade(depth_mm):
    """Height field (mm, indentation into the gel) -> RGB in [0, 1] lit by red / green / blue lights from three sides."""
    from scipy.ndimage import gaussian_filter
    depth_mm = gaussian_filter(depth_mm, 1.2)        # sub-pixel noise of the rasterised coat surface
    gy, gx = np.gradient(depth_mm, PIXEL_MM)
    n = np.stack([gx, gy, np.ones_like(depth_mm)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    elev = np.deg2rad(35.0)
    rgb = np.empty(depth_mm.shape + (3,))
    for c, az in enumerate(np.deg2rad([90.0, 210.0, 330.0])):
        L = np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
        rgb[..., c] = n @ L
    base = np.array([0.50, 0.53, 0.56])
    out = base + 1.6 * (rgb - np.sin(elev))      # flat gel = base colour; slopes tint toward the facing light
    contact = np.clip(depth_mm / 0.25, 0.0, 1.0)[..., None]
    out = out * (1.0 - 0.30 * contact) + 0.30 * contact * np.array([0.96, 0.94, 0.90])   # pressed areas read brighter
    return np.clip(out, 0.0, 1.0)


def pad_shear(d, names, H, W, grid=(9, 8)):
    """Tangential displacement of every pad's coat surface relative to the first record, in the pad frame (fitted per
    record to the pinned back-face vertices). Returns (base (G, 2) display px [col, row of the flipped image],
    flow (records, pads, G, 2) mm along pad (x, y), peak (records, pads) mm), or None when the recording has no pad
    vertices or they do not match the pad asset."""
    import sys
    from scipy.spatial import Delaunay
    here = osp.dirname(osp.abspath(__file__))
    for q in (osp.join(here, "..", ".."), osp.join(here, "..", "lab_tactile")):   # repo stiff_physics.sensors, lab_pad
        sys.path.insert(0, osp.abspath(q))
    try:
        from lab_pad import load_pad_asset
        asset = load_pad_asset()
    except Exception as e:                           # noqa: BLE001
        print(f"[panel] no shear overlay ({e})")
        return None
    fem = [str(x) for x in d["fem_names"]]
    if any(n not in fem for n in names):
        return None
    st, co = asset.stick_mask, asset.coat_mask
    A = asset.points[st]; Ac = A.mean(0)
    rest = asset.points[co, :2]
    gx = (np.arange(grid[0]) + 0.5) / grid[0] * 2 - 1                  # cell centres in [-1, 1]
    gy = (np.arange(grid[1]) + 0.5) / grid[1] * 2 - 1
    ext = np.abs(rest).max(0)
    G = np.stack(np.meshgrid(gx * ext[0], gy * ext[1], indexing="ij"), -1).reshape(-1, 2)
    tri = Delaunay(rest)
    s = tri.find_simplex(G)
    G, s = G[s >= 0], s[s >= 0]
    b = np.einsum("gij,gj->gi", tri.transform[s, :2], G - tri.transform[s, 2])
    Wt = np.zeros((len(G), len(rest)))
    np.put_along_axis(Wt, tri.simplices[s], np.c_[b, 1.0 - b.sum(1)], axis=1)
    n = len(d["t"])
    flow = np.zeros((n, len(names), len(G), 2)); peak = np.zeros((n, len(names)))
    for k, name in enumerate(names):
        V = np.asarray(d[f"fem_verts_{fem.index(name)}"], np.float64)
        if V.shape[1] != asset.n_verts:
            return None
        B = V[:, st]; Bc = B.mean(1, keepdims=True)
        U, _, Vt = np.linalg.svd(np.einsum("ni,rnj->rij", A - Ac, B - Bc))
        sgn = np.sign(np.linalg.det(np.einsum("rji,rkj->rik", Vt, U)))
        Vt[:, 2] *= sgn[:, None]
        R = np.einsum("rji,rkj->rik", Vt, U)                            # world = R local + t
        L = np.einsum("rnj,rjk->rnk", V[:, co] - Bc, R) + Ac            # coat vertices, pad-local
        disp = (L - L[:1])[..., :2] * 1e3
        flow[:, k] = np.einsum("gc,rcj->rgj", Wt, disp)
        peak[:, k] = np.linalg.norm(disp, axis=-1).max(1)
    p = PIXEL_MM * 1e-3
    base = np.c_[G[:, 1] / p + (W - 1) / 2.0, (H - 1) / 2.0 - G[:, 0] / p]   # (col, row) of the flipped image
    return base, flow, peak


def label_of(name):
    # pad_<arm>_<side>: single-arm scenes name the arm "arm"
    _, arm, side = name.split("_", 2)
    s = {"left": "左指", "right": "右指"}[side]
    if arm == "arm":
        return s
    return {"left": "左臂", "right": "右臂"}.get(arm, arm) + " · " + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--every", type=int, default=1, help="render every N records (the video must use the same stride)")
    ap.add_argument("--fmax", type=float, default=0.0, help="force axis limit (N); 0 = from the data")
    ap.add_argument("--style", default="heat", choices=["heat", "gelsight"],
                    help="heat = indentation depth map with a colour bar (default); gelsight = shaded like a camera image")
    ap.add_argument("--dmax", type=float, default=0.0, help="colour-bar top (mm); 0 = 99.9th percentile of the run")
    ap.add_argument("--no-shear", action="store_true", help="no shear arrows")
    ap.add_argument("--shear-mag", type=float, default=0.0, help="arrow magnification; 0 = from the run's 99th-percentile shear")
    a = ap.parse_args()
    d = np.load(a.recording, allow_pickle=True)
    names = [str(x) for x in d["tac_names"]]
    fn = np.asarray(d["tac_fn"], float)             # (records, pads)
    maps = d["tac_maps"]                             # (maps, pads, H, W) uint16 (uint8 in early recordings)
    full = 65535.0 if maps.dtype == np.uint16 else 255.0
    idx = np.asarray(d["tac_map_idx"], int)
    t = np.asarray(d["t"], float)
    n = len(t)
    npad = len(names)
    fp = _cjk_font()
    os.makedirs(a.out, exist_ok=True)
    fmax = a.fmax if a.fmax > 0 else max(1.0, float(np.percentile(fn.max(1), 99.5)) * 1.15)
    dmax = a.dmax if a.dmax > 0 else max(0.2, float(np.percentile(maps.max(axis=(1, 2, 3)), 99.9)) / full * SCALE_M * 1e3)
    from scipy.ndimage import gaussian_filter
    shear = None if a.no_shear else pad_shear(d, names, maps.shape[2], maps.shape[3])
    if shear is not None:
        base, flow, speak = shear
        sp_px = np.median(np.diff(np.unique(np.round(base[:, 0], 3))))  # arrow spacing (px)
        q99 = float(np.percentile(np.linalg.norm(flow, axis=-1), 99.0))
        mag = a.shear_mag if a.shear_mag > 0 else min([m for m in (1, 2, 3, 5, 10, 20, 30, 50, 100)
                                                       if m * q99 / PIXEL_MM >= 0.45 * sp_px] or [100])
    colors = ["#e4572e", "#2e86ab", "#f3a712", "#59a96a"]
    dpi = 100
    fig = plt.figure(figsize=(a.width / dpi, a.height / dpi), dpi=dpi, facecolor="#f4f4f2")
    rows, cols = (1, 2) if npad <= 2 else (2, 2)
    W, H = a.width, a.height
    w = 0.44 if npad <= 2 else 0.30                               # image axes width (figure fraction)
    h = w * W * (200.0 / 176.0) / H                               # pad image aspect (H x W = 200 x 176 px)
    lab = 0.065                                                    # room for the two-line title of each image
    top = 0.925
    axes = []
    for k in range(npad):
        r, c = divmod(k, cols)
        x0 = 0.5 - w - 0.015 + c * (w + 0.03)
        y0 = top - lab - h - r * (h + lab)
        axes.append(fig.add_axes([x0, y0, w, h]))
    y_img_bottom = top - rows * (h + lab)
    axf = fig.add_axes([0.14, 0.065, 0.80, max(0.10, y_img_bottom - 0.065 - 0.13)])
    cax = fig.add_axes([0.30, y_img_bottom - 0.055, 0.40, 0.012])
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0.0, dmax), cmap="inferno")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7)
    cb.set_label("颜色：压入深度 (mm)" + (f"　　箭头：表面切向位移（放大 {mag:g} 倍）" if shear is not None else ""),
                 fontproperties=fp, fontsize=8)
    for i in range(0, n, a.every):
        j = int(np.searchsorted(idx, i, side="right") - 1)
        for k, ax in enumerate(axes):
            ax.clear(); ax.set_xticks([]); ax.set_yticks([])
            dep = maps[max(j, 0), k].astype(float) / full * SCALE_M * 1e3 if j >= 0 else np.zeros(maps.shape[2:])
            if a.style == "gelsight":
                ax.imshow(gelsight_shade(dep[::-1]), interpolation="bilinear")     # fingertip up
            else:
                ax.imshow(gaussian_filter(dep, 2.0)[::-1], cmap="inferno", vmin=0.0, vmax=dmax, interpolation="bilinear")
            if shear is not None:
                u = flow[i, k, :, 1] / PIXEL_MM * mag; v = -flow[i, k, :, 0] / PIXEL_MM * mag
                ax.quiver(base[:, 0], base[:, 1], u, v, angles="xy", scale_units="xy", scale=1.0, color="#3fe0ff",
                          width=0.010, headwidth=3.5, headlength=4.0, minlength=0.0)
                ax.set_xlim(-0.5, maps.shape[3] - 0.5); ax.set_ylim(maps.shape[2] - 0.5, -0.5)
            for s in ax.spines.values():
                s.set_color(colors[k % 4]); s.set_linewidth(2.0)
            sh_txt = f" · 切向 {speak[i, k]:.2f}" if shear is not None else ""
            if npad <= 2:
                title = f"{label_of(names[k])}\n压入 {dep.max():.2f}{sh_txt} mm · {fn[i, k]:.1f} N"
            else:                     # narrow 2 x 2 grid: force on the name line
                title = f"{label_of(names[k])} · {fn[i, k]:.1f} N\n压入 {dep.max():.2f}{sh_txt} mm"
            ax.set_title(title, fontproperties=fp, fontsize=9.5 if npad <= 2 else 9.0)
        axf.clear()
        for k in range(npad):
            axf.plot(t[: i + 1], fn[: i + 1, k], color=colors[k % 4], lw=1.4, label=label_of(names[k]))
        axf.axvline(t[i], color="#555555", lw=0.8)
        axf.set_xlim(0.0, float(t[-1])); axf.set_ylim(0.0, fmax)
        axf.set_xlabel("时间 (s)", fontproperties=fp, fontsize=9); axf.set_ylabel("法向力 (N)", fontproperties=fp, fontsize=9)
        axf.tick_params(labelsize=8)
        axf.legend(prop=fp if fp is None else font_manager.FontProperties(fname=fp.get_file(), size=8), loc="upper left", frameon=False)
        fig.texts.clear()
        fig.text(0.5, 0.965, "指尖触觉（凝胶垫的压入与切向位移）" if shear is not None else "指尖触觉（凝胶垫压入深度）",
                 ha="center", va="center", fontproperties=fp, fontsize=13)
        fig.savefig(osp.join(a.out, f"p{i // a.every:05d}.png"), dpi=dpi, facecolor=fig.get_facecolor())
    print(f"[panel] {len(range(0, n, a.every))} frames -> {a.out} (pads {names}, force axis {fmax:.1f} N"
          + (f", shear arrows x{mag:g}" if shear is not None else "") + ")")


if __name__ == "__main__":
    main()
