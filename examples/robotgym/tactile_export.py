#!/usr/bin/env python3
"""Public tactile data for a robotgym recording made with --tactile: one npz with, for every fingertip gel pad, the
normal force, the indentation image and the tangential (shear) displacement field of the gel surface at every
recorded frame, plus a plain-text description of every array (key "readme"). Also used by usd_public_copy.py
--tactile for the per-pad telemetry curves.

    python examples/robotgym/tactile_export.py recording.npz --out robotgym_bottle_slow_tactile_data.npz
"""
import argparse
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from tactile_panel import PIXEL_MM, SCALE_M, pad_shear  # noqa: E402

README = """StiffGIPC tactile fingertips (VBTS gel pads), one entry per recorded video frame.
Each finger carries a 15.9 x 14.0 x 2.5 mm FEM gel pad (E 0.5 MPa, nu 0.4) simulated in the same IPC solve as the
scene; the readings below come from the simulated gel deformation and contact forces.
  t                     (T,)          s, simulation time of each frame
  pad_names             (P,)          pad_<arm>_<finger>
  normal_force_N        (T, P)        contact force along the pad normal (sum of per-vertex compressive components)
  indentation           (T, P, H, W)  uint16; indentation of the coat surface into the gel, metres = value * indentation_scale_m;
                                      orthographic depth camera behind the gel, H x W = 200 x 176 px, pixel_size_mm;
                                      row = pad x (along the finger, increasing toward the fingertip), column = pad y
                                      (across the finger); relative to the pad's first reading
  pad_x_of_row_mm       (H,)          pad-local x of each image row (mm)
  pad_y_of_col_mm       (W,)          pad-local y of each image column (mm)
  peak_indentation_mm   (T, P)        image maximum
  shear_grid_mm         (G, 2)        pad-local (x, y) of the points where the shear field is sampled (rest positions)
  shear_mm              (T, P, G, 2)  tangential displacement (x, y) of the gel surface at those points, relative to frame 0,
                                      in the pad frame (pad frame fitted to the gel's pinned back face every frame)
  peak_shear_mm         (T, P)        largest tangential displacement over all surface vertices
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(a.recording, allow_pickle=True)
    names = [str(x) for x in d["tac_names"]]
    t = np.asarray(d["t"], np.float32)
    fn = np.asarray(d["tac_fn"], np.float32)
    maps, idx = d["tac_maps"], np.asarray(d["tac_map_idx"], int)
    if maps.dtype != np.uint16:            # early recordings: uint8 = indentation / 2 mm * 255
        maps = (maps.astype(np.uint32) * 65535 // 255).astype(np.uint16)
    H, W = maps.shape[2:]
    j = np.clip(np.searchsorted(idx, np.arange(len(t)), side="right") - 1, 0, None)
    ind = maps[j]                           # one image per frame (the latest one recorded)
    ind[np.arange(len(t)) < idx[0]] = 0
    scale = SCALE_M / 65535.0
    sh = pad_shear(d, names, H, W)
    p = PIXEL_MM
    out = dict(readme=np.array(README), t=t, pad_names=np.array(names), normal_force_N=fn, indentation=ind,
               indentation_scale_m=np.float64(scale), pixel_size_mm=np.float64(p),
               pad_x_of_row_mm=((np.arange(H) - (H - 1) / 2.0) * p).astype(np.float32),
               pad_y_of_col_mm=((np.arange(W) - (W - 1) / 2.0) * p).astype(np.float32),
               peak_indentation_mm=(ind.reshape(len(t), len(names), -1).max(-1) * scale * 1e3).astype(np.float32))
    if sh is not None:
        base, flow, peak = sh
        grid = np.c_[((H - 1) / 2.0 - base[:, 1]) * p, (base[:, 0] - (W - 1) / 2.0) * p]   # back to pad (x, y) mm
        out.update(shear_grid_mm=grid.astype(np.float32), shear_mm=flow.astype(np.float32), peak_shear_mm=peak.astype(np.float32))
    np.savez_compressed(a.out, **out)
    mb = osp.getsize(a.out) / 1e6
    print(f"[tactile_export] {len(t)} frames x {len(names)} pads -> {a.out} ({mb:.1f} MB); peak force "
          f"{fn.max():.1f} N, peak indentation {out['peak_indentation_mm'].max():.2f} mm"
          + (f", peak shear {out['peak_shear_mm'].max():.2f} mm" if sh is not None else ""))


if __name__ == "__main__":
    main()
