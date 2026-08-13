"""Piezoresistive taxel-array tactile sensor on a simulated gel pad.

A piezoresistive skin (FSR sheet, e-skin taxel grid) is the OTHER major tactile
modality next to VBTS: no camera, no optics — a rows x cols grid of taxels whose
electrical resistance drops under local pressure. Where the VBTS pipeline turns
the gel's deformed GEOMETRY into an image, this turns the gel's contact FORCE
FIELD into a coarse pressure map. Both live on the same FEM gel and can be read
simultaneously from one simulation step.

What the engine provides and why it is sufficient
-------------------------------------------------
``get_vertex_contact_forces`` returns per-vertex IPC contact forces in Newtons
(components: barrier "normal", solver-lagged "friction_lagged", or "total"),
validated on the shear rig (measured slip ratio |Ft|/|Fn| = 0.600 +- 0.008 with
mu = 0.6). A taxel is nothing but the integral of surface traction over its
cell, so binning per-vertex forces into taxel cells IS the sensor's physical
input. What the FEM does NOT model is the sensing material's own electronics --
that is the transduction model below, which is explicit and swappable.

Physics -> signal chain, each step honest about what it is:

  1. per-vertex world force  F_i           (IPC solve, Newtons — exact)
  2. rotate into sensor frame; keep the compressive normal component F_z>0 and
     tangential (F_x, F_y)                 (frame bookkeeping — exact)
  3. bin by REST (x, y) into cells         (taxel aggregation — exact up to
     lateral vertex motion under shear, ~0.05 mm against >= 1 mm pitch)
  4. conductance G = G0 + g_per_N * F      (FSR transduction MODEL: conductance
     roughly linear in force is the standard FSR datasheet behavior)
  5. voltage divider + ADC quantisation    (readout circuit MODEL)

Steps 4-5 are parameterised, not physics: calibrate `g_per_N`/`r_off_ohm`
against a real sheet to emulate a specific product. Conservation is checkable
at step 3 and `read()` reports it every call: sum of taxel normal forces vs the
gel's total normal contact force.

Mesh-resolution caveat: per-vertex IPC forces concentrate on contact vertices
(each vertex carries roughly its Voronoi share of the traction), so a taxel
needs several vertices to average over. With the 1e-09 pad (~1.3 mm vertex
spacing) a 2 mm taxel holds ~2-3 coat vertices; the 1e-10 pad (~0.55 mm) holds
~13. `PiezoTaxelArray.verts_per_taxel()` reports the actual histogram — check
it before trusting a fine pitch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TaxelArrayConfig:
    """Geometry + transduction of one taxel sheet. Defaults approximate a
    16x16 FSR array at 2 mm pitch (32 x 32 mm active area)."""

    rows: int = 16
    cols: int = 16
    pitch_mm: float = 2.0
    centre_xy_mm: tuple[float, float] = (0.0, 0.0)  # in sensor-local xy

    # ---- transduction (FSR-like): G = 1/r_off + g_per_N * F ---------------
    r_off_ohm: float = 1.0e6        # zero-load resistance
    g_per_N: float = 2.0e-4         # conductance sensitivity, S per newton
    noise_g_rms: float = 2.0e-7     # conductance-referred noise, S
    # ---- readout circuit: divider  v = v_ref * r_pull / (r_pull + R) ------
    v_ref: float = 3.3
    r_pullup_ohm: float = 10.0e3
    adc_bits: int = 12
    seed: int | None = 0

    @property
    def active_mm(self) -> tuple[float, float]:
        return (self.rows * self.pitch_mm, self.cols * self.pitch_mm)


class PiezoTaxelArray:
    """Bin a gel's per-vertex contact forces into a taxel pressure image.

    Reads the SAME GelBody the VBTS camera reads; the two sensors coexist on
    one pad and one physics step.
    """

    def __init__(self, gel, cfg: TaxelArrayConfig | None = None):
        self.gel = gel
        self.cfg = cfg or TaxelArrayConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        c = self.cfg
        # vertex -> taxel assignment from REST sensor-local positions.
        # All gel vertices are assigned (not just coat): contact force lives on
        # surface vertices anyway, and including everything makes the
        # conservation check exact rather than "most of it".
        p = gel.asset.points * 1e3                     # mm, sensor-local rest
        x = p[:, 0] - c.centre_xy_mm[0] + c.rows * c.pitch_mm / 2.0
        y = p[:, 1] - c.centre_xy_mm[1] + c.cols * c.pitch_mm / 2.0
        r = np.floor(x / c.pitch_mm).astype(np.int64)
        col = np.floor(y / c.pitch_mm).astype(np.int64)
        ok = (r >= 0) & (r < c.rows) & (col >= 0) & (col < c.cols)
        self._vert_ok = ok
        self._vert_cell = np.where(ok, r * c.cols + col, 0)
        self._n_cells = c.rows * c.cols

    # ------------------------------------------------------------------ info
    def verts_per_taxel(self) -> np.ndarray:
        """(rows, cols) count of gel vertices feeding each taxel."""
        cnt = np.bincount(self._vert_cell[self._vert_ok], minlength=self._n_cells)
        return cnt.reshape(self.cfg.rows, self.cfg.cols)

    def taxel_centres_mm(self) -> np.ndarray:
        """(rows, cols, 2) sensor-local xy of each taxel centre, mm."""
        c = self.cfg
        rr = (np.arange(c.rows) + 0.5) * c.pitch_mm - c.rows * c.pitch_mm / 2.0
        cc = (np.arange(c.cols) + 0.5) * c.pitch_mm - c.cols * c.pitch_mm / 2.0
        gx, gy = np.meshgrid(rr + c.centre_xy_mm[0], cc + c.centre_xy_mm[1],
                             indexing="ij")
        return np.stack([gx, gy], axis=-1)

    # ------------------------------------------------------------------ read
    def read(self, sensor_tf: np.ndarray | None = None,
             components: str = "total", noisy: bool = True) -> dict:
        """One taxel-array frame from the current engine state.

        Args:
            sensor_tf: current sensor pose (as in observe()); default = the
                gel's build pose, correct for translation-only drives.
            components: which engine force components feed the sheet.
        Returns dict with, all (rows, cols) unless noted:
            force_N        compressive normal force per taxel
            pressure_kPa   force / taxel area
            shear_N        (rows, cols, 2) tangential force per taxel
            resistance_ohm, v_out, adc   the transduced readout chain
            conservation   scalar: taxel-sum / gel-total normal force
        """
        c = self.cfg
        tf = self.gel.world_tf if sensor_tf is None else np.asarray(sensor_tf)
        R = tf[:3, :3]

        Fw = self.gel.contact_force_map(components)     # (N,3) world newtons
        Fl = Fw @ R                                     # world -> sensor frame
        fz = np.clip(Fl[:, 2], 0.0, None)               # compressive only
        cell = self._vert_cell
        ok = self._vert_ok

        f_cell = np.bincount(cell[ok], weights=fz[ok], minlength=self._n_cells)
        sx = np.bincount(cell[ok], weights=Fl[ok, 0], minlength=self._n_cells)
        sy = np.bincount(cell[ok], weights=Fl[ok, 1], minlength=self._n_cells)
        F = f_cell.reshape(c.rows, c.cols)
        shear = np.stack([sx, sy], axis=-1).reshape(c.rows, c.cols, 2)

        area_m2 = (c.pitch_mm * 1e-3) ** 2
        pressure_kPa = F / area_m2 * 1e-3

        # ---- transduction ------------------------------------------------
        G = 1.0 / c.r_off_ohm + c.g_per_N * F
        if noisy and c.noise_g_rms > 0:
            G = np.clip(G + self._rng.normal(0.0, c.noise_g_rms, G.shape),
                        1e-12, None)
        R_ohm = 1.0 / G
        v = c.v_ref * c.r_pullup_ohm / (c.r_pullup_ohm + R_ohm)
        levels = (1 << c.adc_bits) - 1
        adc = np.round(v / c.v_ref * levels).astype(np.int32)

        total = float(fz[ok].sum())
        gel_total = float(np.clip(Fw @ R, 0, None)[:, 2].sum()) or 1e-12
        return dict(force_N=F, pressure_kPa=pressure_kPa, shear_N=shear,
                    resistance_ohm=R_ohm, v_out=v, adc=adc,
                    conservation=total / gel_total,
                    total_normal_N=total)
