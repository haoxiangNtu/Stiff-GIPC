"""Full-hand pressure-glove API over simulated taxel arrays (vendor-neutral).

Commercial tactile data gloves expose one full-hand pressure matrix
(63 x 39 in the current products) with per-finger/palm region access, ADC
counts 0..254 that deliberately do NOT claim newton calibration, contact-point
statistics, and an IMU. This module gives the simulation the same API shape:

  * a 63 x 39 canvas; each REGION (thumb/index/middle/ring/pinky/palm) is a
    PiezoTaxelArray placed at a documented slot in the canvas. Cells outside
    any region read 0 — the sim hand only carries films where its real
    counterpart does (the L20 has fingertip films), and the canvas is honest
    about it;
  * ADC 0..254 from the taxel divider voltage (v_out / v_ref * 254), with a
    tare (去皮) offset like the USB tare command;
  * data analysis: valid point count, ADC sum / max / mean, and the contact
    points above a threshold;
  * an IMU derived from the palm body's rigid motion: specific force
    (acceleration minus gravity, body frame) and angular velocity — exactly
    what a strapdown IMU measures, computed by finite differences.

Force in newtons stays available per region (`read()["force_N"]`) because the
simulation HAS ground truth — the ADC path just reproduces what the glove's
electronics would show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GLOVE_ROWS, GLOVE_COLS = 63, 39

# region -> (row0, col0) canvas slot of its taxel array (rows x cols follow
# the array's own config). Fingertips across the top, palm centred below.
DEFAULT_SLOTS = {
    "thumb": (2, 1),
    "index": (2, 9),
    "middle": (0, 17),
    "ring": (2, 25),
    "pinky": (4, 32),
    "palm": (34, 13),
}


@dataclass
class _Region:
    array: object                 # PiezoTaxelArray or duck-type with .cfg/.read
    row0: int
    col0: int
    tf_fn: object = None          # () -> current 4x4 sensor tf (None = static)


class GloveIMU:
    """Strapdown IMU on the palm body: feed the palm's world pose every frame.

    read() returns the SPECIFIC FORCE f_b = R^T (a_world - g_world) and the
    body angular velocity — an ideal (noise-free) 6-axis IMU.
    """

    def __init__(self, dt: float, gravity=(0.0, -9.81, 0.0)):
        self.dt = dt
        self.g = np.asarray(gravity, float)
        self._T = []

    def update(self, T_palm: np.ndarray):
        self._T.append(np.asarray(T_palm, float).copy())
        if len(self._T) > 3:
            self._T.pop(0)

    def read(self) -> dict:
        if len(self._T) < 3:
            R = self._T[-1][:3, :3] if self._T else np.eye(3)
            return dict(accel=-(R.T @ self.g), gyro=np.zeros(3))
        T0, T1, T2 = self._T
        a_w = (T2[:3, 3] - 2 * T1[:3, 3] + T0[:3, 3]) / self.dt ** 2
        R = T2[:3, :3]
        dR = T2[:3, :3] @ T1[:3, :3].T          # world-frame increment
        w_skew = (dR - dR.T) / 2.0
        w_world = np.array([w_skew[2, 1], w_skew[0, 2], w_skew[1, 0]]) / self.dt
        return dict(accel=R.T @ (a_w - self.g), gyro=R.T @ w_world)


class TactileGlove:
    """One simulated pressure glove: regions on a 63 x 39 canvas + IMU.

    Args:
        dt: engine step time.
        hand: "left" | "right" (reported in device info, like the real query).
        serial: device id; auto-assigned when omitted.
    """

    _registry: dict[str, "TactileGlove"] = {}
    _seq = 0

    def __init__(self, dt: float = 0.005, hand: str = "right",
                 serial: str | None = None):
        if serial is None:
            TactileGlove._seq += 1
            serial = f"GLV{TactileGlove._seq:06d}"
        self.serial = serial
        self.hand = hand
        self.firmware = "sim-1.0"
        self.dt = dt
        self.regions: dict[str, _Region] = {}
        self.imu = GloveIMU(dt)
        self.channel = "serial"
        self.frame_rate = 1.0 / dt
        self._tare = np.zeros((GLOVE_ROWS, GLOVE_COLS), np.float64)
        TactileGlove._registry[serial] = self

    # ------------------------------------------------------------ connection
    @classmethod
    def open(cls, serial_or_index=0) -> "TactileGlove":
        """USB-style open: look up a registered sim glove."""
        if isinstance(serial_or_index, str):
            return cls._registry[serial_or_index]
        return list(cls._registry.values())[int(serial_or_index)]

    open_ble = open                              # same lookup, BLE spelling

    def device_info(self) -> dict:
        return dict(serial=self.serial, hand=self.hand,
                    firmware=self.firmware, regions=sorted(self.regions))

    def set_channel(self, channel: str):
        assert channel in ("serial", "ble")
        self.channel = channel

    def set_frame_rate(self, hz: float):
        self.frame_rate = float(hz)

    def close(self):
        TactileGlove._registry.pop(self.serial, None)
        self.regions = {}

    # -------------------------------------------------------------- regions
    def add_region(self, name: str, taxel_array, row0: int | None = None,
                   col0: int | None = None, tf_fn=None):
        """Place a taxel array at a canvas slot (defaults from DEFAULT_SLOTS)."""
        if row0 is None or col0 is None:
            row0, col0 = DEFAULT_SLOTS[name]
        c = taxel_array.cfg
        if row0 + c.rows > GLOVE_ROWS or col0 + c.cols > GLOVE_COLS:
            raise ValueError(f"region {name} ({c.rows}x{c.cols}@{row0},{col0}) "
                             f"exceeds the {GLOVE_ROWS}x{GLOVE_COLS} canvas")
        self.regions[name] = _Region(taxel_array, row0, col0, tf_fn)

    def region_slice(self, name: str) -> tuple[slice, slice]:
        r = self.regions[name]
        c = r.array.cfg
        return (slice(r.row0, r.row0 + c.rows), slice(r.col0, r.col0 + c.cols))

    # ----------------------------------------------------------------- data
    def tare(self):
        """Capture the current canvas as the zero offset (their USB 去皮)."""
        self._tare = self._raw_canvas()[0]

    def _raw_canvas(self):
        adc = np.zeros((GLOVE_ROWS, GLOVE_COLS), np.float64)
        force = {}
        for name, r in self.regions.items():
            tf = r.tf_fn() if r.tf_fn else None
            out = r.array.read(sensor_tf=tf)
            c = r.array.cfg
            counts = out["v_out"] / c.v_ref * 254.0
            adc[self.region_slice(name)] = counts
            force[name] = float(out["force_N"].sum())
        return adc, force

    def read(self, threshold: int = 5) -> dict:
        """One glove frame.

        Returns:
            matrix        (63, 39) uint8 ADC counts 0..254, tared
            force_N       {region: newtons} (sim ground truth, not on the ADC)
            valid_count   cells with adc > threshold
            adc_sum/adc_max/adc_mean   over valid cells
            contacts      (K, 3) int array of [row, col, adc] above threshold
            imu           {accel (3,), gyro (3,)} palm-frame
        """
        raw, force = self._raw_canvas()
        m = np.clip(raw - self._tare, 0.0, 254.0)
        mi = np.round(m).astype(np.uint8)
        mask = mi > threshold
        contacts = np.argwhere(mask)
        contacts = np.concatenate([contacts, mi[mask][:, None]], axis=1) \
            if len(contacts) else np.zeros((0, 3), np.int64)
        return dict(
            matrix=mi, force_N=force,
            valid_count=int(mask.sum()),
            adc_sum=int(mi[mask].sum()) if mask.any() else 0,
            adc_max=int(mi.max()),
            adc_mean=float(mi[mask].mean()) if mask.any() else 0.0,
            contacts=contacts,
            imu=self.imu.read(),
        )

    def region(self, name: str, frame: dict) -> np.ndarray:
        """The (rows, cols) sub-matrix of one region from a read() frame."""
        return frame["matrix"][self.region_slice(name)]
