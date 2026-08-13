"""Tactile sensor layer for StiffGIPC: VBTS + piezoresistive taxel arrays.

Ships as `stiff_physics.sensors` inside the engine repo (canonical release
location, synced from the dev workspace by tools/sync_sensors.sh); the same
package also imports as `stiff_tactile` from the dev workspace. STRICTLY
layered: this package imports stiff_physics; the engine core never imports it.

Ported from Taccel (MIT, https://github.com/Taccel-Simulator/Taccel): the sensor
asset / marker / depth / RGB layer, with warp_ipc replaced by stiff_physics.

    from stiff_tactile import SensorAsset, GelBody, observe

    asset = SensorAsset.from_taccel_fabrication(".../single_sensor/tac_fabr_1e-07.json")
    gel   = GelBody(asset, engine, world_tf).add_fixed()
    engine.finalize()
    ...
    engine.step()
    obs = gel.observe()          # depth / normal / marker flow
    f_N = gel.contact_force()    # Newtons
"""

from .observation import (
    depth_from_coat_surface,
    depth_to_normal,
    marker_flow,
    markers_to_pixels,
    observe,
)
from .sensor_asset import CameraConfig, MaterialConfig, SensorAsset

__all__ = [
    "SensorAsset",
    "CameraConfig",
    "MaterialConfig",
    "observe",
    "depth_from_coat_surface",
    "depth_to_normal",
    "marker_flow",
    "markers_to_pixels",
    "GelBody",
    "TactileRenderer",
]


def __getattr__(name):  # lazy: GelBody needs stiff_physics, TactileRenderer needs torch
    if name == "GelBody":
        from .backend_stiffgipc import GelBody

        return GelBody
    if name == "TactileRenderer":
        from .render import TactileRenderer

        return TactileRenderer
    raise AttributeError(f"module 'stiff_tactile' has no attribute {name!r}")
