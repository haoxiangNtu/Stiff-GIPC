"""stiff_physics.sensors - tactile sensor layer for StiffGIPC (VBTS + piezoresistive).

Canonical release location of the sensor layer, shipping with the engine wheel.
STRICTLY layered: this package imports stiff_physics; the engine core never
imports this package. The C++/CUDA core carries only sensor-agnostic
primitives (per-vertex contact forces, MAS permutation getters, content-keyed
mesh cache, device vertex pointer); everything sensor-specific -- assets,
cameras, rasteriser, n2rgb MLP, taxel transduction -- lives here in Python,
where it can iterate without an engine rebuild.

Dev workspace with the experiment scripts: /home/ps/Downloads/stiff_tactile
(sync here with tools/sync_sensors.sh).

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
