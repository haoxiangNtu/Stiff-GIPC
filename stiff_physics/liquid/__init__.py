"""Liquids for StiffGIPC: MPM Lite (Feng et al., SIGGRAPH 2026) water on a
sparse Cartesian grid, one-way coupled to the engine's rigid/soft surfaces.
Requires `warp-lang` (NVIDIA Warp); CPU and CUDA devices both work."""
from .liquid import LiquidMPM, EngineLiquidCoupler, BC_FREE, BC_STICKY, BC_SLIP, BC_SEPARABLE  # noqa: F401
