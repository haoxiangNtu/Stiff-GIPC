"""Load Taccel's ``pad_maxv=*.pkl`` fabrication metadata without PyVista.

Taccel builds those masks from ``pv.UnstructuredGrid.points``, so the pickles
contain ``pyvista.core.pyvista_ndarray.pyvista_ndarray`` instances and a plain
``pickle.load`` raises ModuleNotFoundError on a PyVista-free environment.
``pyvista_ndarray`` is just an ``np.ndarray`` subclass, so a stub is enough.
"""

from __future__ import annotations

import pickle
import sys
import types

import numpy as np


class _PyVistaNdarrayStub(np.ndarray):
    def __array_finalize__(self, obj):  # noqa: D105
        pass


def install_pyvista_stub() -> None:
    """Register a minimal fake ``pyvista`` in sys.modules (no-op if real one exists)."""
    try:
        import pyvista  # noqa: F401

        return
    except ImportError:
        pass

    root = sys.modules.setdefault("pyvista", types.ModuleType("pyvista"))
    core = sys.modules.setdefault("pyvista.core", types.ModuleType("pyvista.core"))
    mod = types.ModuleType("pyvista.core.pyvista_ndarray")
    mod.pyvista_ndarray = _PyVistaNdarrayStub
    sys.modules["pyvista.core.pyvista_ndarray"] = mod
    core.pyvista_ndarray = mod
    root.core = core


def load_fabrication_metadata(path: str) -> dict:
    """Load one gel-pad metadata pickle, normalised to plain numpy arrays.

    Keys (as written by Taccel ``examples/fabricate_sensor.py``):
        stick_mask          (N_body,) bool  -- vertices rigidly attached to the carrier
        coat_mask           (N_body,) bool  -- vertices on the reflective sensing surface
        coat_mask_surf      (N_surf,) bool  -- same, in PyVista extract_surface() order
        marker_vert_idx     (K, 3) int      -- body vertex ids of each marker's host triangle
        marker_vert_idx_surf(K, 3) int      -- same, surface order
        marker_bc_coords    (K, 3) float    -- barycentric weights inside that triangle
    """
    install_pyvista_stub()
    with open(path, "rb") as f:
        raw = pickle.load(f)

    out = {}
    for k, v in raw.items():
        if isinstance(v, np.ndarray):
            out[k] = np.asarray(v).view(np.ndarray).copy()
        elif isinstance(v, (list, tuple)):
            out[k] = np.asarray(v)
        else:
            out[k] = v
    return out
