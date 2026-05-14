"""Prelude: bypass daily-dev's scikit-build editable finder so that
`import stiff_physics` resolves to THIS hybrid-mesh worktree instead.

Usage in a demo: add `import _use_dailyv2_engine  # noqa` BEFORE any
stiff_physics import.  Despite the name (kept for compatibility with
m5_drive_joint2_test.py), this resolves to the worktree containing
this file (= hybrid-mesh worktree).

The PYTHONPATH must include this worktree's root and build_312 for the
prelude to find them.
"""
import os, sys

# Drop the scikit-build redirecting finder injected by daily-dev's editable
# install (registered via _stiff_physics_editable.pth in the venv).
sys.meta_path[:] = [f for f in sys.meta_path
                    if type(f).__name__ != 'ScikitBuildRedirectingFinder']

# Drop daily-dev's own path that the .pth file added at site init.
sys.path[:] = [p for p in sys.path if p != '/home/ps/Downloads/Stiff-GIPC']

# Make sure THIS worktree is at the very front of sys.path.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
BUILD = os.path.join(HERE, 'build_312')
if BUILD not in sys.path:
    sys.path.insert(0, BUILD)

# Drop any already-imported stiff_physics (from prior daily-dev resolution
# during pre-import warmup); next import will pick up THIS worktree.
for mod in [k for k in sys.modules if k.startswith('stiff_physics') or k == 'pystiffgipc']:
    del sys.modules[mod]
