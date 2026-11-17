import warp as wp
from ..types import real

_0 = wp.constant(real(0.0))
_1 = wp.constant(real(1.0))
_n1 = wp.constant(real(-1.0))
_2 = wp.constant(real(2.0))
_3 = wp.constant(real(3.0))
_4 = wp.constant(real(4.0))
_05 = wp.constant(real(0.5))
_13 = wp.constant(real(1.0/3.0))
_14 = wp.constant(real(0.25))

s_min = wp.constant(real(1e-6))
dPdF_use_project = wp.constant(wp.int32(1))
dPdF_eps_small = wp.constant(real(1e-6))
dPdF_eps_pd = wp.constant(real(1e-10))