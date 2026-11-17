import numpy as np
import warp as wp
from warp._src.types import matrix, vector

class Material:
    water    = 0
    elastic  = 1 # no return mapping
    vonmises = 2 # von mises plasticity
    sand     = 3 # drucker-prager plasticity
    nacc     = 4 # NACC https://dl.acm.org/doi/10.1145/3306346.3322949
    snow13   = 5 # https://dl.acm.org/doi/10.1145/2461912.2461948
    foam     = 6

class Constitutive:
    stvk       = 0
    neohookean = 1
    pressure   = 2

USE_FLOAT64 = True

scalar = np.float64 if USE_FLOAT64 else np.float32
real = wp.float64 if USE_FLOAT64 else wp.float32
vec2 = wp.vec2f if not USE_FLOAT64 else wp.vec2d
vec3 = wp.vec3f if not USE_FLOAT64 else wp.vec3d
vec4 = wp.vec4f if not USE_FLOAT64 else wp.vec4d
mat22 = wp.mat22f if not USE_FLOAT64 else wp.mat22d
mat33 = wp.mat33f if not USE_FLOAT64 else wp.mat33d
mat44 = wp.mat44f if not USE_FLOAT64 else wp.mat44d

class vec6f(vector(length=6, dtype=wp.float32)):
    pass
class vec6d(vector(length=6, dtype=wp.float64)):
    pass
class vec8f(vector(length=8, dtype=wp.float32)):
    pass
class vec8d(vector(length=8, dtype=wp.float64)):
    pass
class vec9f(vector(length=9, dtype=wp.float32)):
    pass
class vec9d(vector(length=9, dtype=wp.float64)):
    pass
class mat99f(matrix(shape=(9, 9), dtype=wp.float32)):
    pass
class mat99d(matrix(shape=(9, 9), dtype=wp.float64)):
    pass

mat99 = mat99f if not USE_FLOAT64 else mat99d
vec8 = vec8f if not USE_FLOAT64 else vec8d
vec9 = vec9f if not USE_FLOAT64 else vec9d
vec6 = vec6f if not USE_FLOAT64 else vec6d
from typing import Union, Tuple, Optional

GridSize = Union[wp.vec2i, wp.vec3i]

@wp.struct
class MaterialParams:
    cons_type: int = Constitutive.stvk
    mate_type: int = Material.elastic
    mu_lam_kappa: vec3 = vec3(real(0.0))
    friction_angle: real = real(0.0)
    yield_stress: real = real(0.0)

@wp.kernel
def set_material_params(
    psi_params: wp.array(dtype=MaterialParams),
    material_k: int,
    material_params: MaterialParams,
):
    idx = wp.tid()
    if idx == 0: psi_params[material_k] = material_params

@wp.struct
class QuadratureScratch2D:
    # for SVD, Fe = U diag(Sigma) V^T
    U: mat22
    V: mat22
    Sigma: vec2
    # Ft: mat22
    # for stvk hencky dPdF
    psi00: real
    psi01: real
    psi11: real
    m01_plus_p01_2: real
    m01_sub__p01_2: real

@wp.struct
class QuadratureScratch3D:
    # for SVD, Fe = U diag(Sigma) V^T
    U: mat33
    V: mat33
    Sigma: vec3
    # Ft: mat33
    # for stvk hencky dPdF matrix-free contraction
    Aij: vec6
    B012: vec6
    tau: mat33

@wp.struct
class QuadratureScratch3D_PBNG:
    U: mat33
    V: mat33
    Sigma: vec3
    tau: mat33
    # K: mat99 # PBNG tangent