from ..types import *
import math
from typing import Any, Union, Optional, Callable, Tuple
import warp as wp
import warp.sparse as wps
from warp.optim.linear import aslinearoperator, LinearOperator
from warp._src.types import type_length, type_scalar_type
from .tiled_dot import VecTiledDot

_0 = wp.constant(real(0.0))

def _as_scalar_array(x: wp.array):
    scalar_type = type_scalar_type(x.dtype)
    if scalar_type == x.dtype:
        return x

    dlen = type_length(x.dtype)
    arr = wp.array(
        ptr=x.ptr,
        shape=(*x.shape[:-1], x.shape[-1] * dlen),
        strides=(*x.strides[:-1], x.strides[-1] // dlen),
        dtype=scalar_type,
        device=x.device,
        grad=None if x.grad is None else _as_scalar_array(x.grad),
    )
    arr._ref = x
    return arr


def _repeat_first(arr: wp.array):
    # returns a view of the first element repeated arr.shape[0] times
    view = wp.array(
        ptr=arr.ptr,
        shape=arr.shape,
        dtype=arr.dtype,
        strides=(0, *arr.strides[1:]),
        device=arr.device,
    )
    view._ref = arr
    return view


def _get_dtype_epsilon(dtype):
    if dtype == wp.float64:
        return 1.0e-16
    elif dtype == wp.float16:
        return 1.0e-4

    return 1.0e-8


def _get_tolerances(dtype, tol, atol):
    eps_tol = _get_dtype_epsilon(dtype)
    default_tol = eps_tol ** (3 / 4)
    min_tol = eps_tol ** (9 / 4)

    if tol is None and atol is None:
        tol = atol = default_tol
    elif tol is None:
        tol = atol
    elif atol is None:
        atol = tol

    atol = max(atol, min_tol)
    return tol, atol


@wp.kernel
def _initialize_tolerance(
    rtol: Any,
    atol: Any,
    r_norm_sq: wp.array(dtype=Any),
    atol_sq: wp.array(dtype=Any),
):
    atol = wp.max(rtol * wp.sqrt(r_norm_sq[0]), atol)
    atol_sq[0] = atol * atol


def _initialize_absolute_tolerance(
    b: wp.array,
    tol: float,
    atol: float,
    tiled_dot: VecTiledDot,
    atol_sq: wp.array,
):
    scalar_type = atol_sq.dtype

    # Compute b norm to define absolute tolerance
    tiled_dot.compute(b, b)
    b_norm_sq = tiled_dot.col(0)

    rtol, atol = _get_tolerances(scalar_type, tol, atol)
    wp.launch(
        kernel=_initialize_tolerance,
        dim=1,
        device=b.device,
        inputs=[scalar_type(rtol), scalar_type(atol), b_norm_sq, atol_sq],
    )


@wp.kernel
def _cg_kernel_1(
    tol: wp.array(dtype=Any),
    resid: wp.array(dtype=Any),
    zTr_old: wp.array(dtype=Any),
    p_Ap: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
):
    i = wp.tid()

    alpha = wp.where(resid[0] > tol[0], zTr_old[0] / p_Ap[0], zTr_old.dtype(0.0))

    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * Ap[i]

@wp.kernel
def _cg_kernel_2(
    tol: wp.array(dtype=Any),
    resid_new: wp.array(dtype=Any),
    zTr_old: wp.array(dtype=Any),
    zTr_new: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
):
    i = wp.tid()
    cond = resid_new[0] > tol[0]
    beta = wp.where(cond, zTr_new[0] / zTr_old[0], zTr_old.dtype(0.0))

    p[i] = z[i] + beta * p[i]

@wp.kernel
def _dot_reduce(p: wp.array(dtype=Any), z: wp.array(dtype=Any), out: wp.array(dtype=real)):
    i = wp.tid()
    wp.atomic_add(out, 0, wp.dot(p[i], z[i]))

def dot_array(a, b) -> scalar:
    tmp_dot = wp.zeros(1, dtype=real, device=a.device)
    wp.launch(_dot_reduce, dim=a.shape[0], inputs=(a, b, tmp_dot), device=a.device)
    return scalar(tmp_dot.numpy().reshape(-1)[0])

def tile_dot_array(tiled_dot: VecTiledDot, a: wp.array, b: wp.array) -> scalar:
    tiled_dot.compute(a, b, col_offset=0)
    return scalar(tiled_dot.col(0).numpy().reshape(-1)[0])

_Matrix = Union[wp.array, wps.BsrMatrix, LinearOperator]
def cg( # x must be zero-initialized
    A: _Matrix,
    b: wp.array,
    x: wp.array,
    tol: Optional[float] = None,
    atol: Optional[float] = None,
    maxiter: Optional[int] = 0,
    check_every: Optional[int] = 10,
    projection: Callable[[wp.array], None] = None,
    M_inv: Optional[_Matrix] = None, # inverse of preconditioner
    use_cuda_graph=True,
) -> Union[Tuple[int, float, float], Tuple[wp.array, wp.array, wp.array]]:
    A = aslinearoperator(A)
    M_inv = aslinearoperator(M_inv)

    if maxiter == 0:
        maxiter = A.shape[0]
    device = A.device
    scalar_type = A.scalar_type

    # Temp storage
    r_and_z = wp.empty((2, b.shape[0]), dtype=b.dtype, device=device)
    p_and_Ap = wp.empty_like(r_and_z)
    residuals = wp.empty(2, dtype=scalar_type, device=device)

    tiled_dot = VecTiledDot(max_length=b.shape[0], device=device, scalar_type=scalar_type, vec_dtype=b.dtype, max_column_count=2, tile_size=512)

    # (r, r) -- so we can compute r.z and r.r at once
    r_and_r = _repeat_first(r_and_z)
    if M_inv is None:
        # without preconditioner r == z
        r_and_z = r_and_r
        zTr_new = tiled_dot.col(0)
    else:
        zTr_new = tiled_dot.col(1)

    # all references to the same memory
    r, z = r_and_z[0], r_and_z[1]
    r_norm_sq = tiled_dot.col(0)
    p, Ap = p_and_Ap[0], p_and_Ap[1]
    zTr_old, atol_sq = residuals[0:1], residuals[1:2]
    
    # Not strictly necessary, but makes it more robust
    Ap.zero_()
    z.zero_()

    _initialize_absolute_tolerance(b, tol, atol, tiled_dot, atol_sq)

    # KEY (0): Initialize residual r = b - A x (r = 0 if x = 0), project
    r.assign(b) # Initialize residual: r = b - A x (r = 0 if x = 0)
    if projection is not None: projection(r)

    def update_zproj_rnorm_zTr():
        if M_inv is None:
            # z = r, no need to update
            tiled_dot.compute(r, r)
        else:
            M_inv.matvec(r, z, z, alpha=1.0, beta=0.0) # z = M^-1 r
            if projection is not None: projection(z)
            tiled_dot.compute(r_and_r, r_and_z)

    update_zproj_rnorm_zTr()

    p.assign(z) # update p

    _atol = math.sqrt(atol_sq.numpy().reshape(-1)[0])
    _atol_sq = _atol**2

    cur_iter = 0
    _err_sq = r_norm_sq.numpy().reshape(-1)[0]
    _err = math.sqrt(_err_sq)
    if _err_sq <= _atol_sq:
        return cur_iter, _err, _atol

    graph = None

    def do_iteration():
        zTr_old.assign(zTr_new)

        # KEY (3): Ap = A * p, project
        A.matvec(p, Ap, Ap, alpha=1, beta=0)
        if projection is not None: projection(Ap)
        tiled_dot.compute(p, Ap, col_offset=1)
        p_Ap = tiled_dot.col(1)

        wp.launch( # update x, r
            kernel=_cg_kernel_1,
            dim=x.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, zTr_old, p_Ap, x, r, p, Ap],
        )

        update_zproj_rnorm_zTr() # update z, r_norm_sq, zTr_new

        wp.launch( # update p
            kernel=_cg_kernel_2,
            dim=z.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, zTr_old, zTr_new, z, p],
        )

    while True:
        # Do not do graph capture at first iteration -- modules may not be loaded yet
        if device.is_cuda and use_cuda_graph and cur_iter > 0:
            if graph is None:
                with wp.ScopedCapture(force_module_load=False) as capture:
                    do_iteration()
                graph = capture.graph
            wp.capture_launch(graph)
        else:
            do_iteration()

        cur_iter += 1

        if cur_iter >= maxiter:
            break

        if cur_iter % check_every == 0:
            _err_sq = r_norm_sq.numpy().reshape(-1)[0]

            if _err_sq <= _atol_sq:
                break

    _err_sq = r_norm_sq.numpy().reshape(-1)[0]
    _err = math.sqrt(_err_sq)

    return cur_iter, _err, _atol

def matrix_free_cg(  # x must be zero-initialized
    A_matvec: Callable[[wp.array, wp.array], None],
    b: wp.array,
    x: wp.array,
    tol: Optional[float] = None,
    atol: Optional[float] = None,
    maxiter: Optional[int] = 0,
    check_every: Optional[int] = 10,
    M_inv_matvec: Callable[[wp.array, wp.array], None] = None,  # inverse of preconditioner
    use_cuda_graph=True,
) -> Union[Tuple[int, float, float], Tuple[wp.array, wp.array, wp.array]]:
    if maxiter == 0:
        maxiter = b.shape[0]
    device = b.device
    scalar_type = real

    # Temp storage
    r_and_z = wp.empty((2, b.shape[0]), dtype=b.dtype, device=device)
    p_and_Ap = wp.empty_like(r_and_z)
    residuals = wp.empty(2, dtype=scalar_type, device=device)

    tiled_dot = VecTiledDot(
        max_length=b.shape[0],
        device=device,
        scalar_type=scalar_type,
        vec_dtype=b.dtype,
        max_column_count=2,
        tile_size=512,
    )

    # (r, r) -- so we can compute r.z and r.r at once
    r_and_r = _repeat_first(r_and_z)
    if M_inv_matvec is None:
        # without preconditioner r == z
        r_and_z = r_and_r
        zTr_new = tiled_dot.col(0)
    else:
        zTr_new = tiled_dot.col(1)

    # all references to the same memory
    r, z = r_and_z[0], r_and_z[1]
    r_norm_sq = tiled_dot.col(0)
    p, Ap = p_and_Ap[0], p_and_Ap[1]
    zTr_old, atol_sq = residuals[0:1], residuals[1:2]

    # Not strictly necessary, but makes it more robust
    Ap.zero_()
    z.zero_()

    _initialize_absolute_tolerance(b, tol, atol, tiled_dot, atol_sq)

    # KEY (0): Initialize residual r = b - A x (r = b if x = 0), project
    # ⚠️ WARNING: ensure b is already projected
    r.assign(b)  # Initialize residual: r = b - A x (r = b if x = 0)

    def update_zproj_rnorm_zTr():
        if M_inv_matvec is None:
            # z = r, no need to update
            tiled_dot.compute(r, r)
        else:
            M_inv_matvec(r, z)  # z = M^-1 r
            tiled_dot.compute(r_and_r, r_and_z)

    update_zproj_rnorm_zTr()

    p.assign(z)  # update p

    _atol = math.sqrt(atol_sq.numpy().reshape(-1)[0])
    _atol_sq = _atol**2

    cur_iter = 0
    _err_sq = r_norm_sq.numpy().reshape(-1)[0]
    _err = math.sqrt(_err_sq)

    graph = None

    def do_iteration():
        zTr_old.assign(zTr_new)

        # KEY (3): Ap = A * p, project
        # ⚠️ WARNING: ensure Ap is already projected
        A_matvec(p, Ap)

        tiled_dot.compute(p, Ap, col_offset=1)
        p_Ap = tiled_dot.col(1)

        wp.launch(  # update x, r
            kernel=_cg_kernel_1,
            dim=x.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, zTr_old, p_Ap, x, r, p, Ap],
        )

        update_zproj_rnorm_zTr()  # update z, r_norm_sq, zTr_new

        wp.launch(  # update p
            kernel=_cg_kernel_2,
            dim=z.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, zTr_old, zTr_new, z, p],
        )

    while True:
        # Do not do graph capture at first iteration -- modules may not be loaded yet
        if device.is_cuda and use_cuda_graph and cur_iter > 0:
            if graph is None:
                with wp.ScopedCapture(force_module_load=False) as capture:
                    do_iteration()
                graph = capture.graph
            wp.capture_launch(graph)
        else:
            do_iteration()

        cur_iter += 1

        if cur_iter >= maxiter:
            break

        if cur_iter % check_every == 0:
            _err_sq = r_norm_sq.numpy().reshape(-1)[0]
            if _err_sq <= _atol_sq:
                break

    _err_sq = r_norm_sq.numpy().reshape(-1)[0]
    _err = math.sqrt(_err_sq)

    return cur_iter, _err, _atol