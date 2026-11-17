import numpy as np
try:
    import numba as nb
except ImportError:  # pragma: no cover
    class _NB:
        @staticmethod
        def njit(f=None, **k):
            return f if f is not None else (lambda g: g)
    nb = _NB()
import math

@nb.njit
def poisson_disk_2d(radius, size_x, size_y, k=30, max_points=1_000_000):
    cell_size = radius / math.sqrt(2.0)
    gx = int(size_x / cell_size) + 1
    gy = int(size_y / cell_size) + 1
    grid = -np.ones((gx, gy), dtype=np.int64)

    samples = np.empty((max_points, 2), dtype=np.float64)
    active  = np.empty(max_points, dtype=np.int64)

    # neighbor offsets 5x5 = 25
    neigh = np.empty((25, 2), dtype=np.int64)
    t = 0
    for j in range(-2, 3):
        for i in range(-2, 3):
            neigh[t, 0] = i
            neigh[t, 1] = j
            t += 1

    rr = radius * radius

    # init
    n_samples = 0
    n_active = 0

    x0 = np.random.random() * size_x
    y0 = np.random.random() * size_y
    samples[0, 0] = x0
    samples[0, 1] = y0
    n_samples = 1
    active[0] = 0
    n_active = 1

    cx = int(x0 / cell_size)
    cy = int(y0 / cell_size)
    grid[cx, cy] = 0

    while n_active > 0 and n_samples < max_points:
        idx_in_active = int(np.random.random() * n_active)
        base_idx = active[idx_in_active]
        bx = samples[base_idx, 0]
        by = samples[base_idx, 1]

        found = False
        for _ in range(k):
            phi = np.random.random() * (2.0 * math.pi)
            r = radius * (1.0 + np.random.random())  # [r,2r)
            x = bx + r * math.cos(phi)
            y = by + r * math.sin(phi)

            if x < 0.0 or x >= size_x or y < 0.0 or y >= size_y:
                continue

            ccx = int(x / cell_size)
            ccy = int(y / cell_size)

            ok = True
            for tt in range(25):
                nx = ccx + neigh[tt, 0]
                ny = ccy + neigh[tt, 1]
                if nx < 0 or ny < 0 or nx >= gx or ny >= gy:
                    continue
                s_idx = grid[nx, ny]
                if s_idx != -1:
                    dx = samples[s_idx, 0] - x
                    dy = samples[s_idx, 1] - y
                    if dx*dx + dy*dy < rr:
                        ok = False
                        break

            if ok:
                samples[n_samples, 0] = x
                samples[n_samples, 1] = y
                grid[ccx, ccy] = n_samples
                active[n_active] = n_samples
                n_samples += 1
                n_active += 1
                found = True
                break

        if not found:
            # swap-pop remove
            n_active -= 1
            active[idx_in_active] = active[n_active]

    return samples[:n_samples]

@nb.njit
def poisson_disk_3d(radius, size_x, size_y, size_z, k=30, max_points=1_000_000):
    cell_size = radius/ math.sqrt(3.0)
    gx = int(size_x / cell_size) + 1
    gy = int(size_y / cell_size) + 1
    gz = int(size_z / cell_size) + 1
    grid = -np.ones((gx, gy, gz), dtype=np.int64)

    samples = np.empty((max_points, 3), dtype=np.float64)
    active  = np.empty(max_points, dtype=np.int64)

    # neighbor offsets 5x5x5 = 125
    neigh = np.empty((125, 3), dtype=np.int64)
    t = 0
    for k0 in range(-2, 3):
        for j0 in range(-2, 3):
            for i0 in range(-2, 3):
                neigh[t, 0] = i0
                neigh[t, 1] = j0
                neigh[t, 2] = k0
                t += 1

    rr = radius * radius

    # init
    x0 = np.random.random() * size_x
    y0 = np.random.random() * size_y
    z0 = np.random.random() * size_z
    samples[0, 0] = x0
    samples[0, 1] = y0
    samples[0, 2] = z0
    n_samples = 1
    active[0] = 0
    n_active = 1

    cx = int(x0 / cell_size)
    cy = int(y0 / cell_size)
    cz = int(z0 / cell_size)
    grid[cx, cy, cz] = 0

    while n_active > 0 and n_samples < max_points:
        idx_in_active = int(np.random.random() * n_active)
        base_idx = active[idx_in_active]
        bx = samples[base_idx, 0]
        by = samples[base_idx, 1]
        bz = samples[base_idx, 2]

        found = False
        for _ in range(k):
            # random direction (gaussian normalize)
            vx = np.random.normal()
            vy = np.random.normal()
            vz = np.random.normal()
            nrm = math.sqrt(vx*vx + vy*vy + vz*vz)
            if nrm == 0.0:
                continue
            vx /= nrm; vy /= nrm; vz /= nrm

            r = radius * (1.0 + np.random.random())
            x = bx + vx * r
            y = by + vy * r
            z = bz + vz * r

            if x < 0.0 or x >= size_x or y < 0.0 or y >= size_y or z < 0.0 or z >= size_z:
                continue

            ccx = int(x / cell_size)
            ccy = int(y / cell_size)
            ccz = int(z / cell_size)

            ok = True
            for tt in range(125):
                nx = ccx + neigh[tt, 0]
                ny = ccy + neigh[tt, 1]
                nz = ccz + neigh[tt, 2]
                if nx < 0 or ny < 0 or nz < 0 or nx >= gx or ny >= gy or nz >= gz:
                    continue
                s_idx = grid[nx, ny, nz]
                if s_idx != -1:
                    dx = samples[s_idx, 0] - x
                    dy = samples[s_idx, 1] - y
                    dz = samples[s_idx, 2] - z
                    if dx*dx + dy*dy + dz*dz < rr:
                        ok = False
                        break

            if ok:
                samples[n_samples, 0] = x
                samples[n_samples, 1] = y
                samples[n_samples, 2] = z
                grid[ccx, ccy, ccz] = n_samples
                active[n_active] = n_samples
                n_samples += 1
                n_active += 1
                found = True
                break

        if not found:
            n_active -= 1
            active[idx_in_active] = active[n_active]

    return samples[:n_samples]

def poisson_in_box(radius, size, offset):
    size = np.asarray(size, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    pts = poisson_disk_2d(radius/1.262, size[0], size[1]) if len(size) == 2 else poisson_disk_3d(radius/1.182, size[0], size[1], size[2])
    return pts + offset

def sample_vtk(points, tets, N):
    tet_4v = points[tets]

    p0 = tet_4v[:, 0, :]
    p1 = tet_4v[:, 1, :]
    p2 = tet_4v[:, 2, :]
    p3 = tet_4v[:, 3, :]

    a = p1 - p0
    b = p2 - p0
    c = p3 - p0

    tet_vols = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0
    total_vol = tet_vols.sum()

    vol_prob = tet_vols / total_vol

    ids = np.random.choice(len(tet_vols), size=N, replace=True, p=vol_prob)
    u = np.random.random((N, 4))
    e = -np.log(u)                 # Exp(1)
    w = e / e.sum(axis=1, keepdims=True)  # (n,4), sum to 1
    bary = w
    samples = (bary[:, 0:1] * p0[ids] +
            bary[:, 1:2] * p1[ids] +
            bary[:, 2:3] * p2[ids] +
            bary[:, 3:4] * p3[ids])
    return samples

if __name__ == "__main__":
    import trimesh
    
    dx = 8e-3
    ppc = 5
    dim = 3
    radius = dx / np.power(ppc, 1.0/dim) 
    size = np.array([0.16] * dim)
    p = poisson_in_box(radius, size, offset=np.array([0.0] * dim))
    n_cells = int(np.prod(size/dx))
    sampled_ppc = p.shape[0] / n_cells
    print('[PPC] expected ~', ppc, 'average ~', sampled_ppc)
    print('[Particles] expected ~', n_cells * ppc, 'actual ~', p.shape[0])

    trimesh.points.PointCloud(p).export('poisson_particles.ply')