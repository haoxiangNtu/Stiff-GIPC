"""Explicit hand inertia and actuator stiffness, independent of collision meshes.

Authored asset inertia is not hardware calibration. Drive stiffness can retain
an experimental baseline, but is expressed in Nm/rad rather than mass ratios.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def load_hand_properties(path, joint_names):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get('schema') != 1 or data.get('units') != {
            'mass': 'kg', 'length': 'm', 'inertia': 'kg*m^2', 'stiffness': 'Nm/rad'}:
        raise ValueError('hand properties require schema 1 and explicit SI units')
    links = data['links']; drives = data['drives']
    if set(drives) != set(joint_names):
        raise ValueError('hand properties must specify every finger joint exactly once')
    for name, p in links.items():
        m = float(p['mass']); c = np.asarray(p['com'], float); I = np.asarray(p['inertia'], float)
        if not (np.isfinite(m) and m > 0 and c.shape == (3,) and I.shape == (3, 3)
                and np.isfinite(c).all() and np.isfinite(I).all()):
            raise ValueError(f'invalid hand inertia: {name}')
        if not np.allclose(I, I.T, atol=1e-14, rtol=1e-10):
            raise ValueError(f'asymmetric hand inertia: {name}')
        # The centered second moment must be positive definite for ABD.
        if np.linalg.eigvalsh(.5*np.trace(I)*np.eye(3)-I).min() <= 0:
            raise ValueError(f'nonphysical or degenerate hand inertia: {name}')
    for name, d in drives.items():
        k = float(d['stiffness_nm_per_rad'])
        if not (np.isfinite(k) and k > 0 and d['parent'] in links and d['child'] in links
                and d['parent'] != d['child']):
            raise ValueError(f'invalid hand drive: {name}')
    return data, hashlib.sha256(raw).hexdigest()


def load_frame_inertia(prop, T):
    T = np.asarray(T, float)
    if T.shape != (4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):
        raise ValueError('hand inertial transform must be a rigid SI transform')
    R = T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or np.linalg.det(R) <= 0:
        raise ValueError('hand inertial transform must be a rigid SI transform')
    c = R@np.asarray(prop['com'])+T[:3,3]
    I = R@np.asarray(prop['inertia'])@R.T
    return float(prop['mass']), c, I


def apply_hand_inertia(engine, bodies, link_T0, data):
    expected = {n for n in bodies if n.startswith(('left_', 'right_'))}
    if set(data['links']) != expected:
        raise ValueError('hand inertia link names do not match the loaded hand')
    for name, p in data['links'].items():
        mass, com, inertia = load_frame_inertia(p, link_T0[name])
        body_id = bodies[name].body_id
        # Both calls are necessary in the current engine: the scalar body-mass
        # buffer used by drives is assembled separately from the dyadic inertia.
        engine.set_abd_body_mass(body_id, mass)
        engine.set_abd_body_inertia(body_id, mass, com, inertia)


def drive_strength_ratio(data, name, dt, joint_ratio):
    if not (np.isfinite(dt) and dt > 0 and np.isfinite(joint_ratio) and joint_ratio > 0):
        raise ValueError('dt and joint_ratio must be finite and positive')
    d = data['drives'][name]
    mass = data['links'][d['parent']]['mass'] + data['links'][d['child']]['mass']
    # Engine energy is an incremental potential with the driving term outside
    # the dt^2 multiplier: its small-angle physical stiffness is sr*r*m/dt^2.
    return d['stiffness_nm_per_rad'] * dt**2 / (joint_ratio * mass)
