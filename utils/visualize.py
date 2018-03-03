import numpy as np


def slerp(a, b, miu):
    """Spherical interpolation between a and b. miu is in [0, 1]"""
    if miu < 0 or miu > 1:
        raise ValueError('miu must be in [0, 1] but it is %d' % miu)
    if miu == 0:
        return a
    if miu == 1:
        return b

    omega = np.arccos(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))
    sin_omega = np.sin(omega)
    return np.sin((1.0 - miu) * omega) / sin_omega * a + np.sin(miu * omega) / sin_omega * b


def lerp(a, b, miu):
    """Linear interpolation between a and b. miu is in [0, 1]"""
    if miu < 0 or miu > 1:
        raise ValueError('miu must be in [0, 1] but it is %d' % miu)
    return miu * a + (1. - miu) * b


def get_interpolated_batch(a, b, batch_size=64):

    step_size = 1 / batch_size
    interp = []
    for val in np.arange(0.0, 1.0, step_size):
        interp.append(slerp(val, a, b))
    return interp
