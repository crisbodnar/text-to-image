import numpy as np
import tensorflow as tf


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


def get_interpolated_batch(a, b, batch_size=64, method='slerp'):
    """Generates a batch of given size of interpolations between a, b"""

    step_size = 1 / batch_size
    interp = []
    for val in np.arange(0.0, 1.0, step_size):
        if method == 'slerp':
            interp.append(slerp(a, b, val))
        elif method == 'lerp':
            interp.append(lerp(a, b, val))
    return interp


def interp_z(sess, gen_op, cond_sample, z1, z2, z='z:0', cond='cond:0', method='slerp', bs=64):
    """Generates a set of interpolated images in the space of the noise vector z"""

    z_batch = get_interpolated_batch(z1, z2, bs, method=method)
    return sess.run(gen_op, feed_dict={z: z_batch, cond: cond_sample})

