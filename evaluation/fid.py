#!/usr/bin/env python3
# This code is a modification of https://github.com/bioinf-jku/TTUR/blob/master/fid.py
""" Calculates the Frechet Inception Distance (FID) to evalulate GANs.

The FID metric calculates the distance between two distributions of x.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
x that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import os
import tensorflow as tf
from scipy import linalg
import warnings
from models.inception.model import load_inception_inference
from utils.utils import load_inception_data, prep_incep_img


# Flags and constants
# ------------------------------------------------------------------------------
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/inception/flowers/model.ckpt',
                           """Path where to read model checkpoints.""")
tf.app.flags.DEFINE_string('real_img_folder', './test1', """Path where to load the real x """)
tf.app.flags.DEFINE_string('gen_img_folder', './test2', """Path where to load the generated x """)
tf.app.flags.DEFINE_integer('num_classes', 20, """Number of classes """)  # 20 for flowers
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('gpu', 1, "The ID of GPU to use")

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


class InvalidFIDException(Exception):
    pass

# -------------------------------------------------------------------------------


def get_activations(images, sess, batch_size, act_op, verbose=False):
    """Calculates the activations of the pool_3 layer for all x.

    Params:
    -- x      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the x numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num x, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)

    d0 = len(images)
    if batch_size > d0:
        msg = "batch size is bigger than the data size"
        raise RuntimeError(msg)

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        end = start + batch_size
        batch = []
        for j in range(start, end):
            batch.append(prep_incep_img(images[j]))

        pred = sess.run(act_op, {'inputs:0': batch})
        pred_arr[start:end] = pred
    if verbose:
        print(" done")
    return pred_arr


# -------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# -------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, batch_size, act_op, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- x      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the x numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, act_op, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def save_activation_statistics(mu, sigma, path):
    if os.path.exists(path):
        raise RuntimeError('Path {} already exists. Statistics not saved'.format(path))

    os.makedirs(os.path.dirname(path))
    np.savez(path, mu=mu, sigma=sigma)


def compute_and_save_activation_statistics(img_path, sess, bs, act_op, save_path, verbose=False):
    x = load_inception_data(img_path)
    mu, sigma = calculate_activation_statistics(x, sess, bs, act_op, verbose=verbose)

    save_activation_statistics(mu, sigma, save_path)

# -------------------------------------------------------------------------------


def _handle_path(path, sess, act_op):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        x = load_inception_data(path)
        m, s = calculate_activation_statistics(x, sess, FLAGS.batch_size, act_op, verbose=True)
    return m, s


def calculate_fid_given_paths():
    """ Calculates the FID of two paths. """
    real_img_path = FLAGS.real_img_folder
    gen_img_path = FLAGS.gen_img_folder
    if not os.path.exists(real_img_path):
        raise RuntimeError("Invalid path: %s" % real_img_path)
    if not os.path.exists(gen_img_path):
        raise RuntimeError("Invalid path %s" % gen_img_path)

    with tf.Session() as sess:
        _, layers = load_inception_inference(sess, FLAGS.num_classes, FLAGS.batch_size, FLAGS.checkpoint_dir)
        pool3 = layers['PreLogits']
        act_op = tf.reshape(pool3, shape=[FLAGS.batch_size, -1])

        m1, s1 = _handle_path(real_img_path, sess, act_op)
        m2, s2 = _handle_path(gen_img_path, sess, act_op)
        fid_dist = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_dist


if __name__ == "__main__":
    fid_value = calculate_fid_given_paths()
    print("FID: ", fid_value)
