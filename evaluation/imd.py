"""A library for the Inception match score to evaluate conditional generative models for images"""

import tensorflow as tf
from scipy import spatial
import numpy as np

from utils.utils import load_inception_data, prep_incep_img
from models.inception.model import load_inception_inference

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/inception/flowers/model.ckpt',
                           """Path where to read model checkpoints.""")
tf.app.flags.DEFINE_string('real_img_folder', './evaluation/data/gen/', """Path where to load the real x """)
tf.app.flags.DEFINE_string('gen_img_folder', './evaluation/data/real/', """Path where to load the real x """)
tf.app.flags.DEFINE_integer('num_classes', 20, """Number of classes """)  # 20 for flowers
tf.app.flags.DEFINE_integer('splits', 10, """Number of splits """)
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('gpu', 1, "The ID of GPU to use")


def get_cosine_dist(real_img_act, gen_img_act):
    """
    Computes the Inception Match Distance
    :param gen_img_act: A batch of mixed['pre_logits'] activations for the generated x
    :param real_img_act: A batch of mixed['pre_logits'] activations for the real x
    :return: The inception match score
    """

    print(gen_img_act.shape)
    print(real_img_act.shape)

    cos_dist = []
    for idx in range(FLAGS.batch_size):
        dist = spatial.distance.cosine(gen_img_act[idx], real_img_act[idx])
        cos_dist.append(dist)

    return cos_dist


def compute_imd(sess, real_img, gen_img, act_op, verbose=False):
    assert(len(real_img) == len(gen_img))
    assert (type(real_img[0]) == np.ndarray)
    assert (type(gen_img[0]) == np.ndarray)
    assert (len(real_img[0].shape) == 3)
    assert (len(gen_img[0].shape) == 3)
    assert (np.max(real_img[0]) > 10)
    assert (np.min(gen_img[0]) >= 0.0)

    batch_size = FLAGS.batch_size
    d0 = len(real_img)
    if batch_size > d0:
        msg = "batch size is bigger than the data size"
        raise RuntimeError(msg)

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    distances = np.empty(n_used_imgs)
    for i in range(n_batches):
        if verbose:
            print("\rComputing batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        end = start + batch_size
        r_img_batch = []
        g_img_batch = []
        for j in range(start, end):
            r_img_batch.append(prep_incep_img(real_img[j]))
            g_img_batch.append(prep_incep_img(gen_img[j]))

        pred_real = sess.run(act_op, {'inputs:0': r_img_batch})
        pred_gen = sess.run(act_op, {'inputs:0': g_img_batch})
        distances[start:end] = get_cosine_dist(pred_real, pred_gen)

    if verbose:
        print(" done")

    return print('Mean {}, Std: {}'.format(np.mean(distances), np.std(distances)))


def main(unused_argv=None):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % FLAGS.gpu):
                _, layers = load_inception_inference(sess, FLAGS.num_classes, FLAGS.batch_size, FLAGS.checkpoint_dir)

                pool3 = layers['PreLogits']
                act_op = tf.reshape(pool3, shape=[FLAGS.batch_size, -1])

                real_images = load_inception_data(FLAGS.real_img_folder, alphabetic=True)
                gen_images = load_inception_data(FLAGS.gen_img_folder, alphabetic=True)
                compute_imd(sess, real_images, gen_images, act_op)


if __name__ == '__main__':
    tf.app.run()
