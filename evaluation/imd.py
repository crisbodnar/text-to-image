"""A library for the Inception match score to evaluate conditional generative models for images"""

import tensorflow as tf
from scipy import spatial

from utils.utils import load_inception_data, preprocess_inception_images
from evaluation.inception_inference import inference

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/inception/flowers/model.ckpt',
                           """Path where to read model checkpoints.""")
tf.app.flags.DEFINE_string('real_image_folder', './evaluation/data/gen/', """Path where to load the real images """)
tf.app.flags.DEFINE_string('gen_image_folder', './evaluation/data/real/', """Path where to load the real images """)
tf.app.flags.DEFINE_integer('num_classes', 20, """Number of classes """) # 20 for flowers
tf.app.flags.DEFINE_integer('splits', 10, """Number of splits """)
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('gpu', 1, "The ID of GPU to use")

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

real_img_folder = FLAGS.real_image_folder
gen_img_folder = FLAGS.gen_image_folder


def get_cosine_dist(gen_img_act, real_img_act):
    """
    Computes the Inception Match Distance
    :param gen_img_act: A batch of mixed['pre_logits'] activations for the generated images
    :param real_img_act: A batch of mixed['pre_logits'] activations for the real images
    :return: The inception match score
    """

    print(gen_img_act.shape)
    print(real_img_act.shape)

    cos_dist = []
    for idx in range(FLAGS.batch_size):
        dist = spatial.distance.cosine(gen_img_act[idx], real_img_act[idx])
        cos_dist.append(dist)

    return cos_dist


def compute_imd(sess, real_img, gen_img, act_op):
    assert(len(real_img) == len(gen_img))

    r_inp = []
    g_inp = []
    for idx in range(len(real_img)):
        r_inp.append(preprocess_inception_images(real_img[idx]))
        g_inp.append(preprocess_inception_images(gen_img[idx]))

    r_act = sess.run(act_op, feed_dict={'inputs:0': r_inp})
    g_act = sess.run(act_op, feed_dict={'inputs:0': g_inp})

    print(get_cosine_dist(r_act, g_act))


def main(unused_argv=None):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % FLAGS.gpu):
                # Number of classes in the Dataset label set plus 1.
                # Label 0 is reserved for an (unused) background class.
                num_classes = FLAGS.num_classes + 1

                # Build a Graph that computes the high level convolutional features of the images
                inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, 299, 299, 3], name='inputs')

                _, end_points = inference(inputs, num_classes)

                pre_logits = end_points['pre_logits']
                act_op = tf.reshape(pre_logits, shape=[FLAGS.batch_size, -1])

                # Restore the moving average version of the
                # learned variables for eval.
                variable_averages = \
                    tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, FLAGS.checkpoint_dir)
                print('Restore the model from %s).' % FLAGS.checkpoint_dir)

                real_images = load_inception_data(real_img_folder, alphabetic=True)
                gen_images = load_inception_data(gen_img_folder, alphabetic=True)
                compute_imd(sess, real_images, gen_images, act_op)


if __name__ == '__main__':
    tf.app.run()