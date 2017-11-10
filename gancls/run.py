import os
import scipy.misc
import numpy as np

from gancls.model import GANCLS
from gancls.utils import pp, show_all_variables
from preprocess.dataset import TextDataset

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 600, "Epoch to train [600]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_string("dataset", "flowers", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    datadir = './data/%s' % FLAGS.dataset
    embedding_type = 'cnn-rnn'
    dataset = TextDataset(datadir, embedding_type, 1)

    filename_test = '%s/test' % datadir
    dataset.test = dataset.get_data(filename_test)

    filename_train = '%s/train' % datadir
    dataset.train = dataset.get_data(filename_train)

    with tf.Session(config=run_config) as sess:
        gancls = GANCLS(
                sess,
                dataset=dataset,
                output_size=FLAGS.output_size,
                batch_size=FLAGS.batch_size,
                checkpoint_dir=FLAGS.checkpoint_dir
            )

        show_all_variables()

        if FLAGS.train:
            gancls.train(FLAGS)
        else:
            if not gancls.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()
