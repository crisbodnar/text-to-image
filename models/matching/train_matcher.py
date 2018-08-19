import os

from models.matching.matcher import MatchingModule
from models.matching import matcher
from preprocess.dataset import TextDataset

import tensorflow as tf


def main(_):

    if not os.path.exists(matcher.CHECKPOINT_DIR):
        os.makedirs(matcher.CHECKPOINT_DIR)
    if not os.path.exists(matcher.LOGS_DIR):
        os.makedirs(matcher.LOGS_DIR)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    datadir = matcher.DATASET_DIR
    dataset = TextDataset(datadir, 299)

    # We train the matcher on the test dataset.
    filename_test = '%s/test' % datadir
    dataset.test = dataset.get_data(filename_test)

    with tf.Session(config=run_config) as sess:
        matcher_module = MatchingModule(sess, dataset)
        matcher_module.train()


if __name__ == '__main__':
    tf.app.run()