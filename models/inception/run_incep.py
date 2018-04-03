import os

from models.inception.trainer import InceptionTrainer
from utils.utils import pp
from utils.config import config_from_yaml
from preprocess.dataset import TextDataset

import tensorflow as tf

flags = tf.app.flags
# flags.DEFINE_string('cfg', './models/inception/cfg/flowers.yaml',
#                     'Relative path to the config of the model [./models/inception/cfg/flowers.yaml]')
flags.DEFINE_string('cfg', './models/inception/cfg/birds.yaml',
                    'Relative path to the config of the model [./models/inception/cfg/birds.yaml]')
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    cfg = config_from_yaml(FLAGS.cfg)

    if not os.path.exists(cfg.CHECKPOINT_DIR):
        os.makedirs(cfg.CHECKPOINT_DIR)
    if not os.path.exists(cfg.LOGS_DIR):
        os.makedirs(cfg.LOGS_DIR)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    datadir = cfg.DATASET_DIR
    dataset = TextDataset(datadir, 299)

    # We train inception on the test dataset which contains completely other classes from the train dataset
    # (used in GAN training). This is needed for a correct evaluation of the Inception/FID score.
    filename_test = '%s/test' % datadir
    dataset.test = dataset.get_data(filename_test)

    with tf.Session(config=run_config) as sess:
        if cfg.TRAIN.FLAG:
            stage_i_trainer = InceptionTrainer(
                sess=sess,
                dataset=dataset,
                cfg=cfg,
            )
            stage_i_trainer.train()
        else:
            pass


if __name__ == '__main__':
    tf.app.run()
