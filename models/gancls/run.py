import os

from models.gancls.eval_gancls import GanClsEval
from models.gancls.model import GanCls
from models.gancls.trainer import GanClsTrainer
from models.gancls.visualize_gancls import GanClsVisualizer
from utils.utils import pp, show_all_variables
from utils.config import config_from_yaml
from preprocess.dataset import TextDataset

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('cfg', './models/gancls/cfg/flowers.yml',
                    'Relative path to the config of the model [./gancls/cfg/flowers.yml]')
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    cfg = config_from_yaml(FLAGS.cfg)

    if not os.path.exists(cfg.CHECKPOINT_DIR):
        os.makedirs(cfg.CHECKPOINT_DIR)
    if not os.path.exists(cfg.SAMPLE_DIR):
        os.makedirs(cfg.SAMPLE_DIR)
    if not os.path.exists(cfg.LOGS_DIR):
        os.makedirs(cfg.LOGS_DIR)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    datadir = cfg.DATASET_DIR
    dataset = TextDataset(datadir, 64)

    filename_test = '%s/test' % datadir
    dataset._test = dataset.get_data(filename_test)

    filename_train = '%s/train' % datadir
    dataset.train = dataset.get_data(filename_train)

    with tf.Session(config=run_config) as sess:

        if cfg.EVAL.FLAG:
            gancls = GanCls(cfg, build_model=False)
            eval = GanClsEval(
                sess=sess,
                model=gancls,
                dataset=dataset,
                cfg=cfg)
            eval.evaluate_inception()
        elif cfg.TRAIN.FLAG:
            gancls = GanCls(cfg)
            show_all_variables()
            gancls_trainer = GanClsTrainer(
                sess=sess,
                model=gancls,
                dataset=dataset,
                cfg=cfg,
            )
            gancls_trainer.train()
        else:
            gancls = GanCls(cfg, build_model=False)
            gancls_visualiser = GanClsVisualizer(
                sess=sess,
                model=gancls,
                dataset=dataset,
                config=cfg,
            )
            gancls_visualiser.visualize()


if __name__ == '__main__':
    tf.app.run()
