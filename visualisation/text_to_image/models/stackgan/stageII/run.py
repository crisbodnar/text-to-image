import os

from models.stackgan.stageI.model import ConditionalGan as ConditionalGanStageI
from models.stackgan.stageII.eval_stageii import StageIIEval
from models.stackgan.stageII.model import ConditionalGan
from models.stackgan.stageII.trainer import ConditionalGanTrainer
from models.stackgan.stageII.visualize_stageiI import StageIIVisualizer
from utils.utils import pp, show_all_variables
from utils.config import config_from_yaml
from preprocess.dataset import TextDataset

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('cfg_stage_I', './models/stackgan/stageI/cfg/flowers.yml',
                    'Relative path to the config of the model [./models/stackgan/stageI/cfg/flowers.yml]')
flags.DEFINE_string('cfg_stage_II', './models/stackgan/stageII/cfg/flowers.yml',
                    'Relative path to the config of the model [./models/stackgan/stageII/cfg/flowers.yml]')
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    cfg_stage_i = config_from_yaml(FLAGS.cfg_stage_I)
    cfg = config_from_yaml(FLAGS.cfg_stage_II)

    if not os.path.exists(cfg.CHECKPOINT_DIR):
        os.makedirs(cfg.CHECKPOINT_DIR)
    if not os.path.exists(cfg.SAMPLE_DIR):
        os.makedirs(cfg.SAMPLE_DIR)
    if not os.path.exists(cfg.LOGS_DIR):
        os.makedirs(cfg.LOGS_DIR)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    datadir = cfg.DATASET_DIR
    dataset = TextDataset(datadir, 256)

    filename_test = '%s/test' % datadir
    dataset._test = dataset.get_data(filename_test)

    filename_train = '%s/train' % datadir
    dataset.train = dataset.get_data(filename_train)

    with tf.Session(config=run_config) as sess:
        if cfg.EVAL.FLAG:
            stage_i = ConditionalGanStageI(cfg_stage_i, build_model=False)
            stage_ii = ConditionalGan(stage_i, cfg, build_model=False)
            stage_ii_eval = StageIIEval(
                sess=sess,
                model=stage_ii,
                dataset=dataset,
                cfg=cfg,
            )
            stage_ii_eval.evaluate_inception()

        elif cfg.TRAIN.FLAG:
            stage_i = ConditionalGanStageI(cfg_stage_i, build_model=False)
            stage_ii = ConditionalGan(stage_i, cfg)
            show_all_variables()
            stage_ii_trainer = ConditionalGanTrainer(
                sess=sess,
                model=stage_ii,
                dataset=dataset,
                cfg=cfg,
                cfg_stage_i=cfg_stage_i,
            )
            stage_ii_trainer.train()
        else:
            stage_i = ConditionalGanStageI(cfg_stage_i, build_model=False)
            stage_ii = ConditionalGan(stage_i, cfg, build_model=False)
            stage_ii_eval = StageIIVisualizer(
                sess=sess,
                model=stage_ii,
                dataset=dataset,
                cfg=cfg,
            )
            stage_ii_eval.visualize()


if __name__ == '__main__':
    tf.app.run()
