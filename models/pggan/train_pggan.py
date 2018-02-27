import tensorflow as tf

from models.pggan.pggan import PGGAN
from preprocess.dataset import TextDataset
from utils.config import config_from_yaml
import os

flags = tf.app.flags
flags.DEFINE_string('cfg', './models/pggan/cfg/flowers.yml',
                    'Relative path to the config of the model [./models/pggan/cfg/flowers.yml]')
FLAGS = flags.FLAGS

if __name__ == "__main__":

    stage = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    prev_stage = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]

    for i in range(0, len(stage)):

        t = False if (i % 2 == 0) else True

        cfg = config_from_yaml(FLAGS.cfg)

        batch_size = 16
        max_iters = 37500
        sample_size = 512
        GAN_learn_rate = 1e-4

        pggan_checkpoint_dir_write = os.path.join(cfg.CHECKPOINT_DIR, 'stage%d/' % stage[i])
        pggan_checkpoint_dir_read = os.path.join(cfg.CHECKPOINT_DIR, 'stage%d/' % prev_stage[i])

        if t:
            sample_path = os.path.join(cfg.SAMPLE_DIR, 'stage_t%d/' % stage[i])
            logs_dir = os.path.join(cfg.LOGS_DIR, 'stage_t%d/' % stage[i])
        else:
            sample_path = os.path.join(cfg.SAMPLE_DIR, 'stage%d/' % stage[i])
            logs_dir = os.path.join(cfg.LOGS_DIR, 'stage%d/' % stage[i])

        if not os.path.exists(pggan_checkpoint_dir_write):
            os.makedirs(pggan_checkpoint_dir_write)
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        if not os.path.exists(pggan_checkpoint_dir_read):
            os.makedirs(pggan_checkpoint_dir_read)

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        datadir = cfg.DATASET_DIR
        dataset = TextDataset(datadir, cfg.MODEL.SIZES[stage[i] - 1])

        filename_test = '%s/test' % datadir
        dataset.test = dataset.get_data(filename_test)

        filename_train = '%s/train' % datadir
        dataset.train = dataset.get_data(filename_train)

        tf.reset_default_graph()

        pggan = PGGAN(batch_size=batch_size, max_iters=max_iters,
                      model_path=pggan_checkpoint_dir_write, read_model_path=pggan_checkpoint_dir_read,
                      data=dataset, sample_size=sample_size,
                      sample_path=sample_path, log_dir=logs_dir, learn_rate=GAN_learn_rate, stage=stage[i],
                      t=t)

        pggan.train()

