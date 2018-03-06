import tensorflow as tf

from models.resgan.resgan import ResGAN
from preprocess.dataset import TextDataset
from utils.config import config_from_yaml
import os

flags = tf.app.flags
flags.DEFINE_string('cfg', './models/resgan/cfg/flowers.yml',
                    'Relative path to the config of the model [./models/resgan/cfg/flowers.yml]')
FLAGS = flags.FLAGS

if __name__ == "__main__":

    for stage in range(4, 9):

        cfg = config_from_yaml(FLAGS.cfg)

        batch_size = cfg.MODEL.BS[stage - 1]
        max_iters = cfg.MODEL.KITERS[stage - 1] * 1000

        sample_size = 512
        GAN_learn_rate = 1e-4

        pggan_checkpoint_dir_write = os.path.join(cfg.CHECKPOINT_DIR, 'stage%d/' % stage)
        pggan_checkpoint_dir_read = os.path.join(cfg.CHECKPOINT_DIR, 'stage%d/' % (stage - 1))
        sample_path = os.path.join(cfg.SAMPLE_DIR, 'stage%d/' % stage)
        logs_dir = os.path.join(cfg.LOGS_DIR, 'stage%d/' % stage)

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
        os = cfg.MODEL.SIZES[stage - 1]
        dataset = TextDataset(datadir, os)

        filename_test = '%s/test' % datadir
        dataset.test = dataset.get_data(filename_test)

        filename_train = '%s/train' % datadir
        dataset.train = dataset.get_data(filename_train)

        pggan = ResGAN(batch_size=batch_size, max_iters=max_iters,
                       model_path=pggan_checkpoint_dir_write, read_model_path=pggan_checkpoint_dir_read,
                       data=dataset, sample_size=sample_size,
                       sample_path=sample_path, log_dir=logs_dir, learn_rate=GAN_learn_rate, stage=stage, os=os)

        pggan.train()

