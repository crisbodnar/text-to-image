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

    stage = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    prev_stage = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]

    for i in range(9, len(stage)):

        t = False if (i % 2 == 0) else True

        cfg = config_from_yaml(FLAGS.cfg)

        batch_size = 32
        scale_factor = 1
        if stage[i] == 7:
            batch_size = 8
            scale_factor = 2
        print(batch_size)

        if stage[i] <= 4 or t:
            max_iters = 20000 * scale_factor
        else:
            max_iters = 50000 * scale_factor

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
        if not os.path.exists(pggan_checkpoint_dir_read) and stage > 1:
            os.makedirs(pggan_checkpoint_dir_read)

        datadir = cfg.DATASET_DIR
        dataset = TextDataset(datadir, cfg.MODEL.SIZES[stage[i] - 1])

        filename_test = '%s/test' % datadir
        dataset.test = dataset.get_data(filename_test)

        filename_train = '%s/train' % datadir
        dataset.train = dataset.get_data(filename_train)

        pggan = ResGAN(batch_size=batch_size, max_iters=max_iters,
                       model_path=pggan_checkpoint_dir_write, read_model_path=pggan_checkpoint_dir_read,
                       data=dataset, sample_size=sample_size,
                       sample_path=sample_path, log_dir=logs_dir, learn_rate=GAN_learn_rate, stage=stage[i],
                       t=t)

        pggan.train()

