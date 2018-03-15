import tensorflow as tf

from models.pggan.pggan import PGGAN
from utils.config import config_from_yaml
from utils.utils import make_gif
from utils.visualize import *
from utils.saver import load
import os

flags = tf.app.flags
flags.DEFINE_string('cfg', './models/pggan/cfg/flowers.yml',
                    'Relative path to the config of the model [./models/pggan/cfg/flowers.yml]')
FLAGS = flags.FLAGS

if __name__ == "__main__":
    cfg = config_from_yaml(FLAGS.cfg)

    datadir = cfg.DATASET_DIR
    dataset = TextDataset(datadir, 64)

    filename_test = '%s/test' % datadir
    dataset.test = dataset.get_data(filename_test)

    filename_train = '%s/train' % datadir
    dataset.train = dataset.get_data(filename_train)

    batch_size = 16
    scale_factor = 1
    sample_size = 512
    GAN_learn_rate = 1e-4
    stage = 7

    pggan_checkpoint_dir_read = os.path.join(cfg.CHECKPOINT_DIR, 'stage%d/' % stage)
    samples_dir = cfg.CHECKPOINT_DIR
    if not os.path.exists(pggan_checkpoint_dir_read):
        os.makedirs(pggan_checkpoint_dir_read)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    pggan = PGGAN(batch_size=batch_size, max_iters=-1, model_path='', read_model_path=pggan_checkpoint_dir_read,
                  data=dataset, sample_size=sample_size, sample_path='', log_dir='', learn_rate=GAN_learn_rate,
                  stage=stage, t=False, build_model=False)

    cond = tf.placeholder(tf.float32, [None, 1024], name='cond')
    z = tf.placeholder(tf.float32, [None, sample_size], name='z')
    gen_op, _, _ = pggan.generator(z, cond, stages=stage, t=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.global_variables('g_net'))
        could_load = load(saver, sess, pggan_checkpoint_dir_read)
        if not could_load:
            raise RuntimeError('Could not load stage %d' % stage[i])
        
        for idx in range(6):
            dataset_pos = np.random.randint(0, dataset.test.num_examples)
            # Interpolation in z space:
            # ---------------------------------------------------------------------------------------------------------
            _, cond, _, captions = dataset.test.next_batch_test(1, dataset_pos, 1)
            cond = np.squeeze(cond, axis=0)
            caption = captions[0][0]
    
            samples = gen_noise_interp_img(sess, gen_op, cond, sample_size, batch_size)
            save_cap_batch(samples, caption, '{}/{}_visual/z_interp/z_interp{}.png'.format(samples_dir,
                                                                                           dataset.name,
                                                                                           idx))
            # Interpolation in embedding space:
            # ---------------------------------------------------------------------------------------------------------
    
            _, cond, _, caps = dataset.test.next_batch_test(2, dataset_pos, 1)
            cond = np.squeeze(cond, axis=0)
            cond1, cond2 = cond[0], cond[1]
            cap1, cap2 = caps[0][0], caps[1][0]
    
            samples = gen_cond_interp_img(sess, gen_op, cond1, cond2, sample_size, batch_size)
            save_interp_cap_batch(samples, cap1, cap2,
                                  '{}/{}_visual/cond_interp/cond_interp{}.png'.format(samples_dir, dataset.name, idx))
    
            # Generate captioned image
            # ---------------------------------------------------------------------------------------------------------
            _, conditions, _, captions = dataset.test.next_batch_test(1, dataset_pos, 1)
            conditions = np.squeeze(conditions, axis=0)
            caption = captions[0][0]
            samples = gen_captioned_img(sess, gen_op, conditions, sample_size, batch_size)
    
            save_cap_batch(samples, caption, '{}/{}_visual/cap/cap{}.png'.format(samples_dir,
                                                                                 dataset.name, idx))





