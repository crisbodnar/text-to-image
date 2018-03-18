import tensorflow as tf

from models.pggan.pggan import PGGAN
from utils.config import config_from_yaml
from utils.visualize import *
from utils.saver import load
import os

flags = tf.app.flags
flags.DEFINE_string('cfg', './models/pggan/cfg/flowers.yml',
                    'Relative path to the config of the model [./models/pggan/cfg/flowers.yml]')
FLAGS = flags.FLAGS

if __name__ == "__main__":

    stage = [1, 2, 3, 4, 5, 6, 7]

    all_samples = []

    cfg = config_from_yaml(FLAGS.cfg)

    batch_size = 16
    datadir = cfg.DATASET_DIR
    dataset = TextDataset(datadir, 64)

    filename_test = '%s/test' % datadir
    dataset.test = dataset.get_data(filename_test)

    filename_train = '%s/train' % datadir
    dataset.train = dataset.get_data(filename_train)

    z_sample = np.random.standard_normal((batch_size, 512))

    dataset_pos = np.random.randint(0, dataset.test.num_examples)
    _, conditions, _, captions = dataset.test.next_batch_test(batch_size, dataset_pos, 1)
    conditions = np.squeeze(conditions, 0)

    for i in range(0, len(stage)):
        scale_factor = 1

        sample_size = 512
        GAN_learn_rate = 1e-4

        pggan_checkpoint_dir_read = os.path.join(cfg.CHECKPOINT_DIR, 'stage%d/' % stage[i])
        if not os.path.exists(pggan_checkpoint_dir_read):
            os.makedirs(pggan_checkpoint_dir_read)

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        pggan = PGGAN(batch_size=batch_size, max_iters=-1, model_path='', read_model_path=pggan_checkpoint_dir_read,
                      data=dataset, sample_size=sample_size, sample_path='', log_dir='', learn_rate=GAN_learn_rate,
                      stage=stage[i], t=False, build_model=False)

        cond = tf.placeholder(tf.float32, [None, 1024], name='cond')
        z = tf.placeholder(tf.float32, [None, sample_size], name='z')
        gen_op, _, _ = pggan.generator(z, cond, stages=stage[i], t=False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(tf.global_variables('g_net'))
            could_load = load(saver, sess, pggan_checkpoint_dir_read)
            if not could_load:
                raise RuntimeError('Could not load stage %d' % stage[i])

            samples = sess.run(gen_op, feed_dict={'z:0': z_sample, 'cond:0': conditions})
            samples = np.clip(samples, -1., 1.)
            all_samples.append(samples)

        tf.reset_default_graph()

    all_samples = gen_pggan_sample(all_samples, 128, interp='nearest')

    for idx in range(batch_size):
        caption = captions[idx][0]
        img_for_this_desc = []
        for stg in range(stage[-1]):
            img_for_this_desc.append(all_samples[stg, idx, :, :, :])

        img_for_this_desc = np.array(img_for_this_desc)
        save_cap_batch(img_for_this_desc, caption, '{}/{}_visual/stages/stages{}.png'.format('samples/PGGAN/flowers',
                                                                                             'flowers', idx), split=35)






