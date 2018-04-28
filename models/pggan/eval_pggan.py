import tensorflow as tf
from evaluation import inception_score

from models.inception.model import load_inception_inference
from models.pggan.pggan import PGGAN
from utils.config import config_from_yaml
from utils.utils import make_gif, prep_incep_img
from utils.visualize import *
from utils.saver import load
import os

flags = tf.app.flags
# flags.DEFINE_string('cfg', './models/pggan/cfg/flowers.yml',
#                     'Relative path to the config of the model [./models/pggan/cfg/flowers.yml]')
flags.DEFINE_string('cfg', './models/pggan/cfg/birds.yml',
                    'Relative path to the config of the model [./models/pggan/cfg/birds.yml]')
FLAGS = flags.FLAGS

if __name__ == "__main__":
    cfg = config_from_yaml(FLAGS.cfg)

    datadir = cfg.DATASET_DIR
    dataset = TextDataset(datadir, 256)

    filename_test = '%s/test' % datadir
    dataset.test = dataset.get_data(filename_test)

    filename_train = '%s/train' % datadir
    dataset.train = dataset.get_data(filename_train)

    batch_size = 64
    scale_factor = 1
    sample_size = 128
    stage = 7
    incep_batch_size = batch_size
    incep_checkpoint_dir = cfg.EVAL.INCEP_CHECKPOINT_DIR

    pggan_checkpoint_dir_read = os.path.join(cfg.CHECKPOINT_DIR, 'stage%d/' % stage)
    samples_dir = cfg.SAMPLE_DIR
    if not os.path.exists(pggan_checkpoint_dir_read):
        os.makedirs(pggan_checkpoint_dir_read)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    pggan = PGGAN(batch_size=batch_size, steps=None,
                  check_dir_write='', check_dir_read=pggan_checkpoint_dir_read,
                  dataset=dataset, sample_path=None, log_dir=None, stage=stage,
                  trans=False, build_model=False)

    cond = tf.placeholder(tf.float32, [None, 1024], name='cond')
    z = tf.placeholder(tf.float32, [None, sample_size], name='z')
    gen_op, _, _ = pggan.generator(z, cond, stages=stage, t=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.global_variables('g_net'))
        could_load = load(saver, sess, pggan_checkpoint_dir_read)
        if not could_load:
            raise RuntimeError('Could not load stage %d' % stage)

        logits, _ = load_inception_inference(sess, cfg.EVAL.NUM_CLASSES, incep_batch_size, incep_checkpoint_dir)
        pred_op = tf.nn.softmax(logits)

        size = 50000
        n_batches = size // batch_size

        all_preds = []
        for i in range(n_batches):
            print("\rGenerating batch %d/%d" % (i + 1, n_batches), end="", flush=True)

            sample_z = np.random.normal(0, 1, size=(batch_size, sample_size))
            _, _, embed, _, _ = dataset.test.next_batch(batch_size, 4, embeddings=True)

            # Generate a batch and scale it up for inception
            gen_batch = sess.run(gen_op, feed_dict={z: sample_z, cond: embed})
            gen_batch = np.clip(gen_batch, -1., 1.)

            samples = denormalize_images(gen_batch)
            incep_samples = np.empty((batch_size, 299, 299, 3))
            for sample_idx in range(batch_size):
                incep_samples[sample_idx] = prep_incep_img(samples[sample_idx])

            # Run prediction for current batch
            pred = sess.run(pred_op, feed_dict={'inputs:0': incep_samples})
            all_preds.append(pred)

        # Get rid of the first dimension
        all_preds = np.concatenate(all_preds, 0)

        print('\nComputing inception score...')
        mean, std = inception_score.get_inception_from_predictions(all_preds, 10)
        print('Inception Score | mean:', "%.2f" % mean, 'std:', "%.2f" % std)





