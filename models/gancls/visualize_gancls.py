from models.gancls.model import GanCls
from utils.utils import save_images, get_balanced_factorization, make_gif
from utils.saver import load
from utils.visualize import *
from preprocess.dataset import TextDataset
import tensorflow as tf
import numpy as np


class GanClsVisualizer(object):
    def __init__(self, sess: tf.Session, model: GanCls, dataset: TextDataset, config):
        self.sess = sess
        self.model = model
        self.dataset = dataset
        self.config = config
        self.samples_dir = self.config.SAMPLE_DIR

    def visualize(self):
        z = tf.placeholder(tf.float32, [self.model.batch_size, self.model.z_dim], name='z')
        cond = tf.placeholder(tf.float32, [self.model.batch_size] + [self.model.embed_dim], name='cond')
        gen = self.model.generator(z, cond, is_training=False)

        saver = tf.train.Saver(tf.global_variables('g_net'))
        could_load, _ = load(saver, self.sess, self.config.CHECKPOINT_DIR)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise LookupError('Could not load any checkpoints')

        # Interpolation in z space:
        # ---------------------------------------------------------------------------------------------------------
        _, _, cond, _, _ = self.dataset.test.next_batch(1, window=4, embeddings=True)
        cond = np.tile(cond, reps=[64, 1])

        samples = gen_noise_interp_img(self.sess, gen, cond, self.model.z_dim, self.model.batch_size)
        save_images(samples, get_balanced_factorization(samples.shape[0]),
                    '{}/{}_visual/test1.png'.format(self.samples_dir, self.dataset.name))

        # Interpolation in embedding space:
        # ---------------------------------------------------------------------------------------------------------

        _, cond, _, _ = self.dataset.test.next_batch_test(2, 0, 1)
        cond = np.squeeze(cond, axis=0)
        cond1, cond2 = cond[0], cond[1]

        samples = gen_cond_interp_img(self.sess, gen, cond1, cond2, self.model.z_dim, self.model.batch_size)
        save_images(samples, get_balanced_factorization(samples.shape[0]),
                    '{}/{}_visual/test2.png'.format(self.samples_dir, self.dataset.name))
        make_gif(samples, '{}/{}_visual/test2.gif'.format(self.samples_dir, self.dataset.name))

        # Generate captioned image
        # ---------------------------------------------------------------------------------------------------------
        _, conditions, _, captions = self.dataset.test.next_batch_test(2, 0, 1)
        conditions = np.squeeze(conditions, axis=0)
        caption = captions[0][0]
        samples = gen_captioned_img(self.sess, gen, conditions[0], self.model.z_dim, self.model.batch_size)

        save_cap_batch(samples, caption, '{}/{}_visual/caption.png'.format(self.samples_dir, self.dataset.name))







