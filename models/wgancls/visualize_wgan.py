from models.wgancls.model import WGanCls
from utils.utils import save_images, get_balanced_factorization
from utils.saver import load
from utils.visualize import *
from utils.ops import NHWC
from preprocess.dataset import TextDataset
import tensorflow as tf
import numpy as np


class WGanClsVisualizer(object):
    def __init__(self, sess: tf.Session, model: WGanCls, dataset: TextDataset, config):
        self.sess = sess
        self.model = model
        self.dataset = dataset
        self.config = config
        self.samples_dir = self.config.SAMPLE_DIR

    def visualize(self):
        z = tf.placeholder(tf.float32, [self.model.batch_size, self.model.z_dim], name='z')
        phi = tf.placeholder(tf.float32, [self.model.batch_size] + [self.model.embed_dim], name='cond')
        eval_gen, _, _ = self.model.generator(z, phi, is_training=False, df=NHWC)

        saver = tf.train.Saver(tf.global_variables('g_net'))
        could_load, _ = load(saver, self.sess, self.config.CHECKPOINT_DIR)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise RuntimeError('Could not load the checkpoints of the generator')

        sample_z = np.random.normal(0, 1, size=(2, self.model.z_dim))
        z1, z2 = sample_z[0], sample_z[1]
        _, _, cond, _, _ = self.dataset.test.next_batch(1, window=4, embeddings=True)
        cond = np.tile(cond, reps=[64, 1])

        sample_z = get_interpolated_batch(z1, z2, method='slerp')

        samples = self.sess.run(eval_gen,
                                feed_dict={
                                    z: sample_z,
                                    phi: cond,
                                })

        save_images(samples, get_balanced_factorization(samples.shape[0]),
                    '{}/{}_visual/test.png'.format(self.samples_dir, self.dataset.name))
        # ---------------------------------------------

        sample_z = np.random.normal(0, 1, size=(1, self.model.z_dim))
        sample_z = np.tile(sample_z, reps=[64, 1])

        _, _, cond, _, _ = self.dataset.train.next_batch(2, window=4, embeddings=True)
        cond1, cond2 = cond[0], cond[1]

        cond = get_interpolated_batch(cond1, cond2, method='slerp')

        samples = self.sess.run(eval_gen,
                                feed_dict={
                                    z: sample_z,
                                    phi: cond,
                                })

        save_images(samples, get_balanced_factorization(samples.shape[0]),
                    '{}/{}_visual/test2.png'.format(self.samples_dir, self.dataset.name))

        sample_z = np.random.normal(0, 1, (self.model.sample_num, self.model.z_dim))
        _, sample_cond, _, captions = self.dataset.test.next_batch_test(self.model.sample_num, 0, 1)
        sample_cond = np.squeeze(sample_cond, axis=0)

        samples2 = self.sess.run(eval_gen,
                                feed_dict={
                                    z: sample_z,
                                    phi: sample_cond,
                                })

        save_images(samples2, get_balanced_factorization(samples2.shape[0]),
                    '{}/{}_visual/test3.png'.format(self.samples_dir, self.dataset.name))




