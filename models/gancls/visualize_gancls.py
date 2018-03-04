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
        self.sampler = model.sampler
        self.dataset = dataset
        self.config = config
        self.saver = tf.train.Saver()
        self.samples_dir = self.config.SAMPLE_DIR

        could_load, _ = load(self.saver, self.sess, self.config.CHECKPOINT_DIR)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise LookupError('Could not load any checkpoints')

    def visualize(self):
        z = np.random.uniform(-1, 1, size=(2, self.model.z_dim))
        z1, z2 = z[0], z[1]
        _, _, cond, _, _ = self.dataset.test.next_batch(1, window=4, embeddings=True)
        cond = np.tile(cond, reps=[64, 1])

        sample_z = get_interpolated_batch(z1, z2, method='lerp')

        samples = self.sess.run(self.model.sampler,
                                feed_dict={
                                    self.model.z_sample: sample_z,
                                    self.model.phi_sample: cond,
                                })

        save_images(samples, get_balanced_factorization(samples.shape[0]),
                    '{}/{}_visual/test1.png'.format(self.samples_dir, self.dataset.name))
        # ---------------------------------------------

        sample_z = np.random.uniform(-1, 1, size=(64, self.model.z_dim))

        _, cond, _, _ = self.dataset.test.next_batch_test(2, 0, 1)
        cond = np.squeeze(cond, axis=0)
        cond1, cond2 = cond[0], cond[1]

        cond = get_interpolated_batch(cond1, cond2, method='slerp')

        samples = self.sess.run(self.model.sampler,
                                feed_dict={
                                    self.model.z_sample: sample_z,
                                    self.model.phi_sample: cond,
                                })

        save_images(samples, get_balanced_factorization(samples.shape[0]),
                    '{}/{}_visual/test2.png'.format(self.samples_dir, self.dataset.name))
        make_gif(samples, '{}/{}_visual/test2.gif'.format(self.samples_dir, self.dataset.name))


