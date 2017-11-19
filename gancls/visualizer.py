from random import randint

from gancls.model import GanCls
from gancls.utils import load, save_images, image_manifold_size, visualize
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

        could_load, _ = load(model.directory, self.config.checkpoint_dir, sess, self.saver)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise LookupError('Could not load any checkpoint')

    def visualize(self):
        # TODO: Solve bug with the generator which generates unmatched images.
        sample_z = np.random.uniform(-1, 1, size=(self.model.sample_num, self.model.z_dim))
        _, sample_embed, _, captions = self.dataset.train.next_batch_test(self.model.sample_num,
                                                                          randint(0, self.dataset.test.num_examples), 1)
        sample_embed = np.squeeze(sample_embed, axis=0)

        samples = self.sess.run(self.model.sampler,
                                feed_dict={
                                    self.model.z_sample: sample_z,
                                    self.model.phi_sample: sample_embed,
                                })
        save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/{}/{}/test.png'.format(self.config.test_dir, self.model.name, self.dataset.name))

        visualize(self.sess, self.model, self.config, 5)


