from random import randint

from models.wgancls.model import WGanCls
from utils.utils import save_images, image_manifold_size
from utils.saver import load
from utils.utils import denormalize_images
from preprocess.dataset import TextDataset
from preprocess.utils import closest_image
import tensorflow as tf
import numpy as np
from evaluation import fid
from evaluation.inception import load_inception_network


class WGanClsEval(object):
    def __init__(self, sess: tf.Session, model: WGanCls, dataset: TextDataset, cfg):
        self.sess = sess
        self.model = model
        self.dataset = dataset
        self.cfg = cfg
        self.bs = self.cfg.EVAL.SAMPLE_SIZE

    def evaluate(self):
        _, layers = load_inception_network(self.sess, 20, 64, self.cfg.EVAL.INCEP_CHECKPOINT_DIR)
        pool3 = layers['pool3']
        act_op = tf.reshape(pool3, shape=[64, -1])

        fid.compute_and_save_activation_statistics(self.cfg.R_IMG_PATH, self.sess, 64, act_op,
                                                   self.cfg.EVAL.ACT_STAT_PATH, verbose=True)

        stats = np.load(self.cfg.EVAL.ACT_STAT_PATH)
        mu_real = stats['mu']
        sigma_real = stats['sigma']

        z = tf.placeholder(tf.float32, [self.bs, self.model.z_dim], name='real_images')
        cond = tf.placeholder(tf.float32, [self.bs] + [self.model.embed_dim], name='cond')
        eval_gen = self.model.generator(z, cond, reuse=False)

        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if var.name.startswith('g_net')]
        saver = tf.train.Saver(g_vars)

        could_load, _ = load(saver, self.sess, self.cfg.CHECKPOINT_DIR)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise LookupError('Could not load any checkpoints')

        sample_z = np.random.uniform(-1, 1, size=(self.bs, self.model.z_dim))
        _, _, embed, _, _ = self.dataset.train.next_batch(self.bs, 4)

        fid_size = self.cfg.EVAL.SIZE
        incep_batch_size = self.cfg.EVAL.INCEP_BATCH_SIZE
        n_batches = fid_size // self.bs

        w, h, c = self.model.image_dims[0], self.model.image_dims[1], self.model.image_dims[2]
        samples = np.zeros((n_batches * self.bs, w, h, c))
        for i in range(n_batches):
            start = i * self.bs
            end = start + self.bs

            samples[start: end] = self.sess.run(eval_gen, feed_dict={z: sample_z, cond: embed})

        samples = denormalize_images(samples)

        mu_gen, sigma_gen = fid.calculate_activation_statistics(samples, self.sess, incep_batch_size, act_op,
                                                                verbose=True)
        print("calculate FID:", end=" ", flush=True)
        try:
            FID = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        except Exception as e:
            print(e)
            FID = 500

        print(FID)






