from random import randint

import tensorflow as tf

from models.wgancls.model import WGanCls
from utils.utils import save_images, image_manifold_size, save_captions
from utils.saver import save, load
from preprocess.dataset import TextDataset
import numpy as np
import time


class WGanClsTrainer(object):
    def __init__(self, sess: tf.Session, model: WGanCls, dataset: TextDataset, cfg):
        self.sess = sess
        self.model = model
        self.dataset = dataset
        self.cfg = cfg

    def define_summaries(self):
        self.summary_op = tf.summary.merge_all([
            tf.summary.image('x', self.model.x),
            tf.summary.image('G_img', self.model.G),

            tf.summary.histogram('z', self.model.z),
            tf.summary.histogram('z_sample', self.model.z_sample),

            tf.summary.scalar('G_kl_loss', self.model.G_kl_loss),
            tf.summary.scalar('G_wass_loss', self.model.G_wass_loss),
            tf.summary.scalar('G_loss', self.model.G_loss),

            tf.summary.scalar('wass_dist', self.model.wass_dist),
            tf.summary.scalar('D_grad_penalty', self.model.gradient_penalty),
            tf.summary.scalar('D_loss', self.model.D_loss),

        ])

        self.writer = tf.summary.FileWriter(self.cfg.LOGS_DIR, self.sess.graph)

    def train(self):
        self.define_summaries()

        self.saver = tf.train.Saver(max_to_keep=self.cfg.TRAIN.CHECKPOINTS_TO_KEEP)

        sample_z = np.random.normal(0, 1, (self.model.sample_num, self.model.z_dim))
        _, sample_cond, _, captions = self.dataset.test.next_batch_test(self.model.sample_num, 0, 1)
        sample_cond = np.squeeze(sample_cond, axis=0)
        print('Conditionals sampler shape: {}'.format(sample_cond.shape))

        print(captions)
        save_captions(self.cfg.SAMPLE_DIR, captions)
        exit(-1)

        counter = 1
        start_time = time.time()
        tf.global_variables_initializer().run()

        could_load, checkpoint_counter = load(self.saver, self.sess, self.cfg.CHECKPOINT_DIR)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for idx in range(self.cfg.TRAIN.MAX_STEPS):
            epoch_size = self.dataset.train.num_examples // self.model.batch_size
            epoch = idx // epoch_size

            images, wrong_images, embed, _, _ = self.dataset.train.next_batch(self.model.batch_size, 4)
            batch_z = np.random.normal(0, 1, (self.model.batch_size, self.model.z_dim))

            _, err_d, = self.sess.run([self.model.D_optim, self.model.D_loss],
                                      feed_dict={
                                          self.model.x: images,
                                          self.model.cond: embed,
                                          self.model.z: batch_z
                                      })

            # Update G network
            if np.mod(idx, self.cfg.TRAIN.N_CRITIC) == 0:
                _, err_g, summary_str = self.sess.run([self.model.G_optim, self.model.G_loss, self.summary_op],
                                                      feed_dict={
                                                          self.model.x: images,
                                                          self.model.z: batch_z,
                                                          self.model.cond: embed
                                                      })
                self.writer.add_summary(summary_str, counter)

                print("Epoch: [%2d] [%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, idx, time.time() - start_time, err_d, err_g))

            if np.mod(counter, self.cfg.TRAIN.SAMPLE_PERIOD) == 0:
                try:
                    samples = self.sess.run(self.model.sampler,
                                            feed_dict={
                                                self.model.z_sample: sample_z,
                                                self.model.embed_sample: sample_cond,
                                            })
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                '{}train_{:02d}_{:04d}.png'.format(self.cfg.SAMPLE_DIR, epoch, idx))

                except Exception as e:
                    print("Failed to generate sample image")
                    print(type(e))
                    print(e.args)
                    print(e)

            if np.mod(counter, 500) == 2:
                save(self.saver, self.sess, self.cfg.CHECKPOINT_DIR, counter)
