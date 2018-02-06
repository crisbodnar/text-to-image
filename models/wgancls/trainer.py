from random import randint

import tensorflow as tf

from models.wgancls.model import WGanCls
from utils.utils import save_images, image_manifold_size
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
        _, sample_embed, _, captions = self.dataset.test.next_batch_test(self.model.sample_num,
                                                                         randint(0, self.dataset.test.num_examples), 1)
        sample_embed = np.squeeze(sample_embed, axis=0)
        print(sample_embed.shape)

        # Display the captions of the sampled images
        print('\nCaptions of the sampled images:')
        for caption_idx, caption_batch in enumerate(captions):
            print('{}: {}'.format(caption_idx + 1, caption_batch[0]))
        print()

        counter = 1
        start_time = time.time()
        tf.global_variables_initializer().run()

        could_load, checkpoint_counter = load(self.saver, self.sess, self.cfg.CHECKPOINT_DIR)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(self.cfg.TRAIN.EPOCH):
            # Updates per epoch are given by the training data size / batch size
            updates_per_epoch = self.dataset.train.num_examples // self.model.batch_size

            for idx in range(0, updates_per_epoch):
                images, wrong_images, embed, _, _ = self.dataset.train.next_batch(self.model.batch_size, 4)
                batch_z = np.random.normal(0, 1, (self.model.batch_size, self.model.z_dim))


                    self.sess.run([self.model.D_optim],
                                  feed_dict={
                                        self.model.x: images,
                                        self.model.cond: embed,
                                        self.model.z: batch_z
                                  })

                # Update G network
                _, err_g, summary_str = self.sess.run([self.model.G_optim, self.model.G_loss, self.G_merged_summ],
                                                      feed_dict={self.model.z: batch_z, self.model.cond: embed})
                self.writer.add_summary(summary_str, counter)

                # Update D one more time after G
                _, err_d, summary_str = self.sess.run([self.D_optim, self.D_loss, self.D_merged_summ],
                                                      feed_dict={
                                                          self.model.x: images,
                                                          self.model.wrong_inputs: wrong_images,
                                                          self.model.cond: embed,
                                                          self.model.z: batch_z
                                                      })
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, idx, updates_per_epoch,
                         time.time() - start_time, err_d, err_g))

                if np.mod(counter, 100) == 0:
                    try:
                        samples = self.sess.run(self.model.sampler,
                                                feed_dict={
                                                            self.model.z_sample: sample_z,
                                                            self.model.embed_sample: sample_embed,
                                                          })
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    '{}train_{:02d}_{:04d}.png'.format(self.cfg.SAMPLE_DIR, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (err_d, err_g))

                        # Display the captions of the sampled images
                        print('\nCaptions of the sampled images:')
                        for caption_idx, caption_batch in enumerate(captions):
                            print('{}: {}'.format(caption_idx + 1, caption_batch[0]))
                        print()
                    except Exception as e:
                        print("Failed to generate sample image")
                        print(type(e))
                        print(e.args)
                        print(e)

                if np.mod(counter, 500) == 2:
                    save(self.saver, self.sess, self.cfg.CHECKPOINT_DIR, counter)
