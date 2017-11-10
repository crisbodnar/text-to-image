from random import randint

import tensorflow as tf
from gancls.model import GanCls
from gancls.utils import save_images, image_manifold_size
import numpy as np
import time
import os
from preprocess.dataset import TextDataset


class GanClsTrainer(object):
    def __init__(self, sess: tf.Session, model: GanCls, dataset: TextDataset, config):
        self.sess = sess
        self.model = model
        self.dataset = dataset
        self.config = config

    def define_losses(self):
        self.D_synthetic_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model.D_synthetic_logits,
                                                    labels=tf.zeros_like(self.model.D_synthetic)))
        self.D_real_match_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model.D_real_match_logits,
                                                    labels=tf.fill(self.model.D_real_match.get_shape(), 0.9)))
        self.D_real_mismatch_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model.D_real_mismatch_logits,
                                                    labels=tf.zeros_like(self.model.D_real_mismatch)))
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model.D_synthetic_logits,
                                                    labels=tf.ones_like(self.model.D_synthetic)))

        self.D_synthetic_loss_summ = tf.summary.histogram('d_synthetic_sum_loss', self.D_synthetic_loss)
        self.D_real_match_loss_summ = tf.summary.histogram('d_real_match_sum_loss', self.D_real_match_loss)
        self.D_real_mismatch_loss_summ = tf.summary.histogram('d_real_mismatch_sum_loss', self.D_real_mismatch_loss)

        alpha = 0.5
        self.D_loss = self.D_real_match_loss + alpha * self.D_real_mismatch_loss + (1.0 - alpha) * self.D_synthetic_loss

        self.G_loss_summ = tf.summary.scalar("g_loss", self.G_loss)
        self.D_loss_summ = tf.summary.scalar("d_loss", self.D_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

        self.D_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
            .minimize(self.D_loss, var_list=self.d_vars)
        self.G_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
            .minimize(self.G_loss, var_list=self.g_vars)

    def define_summaries(self):
        self.D_synthetic_summ = tf.summary.histogram('d_synthetic_sum', self.model.D_synthetic)
        self.D_real_match_summ = tf.summary.histogram('d_real_match_sum', self.model.D_real_match)
        self.D_real_mismatch_summ = tf.summary.histogram('d_real_mismatch_sum', self.model.D_real_mismatch)
        self.G_summ = tf.summary.image("g_sum", self.model.G)
        self.z_sum = tf.summary.histogram("z", self.model.z)
        self.G_merged_summ = tf.summary.merge([self.z_sum, self.G_summ])
        self.D_merged_summ = tf.summary.merge([self.z_sum, self.D_real_mismatch_summ,
                                               self.D_real_match_summ, self.D_synthetic_summ])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    def train(self):
        self.define_losses()
        self.define_summaries()

        tf.global_variables_initializer().run()

        # TODO: There is a bug which enforces the sample num to be the bath size.
        sample_z = np.random.uniform(-1, 1, size=(self.model.sample_num, self.model.z_dim))
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
        could_load, checkpoint_counter = self.load(self.config.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(self.config.epoch):
            # Updates per epoch are given by the training data size / batch size
            updates_per_epoch = self.dataset.train.num_examples // self.model.batch_size

            for idx in range(0, updates_per_epoch):
                images, wrong_images, embed, _, _ = self.dataset.train.next_batch(self.model.batch_size, 4)

                batch_z = np.random.uniform(-1, 1, [self.config.batch_size, self.model.z_dim]).astype(np.float32)

                # Update D network
                _, err_d_real_match, err_d_real_mismatch, err_d_fake, err_d, summary_str = self.sess.run(
                    [self.D_optim, self.D_real_match_loss, self.D_real_mismatch_loss, self.D_synthetic_loss,
                     self.D_loss, self.D_merged_summ],
                    feed_dict={
                        self.model.inputs: images,
                        self.model.wrong_inputs: wrong_images,
                        self.model.phi_inputs: embed,
                        self.model.z: batch_z
                    })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, err_g, summary_str = self.sess.run([self.G_optim, self.G_loss, self.G_merged_summ],
                                                      feed_dict={self.model.z: batch_z, self.model.phi_inputs: embed})
                self.writer.add_summary(summary_str, counter)

                # # Run G_optim twice to make sure that d_loss does not go to zero (different from paper)
                # _, err_g, summary_str = self.sess.run([G_optim, self.G_loss, self.G_merged_summ],
                #                                       feed_dict={self.z: batch_z, self.phi_inputs: embed})
                # self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, updates_per_epoch,
                         time.time() - start_time, err_d, err_g))

                if np.mod(counter, 100) == 1:
                    try:
                        samples = self.sess.run(self.model.sampler,
                                                feed_dict={
                                                            self.model.z_sample: sample_z,
                                                            self.model.phi_sample: sample_embed,
                                                          })
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/{}/train_{:02d}_{:04d}.png'.format(self.config.sample_dir, 'GANCLS', epoch,
                                                                             idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (err_d, err_g))
                    except Exception as excep:
                        print("one pic error!...")
                        print(excep)

                if np.mod(counter, 500) == 2:
                    self.save(self.config.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}/{}_{}_{}".format(
            self.model.name, self.dataset.dataset_name, self.model.batch_size, self.model.output_size
        )

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model.name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
