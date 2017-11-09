from __future__ import division
import os
import time

from gancls.ops import *
from gancls.utils import *
from random import randint


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class GANCLS(object):
    def __init__(self, sess, crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64, z_dim=100, c_phi_dim=128,
                 phi_dim=1024, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, dataset=None):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          phi_dim: (optional) Dimension of the text embedding used to condition the GAN.
          c_phi_dim: (optional) The dimension of the compressed embedding which is appended to the z vector.
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.dataset = dataset
        self.crop = crop

        self.batch_size = batch_size
        self.batch_size = batch_size
        self.sample_num = sample_num

        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim
        self.phi_dim = phi_dim
        self.c_phi_dim = c_phi_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        self.c_dim = 3

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        image_dims = self.dataset.image_shape

        # Define the input tensor by appending the batch size dimension to the image dimension
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.wrong_inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='wrong_images')
        self.phi_inputs = tf.placeholder(tf.float32, [self.batch_size] + [self.phi_dim], name='phi_inputs')

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.z_sample =  tf.placeholder(tf.float32, [self.sample_num] + [self.z_dim], name='z_sample')
        self.phi_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.phi_dim], name='phi_sample')

        self.G = self.generator(self.z, self.phi_inputs)
        self.D_synthetic, self.D_synthetic_logits = self.discriminator(self.G, self.phi_inputs, reuse=False)
        self.D_real_match, self.D_real_match_logits = self.discriminator(self.inputs, self.phi_inputs, reuse=True)
        self.D_real_mismatch, self.D_real_mismatch_logits = self.discriminator(self.wrong_inputs, self.phi_inputs, reuse=True)

        self.sampler = self.sampler(self.z_sample, self.phi_sample)

        self.D_synthetic_summ = histogram_summary('d_synthetic_sum', self.D_synthetic)
        self.D_real_match_summ = histogram_summary('d_real_match_sum', self.D_real_match)
        self.D_real_mismatch_summ = histogram_summary('d_real_mismatch_sum', self.D_real_mismatch)
        self.G_summ = image_summary("g_sum", self.G)

        self.D_synthetic_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_synthetic_logits,
                                                    labels=tf.zeros_like(self.D_synthetic)))
        self.D_real_match_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_match_logits,
                                                    labels=tf.fill(self.D_real_match.get_shape(), 0.9)))
        self.D_real_mismatch_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_mismatch_logits,
                                                    labels=tf.zeros_like(self.D_real_mismatch)))
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_synthetic_logits,
                                                    labels=tf.ones_like(self.D_synthetic)))

        self.D_synthetic_loss_summ = histogram_summary('d_synthetic_sum_loss', self.D_synthetic_loss)
        self.D_real_match_loss_summ = histogram_summary('d_real_match_sum_loss', self.D_real_match_loss)
        self.D_real_mismatch_loss_summ = histogram_summary('d_real_mismatch_sum_loss', self.D_real_mismatch_loss)

        alpha = 0.5
        self.D_loss = self.D_real_match_loss + alpha * self.D_real_mismatch_loss + (1.0 - alpha) * self.D_synthetic_loss

        self.G_loss_summ = scalar_summary("g_loss", self.G_loss)
        self.D_loss_summ = scalar_summary("d_loss", self.D_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        D_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.D_loss, var_list=self.d_vars)
        G_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.G_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.G_merged_summ = merge_summary([self.z_sum, self.G_summ])
        self.D_merged_summ = merge_summary([self.z_sum, self.D_real_mismatch_summ, self.D_real_match_summ,
                                           self.D_synthetic_summ])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        # TODO: There is a bug which enforces the sample num to be the bath size.
        # TODO: Find out why the Images which are generated are not consistent
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        _, sample_phi, _, captions = self.dataset.test.next_batch_test(self.sample_num,
                                                                       randint(0, self.dataset.test.num_examples), 1)
        sample_phi = np.squeeze(sample_phi, axis=0)
        print(sample_phi.shape)

        # Display the captions of the sampled images
        print('\nCaptions of the sampled images:')
        for caption_idx, caption_batch in enumerate(captions):
            print('{}: {}'.format(caption_idx + 1, caption_batch[0]))
        print()

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(config.epoch):
            # Updates per epoch are given by the training data size / batch size
            updates_per_epoch = self.dataset.train.num_examples // self.batch_size

            for idx in range(0, updates_per_epoch):
                images, wrong_images, phi, _, _ = self.dataset.train.next_batch(self.batch_size, 4)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, err_d_real_match, err_d_real_mismatch, err_d_fake, err_d, summary_str = self.sess.run(
                    [D_optim, self.D_real_match_loss, self.D_real_mismatch_loss, self.D_synthetic_loss,
                     self.D_loss, self.D_merged_summ],
                    feed_dict={
                        self.inputs: images,
                        self.wrong_inputs: wrong_images,
                        self.phi_inputs: phi,
                        self.z: batch_z
                    })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, err_g, summary_str = self.sess.run([G_optim, self.G_loss, self.G_merged_summ],
                                                      feed_dict={self.z: batch_z, self.phi_inputs: phi})
                self.writer.add_summary(summary_str, counter)

                # # Run G_optim twice to make sure that d_loss does not go to zero (different from paper)
                # _, err_g, summary_str = self.sess.run([G_optim, self.G_loss, self.G_merged_summ],
                #                                       feed_dict={self.z: batch_z, self.phi_inputs: phi})
                # self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, updates_per_epoch,
                         time.time() - start_time, err_d, err_g))

                if np.mod(counter, 100) == 1:
                    try:
                        samples = self.sess.run(self.sampler, feed_dict={self.z_sample: sample_z,
                                                                         self.phi_sample: sample_phi,
                                                                         })
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, 'GANCLS', epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (err_d, err_g))
                    except Exception as excep:
                        print("one pic error!...")
                        print(excep)

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, phi, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # Compress the conditional phi vector using a fully connected layer
            d_fc_phi_w = tf.get_variable('d_fc_phi_w', [self.phi_dim, self.c_phi_dim],
                                         initializer=tf.random_normal_initializer(stddev=0.02))
            d_fc_phi_b = tf.get_variable('d_fc_phi_b', [self.c_phi_dim],
                                         initializer=tf.random_normal_initializer(stddev=0.02))
            c_phi = lrelu(tf.matmul(phi, d_fc_phi_w) + d_fc_phi_b, name='d_c_phi')

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))

            # Concatenate in depth the compressed text feature vectors to the last hidden layer
            concat_w = h3.get_shape().as_list()[1]
            concat_h = h3.get_shape().as_list()[1]
            depth_c_phi = tf.reshape(c_phi, [self.batch_size, concat_w, concat_h, -1])
            concat_h3_phi = tf.concat([h3, depth_c_phi], 3, name='d_concat_h3_phi')

            h4 = linear(tf.reshape(concat_h3_phi, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, phi):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # Compress the conditional phi vector using a fully connected layer
            g_fc_phi_w = tf.get_variable('g_fc_phi_w', [self.phi_dim, self.c_phi_dim],
                                         initializer=tf.random_normal_initializer(stddev=0.02))
            g_fc_phi_b = tf.get_variable('g_fc_phi_b', [self.c_phi_dim],
                                         initializer=tf.random_normal_initializer(stddev=0.02))
            c_phi = lrelu(tf.matmul(phi, g_fc_phi_w) + g_fc_phi_b, name='g_c_phi')

            # Append the compressed phi vector to the z noise vector
            z_concat = tf.concat([z, c_phi], 1, name='g_z_concat')

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z_concat, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, z, phi):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # Compress the conditional phi vector using a fully connected layer
            g_fc_phi_w = tf.get_variable('g_fc_phi_w', [self.phi_dim, self.c_phi_dim],
                                         initializer=tf.random_normal_initializer(stddev=0.02))
            g_fc_phi_b = tf.get_variable('g_fc_phi_b', [self.c_phi_dim],
                                         initializer=tf.random_normal_initializer(stddev=0.02))
            c_phi = lrelu(tf.matmul(phi, g_fc_phi_w) + g_fc_phi_b, name='g_c_phi')

            # Append the compressed phi vector to the z noise vector
            z_concat = tf.concat([z, c_phi], 1, name='g_z_concat')

            # project `z` and reshape
            h0 = tf.reshape(linear(z_concat, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                            [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

            print(tf.shape(h4))
            return tf.nn.tanh(h4)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "GANCLS.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
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
