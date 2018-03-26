import tensorflow as tf
import time

from utils.ops import lrelu_act, conv2d, fc, upscale, pool, conv2d_transpose, layer_norm
from utils.utils import save_images, get_balanced_factorization, show_all_variables, save_captions, print_vars, \
    initialize_uninitialized
from utils.saver import load, save
import numpy as np
import sys


class PGGAN(object):

    # build model
    def __init__(self, batch_size, steps, check_dir_write, check_dir_read, dataset, sample_path, log_dir, stage, trans,
                 build_model=True):

        self.batch_size = batch_size
        self.steps = steps
        self.check_dir_write = check_dir_write
        self.check_dir_read = check_dir_read
        self.dataset = dataset
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.stage = stage
        self.trans = trans

        self.z_dim = 512
        self.embed_dim = 1024
        self.out_size = 4 * pow(2, stage - 1)
        self.channel = 3
        self.sample_num = 64
        self.alpha = tf.Variable(initial_value=0.0, trainable=False, name='alpha')

        if build_model:
            self.build_model()
            self.define_losses()
            self.define_summaries()

    def build_model(self):
        # Define the input tensor by appending the batch size dimension to the image dimension
        self.iter = tf.placeholder(tf.int32, shape=None)
        self.x = tf.placeholder(tf.float32, [None, self.out_size, self.out_size, self.channel], name='x')
        self.x_mismatch = tf.placeholder(tf.float32,
                                         [None, self.out_size, self.out_size, self.channel],
                                         name='x_mismatch')
        self.cond = tf.placeholder(tf.float32, [None, self.embed_dim], name='cond')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.epsilon = tf.placeholder(tf.float32, [None, 1, 1, 1], name='eps')

        self.z_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.z_dim], name='z_sample')
        self.cond_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.embed_dim], name='cond_sample')

        self.G, self.embed_mean, self.embed_log_sigma \
            = self.generator(self.z, self.cond, stages=self.stage, t=self.trans)
        self.Dg_logit, self.Dgm_logit = self.discriminator(self.G, self.cond, reuse=False, stages=self.stage,
                                                           t=self.trans)
        self.Dx_logit, self.Dxma_logit = self.discriminator(self.x, self.cond, reuse=True, stages=self.stage,
                                                            t=self.trans)
        _, self.Dxmi_logit = self.discriminator(self.x_mismatch, self.cond, reuse=True, stages=self.stage, t=self.trans)

        self.sampler, _, _ = self.generator(self.z_sample, self.cond_sample, reuse=True, stages=self.stage,
                                            t=self.trans)
        self.alpha_assign = tf.assign(self.alpha,
                                      (tf.cast(tf.cast(self.iter, tf.float32) / self.steps, tf.float32)))

        self.d_vars = tf.trainable_variables('d_net')
        self.g_vars = tf.trainable_variables('g_net')

        show_all_variables()

    def get_gradient_penalty(self, x, y):
        grad_y = tf.gradients(y, [x])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_y), reduction_indices=[1, 2, 3]))
        return tf.reduce_mean(tf.square(slopes - 1.))

    def ce_loss(self, logits, label_val):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                      labels=tf.fill(logits.get_shape(), label_val)))

    def define_losses(self):
        self.D_real_match_loss = self.ce_loss(self.Dxma_logit, 1.0)
        self.D_real_mismatch_loss = self.ce_loss(self.Dxmi_logit, 0.0)
        self.D_g_match_loss = self.ce_loss(self.Dgm_logit, 0.0)

        self.D_loss_real = tf.reduce_mean(tf.square(self.Dx_logit - 1))
        self.D_loss_fake = tf.reduce_mean(tf.square(self.Dg_logit))

        self.D_realism_loss = self.D_loss_real + self.D_loss_fake
        self.D_matching_loss = self.D_real_match_loss + self.D_real_mismatch_loss + self.D_g_match_loss

        self.D_loss = self.D_realism_loss + self.D_matching_loss

        self.G_kl_loss = self.kl_std_normal_loss(self.embed_mean, self.embed_log_sigma)
        self.G_gan_loss = tf.reduce_mean(tf.square(self.Dg_logit - 0.5))
        self.G_match_loss = self.ce_loss(self.Dgm_logit, 1.0)

        self.kl_coeff = 1.0
        self.G_loss = self.G_gan_loss + 0.5 * self.G_match_loss + self.kl_coeff * self.G_kl_loss

        self.D_optimizer = tf.train.AdamOptimizer(0.00005, beta1=0.0, beta2=0.9)
        self.G_optimizer = tf.train.AdamOptimizer(0.00005, beta1=0.0, beta2=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([self.alpha_assign] + update_ops):
            self.D_optim = self.D_optimizer.minimize(self.D_loss, var_list=self.d_vars)
            self.G_optim = self.G_optimizer.minimize(self.G_loss, var_list=self.g_vars)

        # variables to save
        vars_to_save = self.get_variables_up_to_stage(self.stage)
        print('Length of the vars to save: %d' % len(vars_to_save))
        print('\n\nVariables to save:')
        print_vars(vars_to_save)
        self.saver = tf.train.Saver(vars_to_save, max_to_keep=2)

        # variables to restore
        self.restore = None
        if self.stage > 1 and self.trans:
            vars_to_restore = self.get_variables_up_to_stage(self.stage - 1)
            print('Length of the vars to restore: %d' % len(vars_to_restore))
            print('\n\nVariables to restore:')
            print_vars(vars_to_restore)
            self.restore = tf.train.Saver(vars_to_restore)

    def define_summaries(self):
        self.summary_op = tf.summary.merge([
            tf.summary.image('x', self.x),
            tf.summary.image('G_img', self.G),

            tf.summary.histogram('z', self.z),
            tf.summary.histogram('z_sample', self.z_sample),

            tf.summary.scalar('Gm_loss', self.G_match_loss),
            tf.summary.scalar('G_gan_loss', self.G_gan_loss),
            tf.summary.scalar('G_loss', self.G_loss),
            tf.summary.scalar('alpha', self.alpha),
            tf.summary.scalar('kl_loss', self.G_kl_loss),

            tf.summary.scalar('D_syntehtic_loss', self.D_loss_fake),
            tf.summary.scalar('D_loss_real', self.D_loss_real),
            tf.summary.scalar('D_real_match_loss', self.D_real_match_loss),
            tf.summary.scalar('D_real_mismatch_loss', self.D_real_mismatch_loss),
            # tf.summary.scalar('D_realism_loss', self.D_realism_loss),
            # tf.summary.scalar('D_matching_loss', self.D_matching_loss),
            tf.summary.scalar('D_g_match_loss', self.D_g_match_loss),
            tf.summary.scalar('D_loss', self.D_loss)
        ])

    # do train
    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.stage != 1:
                if self.trans:
                    could_load = load(self.restore, sess, self.check_dir_read)
                    if not could_load:
                        raise RuntimeError('Could not load previous stage during transition')
                else:
                    could_load = load(self.saver, sess, self.check_dir_read)
                    if not could_load:
                        raise RuntimeError('Could not load current stage')

            # variables to init
            vars_to_init = initialize_uninitialized(sess)
            sess.run(tf.variables_initializer(vars_to_init))

            sample_z = np.random.normal(0, 1, (self.sample_num, self.z_dim))
            _, sample_cond, _, captions = self.dataset.test.next_batch_test(self.sample_num, 0, 1)
            sample_cond = np.squeeze(sample_cond, axis=0)
            print('Conditionals sampler shape: {}'.format(sample_cond.shape))

            save_captions(self.sample_path, captions)
            start_time = time.time()

            start_point = 0
            for idx in range(start_point + 1, self.steps):
                epoch_size = self.dataset.train.num_examples // self.batch_size
                epoch = idx // epoch_size

                images, wrong_images, embed, _, _ = self.dataset.train.next_batch(self.batch_size, 4,
                                                                                  wrong_img=True,
                                                                                  embeddings=True)
                batch_z = np.random.normal(0, 1, (self.batch_size, self.z_dim))
                eps = np.random.uniform(0., 1., size=(self.batch_size, 1, 1, 1))

                feed_dict = {
                    self.x: images,
                    self.x_mismatch: wrong_images,
                    self.cond: embed,
                    self.z: batch_z,
                    self.epsilon: eps,
                    self.z_sample: sample_z,
                    self.cond_sample: sample_cond,
                    self.iter: idx,
                }

                _, err_d = sess.run([self.D_optim, self.D_loss], feed_dict=feed_dict)
                _, err_g = sess.run([self.G_optim, self.G_loss], feed_dict=feed_dict)

                if np.mod(idx, 20) == 0:
                    summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, idx)

                    print("Epoch: [%2d] [%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                          % (epoch, idx, time.time() - start_time, err_d, err_g))

                if np.mod(idx, 2000) == 0:
                    try:
                        samples = sess.run(self.sampler, feed_dict={
                                                    self.z_sample: sample_z,
                                                    self.cond_sample: sample_cond})
                        samples = np.clip(samples, -1., 1.)
                        if self.out_size > 256:
                            samples = samples[:4]

                        save_images(samples, get_balanced_factorization(samples.shape[0]),
                                    '{}train_{:02d}_{:04d}.png'.format(self.sample_path, epoch, idx))

                    except Exception as e:
                        print("Failed to generate sample image")
                        print(type(e))
                        print(e.args)
                        print(e)

                if np.mod(idx, 2000) == 0 or idx == self.steps - 1:
                    save(self.saver, sess, self.check_dir_write, idx)
                sys.stdout.flush()

        tf.reset_default_graph()

    def discriminator(self, inp, cond, stages, t, reuse=False):
        alpha_trans = self.alpha
        with tf.variable_scope("d_net", reuse=reuse):
            x_iden = None
            if t:
                x_iden = pool(inp, 2)
                x_iden = self.from_rgb(x_iden, stages - 2)

            x = self.from_rgb(inp, stages - 1)

            for i in range(stages - 1, 0, -1):
                with tf.variable_scope(self.get_conv_scope_name(i), reuse=reuse):
                    x = conv2d(x, f=self.get_nf(i), ks=(3, 3), s=(1, 1))
                    x = layer_norm(x, act=lrelu_act())
                    x = conv2d(x, f=self.get_nf(i-1), ks=(3, 3), s=(1, 1))
                    x = layer_norm(x, act=lrelu_act())
                    x = pool(x, 2)
                if i == stages - 1 and t:
                    x = tf.multiply(alpha_trans, x) + tf.multiply(tf.subtract(1., alpha_trans), x_iden)

            with tf.variable_scope(self.get_conv_scope_name(0), reuse=reuse):
                # Real/False branch
                conv_b1 = conv2d(x, f=self.get_nf(0), ks=(3, 3), s=(1, 1))
                conv_b1 = layer_norm(conv_b1, act=lrelu_act())
                conv_b1 = conv2d(conv_b1, f=self.get_nf(0), ks=(4, 4), s=(1, 1), padding='VALID')
                output_b1 = fc(conv_b1, units=1)

                # Match/Mismatch branch
                concat = self.concat_cond(x, cond)
                conv_b2 = conv2d(concat, f=self.get_nf(0), ks=(3, 3), s=(1, 1))
                conv_b2 = layer_norm(conv_b2, act=lrelu_act())
                conv_b2 = conv2d(conv_b2, f=self.get_nf(0), ks=(4, 4), s=(1, 1), padding='VALID')
                output_b2 = fc(conv_b2, units=1)

            return output_b1, output_b2

    def generator(self, z, cond, stages, t, reuse=False, cond_noise=True):
        alpha_trans = self.alpha
        with tf.variable_scope('g_net', reuse=reuse):

            with tf.variable_scope(self.get_conv_scope_name(0), reuse=reuse):
                x = tf.reshape(z, [-1, 1, 1, self.get_nf(0)])
                x = conv2d_transpose(x, f=self.get_nf(0), ks=(4, 4), s=(1, 1), act=tf.nn.relu, padding='VALID')

                mean, log_sigma = self.generate_conditionals(cond)
                cond = self.sample_normal_conditional(mean, log_sigma, cond_noise)

                x = self.concat_cond(x, cond)
                x = conv2d(x, f=self.get_nf(0), ks=(3, 3), s=(1, 1))
                x = layer_norm(x, act=tf.nn.relu)

            x_iden = None
            for i in range(1, stages):

                if (i == stages - 1) and t:
                    x_iden = self.to_rgb(x, stages - 2)
                    x_iden = upscale(x_iden, 2)

                with tf.variable_scope(self.get_conv_scope_name(i), reuse=reuse):
                    x = upscale(x, 2)
                    x = conv2d(x, f=self.get_nf(i), ks=(3, 3), s=(1, 1))
                    x = layer_norm(x, act=tf.nn.relu)
                    x = conv2d(x, f=self.get_nf(i), ks=(3, 3), s=(1, 1))
                    x = layer_norm(x, act=tf.nn.relu)

            x = self.to_rgb(x, stages - 1)

            if t:
                x = tf.multiply(tf.subtract(1., alpha_trans), x_iden) + tf.multiply(alpha_trans, x)

            return x, mean, log_sigma

    def concat_cond(self, x, cond):
        cond_compress = fc(cond, units=128, act=lrelu_act())
        cond_compress = tf.expand_dims(tf.expand_dims(cond_compress, 1), 1)
        cond_compress = tf.tile(cond_compress, [1, 4, 4, 1])
        x = tf.concat([x, cond_compress], axis=3)
        return x

    def get_rgb_name(self, stage):
        return 'rgb_stage_%d' % stage

    def get_conv_scope_name(self, stage):
        return 'conv_stage_%d' % stage

    def get_nf(self, stage):
        return min(1024 // (2 ** stage) * 4, 512)

    def from_rgb(self, x, stage):
        with tf.variable_scope(self.get_rgb_name(stage)):
            return conv2d(x, f=self.get_nf(stage), ks=(1, 1), s=(1, 1), act=lrelu_act())

    def generate_conditionals(self, embeddings):
        """Takes the embeddings, compresses them and builds the statistics for a multivariate normal distribution"""
        mean = fc(embeddings, 128, act=lrelu_act())
        log_sigma = fc(embeddings, 128, act=lrelu_act())
        return mean, log_sigma

    def sample_normal_conditional(self, mean, log_sigma, cond_noise=True):
        if cond_noise:
            epsilon = tf.truncated_normal(tf.shape(mean))
            stddev = tf.exp(log_sigma)
            return mean + stddev * epsilon
        return mean

    def kl_std_normal_loss(self, mean, log_sigma):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mean))
        loss = tf.reduce_mean(loss)
        return loss

    def to_rgb(self, x, stage):
        with tf.variable_scope(self.get_rgb_name(stage)):
            return conv2d(x, f=3, ks=(1, 1), s=(1, 1))

    def get_adam_vars(self, opt, vars_to_train):
        opt_vars = [opt.get_slot(var, name) for name in opt.get_slot_names()
                    for var in vars_to_train
                    if opt.get_slot(var, name) is not None]
        opt_vars.extend(list(opt._get_beta_accumulators()))
        return opt_vars

    def get_variables_up_to_stage(self, stages):
        d_vars_to_save = tf.global_variables('d_net/%s' % self.get_rgb_name(stages - 1))
        g_vars_to_save = tf.global_variables('g_net/%s' % self.get_rgb_name(stages - 1))
        for stage in range(stages):
            d_vars_to_save += tf.global_variables('d_net/%s' % self.get_conv_scope_name(stage))
            g_vars_to_save += tf.global_variables('g_net/%s' % self.get_conv_scope_name(stage))
        return d_vars_to_save + g_vars_to_save








