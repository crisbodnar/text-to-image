import tensorflow as tf
import time

from utils.ops import lrelu_act, conv2d, fc, upscale, pix_norm, pool, conv2d_transpose, layer_norm, layer_norm
from utils.utils import save_images, image_manifold_size, show_all_variables, save_captions, print_vars
from utils.saver import load, save
import numpy as np
import sys


class PGGAN(object):

    # build model
    def __init__(self, batch_size, max_iters, model_path, read_model_path, data, sample_size, sample_path, log_dir,
                 learn_rate, stage, t):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.gan_model_path = model_path
        self.read_model_path = read_model_path
        self.dataset = data
        self.sample_size = sample_size
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learn_rate
        self.stage = stage
        self.trans = t
        self.log_vars = []
        self.channel = 3
        self.sample_num = 64
        self.embed_dim = 1024
        self.output_size = 4 * pow(2, stage - 1)
        self.alpha_tra = tf.Variable(initial_value=0.0, trainable=False, name='alpha_tra')

        self.build_model()
        self.define_losses()
        self.define_summaries()

    def build_model(self):
        # Define the input tensor by appending the batch size dimension to the image dimension
        self.iter = tf.placeholder(tf.int32, shape=None)
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel], name='x')
        self.x_mismatch = tf.placeholder(tf.float32,
                                         [self.batch_size, self.output_size, self.output_size, self.channel],
                                         name='x_mismatch')
        self.cond = tf.placeholder(tf.float32, [self.batch_size, self.embed_dim], name='cond')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.sample_size], name='z')
        self.epsilon = tf.placeholder(tf.float32, [self.batch_size, 1, 1, 1], name='eps')

        self.z_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.sample_size], name='z_sample')
        self.cond_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.embed_dim], name='cond_sample')

        self.G = self.generator(self.z, self.cond, stages=self.stage, t=self.trans)
        self.Dg_logit, self.Dgm_logit = self.discriminator(self.G, self.cond, reuse=False, stages=self.stage, t=self.trans)
        self.Dx_logit, self.Dxma_logit = self.discriminator(self.x, self.cond, reuse=True, stages=self.stage, t=self.trans)
        _, self.Dxm_logit = self.discriminator(self.x_mismatch, self.cond, reuse=True, stages=self.stage, t=self.trans)

        self.x_hat = self.epsilon * self.G + (1. - self.epsilon) * self.x
        self.Dx_hat_logit, _ = self.discriminator(self.x_hat, self.cond, reuse=True, stages=self.stage, t=self.trans)

        self.sampler = self.generator(self.z_sample, self.cond_sample, reuse=True, stages=self.stage, t=self.trans,
                                      is_train=False)
        self.alpha_assign = tf.assign(self.alpha_tra,
                                      (tf.cast(tf.cast(self.iter, tf.float32) / self.max_iters, tf.float32)))

        self.d_vars = tf.trainable_variables('d_net')
        self.g_vars = tf.trainable_variables('g_net')

        show_all_variables()

    def define_losses(self):
        self.D_loss_real_match = -tf.reduce_mean(self.Dx_logit)
        self.D_loss_fake = tf.reduce_mean(self.Dg_logit)
        self.Dm_loss = \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dxm_logit,
                                                                   labels=tf.zeros_like(self.Dxm_logit))) \
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dxma_logit,
                                                                     labels=tf.ones_like(self.Dxma_logit)))
        self.Gm_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dgm_logit,
                                                                              labels=tf.ones_like(self.Dgm_logit)))
        self.D_logits_reg = 0.5 * tf.reduce_mean(tf.square(self.Dx_logit)) \
                            + 0.5 * tf.reduce_mean(tf.square(self.Dxm_logit))

        self.dm_coeff = 1
        self.gm_coeff = 0.1

        grad_Dx_hat = tf.gradients(self.Dx_hat_logit, [self.x_hat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_Dx_hat), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean(tf.square(tf.maximum(0., slopes - 1.)))

        self.D_loss = (self.D_loss_real_match + self.D_loss_fake) + 10.0 * self.gradient_penalty
        self.D_loss += self.dm_coeff * self.Dm_loss
        # self.D_loss += 0.0001 * self.D_logits_reg

        self.G_loss = -self.D_loss_fake + self.gm_coeff * self.Gm_loss

        self.D_optimizer = tf.train.AdamOptimizer(3e-4, beta1=0.0, beta2=0.9)
        self.G_optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.0, beta2=0.9)

        self.D_optim = self.D_optimizer.minimize(self.D_loss, var_list=self.d_vars)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops + [self.alpha_assign]):
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

            tf.summary.scalar('G_loss_wass', -self.D_loss_fake),
            tf.summary.scalar('Gm_loss', self.Gm_loss),
            tf.summary.scalar('G_loss', self.G_loss),
            tf.summary.scalar('alpha', self.alpha_tra),

            tf.summary.scalar('D_loss_real_match', self.D_loss_real_match),
            tf.summary.scalar('D_loss_fake', self.D_loss_fake),
            tf.summary.scalar('D_grad_penalty', self.gradient_penalty),
            tf.summary.scalar('D_logits_reg', self.D_logits_reg),
            tf.summary.scalar('neg_d_loss', -self.D_loss),
            tf.summary.scalar('D_loss', self.D_loss),
            tf.summary.scalar('Dm_loss', self.Dm_loss),
        ])

    # do train
    def train(self):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.stage != 1 and self.stage != 7:
                if self.trans:
                    load(self.restore, sess, self.read_model_path)
                else:
                    load(self.saver, sess, self.read_model_path)

            sample_z = np.random.normal(0, 1, (self.sample_num, self.sample_size))
            _, sample_cond, _, captions = self.dataset.test.next_batch_test(self.sample_num, 0, 1)
            sample_cond = np.squeeze(sample_cond, axis=0)
            print('Conditionals sampler shape: {}'.format(sample_cond.shape))

            save_captions(self.sample_path, captions)
            start_time = time.time()

            start_point = 0
            for idx in range(start_point + 1, self.max_iters):
                epoch_size = self.dataset.train.num_examples // self.batch_size
                epoch = idx // epoch_size

                images, wrong_images, embed, _, _ = self.dataset.train.next_batch(self.batch_size, 4,
                                                                                  wrong_img=True,
                                                                                  embeddings=True)
                batch_z = np.random.normal(0, 1, (self.batch_size, self.sample_size))
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

                # Use TTUR update rule (https://arxiv.org/abs/1706.08500)
                _, err_g = sess.run([self.G_optim, self.G_loss], feed_dict=feed_dict)

                if np.mod(idx, 20) == 0:
                    summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, idx)

                    print("Epoch: [%2d] [%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                          % (epoch, idx, time.time() - start_time, err_d, err_g))

                if np.mod(idx, 500) == 0:
                    try:
                        samples = sess.run(self.sampler, feed_dict={
                                                    self.z_sample: sample_z,
                                                    self.cond_sample: sample_cond,
                                                })
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    '{}train_{:02d}_{:04d}.png'.format(self.sample_path, epoch, idx))

                    except Exception as e:
                        print("Failed to generate sample image")
                        print(type(e))
                        print(e.args)
                        print(e)

                if np.mod(idx, 500) == 0 or idx == self.max_iters - 1:
                    save(self.saver, sess, self.gan_model_path, idx)
                sys.stdout.flush()

        tf.reset_default_graph()

    def discriminator(self, inp, cond, stages, t, reuse=False):
        alpha_trans = self.alpha_tra
        with tf.variable_scope("d_net", reuse=reuse):
            conv_iden = None
            if t:
                conv_iden = pool(inp, 2)
                conv_iden = self.from_rgb(conv_iden, stages - 2)

            conv = self.from_rgb(inp, stages - 1)

            for i in range(stages - 1, 0, -1):
                with tf.variable_scope(self.get_conv_scope_name(i), reuse=reuse):
                    conv = conv2d(conv, f=self.get_nf(i), ks=(3, 3), s=(1, 1))
                    conv = layer_norm(conv, act=lrelu_act())
                    conv = conv2d(conv, f=self.get_nf(i-1), ks=(3, 3), s=(1, 1))
                    conv = layer_norm(conv, act=lrelu_act())
                    conv = pool(conv, 2)
                if i == stages - 1 and t:
                    conv = tf.multiply(alpha_trans, conv) + tf.multiply(tf.subtract(1., alpha_trans), conv_iden)

            with tf.variable_scope(self.get_conv_scope_name(0), reuse=reuse):
                concat = self.concat_cond(conv, cond)

                # Real/False branch
                conv_b1 = conv2d(concat, f=self.get_nf(0), ks=(3, 3), s=(1, 1))
                conv_b1 = layer_norm(conv_b1, act=lrelu_act())
                conv_b1 = conv2d(conv_b1, f=self.get_nf(0), ks=(4, 4), s=(1, 1), padding='VALID')
                conv_b1 = layer_norm(conv_b1, act=lrelu_act())
                conv_b1 = tf.reshape(conv_b1, [-1, self.get_nf(0)])
                output_b1 = fc(conv_b1, units=1)

                # Match/Mismatch branch
                conv_b2 = conv2d(concat, f=self.get_nf(0), ks=(3, 3), s=(1, 1))
                conv_b2 = layer_norm(conv_b2, act=lrelu_act())
                conv_b2 = conv2d(conv_b2, f=self.get_nf(0), ks=(4, 4), s=(1, 1), padding='VALID')
                conv_b2 = layer_norm(conv_b2, act=lrelu_act())
                conv_b2 = tf.reshape(conv_b2, [-1, self.get_nf(0)])
                output_b2 = fc(conv_b2, units=1)

            return output_b1, output_b2

    def generator(self, z_var, cond, stages, t, reuse=False, is_train=True):
        alpha_trans = self.alpha_tra
        with tf.variable_scope('g_net', reuse=reuse):

            with tf.variable_scope(self.get_conv_scope_name(0), reuse=reuse):
                de = tf.reshape(z_var, [-1, 1, 1, self.get_nf(0)])
                de = conv2d_transpose(de, f=self.get_nf(0), ks=(4, 4), s=(1, 1), act=lrelu_act(), padding='VALID')
                de = self.concat_cond(de, cond)
                de = conv2d(de, f=self.get_nf(0), ks=(3, 3), s=(1, 1))
                de = pix_norm(de, act=lrelu_act())

            de_iden = None
            for i in range(1, stages):

                if (i == stages - 1) and t:
                    # To RGB
                    de_iden = self.to_rgb(de, stages - 2)
                    de_iden = upscale(de_iden, 2)

                with tf.variable_scope(self.get_conv_scope_name(i), reuse=reuse):
                    de = upscale(de, 2)
                    de = conv2d(de, f=self.get_nf(i), ks=(3, 3), s=(1, 1))
                    de = pix_norm(de, act=lrelu_act())
                    de = conv2d(de, f=self.get_nf(i), ks=(3, 3), s=(1, 1))
                    de = pix_norm(de, act=lrelu_act())

            de = self.to_rgb(de, stages - 1)

            if stages == 1:
                return de
            if t:
                de = tf.multiply(tf.subtract(1., alpha_trans), de_iden) + tf.multiply(alpha_trans, de)

            return de

    def concat_cond(self, x, cond):
        cond_compress = fc(cond, units=128, act=lrelu_act())
        cond_compress = tf.reshape(cond_compress, shape=[-1, 4, 4, 8])
        x = tf.concat([x, cond_compress], axis=3)
        return x

    def get_rgb_name(self, stage):
        return 'rgb_stage_%d' % stage

    def get_conv_scope_name(self, stage):
        return 'conv_stage_%d' % stage

    def get_nf(self, stage):
        return min(1024 // (2 ** (stage * 1)), 512)

    def from_rgb(self, x, stage):
        with tf.variable_scope(self.get_rgb_name(stage)):
            return conv2d(x, f=self.get_nf(stage), ks=(1, 1), s=(1, 1), act=lrelu_act())

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








