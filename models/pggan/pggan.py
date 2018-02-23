import tensorflow as tf
from utils.ops import lrelu_act, conv2d, fc, upscale, pix_norm, pool, conv2d_transpose
from utils.utils import save_images, image_manifold_size, show_all_variables
import numpy as np
from scipy import ndimage


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
        self.output_size = 4 * pow(2, stage - 1)
        self.images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.sample_size])
        self.alpha_tra = tf.Variable(initial_value=0.0, trainable=False, name='alpha_tra')

    def build_model_PGGan(self):

        self.fake_images = self.generator(self.z, stages=self.stage, t=self.trans, alpha_trans=self.alpha_tra)

        _, self.D_pro_logits = self.discriminator(self.images, reuse=False, stages=self.stage, t=self.trans,
                                                  alpha_trans=self.alpha_tra)
        _, self.G_pro_logits = self.discriminator(self.fake_images, reuse=True, stages=self.stage, t=self.trans,
                                                  alpha_trans=self.alpha_tra)

        show_all_variables()
        # the defination of loss for D and G
        self.D_loss = tf.reduce_mean(self.G_pro_logits) - tf.reduce_mean(self.D_pro_logits)
        self.G_loss = -tf.reduce_mean(self.G_pro_logits)

        # gradient penalty from WGAN-GP
        self.differences = self.fake_images - self.images
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = self.images + (self.alpha * self.differences)
        _, discri_logits = self.discriminator(interpolates, reuse=True, stages=self.stage, t=self.trans,
                                              alpha_trans=self.alpha_tra)
        gradients = tf.gradients(discri_logits, [interpolates])[0]

        ##2 norm
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean(tf.square(tf.maximum(0.0, slopes - 1.)))
        tf.summary.scalar("gp_loss", self.gradient_penalty)

        self.D_origin_loss = self.D_loss

        self.D_loss += 10 * self.gradient_penalty
        self.D_loss += 0.001 * tf.reduce_mean(tf.square(self.D_pro_logits - 0.0))

        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))

        self.d_vars = tf.trainable_variables('d_net')
        self.g_vars = tf.trainable_variables('g_net')

        # variables to save
        self.saver = tf.train.Saver(self.get_variables_up_to_stage(self.stage), max_to_keep=2)
        # variables to restore
        self.restore = None
        if self.stage > 1:
            self.restore = tf.train.Saver(self.get_variables_up_to_stage(self.stage - 1))

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    # do train
    def train(self):

        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_tra_assign = self.alpha_tra.assign(step_pl / self.max_iters)

        opti_D = tf.train.AdamOptimizer(learning_rate=3e-4, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.0, beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.stage != 1 and self.stage != 7:

                if self.trans:
                    self.restore.restore(sess, self.read_model_path)
                else:
                    self.saver.restore(sess, self.read_model_path)

            step = 0
            batch_num = 0
            while step <= self.max_iters:

                # optimization D
                n_critic = 1
                if self.stage == 5 and self.trans:
                    n_critic = 1

                img = None
                for i in range(n_critic):

                    sample_z = np.random.normal(size=[self.batch_size, self.sample_size])
                    img, wrong_img, embed, _, _ = self.dataset.train.next_batch(self.batch_size, 4, embeddings=True,
                                                                                wrong_img=True)

                    sess.run(opti_D, feed_dict={self.images: img, self.z: sample_z})
                    batch_num += 1

                # optimization G
                sess.run(opti_G, feed_dict={self.z: sample_z})

                summary_str = sess.run(summary_op, feed_dict={self.images: img, self.z: sample_z})
                summary_writer.add_summary(summary_str, step)
                # the alpha of fake_in process
                sess.run(alpha_tra_assign, feed_dict={step_pl: step})

                if step % 400 == 0:
                    D_loss, G_loss, D_origin_loss, alpha_tra = sess.run(
                        [self.D_loss, self.G_loss, self.D_origin_loss, self.alpha_tra],
                        feed_dict={self.images: img, self.z: sample_z})
                    print("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (
                        self.stage, step, D_loss, G_loss, D_origin_loss, alpha_tra))

                    fake_images = sess.run(self.fake_images, feed_dict={self.z: sample_z})
                    fake_images = np.clip(fake_images, -1, 1)
                    save_images(fake_images, image_manifold_size(fake_images.shape[0]),
                                '{}train_{:02d}.png'.format(self.sample_path, step))

                if np.mod(step, 4000) == 0 and step != 0:
                    self.saver.save(sess, self.gan_model_path)
                step += 1

            save_path = self.saver.save(sess, self.gan_model_path)
            print("Model saved in file: %s" % save_path)

        tf.reset_default_graph()

    def discriminator(self, inp, reuse=False, stages=1, t=False, alpha_trans=tf.Variable(0, trainable=False)):

        # dis_as_v = []
        with tf.variable_scope("d_net", reuse=reuse):
            conv_iden = None
            if t:
                conv_iden = pool(inp, 2)
                # From RGB
                conv_iden = conv2d(conv_iden, f=self.get_nf(stages - 2), ks=(1, 1), s=(1, 1), act=lrelu_act(),
                                   name=self.get_rgb_name(stages - 2))

            conv = conv2d(inp, f=self.get_nf(stages - 1), ks=(1, 1), s=(1, 1), act=lrelu_act(),
                          name=self.get_rgb_name(stages - 1))

            for i in range(stages - 1, 0):

                with tf.variable_scope(self.get_conv_scope_name(i), reuse=reuse):
                    conv = conv2d(conv, f=self.get_nf(i), ks=(3, 3), s=(1, 1), act=lrelu_act())
                    conv = conv2d(conv, f=self.get_nf(i - 1), ks=(3, 3), s=(1, 1), act=lrelu_act())
                conv = pool(conv, 2)
                if i == stages - 1 and t:
                    conv = tf.multiply(alpha_trans, conv) + tf.multiply(tf.subtract(1., alpha_trans), conv_iden)

            with tf.variable_scope(self.get_conv_scope_name(0), reuse=reuse):
                # conv = MinibatchstateConcat(conv)
                conv = conv2d(conv, f=self.get_nf(1), ks=(3, 3), s=(1, 1), act=lrelu_act())
                conv = conv2d(conv, f=self.get_nf(1), ks=(4, 4), s=(1, 1), act=lrelu_act(), padding='VALID')

                conv = tf.reshape(conv, [self.batch_size, -1])

                # for D
                output = fc(conv, units=1)

            return tf.nn.sigmoid(output), output

    def generator(self, z_var, stages=1, t=False, alpha_trans=tf.Variable(0., trainable=False)):

        with tf.variable_scope('g_net'):

            with tf.variable_scope(self.get_conv_scope_name(0)):
                de = tf.reshape(z_var, [self.batch_size, 1, 1, self.get_nf(1)])
                de = conv2d_transpose(de, f=self.get_nf(1), ks=(4, 4), s=(1, 1), act=lrelu_act(), padding='VALID')
                de = pix_norm(de)
                de = tf.reshape(de, [self.batch_size, 4, 4, tf.cast(self.get_nf(1), tf.int32)])
                de = conv2d(de, f=self.get_nf(1), ks=(3, 3), s=(1, 1), act=lrelu_act())
                de = pix_norm(de)

            de_iden = None
            for i in range(1, stages):

                de = upscale(de, 2)

                if i == stages - 1 and t:
                    # To RGB
                    de_iden = conv2d(de, f=3, ks=(1, 1), s=(1, 1), name=self.get_trans_rgb_name(i))

                with tf.variable_scope(self.get_conv_scope_name(i)):
                    de = pix_norm(conv2d(de, f=self.get_nf(i + 1), ks=(3, 3), s=(1, 1), act=lrelu_act()))
                    de = pix_norm(conv2d(de, f=self.get_nf(i + 1), ks=(3, 3), s=(1, 1), act=lrelu_act()))

            # To RGB
            de = conv2d(de, f=3, ks=(1, 1), s=(1, 1), name=self.get_rgb_name(stages - 1))

            if stages == 1:
                return de
            if t:
                de = tf.multiply(tf.subtract(1, alpha_trans), de_iden) + tf.multiply(alpha_trans, de)

            return de

    def get_trans_rgb_name(self, stage):
        return 'rgb_trans_stage_%d' % stage

    def get_rgb_name(self, stage):
        return 'rgb_stage_%d' % stage

    def get_conv_scope_name(self, stage):
        return 'conv_stage_%d' % stage

    def get_nf(self, stage):
        return min(1024 // (2 ** (stage * 1)), 512)

    def get_variables_up_to_stage(self, stages):
        d_rgb_name = 'd_net/%s' % self.get_rgb_name(stages - 1)
        g_rgb_name = 'g_net/%s' % self.get_rgb_name(stages - 1)
        d_vars_to_save = [var for var in tf.global_variables('d_net') if var.name.startswith(d_rgb_name)]
        g_vars_to_save = [var for var in tf.global_variables('g_net') if var.name.startswith(g_rgb_name)]
        for stage in range(stages):
            d_vars_to_save += tf.global_variables('d_net/%s' % self.get_conv_scope_name(stage))
            g_vars_to_save += tf.global_variables('g_net/%s' % self.get_conv_scope_name(stage))
        return d_vars_to_save + g_vars_to_save








