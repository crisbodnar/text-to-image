import tensorflow as tf
from utils.ops import *


class WGanCls(object):
    def __init__(self, cfg, build_model=True):
        """
        Args:
          cfg: Config specifying all the parameters of the model.
        """

        self.cfg = cfg

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.sample_num = cfg.TRAIN.SAMPLE_NUM

        self.output_size = cfg.MODEL.OUTPUT_SIZE

        self.z_dim = cfg.MODEL.Z_DIM
        self.embed_dim = cfg.MODEL.EMBED_DIM
        self.compressed_embed_dim = cfg.MODEL.COMPRESSED_EMBED_DIM

        self.gf_dim = cfg.MODEL.GF_DIM
        self.df_dim = cfg.MODEL.DF_DIM
        
        self.image_dims = [cfg.MODEL.IMAGE_SHAPE.H, cfg.MODEL.IMAGE_SHAPE.W, cfg.MODEL.IMAGE_SHAPE.D]

        self.global_step = tf.Variable(0, trainable=False)

        if build_model:
            self.build_model()
            self.define_losses()

    def build_model(self):
        # Define the input tensor by appending the batch size dimension to the image dimension
        self.iter = tf.placeholder(tf.int32, shape=None)
        self.learning_rate_d = tf.placeholder(tf.float32, shape=None)
        self.learning_rate_g = tf.placeholder(tf.float32, shape=None)
        self.x = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='real_images')
        self.x_mismatch = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='real_images')
        self.cond = tf.placeholder(tf.float32, [self.batch_size] + [self.embed_dim], name='cond')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.epsilon = tf.placeholder(tf.float32, [self.batch_size, 1, 1, 1], name='eps')

        self.z_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.z_dim], name='z_sample')
        self.cond_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.embed_dim], name='cond_sample')

        self.G, self.embed_mean, self.embed_log_sigma = self.generator(self.z, self.cond, reuse=False)
        self.Dg_logit = self.discriminator(self.G, self.cond, reuse=False)
        self.Dx_logit= self.discriminator(self.x, self.cond, reuse=True)
        self.Dxmi_logit= self.discriminator(self.x_mismatch, self.cond, reuse=True)

        self.x_hat = self.epsilon * self.G + (1. - self.epsilon) * self.x
        self.cond_inp = self.cond + 0.0
        self.Dx_hat_logit = self.discriminator(self.x_hat, self.cond_inp, reuse=True)

        self.sampler, _, _ = self.generator(self.z_sample, self.cond_sample, reuse=True, is_training=False)

        self.d_vars = tf.trainable_variables('d_net')
        self.g_vars = tf.trainable_variables('g_net')

    def get_gradient_penalty(self, x, y):
        grad_y = tf.gradients(y, [x])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_y), reduction_indices=[1, 2, 3]))
        return tf.reduce_mean(tf.maximum(0.0, slopes - 1.)**2)

    def get_gradient_penalty2(self, x, y):
        grad_y = tf.gradients(y, [x])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_y), reduction_indices=[1]))
        return tf.reduce_mean(tf.maximum(0.0, slopes - 1.)**2)

    def define_losses(self):
        # Define the final losses
        kl_coeff = self.cfg.TRAIN.COEFF.KL
        lambda1 = self.cfg.TRAIN.COEFF.LAMBDA

        self.kt = tf.Variable(0.7, trainable=True, name='kt')

        self.D_loss_real = tf.reduce_mean(self.Dx_logit)
        self.D_loss_fake = tf.reduce_mean(self.Dg_logit)
        self.D_loss_mismatch = tf.reduce_mean(self.Dxmi_logit)
        self.wdist = self.D_loss_real - self.D_loss_fake
        self.wdist2 = self.D_loss_real - self.D_loss_mismatch
        self.reg_loss = tf.reduce_mean(tf.square(self.Dxmi_logit))
        self.balance_loss = tf.reduce_mean(tf.square(self.kt * self.wdist2 - self.wdist))

        self.G_kl_loss = self.kl_std_normal_loss(self.embed_mean, self.embed_log_sigma)
        self.real_gp = self.get_gradient_penalty(self.x_hat, self.Dx_hat_logit)
        self.real_gp2 = self.get_gradient_penalty2(self.cond_inp, self.Dx_hat_logit)

        self.D_loss = -self.wdist - self.kt * self.wdist2 + 150.0 * (self.real_gp + self.real_gp2)
        self.G_loss = -self.D_loss_fake + kl_coeff * self.G_kl_loss

        self.D_optim = tf.train.AdamOptimizer(self.learning_rate_d,
                                              beta1=self.cfg.TRAIN.BETA1,
                                              beta2=self.cfg.TRAIN.BETA2) \
            .minimize(self.D_loss, var_list=self.d_vars, global_step=self.global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.kt_optim = tf.train.GradientDescentOptimizer(0.001).minimize(self.balance_loss, var_list=[self.kt])

        with tf.control_dependencies(update_ops):
            self.G_optim = tf.train.AdamOptimizer(self.learning_rate_g,
                                                  beta1=self.cfg.TRAIN.BETA1,
                                                  beta2=self.cfg.TRAIN.BETA2)\
                .minimize(self.G_loss, var_list=self.g_vars)

    def generate_conditionals(self, embeddings):
        """Takes the embeddings, compresses them and builds the statistics for a multivariate normal distribution"""
        lrelu = lambda l: tf.nn.leaky_relu(l, 0.2)

        embeddings = tf.layers.flatten(embeddings)
        mean = fc(embeddings, self.compressed_embed_dim, act=lrelu)
        log_sigma = fc(embeddings, self.compressed_embed_dim, act=lrelu)
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

    def discriminator(self, inputs, embed, reuse=False):
        s16 = self.output_size / 16
        lrelu = lambda l: tf.nn.leaky_relu(l, 0.2)
        inputs = to_nchw(inputs)

        with tf.variable_scope("d_net", reuse=reuse):
            net_ho = conv2d(inputs, self.df_dim, ks=(4, 4), s=(2, 2), act=lrelu, df=NCHW)
            net_h1 = conv2d(net_ho, self.df_dim * 2, ks=(4, 4), s=(2, 2), df=NCHW, act=lrelu)
            net_h2 = conv2d(net_h1, self.df_dim * 4, ks=(4, 4), s=(2, 2), df=NCHW, act=lrelu)
            net_h3 = conv2d(net_h2, self.df_dim * 8, ks=(4, 4), s=(2, 2), df=NCHW)
            # --------------------------------------------------------

            # Residual layer
            net = conv2d(net_h3, self.df_dim * 2, ks=(1, 1), s=(1, 1), padding='valid', df=NCHW, act=lrelu)
            net = conv2d(net, self.df_dim * 4, ks=(3, 3), s=(1, 1), df=NCHW, act=lrelu)
            net = conv2d(net, self.df_dim * 8, ks=(3, 3), s=(1, 1), df=NCHW)
            net_h4 = tf.add(net_h3, net)
            net_h4 = tf.nn.leaky_relu(net_h4, 0.2)
            # --------------------------------------------------------

            # Compress embeddings
            net_embed = fc(embed, self.compressed_embed_dim, act=lrelu)

            # Spatially replicate embeddings in depth
            net_embed = tf.expand_dims(tf.expand_dims(net_embed, 2), 2)
            net_embed = tf.tile(net_embed, [1, 1, 4, 4])
            net_h4_concat = tf.concat([net_h4, net_embed], 1)

            net_h5 = conv2d(net_h4_concat, self.df_dim*8, ks=(3, 3), s=(1, 1), padding='same', df=NCHW, act=lrelu)
            net_h6 = conv2d(net_h5, self.df_dim*8, ks=(1, 1), s=(1, 1), padding='valid', df=NCHW, act=lrelu)

            out = conv2d(net_h6, 1, ks=(4, 4), s=(4, 4), padding='valid', df=NCHW)
            return out

    def generator(self, z, embed, reuse=False, is_training=True, df=NCHW, cond_noise=True):
        s = self.output_size
        s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

        with tf.variable_scope("g_net", reuse=reuse):
            # Sample from the multivariate normal distribution of the embeddings
            mean, log_sigma = self.generate_conditionals(embed)
            net_embed = self.sample_normal_conditional(mean, log_sigma, cond_noise)
            # --------------------------------------------------------

            # Concatenate the sampled embedding with the z vector
            net_input = tf.concat([z, net_embed], 1)
            net_h0 = fc(net_input, self.gf_dim * 8 * s16 * s16, act=None)
            net_h0 = batch_norm(net_h0, train=is_training, act=None, df=df)
            # --------------------------------------------------------
            if df == NCHW:
                net_h0 = tf.reshape(net_h0, [-1, self.gf_dim * 8, s16, s16])
            else:
                net_h0 = tf.reshape(net_h0, [-1, s16, s16, self.gf_dim * 8])

            # Residual layer
            net = conv2d(net_h0, self.gf_dim * 2, ks=(1, 1), s=(1, 1), padding='valid', df=df)
            net = batch_norm(net, train=is_training, act=tf.nn.relu, df=df)
            net = conv2d(net, self.gf_dim * 2, ks=(3, 3), s=(1, 1), df=df)
            net = batch_norm(net, train=is_training, act=tf.nn.relu, df=df)
            net = conv2d(net, self.gf_dim * 8, ks=(3, 3), s=(1, 1), df=df)
            net = batch_norm(net, train=is_training, act=None, df=df)
            net_h1 = tf.add(net_h0, net)
            net_h1 = tf.nn.relu(net_h1)
            # --------------------------------------------------------

            net_h2 = conv2d_transpose(net_h1, self.gf_dim * 4, ks=(4, 4), s=(2, 2), df=df)
            net_h2 = conv2d(net_h2, self.gf_dim * 4, ks=(3, 3), s=(1, 1), df=df)
            net_h2 = batch_norm(net_h2, train=is_training, act=None, df=df)
            # --------------------------------------------------------

            # Residual layer
            net = conv2d(net_h2, self.gf_dim, ks=(1, 1), s=(1, 1), padding='valid', df=df)
            net = batch_norm(net, train=is_training, act=tf.nn.relu, df=df)
            net = conv2d(net, self.gf_dim, ks=(3, 3), s=(1, 1), df=df)
            net = batch_norm(net, train=is_training, act=tf.nn.relu, df=df)
            net = conv2d(net, self.gf_dim * 4, ks=(3, 3), s=(1, 1), df=df)
            net = batch_norm(net, train=is_training, act=None, df=df)
            net_h3 = tf.add(net_h2, net)
            net_h3 = tf.nn.relu(net_h3)
            # --------------------------------------------------------

            net_h4 = conv2d_transpose(net_h3, self.gf_dim * 2, ks=(4, 4), s=(2, 2), df=df)
            net_h4 = conv2d(net_h4, self.gf_dim * 2, ks=(3, 3), s=(1, 1), df=df)
            net_h4 = batch_norm(net_h4, train=is_training, act=tf.nn.relu, df=df)

            net_h5 = conv2d_transpose(net_h4, self.gf_dim, ks=(4, 4), s=(2, 2), df=df)
            net_h5 = conv2d(net_h5, self.gf_dim, ks=(3, 3), s=(1, 1), df=df)
            net_h5 = batch_norm(net_h5, train=is_training, act=tf.nn.relu, df=df)

            net_logits = conv2d_transpose(net_h5, self.image_dims[-1], ks=(4, 4), s=(2, 2), df=df)
            net_logits = conv2d(net_logits, self.image_dims[-1], ks=(3, 3), s=(1, 1), df=df)

            net_output = tf.nn.tanh(net_logits)

            if df == NCHW:
                net_output = to_nhwc(net_output)
            return net_output, mean, log_sigma
