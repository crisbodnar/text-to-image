import tensorflow as tf
from utils.ops import conv2d, conv2d_transpose, layer_norm, batch_norm, fc


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
        
        self.conv_init = tf.contrib.layers.xavier_initializer_conv2d()
        self.fc_init = tf.contrib.layers.xavier_initializer()

        self.global_step = tf.Variable(0, trainable=False)

        if build_model:
            self.build_model()
            self.define_losses()

    def build_model(self):
        # Define the input tensor by appending the batch size dimension to the image dimension
        self.iter = tf.placeholder(tf.int32, shape=None)
        self.x = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='real_images')
        self.x_mismatch = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='real_images')
        self.cond = tf.placeholder(tf.float32, [self.batch_size] + [self.embed_dim], name='cond')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.epsilon = tf.placeholder(tf.float32, [self.batch_size, 1, 1, 1], name='eps')

        self.z_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.z_dim], name='z_sample')
        self.cond_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.embed_dim], name='cond_sample')

        self.G, self.embed_mean, self.embed_log_sigma = self.generator(self.z, self.cond, reuse=False)
        self.Dg, self.Dg_logit, self.Dgm_logit = self.discriminator(self.G, self.cond, reuse=False)
        self.Dx, self.Dx_logit, self.Dxma_logit = self.discriminator(self.x, self.cond, reuse=True)
        _, _, self.Dxm_logit = self.discriminator(self.x_mismatch, self.cond, reuse=True)

        self.x_hat = self.epsilon * self.G + (1. - self.epsilon) * self.x
        self.Dx_hat, self.Dx_hat_logit, _ = self.discriminator(self.x_hat, self.cond, reuse=True)

        self.sampler, _, _ = self.generator(self.z_sample, self.cond_sample, reuse=True, sampler=True,
                                            is_training=False)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if var.name.startswith('d_net')]
        self.g_vars = [var for var in t_vars if var.name.startswith('g_net')]
        
    def define_losses(self):
        # Define the final losses
        kl_coeff = self.cfg.TRAIN.COEFF.KL
        lambda_coeff = self.cfg.TRAIN.COEFF.LAMBDA

        self.D_loss_real_match = -tf.reduce_mean(self.Dx_logit)
        self.D_loss_fake = tf.reduce_mean(self.Dg_logit)
        self.Dm_loss = \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dxm_logit,
                                                                   labels=tf.zeros_like(self.Dxm_logit))) \
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dxma_logit,
                                                                    labels=tf.ones_like(self.Dxma_logit)))
        self.Gm_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dgm_logit,
                                                                    labels=tf.ones_like(self.Dgm_logit)))
        self.G_kl_loss = self.kl_std_normal_loss(self.embed_mean, self.embed_log_sigma)

        grad_Dx_hat = tf.gradients(self.Dx_hat_logit, [self.x_hat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_Dx_hat), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean(tf.square(tf.maximum(0., slopes - 1.)))

        self.D_loss = (self.D_loss_real_match + self.D_loss_fake) + lambda_coeff * self.gradient_penalty + self.Dm_loss
        self.G_loss = -self.D_loss_fake + kl_coeff * self.G_kl_loss + 0.1 * self.Gm_loss

        # decay = tf.maximum(0., 1 - tf.divide(tf.cast(self.iter, tf.float32), self.cfg.TRAIN.MAX_STEPS))
        decay = 1
        self.d_lr = self.cfg.TRAIN.D_LR * decay
        self.g_lr = self.cfg.TRAIN.G_LR * decay

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.D_optim = tf.train.AdamOptimizer(self.d_lr,
                                                  beta1=self.cfg.TRAIN.BETA1,
                                                  beta2=self.cfg.TRAIN.BETA2)\
                .minimize(self.D_loss, var_list=self.d_vars, global_step=self.global_step)
            self.G_optim = tf.train.AdamOptimizer(self.g_lr,
                                                  beta1=self.cfg.TRAIN.BETA1,
                                                  beta2=self.cfg.TRAIN.BETA2)\
                .minimize(self.G_loss, var_list=self.g_vars)

    def generate_conditionals(self, embeddings):
        """Takes the embeddings, compresses them and builds the statistics for a multivariate normal distribution"""
        lrelu = lambda l: tf.nn.leaky_relu(l, 0.2)

        embeddings = tf.layers.flatten(embeddings)
        mean = fc(embeddings, self.compressed_embed_dim, act=lrelu, init=self.fc_init)
        log_sigma = fc(embeddings, self.compressed_embed_dim, act=lrelu, init=self.fc_init)
        return mean, log_sigma

    def sample_normal_conditional(self, mean, log_sigma):
        epsilon = tf.truncated_normal(tf.shape(mean))
        stddev = tf.exp(log_sigma)
        return mean + stddev * epsilon

    def kl_std_normal_loss(self, mean, log_sigma):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mean))
        loss = tf.reduce_mean(loss)
        return loss

    def discriminator(self, inputs, embed, reuse=False):
        s16 = self.output_size / 16
        lrelu = lambda l: tf.nn.leaky_relu(l, 0.2)
        
        with tf.variable_scope("d_net", reuse=reuse):
            net_ho = conv2d(inputs, self.df_dim, ks=(4, 4), s=(2, 2), act=lrelu, init=self.conv_init)
            net_h1 = conv2d(net_ho, self.df_dim * 2, ks=(4, 4), s=(2, 2), init=self.conv_init)
            net_h1 = layer_norm(net_h1, act=lrelu)
            net_h2 = conv2d(net_h1, self.df_dim * 4, ks=(4, 4), s=(2, 2), init=self.conv_init)
            net_h2 = layer_norm(net_h2, act=lrelu)
            net_h3 = conv2d(net_h2, self.df_dim * 8, ks=(4, 4), s=(2, 2), init=self.conv_init)
            net_h3 = layer_norm(net_h3)
            # --------------------------------------------------------

            # Residual layer
            net = conv2d(net_h3, self.df_dim * 2, ks=(1, 1), s=(1, 1), padding='valid', init=self.conv_init)
            net = layer_norm(net, act=lrelu)
            net = conv2d(net, self.df_dim * 2, ks=(3, 3), s=(1, 1), init=self.conv_init)
            net = layer_norm(net, act=lrelu)
            net = conv2d(net, self.df_dim * 8, ks=(3, 3), s=(1, 1), init=self.conv_init)
            net = layer_norm(net)
            net_h4 = tf.add(net_h3, net)
            net_h4 = tf.nn.leaky_relu(net_h4, 0.2)
            # --------------------------------------------------------

            # Compress embeddings
            net_embed = fc(embed, self.compressed_embed_dim, act=lrelu, init=self.fc_init)

            # Append embeddings in depth
            net_embed = tf.reshape(net_embed, [self.batch_size, 4, 4, -1])
            net_h4_concat = tf.concat([net_h4, net_embed], 3)

            net_h4 = conv2d(net_h4_concat, self.df_dim * 8, ks=(1, 1), s=(1, 1), padding='valid', init=self.conv_init)
            net_h4 = layer_norm(net_h4, act=lrelu)
            net_h4 = conv2d(net_h4, self.df_dim * 8, ks=(2, 2), s=(1, 1), init=self.conv_init)
            net_h4 = layer_norm(net_h4, act=lrelu)

            net_logits = conv2d(net_h4, 1, ks=(s16, s16), s=(s16, s16), padding='valid', init=self.conv_init)
            mnet_logits = conv2d(net_h4, 1, ks=(s16, s16), s=(s16, s16), padding='valid', init=self.conv_init)

            return tf.nn.sigmoid(net_logits), net_logits, mnet_logits

    def generator(self, z, embed, reuse=False, sampler=False, is_training=True):
        s = self.output_size
        s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

        with tf.variable_scope("g_net", reuse=reuse):
            # Sample from the multivariate normal distribution of the embeddings
            mean, log_sigma = self.generate_conditionals(embed)
            net_embed = self.sample_normal_conditional(mean, log_sigma)
            # --------------------------------------------------------

            # Concatenate the sampled embedding with the z vector
            net_input = tf.concat([z, net_embed], 1)
            net_h0 = fc(net_input, self.gf_dim * 8 * s16 * s16, act=None, init=self.fc_init)
            net_h0 = batch_norm(net_h0, train=is_training, act=None)
            # --------------------------------------------------------
            net_h0 = tf.reshape(net_h0, [-1, s16, s16, self.gf_dim * 8])

            # Residual layer
            net = conv2d(net_h0, self.gf_dim * 2, ks=(1, 1), s=(1, 1), padding='valid', init=self.conv_init)
            net = batch_norm(net, train=is_training, act=tf.nn.relu)
            net = conv2d(net, self.gf_dim * 2, ks=(3, 3), s=(1, 1), init=self.conv_init)
            net = batch_norm(net, train=is_training, act=tf.nn.relu)
            net = conv2d(net, self.gf_dim * 8, ks=(3, 3), s=(1, 1), padding='same', init=self.conv_init)
            net = batch_norm(net, train=is_training, act=None)
            net_h1 = tf.add(net_h0, net)
            net_h1 = tf.nn.relu(net_h1)
            # --------------------------------------------------------

            net_h2 = conv2d_transpose(net_h1, self.gf_dim * 4, ks=(4, 4), s=(2, 2), init=self.conv_init)
            net_h2 = conv2d(net_h2, self.gf_dim * 4, ks=(3, 3), s=(1, 1), init=self.conv_init)
            net_h2 = batch_norm(net_h2, train=is_training, act=None)
            # --------------------------------------------------------

            # Residual layer
            net = conv2d(net_h2, self.gf_dim, ks=(1, 1), s=(1, 1), padding='valid', init=self.conv_init)
            net = batch_norm(net, train=is_training, act=tf.nn.relu)
            net = conv2d(net, self.gf_dim, ks=(3, 3), s=(1, 1), init=self.conv_init)
            net = batch_norm(net, train=is_training, act=tf.nn.relu)
            net = conv2d(net, self.gf_dim * 4, ks=(3, 3), s=(1, 1), init=self.conv_init)
            net = batch_norm(net, train=is_training, act=None)
            net_h3 = tf.add(net_h2, net)
            net_h3 = tf.nn.relu(net_h3)
            # --------------------------------------------------------

            net_h4 = conv2d_transpose(net_h3, self.gf_dim * 2, ks=(4, 4), s=(2, 2), init=self.conv_init)
            net_h4 = conv2d(net_h4, self.gf_dim * 2, ks=(3, 3), s=(1, 1), init=self.conv_init)
            net_h4 = batch_norm(net_h4, train=is_training, act=tf.nn.relu)

            net_h5 = conv2d_transpose(net_h4, self.gf_dim, ks=(4, 4), s=(2, 2), init=self.conv_init)
            net_h5 = conv2d(net_h5, self.gf_dim, ks=(3, 3), s=(1, 1), init=self.conv_init)
            net_h5 = batch_norm(net_h5, train=is_training, act=tf.nn.relu)

            net_logits = conv2d_transpose(net_h5, self.image_dims[-1], ks=(4, 4), s=(2, 2), init=self.conv_init)
            net_logits = conv2d(net_logits, self.image_dims[-1], ks=(3, 3), s=(1, 1), init=self.conv_init)

            net_output = tf.nn.tanh(net_logits)
            return net_output, mean, log_sigma