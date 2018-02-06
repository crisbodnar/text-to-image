import tensorflow as tf
from utils.ops import conv2d, conv2d_transpose, kl_std_normal_loss


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
        
        self.w_init = tf.random_normal_initializer(stddev=0.02)

        if build_model:
            self.build_model()
            self.define_losses()

    def build_model(self):
        # Define the input tensor by appending the batch size dimension to the image dimension
        self.x = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='real_images')
        self.cond = tf.placeholder(tf.float32, [self.batch_size] + [self.embed_dim], name='cond')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        self.z_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.z_dim], name='z_sample')
        self.cond_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.embed_dim], name='cond_sample')

        self.G, self.embed_mean, self.embed_log_sigma = self.generator(self.z, self.cond, reuse=False)
        self.Dg, self.Dg_logit = self.discriminator(self.G, self.cond, reuse=False)
        self.Dx, self.Dx_logit = self.discriminator(self.x, self.cond, reuse=True)

        epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        self.x_hat = epsilon * self.G + (1. - epsilon) * self.x
        self.Dx_hat, self.Dx_hat_logit = self.discriminator(self.x_hat, self.cond, reuse=True)

        self.sampler, _, _ = self.generator(self.z_sample, self.cond_sample, reuse=True, sampler=True)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if var.name.startswith('d_net')]
        self.g_vars = [var for var in t_vars if var.name.startswith('g_net')]
        
    def define_losses(self):
        self.wass_dist = -tf.reduce_mean(self.Dg_logit) + tf.reduce_mean(self.Dx_logit)
        self.G_kl_loss = kl_std_normal_loss(self.embed_mean, self.embed_log_sigma)
        self.G_wass_loss = -tf.reduce_mean(self.Dg_logit)

        grad_Dx_hat = tf.gradients(self.Dx_hat_logit, [self.x_hat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_Dx_hat), reduction_indices=[-1]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.)**2)

        # Define the final losses
        kl_coeff = self.cfg.TRAIN.COEFF.KL
        lambda_coeff = self.cfg.TRAIN.COEFF.LAMBDA

        self.D_loss = -self.wass_dist + lambda_coeff * self.gradient_penalty
        self.G_loss = self.G_wass_loss + kl_coeff * self.G_kl_loss

        self.G_loss_summ = tf.summary.scalar("g_loss", self.G_loss)
        self.D_loss_summ = tf.summary.scalar("d_loss", self.D_loss)

        self.D_optim = tf.train.AdamOptimizer(self.cfg.TRAIN.D_LR, beta1=self.cfg.TRAIN.D_BETA_DECAY) \
            .minimize(self.D_loss, var_list=self.d_vars)
        self.G_optim = tf.train.AdamOptimizer(self.cfg.TRAIN.G_LR, beta1=self.cfg.TRAIN.G_BETA_DECAY) \
            .minimize(self.G_loss, var_list=self.g_vars)

    def generate_conditionals(self, embeddings):
        """Takes the embeddings, compresses them and builds the statistics for a multivariate normal distribution"""
        lrelu = lambda l: tf.nn.leaky_relu(l, 0.2)

        embeddings = tf.layers.flatten(embeddings)
        mean = tf.layers.dense(embeddings, self.compressed_embed_dim, activation=lrelu,
                               kernel_initializer=self.w_init)
        log_sigma = tf.layers.dense(embeddings, self.compressed_embed_dim,
                                    activation=lrelu, kernel_initializer=self.w_init)
        return mean, log_sigma

    def sample_normal_conditional(self, mean, log_sigma):
        epsilon = tf.truncated_normal(tf.shape(mean))
        stddev = tf.exp(log_sigma)
        return mean + stddev * epsilon

    def discriminator(self, inputs, embed, reuse=False):
        s16 = self.output_size / 16
        lrelu = lambda l: tf.nn.leaky_relu(l, 0.2)
        
        with tf.variable_scope("d_net", reuse=reuse):
            net_ho = conv2d(inputs, self.df_dim, ks=(4, 4), s=(2, 2), act=lrelu, init=self.w_init)
            net_h1 = conv2d(net_ho, self.df_dim * 2, ks=(4, 4), s=(2, 2), act=lrelu, init=self.w_init)
            net_h2 = conv2d(net_h1, self.df_dim * 4, ks=(4, 4), s=(2, 2), act=lrelu, init=self.w_init)
            net_h3 = conv2d(net_h2, self.df_dim * 8, ks=(4, 4), s=(2, 2), init=self.w_init)
            # --------------------------------------------------------

            # Residual layer
            net = conv2d(net_h3, self.df_dim * 2, ks=(1, 1), s=(1, 1), padding='valid', act=lrelu, init=self.w_init)
            net = conv2d(net, self.df_dim * 2, ks=(3, 3), s=(1, 1), act=lrelu, init=self.w_init)
            net = conv2d(net, self.df_dim * 8, ks=(3, 3), s=(1, 1), init=self.w_init)
            net_h4 = tf.add(net_h3, net)
            net_h4 = tf.nn.leaky_relu(net_h4, 0.2)
            # --------------------------------------------------------

            # Compress embeddings
            net_embed = tf.layers.dense(embed, units=self.compressed_embed_dim, activation=lrelu)

            # Append embeddings in depth
            net_embed = tf.reshape(net_embed, [self.batch_size, 4, 4, -1])
            net_h4_concat = tf.concat([net_h4, net_embed], 3)

            net_h4 = conv2d(net_h4_concat, self.df_dim*8, ks=(1, 1), s=(1, 1), padding='valid', act=lrelu,
                            init=self.w_init)
            net_logits = conv2d(net_h4, 1, ks=(s16, s16), s=(s16, s16), padding='valid', init=self.w_init)

            return tf.nn.sigmoid(net_logits), net_logits

    def generator(self, z, embed, reuse=False, sampler=False):
        s = self.output_size
        s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

        with tf.variable_scope("g_net", reuse=reuse):
            # Sample from the multivariate normal distribution of the embeddings
            mean, log_sigma = self.generate_conditionals(embed)
            net_embed = self.sample_normal_conditional(mean, log_sigma)
            # --------------------------------------------------------

            # Concatenate the sampled embedding with the z vector
            net_input = tf.concat([z, net_embed], 1)
            net_h0 = tf.layers.dense(net_input, units=self.gf_dim*8*s16*s16, activation=None,
                                     kernel_initializer=self.w_init)
            # --------------------------------------------------------

            # Reshape based on the number of samples if this is the sampler (instead of the training batch_size).
            if sampler:
                net_h0 = tf.reshape(net_h0, [self.sample_num, s16, s16, -1])
            else:
                net_h0 = tf.reshape(net_h0, [self.batch_size, s16, s16, -1])

            # Residual layer
            net = conv2d(net_h0, self.gf_dim*2, ks=(1, 1), s=(1, 1), padding='valid', act=tf.nn.relu, init=self.w_init)
            net = conv2d(net, self.gf_dim*2, ks=(3, 3), s=(1, 1), act=tf.nn.relu, init=self.w_init)
            net = conv2d(net, self.gf_dim*8, ks=(3, 3), s=(1, 1), init=self.w_init)
            net_h1 = tf.add(net_h0, net)
            net_h1 = tf.nn.relu(net_h1)
            # --------------------------------------------------------

            net_h2 = conv2d_transpose(net_h1, self.gf_dim*4, ks=(4, 4), s=(2, 2), init=self.w_init)
            net_h2 = conv2d(net_h2, self.gf_dim*4, ks=(3, 3), s=(1, 1), init=self.w_init)
            # --------------------------------------------------------

            # Residual layer
            net = conv2d(net_h2, self.gf_dim, ks=(1, 1), s=(1, 1), padding='valid', act=tf.nn.relu, init=self.w_init)
            net = conv2d(net, self.gf_dim, ks=(3, 3), s=(1, 1), act=tf.nn.relu, init=self.w_init)
            net = conv2d(net, self.gf_dim*4, ks=(3, 3), s=(1, 1), init=self.w_init)
            net_h3 = tf.add(net_h2, net)
            net_h3 = tf.nn.relu(net_h3)
            # --------------------------------------------------------

            net_h4 = conv2d_transpose(net_h3, self.gf_dim*2, ks=(4, 4), s=(2, 2), init=self.w_init)
            net_h4 = conv2d(net_h4, self.gf_dim*2, ks=(3, 3), s=(1, 1), act=tf.nn.relu, init=self.w_init)

            net_h5 = conv2d_transpose(net_h4, self.gf_dim, ks=(4, 4), s=(2, 2), init=self.w_init)
            net_h5 = conv2d(net_h5, self.gf_dim, ks=(3, 3), s=(1, 1), act=tf.nn.relu, init=self.w_init)

            net_logits = conv2d_transpose(net_h5, self.image_dims[-1], ks=(4, 4), s=(2, 2), init=self.w_init)
            net_logits = conv2d(net_logits, self.image_dims[-1], ks=(3, 3), s=(1, 1), init=self.w_init)

            net_output = tf.nn.tanh(net_logits)
            return net_output, mean, log_sigma

