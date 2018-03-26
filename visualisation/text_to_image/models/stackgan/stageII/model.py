import tensorflow as tf

from models.stackgan.stageI.model import ConditionalGan as StageI
from utils.ops import batch_norm, conv2d, conv2d_transpose


class ConditionalGan(object):
    def __init__(self, stagei: StageI, cfg, build_model=True):
        """
        Args:
          cfg: Config specifying all the parameters of the model.
        """

        self.name = 'ConditionalGAN/StageII'
        self.stagei = stagei
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
        self.batch_norm_init = {
            'gamma': tf.random_normal_initializer(1., 0.02),
        }

        if build_model:
            self.build_model()

    def build_model(self):
        # Define the input tensor by appending the batch size dimension to the image dimension
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='real_images')
        self.wrong_inputs = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='wrong_images')
        self.embed_inputs = tf.placeholder(tf.float32, [self.batch_size] + [self.embed_dim], name='phi_inputs')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        self.z_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.z_dim], name='z_sample')
        self.embed_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.embed_dim], name='phi_sample')

        stagei_g, _, _ = self.stagei.generator(self.z, self.embed_inputs, reuse=False)
        stagei_g_sampler, _, _ = self.stagei.generator(self.z_sample, self.embed_sample, reuse=True, is_training=False)
        self.G, self.embed_mean, self.embed_log_sigma = self.generator(stagei_g, self.embed_inputs, reuse=False)
        self.D_synthetic, self.D_synthetic_logits = self.discriminator(self.G, self.embed_inputs, reuse=False)
        self.D_real_match, self.D_real_match_logits = self.discriminator(self.inputs, self.embed_inputs, reuse=True)
        self.D_real_mismatch, self.D_real_mismatch_logits = self.discriminator(self.wrong_inputs, self.embed_inputs,
                                                                               reuse=True)
        self.sampler, _, _ = self.generator(stagei_g_sampler, self.embed_sample, reuse=True, is_training=False)

        self.d_vars = tf.trainable_variables('stageII_d_net')
        self.g_vars = tf.trainable_variables('stageII_g_net')

    def generate_conditionals(self, embeddings):
        """Takes the embeddings, compresses them and builds the statistics for a multivariate normal distribution"""
        embeddings = tf.layers.flatten(embeddings)
        mean = tf.layers.dense(embeddings, self.compressed_embed_dim, activation=lambda l: tf.nn.leaky_relu(l, 0.2),
                               kernel_initializer=self.w_init)
        log_sigma = tf.layers.dense(embeddings, self.compressed_embed_dim,
                                    activation=lambda l: tf.nn.leaky_relu(l, 0.2), kernel_initializer=self.w_init)
        return mean, log_sigma

    def sample_normal_conditional(self, mean, log_sigma, cond_noise):
        if cond_noise:
            epsilon = tf.truncated_normal(tf.shape(mean))
            stddev = tf.exp(log_sigma)
            return mean + stddev * epsilon
        return mean

    def discriminator(self, inputs, embed, is_training=True, reuse=False):
        s16 = self.output_size // 64
        lrelu = lambda l: tf.nn.leaky_relu(l, 0.2)

        with tf.variable_scope("stageII_d_net", reuse=reuse):
            net_ho = conv2d(inputs, self.df_dim, ks=(4, 4), s=(2, 2), act=lrelu, init=self.w_init)

            net_h1 = conv2d(net_ho, self.df_dim * 2, ks=(4, 4), s=(2, 2), init=self.w_init)
            net_h1 = batch_norm(net_h1, train=is_training, init=self.batch_norm_init, act=lrelu)

            net_h2 = conv2d(net_h1, self.df_dim * 4, ks=(4, 4), s=(2, 2), init=self.w_init)
            net_h2 = batch_norm(net_h2, train=is_training, init=self.batch_norm_init, act=lrelu)

            net_h3 = conv2d(net_h2, self.df_dim * 8, ks=(4, 4), s=(2, 2), init=self.w_init)
            net_h3 = batch_norm(net_h3, train=is_training, init=self.batch_norm_init, act=lrelu)

            net_h4 = conv2d(net_h3, self.df_dim * 16, ks=(4, 4), s=(2, 2), init=self.w_init)
            net_h4 = batch_norm(net_h4, train=is_training, init=self.batch_norm_init, act=lrelu)

            net_h5 = conv2d(net_h4, self.df_dim * 32, ks=(4, 4), s=(2, 2), init=self.w_init)
            net_h5 = batch_norm(net_h5, train=is_training, init=self.batch_norm_init, act=lrelu)

            net_h6 = conv2d(net_h5, self.df_dim * 16, ks=(4, 4), s=(1, 1), init=self.w_init)
            net_h6 = batch_norm(net_h6, train=is_training, init=self.batch_norm_init, act=lrelu)

            net_h7 = conv2d(net_h6, self.df_dim * 8, ks=(4, 4), s=(1, 1), init=self.w_init)
            net_h7 = batch_norm(net_h7, train=is_training, init=self.batch_norm_init)
            # --------------------------------------------------------

            # Residual layer
            net = conv2d(net_h7, self.df_dim * 2, ks=(1, 1), s=(1, 1), init=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init, act=lrelu)

            net = conv2d(net, self.df_dim * 2, ks=(3, 3), s=(1, 1), init=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init, act=lrelu)

            net = conv2d(net, self.df_dim * 8, ks=(3, 3), s=(1, 1), init=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init)

            net_h8 = tf.add(net, net)
            net_h8 = tf.nn.leaky_relu(net_h8, 0.2)
            # --------------------------------------------------------

            # Compress embeddings
            net_embed = tf.layers.dense(inputs=embed, units=self.compressed_embed_dim, activation=lrelu)

            # Append embeddings in depth
            net_embed = tf.expand_dims(tf.expand_dims(net_embed, 1), 1)
            net_embed = tf.tile(net_embed, [1, s16, s16, 1])
            net_h8_concat = tf.concat([net_h8, net_embed], 3)

            net_h9 = conv2d(net_h8_concat, self.df_dim * 8, ks=(1, 1), s=(1, 1), init=self.w_init)
            net_h9 = batch_norm(net_h9, train=is_training, init=self.batch_norm_init, act=lrelu)

            net_logits = conv2d(net_h9, 1, ks=(s16, s16), s=(s16, s16), init=self.w_init)
            return tf.nn.sigmoid(net_logits), net_logits

    def generator_encode_image(self, image, is_training=True):
        net_h0 = conv2d(image, self.gf_dim, ks=(3, 3), s=(1, 1), act=tf.nn.relu)

        net_h1 = conv2d(net_h0, self.gf_dim * 2, ks=(4, 4), s=(2, 2))
        net_h1 = batch_norm(net_h1, train=is_training, init=self.batch_norm_init, act=tf.nn.relu)

        output_tensor = conv2d(net_h1, self.gf_dim * 4, ks=(4, 4), s=(2, 2))
        output_tensor = batch_norm(output_tensor, train=is_training, init=self.batch_norm_init, act=tf.nn.relu)

        return output_tensor

    def generator_residual_layer(self, input_layer, is_training=True):
        net_h0 = input_layer

        net_h1 = conv2d(net_h0, self.gf_dim * 4, ks=(4, 4), s=(1, 1))
        net_h1 = batch_norm(net_h1, train=is_training, init=self.batch_norm_init, act=tf.nn.relu)

        net_h2 = conv2d(net_h1, self.gf_dim * 4, ks=(4, 4), s=(1, 1))
        net_h2 = batch_norm(net_h2, train=is_training, init=self.batch_norm_init)

        return tf.nn.relu(tf.add(net_h0, net_h2))

    def generator_upsample(self, input_layer, is_training=True):
        net_h0 = conv2d_transpose(input_layer, self.gf_dim * 2, ks=(4, 4), init=self.w_init)
        net_h0 = conv2d(net_h0, self.gf_dim * 2, ks=(3, 3), s=(1, 1))
        net_h0 = batch_norm(net_h0, train=is_training, init=self.batch_norm_init, act=tf.nn.relu)

        net_h1 = conv2d_transpose(net_h0, self.gf_dim, ks=(4, 4), init=self.w_init)
        net_h1 = conv2d(net_h1, self.gf_dim, ks=(3, 3), s=(1, 1))
        net_h1 = batch_norm(net_h1, train=is_training, init=self.batch_norm_init, act=tf.nn.relu)

        net_h2 = conv2d_transpose(net_h1, self.gf_dim // 2, ks=(4, 4), init=self.w_init)
        net_h2 = conv2d(net_h2, self.gf_dim // 2, ks=(3, 3), s=(1, 1))
        net_h2 = batch_norm(net_h2, train=is_training, init=self.batch_norm_init, act=tf.nn.relu)

        net_h3 = conv2d_transpose(net_h2, self.gf_dim // 4, ks=(4, 4), init=self.w_init)
        net_h3 = conv2d(net_h3, self.gf_dim // 4, ks=(3, 3), s=(1, 1))
        net_h3 = batch_norm(net_h3, train=is_training, init=self.batch_norm_init, act=tf.nn.relu)

        return conv2d(net_h3, self.image_dims[-1], ks=(3, 3), s=(1, 1), act=tf.nn.tanh)

    def generator(self, image, embed, is_training=True, reuse=False, cond_noise=True):
        s = 64
        s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

        with tf.variable_scope("stageII_g_net", reuse=reuse):
            encoded_img = self.generator_encode_image(image, is_training=is_training)

            # Sample from the multivariate normal distribution of the embeddings
            mean, log_sigma = self.generate_conditionals(embed)
            net_embed = self.sample_normal_conditional(mean, log_sigma, cond_noise)
            # --------------------------------------------------------

            # Concatenate the encoded image and the embeddings
            net_embed = tf.expand_dims(tf.expand_dims(net_embed, 1), 1)
            net_embed = tf.tile(net_embed, [1, s4, s4, 1])
            imgenc_embed = tf.concat([encoded_img, net_embed], 3)

            pre_res = conv2d(imgenc_embed, self.gf_dim * 4, ks=(3, 3), s=(1, 1))
            pre_res = batch_norm(pre_res, train=is_training, init=self.batch_norm_init, act=tf.nn.relu)

            r_block1 = self.generator_residual_layer(pre_res, is_training=is_training)
            r_block2 = self.generator_residual_layer(r_block1, is_training=is_training)
            r_block3 = self.generator_residual_layer(r_block2, is_training=is_training)
            r_block4 = self.generator_residual_layer(r_block3, is_training=is_training)

            return self.generator_upsample(r_block4, is_training=is_training), mean, log_sigma






