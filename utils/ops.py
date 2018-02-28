import tensorflow as tf

NHWC = 'NHWC'
NCHW = 'NCHW'


def batch_norm(x, train, init=None, act=None, name=None, eps=1e-5, decay=0.9):
    """
    A batch normalization layer with input x

    Parameters:
        x: Input tensor
        train: True if this layer is currently used during training (false for testing)
        init: A dictionary which can contain the keys "gamma" and "beta" specifying the gamma and beta
                      init
        act: The activation function of the layer
        name: Name of the layer
    """

    return tf.contrib.layers.batch_norm(x,
                                        decay=decay,
                                        epsilon=eps,
                                        scale=True,
                                        param_initializers=init,
                                        is_training=train,
                                        scope=name,
                                        fused=True,
                                        activation_fn=act)


def batch_renorm(x, train, init=None, act=None, name=None, eps=1e-5, decay=0.9, df=NHWC):
    """
    A batch normalization layer with input x

    Parameters:
        x: Input tensor
        train: True if this layer is currently used during training (false for testing)
        init: A dictionary which can contain the keys "gamma" and "beta" specifying the gamma and beta
                      init
        act: The activation function of the layer
        name: Name of the layer
    """

    return tf.contrib.layers.batch_norm(x,
                                        decay=decay,
                                        epsilon=eps,
                                        scale=True,
                                        param_initializers=init,
                                        is_training=train,
                                        scope=name,
                                        fused=True,
                                        activation_fn=act,
                                        renorm=True,
                                        data_format=df)


def conv2d(x, f, ks=(4, 4), s=(2, 2), padding='SAME', act=None, init=None, name=None):
    if init is None:
        init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)  # He init
    return tf.layers.conv2d(inputs=x, filters=f, kernel_size=ks, strides=s, padding=padding, activation=act,
                            kernel_initializer=init, name=name)


def conv2d_transpose(x, f, ks=(4, 4), s=(2, 2), padding='SAME', act=None, init=None, name=None):
    if init is None:
        init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)  # He init
    return tf.layers.conv2d_transpose(inputs=x, filters=f, kernel_size=ks, strides=s, padding=padding, activation=act,
                                      kernel_initializer=init, name=name)


def kl_std_normal_loss(mean, log_sigma):
    loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mean))
    loss = tf.reduce_mean(loss)
    return loss


def layer_norm(x, act=None, scope=None):
    return tf.contrib.layers.layer_norm(x, activation_fn=act, scope=scope)


def fc(x, units, act=None, init=None, bias=True, name=None, wdecay=4e-5):
    if init is None:
        init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)  # He init
    return tf.layers.dense(x, units=units, activation=act, kernel_initializer=init, use_bias=bias, name=name)


def lrelu_act(alpha=0.2):
    return lambda x: tf.nn.leaky_relu(x, alpha)


def pix_norm(x, eps=1e-8, act=None):
    if act is not None:
        x = act(x)
    return x / tf.sqrt(tf.reduce_mean(x**2, axis=3, keep_dims=True) + eps)


def pool(x, s=2, p_type='AVG'):
    return tf.nn.pool(x, window_shape=[s, s], pooling_type=p_type, strides=[s, s], padding='SAME')


def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x


def upscale(x, s=2):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h*s, w*s))


def downscale(x, s=2):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h // s, w // s))


def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]


