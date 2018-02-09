import tensorflow as tf


def batch_norm(x, train, init, act=None, name=None, eps=1e-5, decay=0.9):
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
                                        activation_fn=act,
                                        updates_collections=None)


def conv2d(x, f, ks=(4, 4), s=(2, 2), padding='SAME', act=None, init=None):
    return tf.layers.conv2d(inputs=x, filters=f, kernel_size=ks, strides=s, padding=padding, activation=act,
                            kernel_initializer=init)


def conv2d_transpose(x, f, ks=(4, 4), s=(2, 2), padding='SAME', act=None, init=None):
    return tf.layers.conv2d_transpose(inputs=x, filters=f, kernel_size=ks, strides=s, padding=padding, activation=act,
                                      kernel_initializer=init)


def kl_std_normal_loss(mean, log_sigma):
    loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mean))
    loss = tf.reduce_mean(loss)
    return loss


def layer_norm(x, act=None):
    return tf.contrib.layers.layer_norm(x, activation_fn=act)


