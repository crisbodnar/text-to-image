import tensorflow as tf


def batch_normalization(x, is_training, initializer, activation, name):
    with tf.variable_scope(name):
        epsilon = 1e-5
        momentum = 0.9
        layer = tf.contrib.layers.batch_norm(x,
                                             decay=momentum,
                                             epsilon=epsilon,
                                             scale=True,
                                             is_training=is_training,
                                             scope=name,
                                             updates_collections=None)
        if activation is not None:
            return activation(layer)
        return layer