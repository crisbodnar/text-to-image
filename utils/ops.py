import tensorflow as tf


def batch_normalization(x, is_training, initializers, activation, name=None):
    """
    A batch normalization layer with input x

    Parameters:
        x: Input tensor
        is_training: True if this layer is currently used during training (false for testing)
        initializers: A dictionary which can contain the keys "gamma" and "beta" specifying the gamma and beta
                      initializers
        activation: The activation function of the layer
        name: Name of the layer
    """

    epsilon = 1e-5
    momentum = 0.9
    layer = tf.contrib.layers.batch_norm(x,
                                         decay=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         param_initializers=initializers,
                                         is_training=is_training,
                                         scope=name,
                                         updates_collections=None)
    if activation is not None:
        return activation(layer)
    return layer
