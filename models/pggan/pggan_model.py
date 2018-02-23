from utils.ops import conv2d, lrelu, pix_norm
import tensorflow as tf


def residual_block(x, filters, alpha, ks=3, s=1):
    conv_init = tf.contrib.layers.xavier_initializer_conv2d()
    act = lrelu(0.2)

    inp = x
    x = pix_norm(conv2d(x, filters, ks=(ks, ks), s=(s, s), init=conv_init, act=act))
    x = pix_norm(conv2d(x, filters, ks=(ks, ks), s=(s, s), init=conv_init, act=act))

    x = (1. - alpha) * inp + alpha * x
    return x
