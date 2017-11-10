import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


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

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, alpha=0.2, name='lrelu'):
    return tf.maximum(x, alpha*x, name=name)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


            # s_h, s_w = self.output_size, self.output_size
            # s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            # s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            # s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            #
            # # Compress the conditional phi vector using a fully connected layer
            # g_fc_phi_w = tf.get_variable('g_fc_phi_w', [self.embed_dim, self.compressed_embed_dim],
            #                              initializer=tf.random_normal_initializer(stddev=0.02))
            # g_fc_phi_b = tf.get_variable('g_fc_phi_b', [self.compressed_embed_dim],
            #                              initializer=tf.random_normal_initializer(stddev=0.02))
            # c_phi = lrelu(tf.matmul(phi, g_fc_phi_w) + g_fc_phi_b, name='g_c_phi')
            #
            # # Append the compressed phi vector to the z noise vector
            # z_concat = tf.concat([z, c_phi], 1, name='g_z_concat')
            #
            # # project `z` and reshape
            # self.z_, self.h0_w, self.h0_b = linear(z_concat, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)
            #
            # self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            # h0 = tf.nn.relu(self.g_bn0(self.h0))
            #
            # self.h1, self.h1_w, self.h1_b = deconv2d(
            #     h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            # h1 = tf.nn.relu(self.g_bn1(self.h1))
            #
            # h2, self.h2_w, self.h2_b = deconv2d(
            #     h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            # h2 = tf.nn.relu(self.g_bn2(h2))
            #
            # h3, self.h3_w, self.h3_b = deconv2d(
            #     h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            # h3 = tf.nn.relu(self.g_bn3(h3))
            #
            # h4, self.h4_w, self.h4_b = deconv2d(
            #     h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)