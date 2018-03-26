import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import inception
from utils.saver import load


def inception_net(images, num_classes, for_training=False, reuse=False):
    """Build Inception v3 model architecture."""

    with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, endpoints = inception.inception_v3(images,
                                                       dropout_keep_prob=0.8,
                                                       num_classes=num_classes,
                                                       is_training=for_training,
                                                       reuse=reuse,
                                                       scope='InceptionV3')

    return logits, endpoints


def load_inception_inference(sess, num_classes, batch_size, checkpoint_dir):
    """Loads the inception network with the parameters from checkpoint_dir"""
    # Build a Graph that computes the logits predictions from the inference model.
    inputs = tf.placeholder(tf.float32, [batch_size, 299, 299, 3], name='inputs')
    logits, layers = inception_net(inputs, num_classes)

    inception_vars = tf.global_variables('InceptionV3')

    saver = tf.train.Saver(inception_vars)
    print('Restoring Inception model from %s' % checkpoint_dir)

    could_load, _ = load(saver, sess, checkpoint_dir)
    if could_load:
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
    return logits, layers
