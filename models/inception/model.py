import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import inception


def inception_net(images, num_classes, for_training=False):
    """Build Inception v3 model architecture.

    See here for reference: http://arxiv.org/abs/1512.00567

    Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

    Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
    """

    with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, endpoints = inception.inception_v3(images,
                                                       dropout_keep_prob=0.8,
                                                       num_classes=num_classes,
                                                       is_training=for_training)

    return logits, endpoints


def load_inception_network(sess, num_classes, batch_size, checkpoint_dir):
    """Loads the inception network with the parameters from checkpoint_dir"""
    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = num_classes + 1

    # Build a Graph that computes the logits predictions from the inference model.
    inputs = tf.placeholder( tf.float32, [batch_size, 299, 299, 3], name='inputs')

    logits, layers = inference(inputs, num_classes)

    saver = tf.train.Saver([])
    saver.restore(sess, checkpoint_dir)
    print('Restoring model from %s).' % checkpoint_dir)

    return logits, layers
