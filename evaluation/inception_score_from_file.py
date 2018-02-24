# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""

import tensorflow as tf
from utils.utils import load_inception_data
from models.inception.model import load_inception_inference
from evaluation.inception_score import get_inception_score

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/inception/flowers/',
                           """Path where to read model checkpoints.""")
tf.app.flags.DEFINE_string('img_folder', './evaluation/data', """Path where to load the x """)
tf.app.flags.DEFINE_integer('num_classes', 20, """Number of classes """)  # 20 for flowers
tf.app.flags.DEFINE_integer('splits', 10, """Number of splits """)
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('gpu', 1, "The ID of GPU to use")


def main(_):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % FLAGS.gpu):
                logits, _ = load_inception_inference(sess, FLAGS.num_classes, FLAGS.batch_size, FLAGS.checkpoint_dir)
                pred_op = tf.nn.softmax(logits)

                images = load_inception_data(FLAGS.img_folder)
                mean, std = get_inception_score(images, sess, FLAGS.batch_size, FLAGS.splits, pred_op)
                print('mean:', "%.2f" % mean, 'std:', "%.2f" % std)


if __name__ == '__main__':
    tf.app.run()
