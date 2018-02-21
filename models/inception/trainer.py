import tensorflow as tf
from models.inception.model import inception_net
from utils.saver import save, load
from utils.utils import show_all_variables
from preprocess.dataset import TextDataset
import numpy as np
import time
import sys


class InceptionTrainer(object):
    def __init__(self, sess: tf.Session, dataset: TextDataset, cfg):
        self.sess = sess
        self.dataset = dataset
        self.cfg = cfg

    def define_summaries(self):
        self.summary_op = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.image('image', self.x),
            tf.summary.scalar('train_acc', self.train_accuracy),
        ])

        self.writer = tf.summary.FileWriter(self.cfg.LOGS_DIR, self.sess.graph)

    def define_model(self):
        self.x = tf.placeholder(tf.float32, [self.cfg.TRAIN.BATCH_SIZE, 299, 299, 3])
        self.labels = tf.placeholder(tf.int32, [self.cfg.TRAIN.BATCH_SIZE])
        one_hot_labels = tf.one_hot(self.labels, self.cfg.MODEL.CLASSES)
        self.logits, _ = inception_net(self.x, self.cfg.MODEL.CLASSES, for_training=True)

        train_correct_prediction = tf.equal(tf.argmax(one_hot_labels, 1), tf.argmax(self.logits, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

        self.vars = tf.trainable_variables('InceptionV3')
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_labels))
        self.optim = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.5).minimize(self.loss, var_list=self.vars)

        show_all_variables()

    def train(self):
        self.define_model()
        self.define_summaries()
        self.saver = tf.train.Saver(max_to_keep=self.cfg.TRAIN.CHECKPOINTS_TO_KEEP)

        start_time = time.time()

        could_load, checkpoint_counter = load(self.saver, self.sess, self.cfg.CHECKPOINT_DIR)
        if could_load:
            start_point = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_point = 0
            print(" [!] Load failed...")
            tf.global_variables_initializer().run()
        sys.stdout.flush()

        batch_size = self.cfg.MODEL.BATCH_SIZE
        for idx in range(start_point + 1, self.cfg.TRAIN.MAX_STEPS):
            epoch_size = self.dataset.test.num_examples // batch_size
            epoch = idx // epoch_size

            images, _, _, _, labels = self.dataset.test.next_batch(batch_size, labels=True)

            feed_dict = {
                self.x: images,
                self.labels: labels,
            }

            _, err = self.sess.run([self.optim, self.loss], feed_dict=feed_dict)

            summary_period = self.cfg.TRAIN.SUMMARY_PERIOD
            if np.mod(idx, summary_period) == 0:
                summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                self.writer.add_summary(summary_str, idx)

                print("Epoch: [%2d] [%4d] time: %4.4f, loss: %.8f" % (epoch, idx, time.time() - start_time, err))

            if np.mod(idx, 500) == 2:
                save(self.saver, self.sess, self.cfg.CHECKPOINT_DIR, idx)
            sys.stdout.flush()
