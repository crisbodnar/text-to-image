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
        self.class_to_idx = self.dataset.test.class_to_index()
        self.cfg = cfg

    def define_summaries(self):
        self.summary_op = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.image('image', self.x),
            tf.summary.scalar('train_acc', self.train_accuracy),
        ])

        self.writer = tf.summary.FileWriter(self.cfg.LOGS_DIR, self.sess.graph)

    def define_model(self):
        self.x = tf.placeholder(tf.float32, [self.cfg.TRAIN.BATCH_SIZE, 299, 299, 3], name='inputs')
        self.labels = tf.placeholder(tf.int32, [self.cfg.TRAIN.BATCH_SIZE])
        self.logits, layers = inception_net(self.x, self.cfg.MODEL.CLASSES, for_training=True)
        self.pred = tf.nn.softmax(self.logits)

        train_correct_prediction = tf.equal(self.labels, tf.cast(tf.argmax(self.pred, 1), tf.int32))
        self.train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                  labels=self.labels))

        self.vars_to_train = tf.trainable_variables('InceptionV3/Logits') \
            + tf.trainable_variables('InceptionV3/Mixed_7c')

        self.all_vars = tf.global_variables('InceptionV3')
        self.not_to_restore = tf.global_variables('InceptionV3/Logits') \
            + tf.global_variables('InceptionV3/AuxLogits')

        self.pretrained_to_restore = [var for var in self.all_vars if var not in self.not_to_restore]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.RMSPropOptimizer(learning_rate=0.00005)
            self.opt_step = opt.minimize(self.loss, var_list=self.vars_to_train)

        self.opt_vars = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in self.vars_to_train
                         if opt.get_slot(var, name) is not None]
        show_all_variables()

    def train(self):
        self.define_model()
        self.define_summaries()

        start_time = time.time()
        self.saver = tf.train.Saver(max_to_keep=self.cfg.TRAIN.CHECKPOINTS_TO_KEEP)

        if self.cfg.TRAIN.RESTORE_PRETRAIN:
            pretrain_saver = tf.train.Saver(self.pretrained_to_restore)

            # Load the pre-trained layer
            pretrain_saver.restore(self.sess, self.cfg.TRAIN.PRETRAINED_CHECKPOINT_DIR)

            # Initialise the not restored layers and the optimizer variables
            self.sess.run(tf.variables_initializer(self.not_to_restore + self.opt_vars))
            start_point = 0
        else:
            could_load, checkpoint_counter = load(self.saver, self.sess, self.cfg.CHECKPOINT_DIR)
            if could_load:
                start_point = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                raise RuntimeError('Failed to restore the complete Inception model')
        sys.stdout.flush()

        batch_size = self.cfg.TRAIN.BATCH_SIZE
        for idx in range(start_point + 1, self.cfg.TRAIN.MAX_STEPS):
            epoch_size = self.dataset.test.num_examples // batch_size
            epoch = idx // epoch_size

            images, _, _, _, labels = self.dataset.test.next_batch(batch_size, labels=True)

            # Bring the labels in a continuous range: [0, num_classes)
            new_labels = []
            for label in labels:
                new_labels.append(self.class_to_idx[label])

            assert(np.min(images) >= -1.)
            assert(np.max(images) <= 1.)
            assert(np.min(new_labels) >= 0)
            assert(np.max(new_labels) < 50)  # 20 for flowers, 50 for birds

            feed_dict = {
                self.x: images,
                self.labels: new_labels,
            }

            _, err = self.sess.run([self.opt_step, self.loss], feed_dict=feed_dict)

            summary_period = self.cfg.TRAIN.SUMMARY_PERIOD
            if np.mod(idx, summary_period) == 0:
                summary_str, pred = self.sess.run([self.summary_op, self.pred], feed_dict=feed_dict)
                self.writer.add_summary(summary_str, idx)

                print("Epoch: [%2d] [%4d] time: %4.4f, loss: %.8f" % (epoch, idx, time.time() - start_time, err))

            if np.mod(idx, 200) == 0:
                save(self.saver, self.sess, self.cfg.CHECKPOINT_DIR, idx)
            sys.stdout.flush()
