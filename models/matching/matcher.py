import tensorflow as tf
from models.inception.model import inception_net
from preprocess.data_batch import DataBatch
from utils.ops import fc, cosine_similarity
from utils.saver import save, load
from utils.utils import show_all_variables
from preprocess.dataset import TextDataset
import numpy as np
import time
import sys

CLASSES = 1000
ENC_DIM = 300
BATCH_SIZE = 64
TXT_EMBED_DIM = 1024
ALPHA = 0.2
MAX_STEPS = 100000

RESTORE_PRETRAIN = True
PRETRAINED_CHECKPOINT_DIR = './checkpoints/Inception/imagenet/inception_v3.ckpt'
CHECKPOINT_DIR = './checkpoints/matching/birds/'
SUMMARY_PERIOD = 20
LOGS_DIR = './logs/matching/'
DATASET_DIR = './data/birds/'


class MatchingModule(object):
    def __init__(self, sess: tf.Session, dataset: TextDataset):
        self.sess = sess
        self.dataset = dataset
        self.class_to_idx = self.dataset.test.class_to_index()

    def define_summaries(self):
        self.summary_op = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
        ])

        self.writer = tf.summary.FileWriter(LOGS_DIR, self.sess.graph)

    def define_model(self):
        self.inp_image = tf.placeholder(tf.float32, [BATCH_SIZE, 299, 299, 3], name='inp_image')
        self.inp_image_w = tf.placeholder(tf.float32, [BATCH_SIZE, 299, 299, 3], name='inp_image_w')
        self.inp_txt_embed = tf.placeholder(tf.float32, [BATCH_SIZE, TXT_EMBED_DIM], name='inp_txt_embed')
        self.inp_txt_embed_w = tf.placeholder(tf.float32, [BATCH_SIZE, TXT_EMBED_DIM], name='inp_txt_embed_w')

        self.img_enc = self.cnn_embed(self.inp_image)
        self.img_enc_w = self.cnn_embed(self.inp_image_w, reuse=True)
        self.txt_enc = self.dense_embed(self.inp_txt_embed)
        self.txt_enc_w = self.dense_embed(self.inp_txt_embed_w, reuse=True)

        self.vars_to_train = tf.trainable_variables('img_embed') + tf.trainable_variables('dense_embed') \
            + tf.trainable_variables('InceptionV3/Logits') + tf.trainable_variables('InceptionV3/Mixed_7c')

        self.inception_vars = tf.global_variables('InceptionV3')
        self.all_vars = tf.global_variables('img_embed') + tf.global_variables('dense_embed') + self.inception_vars
        self.not_to_restore = tf.global_variables('InceptionV3/Logits') \
            + tf.global_variables('InceptionV3/AuxLogits')

        self.pretrained_to_restore = [var for var in self.inception_vars if var not in self.not_to_restore]

        self.loss = tf.reduce_mean(tf.maximum(0., ALPHA - cosine_similarity(self.img_enc, self.txt_enc)
                                              + cosine_similarity(self.img_enc, self.txt_enc_w))) + \
                    tf.reduce_mean(tf.maximum(0., ALPHA - cosine_similarity(self.img_enc, self.txt_enc)
                                              + cosine_similarity(self.img_enc_w, self.txt_enc)))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.RMSPropOptimizer(learning_rate=0.00005)
            self.opt_step = opt.minimize(self.loss, var_list=self.vars_to_train)

        self.vars_to_init = [var for var in tf.global_variables() if var not in self.pretrained_to_restore]

        show_all_variables()

    def cnn_embed(self, images, for_training=True, reuse=False):
        logits, _ = inception_net(images, 1000, for_training=for_training, reuse=reuse)
        with tf.variable_scope('img_embed', reuse=reuse):
            return fc(logits, ENC_DIM, act=None)

    def dense_embed(self, txt_emb, reuse=False):
        with tf.variable_scope('dense_embed', reuse=reuse):
            return fc(txt_emb, ENC_DIM, act=None)

    def train(self):
        self.define_model()
        self.define_summaries()

        start_time = time.time()
        self.saver = tf.train.Saver(max_to_keep=1)

        if RESTORE_PRETRAIN:
            pretrain_saver = tf.train.Saver(self.pretrained_to_restore)

            # Load the pre-trained layer
            pretrain_saver.restore(self.sess, PRETRAINED_CHECKPOINT_DIR)

            # Initialise the not restored layers and the optimizer variables
            self.sess.run(tf.variables_initializer(self.vars_to_init))
            start_point = 0
        else:
            could_load, checkpoint_counter = load(self.saver, self.sess, CHECKPOINT_DIR)
            if could_load:
                start_point = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                raise RuntimeError('Failed to restore the complete Inception model')
        sys.stdout.flush()

        for idx in range(start_point + 1, MAX_STEPS):
            epoch_size = self.dataset.test.num_examples // BATCH_SIZE
            epoch = idx // epoch_size

            batch: DataBatch = self.dataset.test.next_batch(BATCH_SIZE, 4, wrong_img=True, wrong_emb=True,
                                                            embeddings=True)
            feed_dict = {
                self.inp_image: batch.images,
                self.inp_image_w: batch.wrong_images,
                self.inp_txt_embed: batch.embeddings,
                self.inp_txt_embed_w: batch.wrong_embeddings,
            }

            _, err = self.sess.run([self.opt_step, self.loss], feed_dict=feed_dict)

            summary_period = SUMMARY_PERIOD
            if np.mod(idx, summary_period) == 0:
                summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                self.writer.add_summary(summary_str, idx)

                print("Epoch: [%2d] [%4d] time: %4.4f, loss: %.8f" % (epoch, idx, time.time() - start_time, err))

            if np.mod(idx, 200) == 0:
                save(self.saver, self.sess, CHECKPOINT_DIR, idx)
            sys.stdout.flush()
