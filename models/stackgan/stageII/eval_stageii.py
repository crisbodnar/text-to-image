from models.stackgan.stageII.model import ConditionalGan
from utils.saver import load
from utils.utils import denormalize_images, prep_incep_img
from preprocess.dataset import TextDataset
import tensorflow as tf
import numpy as np
from evaluation import fid, inception_score
from models.inception.model import load_inception_inference
import os


class StageIIEval(object):
    def __init__(self, sess: tf.Session, model: ConditionalGan, dataset: TextDataset, cfg):
        self.sess = sess
        self.model = model
        self.dataset = dataset
        self.cfg = cfg
        self.bs = self.cfg.EVAL.SAMPLE_SIZE

    def evaluate_fid(self):
        incep_batch_size = self.cfg.EVAL.INCEP_BATCH_SIZE
        _, layers = load_inception_inference(self.sess, 20, incep_batch_size,
                                             self.cfg.EVAL.INCEP_CHECKPOINT_DIR)
        pool3 = layers['PreLogits']
        act_op = tf.reshape(pool3, shape=[incep_batch_size, -1])

        if not os.path.exists(self.cfg.EVAL.ACT_STAT_PATH):
            print('Computing activation statistics for real x')
            fid.compute_and_save_activation_statistics(self.cfg.EVAL.R_IMG_PATH, self.sess, incep_batch_size, act_op,
                                                       self.cfg.EVAL.ACT_STAT_PATH, verbose=True)

        print('Loading activation statistics for the real x')
        stats = np.load(self.cfg.EVAL.ACT_STAT_PATH)
        mu_real = stats['mu']
        sigma_real = stats['sigma']

        z = tf.placeholder(tf.float32, [self.bs, self.model.z_dim], name='real_images')
        cond = tf.placeholder(tf.float32, [self.bs] + [self.model.embed_dim], name='cond')
        eval_gen, _, _ = self.model.generator(z, cond, reuse=False)

        saver = tf.train.Saver(tf.global_variables('g_net'))
        could_load, _ = load(saver, self.sess, self.cfg.CHECKPOINT_DIR)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise RuntimeError('Could not load the checkpoints of the generator')

        print('Generating batches...')

        fid_size = self.cfg.EVAL.SIZE
        n_batches = fid_size // self.bs

        w, h, c = self.model.image_dims[0], self.model.image_dims[1], self.model.image_dims[2]
        # Evaluate each bach on inception dynamically to avoid getting out of memory
        for i in range(n_batches):
            sample_z = np.random.normal(0, 1, size=(self.bs, self.model.z_dim))
            images, _, embed, _, _ = self.dataset.test.next_batch(self.bs, 4, embeddings=True)

            samples = denormalize_images(self.sess.run(eval_gen, feed_dict={z: sample_z, cond: embed}))


        print('Computing activation statistics for generated x...')
        mu_gen, sigma_gen = fid.calculate_activation_statistics(samples, self.sess, incep_batch_size, act_op,
                                                                verbose=True)
        print("calculate FID:", end=" ", flush=True)
        try:
            FID = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        except Exception as e:
            print(e)
            FID = 500

        print(FID)

    def evaluate_inception(self):
        incep_batch_size = self.cfg.EVAL.INCEP_BATCH_SIZE
        logits, _ = load_inception_inference(self.sess, self.cfg.EVAL.NUM_CLASSES, incep_batch_size,
                                             self.cfg.EVAL.INCEP_CHECKPOINT_DIR)
        pred_op = tf.nn.softmax(logits)

        z = tf.placeholder(tf.float32, [self.bs, self.model.stagei.z_dim], name='z')
        cond = tf.placeholder(tf.float32, [self.bs] + [self.model.stagei.embed_dim], name='cond')
        stagei_gen, _, _ = self.model.stagei.generator(z, cond, reuse=False, is_training=False)
        eval_gen, _, _ = self.model.generator(stagei_gen, cond, reuse=False, is_training=False)

        saver = tf.train.Saver(tf.global_variables('g_net'))
        could_load, _ = load(saver, self.sess, self.model.stagei.cfg.CHECKPOINT_DIR)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise RuntimeError('Could not load the checkpoints of stage I')

        saver = tf.train.Saver(tf.global_variables('stageII_g_net'))
        could_load, _ = load(saver, self.sess, self.cfg.CHECKPOINT_DIR)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise RuntimeError('Could not load the checkpoints of stage II')

        print('Generating batches...')

        size = self.cfg.EVAL.SIZE
        n_batches = size // self.bs

        all_preds = []
        for i in range(n_batches):
            print("\rGenerating batch %d/%d" % (i + 1, n_batches), end="", flush=True)

            sample_z = np.random.normal(0, 1, size=(self.bs, self.model.z_dim))
            _, _, embed, _, _ = self.dataset.test.next_batch(self.bs, 4, embeddings=True)

            # Generate a batch and scale it up for inception
            gen_batch = self.sess.run(eval_gen, feed_dict={z: sample_z, cond: embed})

            samples = denormalize_images(gen_batch)
            incep_samples = np.empty((self.bs, 299, 299, 3))
            for sample_idx in range(self.bs):
                incep_samples[sample_idx] = prep_incep_img(samples[sample_idx])

            # Run prediction for current batch
            pred = self.sess.run(pred_op, feed_dict={'inputs:0': incep_samples})
            all_preds.append(pred)

        # Get rid of the first dimension
        all_preds = np.concatenate(all_preds, 0)

        print('\nComputing inception score...')
        mean, std = inception_score.get_inception_from_predictions(all_preds, 10)
        print('Inception Score | mean:', "%.2f" % mean, 'std:', "%.2f" % std)





