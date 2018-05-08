from models.stackgan.stageII.model import ConditionalGan
from utils.utils import make_gif
from utils.saver import load
from utils.visualize import *
from preprocess.dataset import TextDataset
import tensorflow as tf
import numpy as np


class StageIIVisualizer(object):
    def __init__(self, sess: tf.Session, model: ConditionalGan, dataset: TextDataset, cfg):
        self.sess = sess
        self.model = model
        self.dataset = dataset
        self.config = cfg
        self.samples_dir = self.config.SAMPLE_DIR

    def visualize(self):
        z = tf.placeholder(tf.float32, [self.model.batch_size, self.model.z_dim], name='z')
        cond = tf.placeholder(tf.float32, [self.model.batch_size] + [self.model.embed_dim], name='cond')
        gen_stagei, _, _ = self.model.stagei.generator(z, cond, is_training=False)
        gen, _, _ = self.model.generator(gen_stagei, cond, is_training=False)
        gen_no_noise, _, _ = self.model.generator(gen_stagei, cond, is_training=False, reuse=True, cond_noise=False)

        saver = tf.train.Saver(tf.global_variables('g_net'))
        could_load, _ = load(saver, self.sess, self.model.stagei.cfg.CHECKPOINT_DIR)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise LookupError('Could not load any checkpoints for stage I')

        saver = tf.train.Saver(tf.global_variables('stageII_g_net'))
        could_load, _ = load(saver, self.sess, self.config.CHECKPOINT_DIR)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise LookupError('Could not load any checkpoints for stage II')

        dataset_pos = np.random.randint(0, self.dataset.test.num_examples)
        for idx in range(0):
            dataset_pos = np.random.randint(0, self.dataset.test.num_examples)
            dataset_pos2 = np.random.randint(0, self.dataset.test.num_examples)

            # Interpolation in z space:
            # ---------------------------------------------------------------------------------------------------------
            _, cond, _, captions = self.dataset.test.next_batch_test(1, dataset_pos, 1)
            cond = np.squeeze(cond, axis=0)
            caption = captions[0][0]

            samples = gen_noise_interp_img(self.sess, gen_no_noise, cond, self.model.z_dim, self.model.batch_size)
            save_cap_batch(samples, caption, '{}/{}_visual/z_interp/z_interp{}.png'.format(self.samples_dir,
                                                                                           self.dataset.name,
                                                                                           idx))
            # Interpolation in embedding space:
            # ---------------------------------------------------------------------------------------------------------

            _, cond1, _, caps1 = self.dataset.test.next_batch_test(1, dataset_pos, 1)
            _, cond2, _, caps2 = self.dataset.test.next_batch_test(1, dataset_pos2, 1)

            cond1 = np.squeeze(cond1, axis=0)
            cond2 = np.squeeze(cond2, axis=0)
            cap1, cap2 = caps1[0][0], caps2[0][0]

            samples = gen_cond_interp_img(self.sess, gen_no_noise, cond1, cond2, self.model.z_dim, self.model.batch_size)
            save_interp_cap_batch(samples, cap1, cap2,
                                  '{}/{}_visual/cond_interp/cond_interp{}.png'.format(self.samples_dir,
                                                                                      self.dataset.name,
                                                                                      idx))
            # make_gif(samples, '{}/{}_visual/cond_interp/gifs/cond_interp{}.gif'.format(self.samples_dir,
            #                                                                            self.dataset.name,
            #                                                                            idx), duration=4)

            # Generate captioned image
            # ---------------------------------------------------------------------------------------------------------
            _, conditions, _, captions = self.dataset.test.next_batch_test(1, dataset_pos, 1)
            conditions = np.squeeze(conditions, axis=0)
            caption = captions[0][0]
            samples = gen_captioned_img(self.sess, gen, conditions, self.model.z_dim, self.model.batch_size)

            save_cap_batch(samples, caption, '{}/{}_visual/cap/cap{}.png'.format(self.samples_dir,
                                                                                 self.dataset.name, idx))

            # Generate Stage I and Stage II images
            # ---------------------------------------------------------------------------------------------------------
            _, cond, _, captions = self.dataset.test.next_batch_test(self.model.batch_size, dataset_pos, 1)
            cond = np.squeeze(cond, axis=0)
            samples = gen_multiple_stage_img(self.sess, [gen_stagei, gen], cond, self.model.z_dim,
                                             self.model.batch_size, size=128)
            text = "Stage I and Stage II"
            save_cap_batch(samples, text, '{}/{}_visual/stages/stage{}.png'.format(self.samples_dir,
                                                                                   self.dataset.name, idx))

        special_flowers = [1126, 908, 398]
        special_birds = [12, 908, 1005]
        for idx, special_pos in enumerate(special_birds):
            print(special_pos)
            # Generate specific image
            # ---------------------------------------------------------------------------------------------------------
            _, conditions, _, captions = self.dataset.test.next_batch_test(1, special_pos, 1)
            conditions = np.squeeze(conditions, axis=0)
            caption = captions[0][0]
            samples = gen_captioned_img(self.sess, gen, conditions, self.model.z_dim, self.model.batch_size)

            save_cap_batch(samples, caption, '{}/{}_visual/special_cap/cap{}.png'.format(self.samples_dir,
                                                                                         self.dataset.name, idx))

        # # Generate some images and their closest neighbours
        # # ---------------------------------------------------------------------------------------------------------
        # _, conditions, _, _ = self.dataset.test.next_batch_test(self.model.batch_size, dataset_pos, 1)
        # conditions = np.squeeze(conditions)
        # samples, neighbours = gen_closest_neighbour_img(self.sess, gen, conditions, self.model.z_dim,
        #                                                 self.model.batch_size, self.dataset)
        # batch = np.concatenate([samples, neighbours])
        # text = 'Generated images and their closest neighbours'
        # save_cap_batch(batch, text, '{}/{}_visual/neighb/neighb.png'.format(self.samples_dir,
        #                                                                     self.dataset.name))









