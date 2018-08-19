"""A library for the Inception match score to evaluate conditional generative models for images"""
import os

from scipy import spatial
import tensorflow as tf
import numpy as np

from models.matching.matcher import MatchingModule
from preprocess.dataset import TextDataset
from utils.utils import load_imgs_with_filenames

FLAGS = tf.app.flags.FLAGS


BATCH_SIZE = 64
TXT_EMBED_DIM = 1024

DATASET_DIR = './data/birds'
IMG_FOLDER = ''


def compute_ims(sess: tf.Session, img_folder, dataset: TextDataset, img_enc_op, txt_enc_op):
    """
    :param dataset:
    :param txt_enc_op:
    :param img_enc_op:
    :param sess: tensorflow sess
    :param img_folder: a folder where the image names are of the type imgId_number.jpg
    """

    sim_vector = []
    for (dirpath, dirnames, filenames) in os.walk(img_folder):
        no_of_images = len(filenames)
        batches = no_of_images // BATCH_SIZE
        for batch in range(batches):
            filenames_in_batch = filenames[batch * BATCH_SIZE : batch * BATCH_SIZE + BATCH_SIZE]
            ids = [filename.split('_')[0] for filename in filenames_in_batch]
            text_embed, _ = dataset.test.get_embeddings_from_ids(ids)

            filepaths_in_batch = [os.path.join(dirpath, filename) for filename in filenames_in_batch]
            images = load_imgs_with_filenames(filepaths_in_batch, 299)

            img_enc, txt_enc = sess.run([img_enc_op, txt_enc_op], feed_dict={
                'inp_image': images,
                'inp_txt_embed': text_embed,
            })

            for idx in range(BATCH_SIZE):
                encoded_img = img_enc[idx, :, :, :]
                encoded_txt = img_enc[idx, :, :, :]
                sim_vector.append(1.0 - spatial.distance.cosine(encoded_img, encoded_txt))

        return sim_vector


def load_matcher_module(sess, dataset: TextDataset):
    matcher = MatchingModule(sess, dataset)

    inp_image = tf.placeholder(tf.float32, [BATCH_SIZE, 299, 299, 3], name='inp_image')
    inp_txt_embed = tf.placeholder(tf.float32, [BATCH_SIZE, TXT_EMBED_DIM], name='inp_txt_embed')

    return matcher.cnn_embed(inp_image, for_training=False), matcher.dense_embed(inp_txt_embed)


def main():
    """Evaluate model on Dataset for a number of steps."""
    results = []

    dataset = TextDataset(DATASET_DIR, 299)

    filename_test = '%s/test' % DATASET_DIR
    dataset.test = dataset.get_data(filename_test)

    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device('/gpu:0'):
                img_enc_op, text_enc_op = load_matcher_module(sess, dataset)
                results += compute_ims(sess, IMG_FOLDER, dataset, img_enc_op, text_enc_op)

    print(np.mean(results), np.std(results))


if __name__ == '__main__':
    tf.app.run()
