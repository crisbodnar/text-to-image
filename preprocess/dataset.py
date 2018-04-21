"""
Parts of this code are taken from: https://github.com/hanzhanggit/StackGAN/blob/master/misc/datasets.py
"""

import numpy as np
from sklearn.externals import joblib
import pickle
import random
import os

FINAL_SIZE_TO_ORIG = {
    4: 4,
    8: 8,
    16: 16,
    32: 38,
    64: 76,
    128: 152,
    256: 304,
    299: 360,
    512: 600,
}


class Dataset(object):
    def __init__(self, images, imsize, embeddings=None,
                 filenames=None, workdir=None,
                 labels=None, aug_flag=True,
                 class_id=None, class_range=None):
        self._images = images
        self._embeddings = embeddings
        self._filenames = filenames
        self.workdir = workdir
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = len(images)
        self._saveIDs = self.saveIDs()

        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._aug_flag = aug_flag
        self._class_id = class_id
        self._class_range = class_range
        self._imsize = imsize
        self._perm = None

    @property
    def images(self):
        return self._images

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def filenames(self):
        return self._filenames

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def saveIDs(self):
        self._saveIDs = np.arange(self._num_examples)
        np.random.shuffle(self._saveIDs)
        return self._saveIDs

    def readCaptions(self, filenames, class_id):
        name = filenames
        if name.find('jpg/') != -1:  # flowers dataset
            class_name = 'class_%05d/' % (class_id + 1)  # Class ids are offset by 1 for classification tasks
            name = name.replace('jpg/', class_name)
        cap_path = '%s/text_c10/%s.txt' %\
                   (self.workdir, name)
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        captions = [cap for cap in captions if len(cap) > 0]
        return captions

    def transform(self, images):
        if self._aug_flag:
            transformed_images = np.zeros([images.shape[0], self._imsize, self._imsize, 3])
            ori_size = images.shape[1]
            for i in range(images.shape[0]):
                h1 = int(np.floor((ori_size - self._imsize) * np.random.random()))
                w1 = int(np.floor((ori_size - self._imsize) * np.random.random()))
                cropped_image = images[i][w1: w1 + self._imsize, h1: h1 + self._imsize, :]
                if random.random() > 0.5:
                    cropped_image = np.fliplr(cropped_image)
                transformed_images[i] = cropped_image
            return transformed_images
        else:
            return images

    def sample_embeddings(self, embeddings, filenames, class_id, sample_num):
        """Returns a mean of the specified number of embeddings (5 available per image)"""
        if len(embeddings.shape) == 2 or embeddings.shape[1] == 1:
            return np.squeeze(embeddings)
        else:
            batch_size, embedding_num, _ = embeddings.shape
            # Take every sample_num captions to compute the mean vector
            sampled_embeddings = []
            sampled_captions = []
            for i in range(batch_size):
                randix = np.random.choice(embedding_num, sample_num, replace=False)
                if sample_num == 1:
                    randix = int(randix)
                    captions = self.readCaptions(filenames[i],
                                                 class_id[i])
                    sampled_captions.append(captions[randix])
                    sampled_embeddings.append(embeddings[i, randix, :])
                else:
                    e_sample = embeddings[i, randix, :]
                    e_mean = np.mean(e_sample, axis=0)
                    sampled_embeddings.append(e_mean)
            sampled_embeddings_array = np.array(sampled_embeddings)
            return np.squeeze(sampled_embeddings_array), sampled_captions

    def next_batch(self, batch_size, window=None, wrong_img=False, embeddings=False, labels=False):
        """Return the next `batch_size` examples from this data set.

        :arg batch_size: the size of the batch
        :arg window: the number of embeddings whose mean to be returned (maximum is 5)
        :arg wrong_img: include the mismatching x in the return list
        :arg embeddings: include the text embedding is the return list
        :arg labels: include the class labels in the return list
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the .data
            self._perm = np.arange(self._num_examples)
            np.random.shuffle(self._perm)

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        current_ids = self._perm[start:end]
        sampled_images = self._images[current_ids]
        sampled_images = sampled_images.astype(np.float32)
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_images = self.transform(sampled_images)
        ret_list = [sampled_images]

        if wrong_img:
            fake_ids = np.random.randint(self._num_examples, size=batch_size)
            collision_flag = (self._class_id[current_ids] == self._class_id[fake_ids])
            fake_ids[collision_flag] = (fake_ids[collision_flag] + np.random.randint(100, 200)) % self._num_examples

            sampled_wrong_images = self._images[fake_ids, :, :, :]
            sampled_wrong_images = sampled_wrong_images.astype(np.float32)
            sampled_wrong_images = sampled_wrong_images * (2. / 255) - 1.
            sampled_wrong_images = self.transform(sampled_wrong_images)
            ret_list.append(sampled_wrong_images)
        else:
            ret_list.append(None)

        if self._embeddings is not None and embeddings:
            filenames = [self._filenames[i] for i in current_ids]
            class_id = [self._class_id[i] for i in current_ids]
            sampled_embeddings, sampled_captions = \
                self.sample_embeddings(self._embeddings[current_ids],
                                       filenames, class_id, window)
            ret_list.append(sampled_embeddings)
            ret_list.append(sampled_captions)
        else:
            ret_list.append(None)
            ret_list.append(None)

        if self._labels is not None and labels:
            class_id = [self._class_id[i] for i in current_ids]
            ret_list.append(class_id)
        else:
            ret_list.append(None)
        return ret_list

    def next_batch_test(self, batch_size, start, max_captions):
        """Return the next `batch_size` examples from this data set."""
        if (start + batch_size) > self._num_examples:
            end = self._num_examples
            start = end - batch_size
        else:
            end = start + batch_size

        sampled_images = self._images[start:end]
        sampled_images = sampled_images.astype(np.float32)
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_images = self.transform(sampled_images)

        sampled_embeddings = self._embeddings[start:end]
        _, embedding_num, _ = sampled_embeddings.shape
        sampled_embeddings_batchs = []

        sampled_captions = []
        sampled_filenames = self._filenames[start:end]
        sampled_class_id = self._class_id[start:end]
        for i in range(len(sampled_filenames)):
            captions = self.readCaptions(sampled_filenames[i],
                                         sampled_class_id[i])
            sampled_captions.append(captions)

        for i in range(np.minimum(max_captions, embedding_num)):
            batch = sampled_embeddings[:, i, :]
            sampled_embeddings_batchs.append(np.squeeze(batch))

        return [sampled_images, sampled_embeddings_batchs,
                self._saveIDs[start:end], sampled_captions]

    @property
    def class_ids(self):
        return self._class_id

    def class_to_index(self):
        class_to_idx = {}
        for idx, class_id in enumerate(np.unique(self._class_id)):
            class_to_idx[class_id] = idx
        return class_to_idx


class TextDataset(object):
    def __init__(self, workdir, size):
        self.size = size
        if size not in FINAL_SIZE_TO_ORIG:
            raise RuntimeError('Size {} not supported'.format(size))
        self.image_filename = '/{}images.pickle'.format(FINAL_SIZE_TO_ORIG[size])

        self.image_shape = [size, size, 3]
        self.image_dim = self.image_shape[0] * self.image_shape[1] * 3
        self.embedding_shape = None

        self._train = None
        self._test = None
        self.workdir = workdir
        self._dataset_name = os.path.basename(os.path.normpath(workdir))

        self.embedding_filename = '/char-CNN-RNN-embeddings.pickle'

    @property
    def train(self) -> Dataset:
        return self._train

    @train.setter
    def train(self, train):
        self._train = train

    @property
    def test(self) -> Dataset:
        return self._test

    @test.setter
    def test(self, test):
        self._test = test

    def get_data(self, pickle_path, aug_flag=True) -> Dataset:
        images = joblib.load(pickle_path + self.image_filename)
        images = np.array(images)
        print('Image shape: ', images.shape)

        with open(pickle_path + self.embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, encoding='bytes')
            embeddings = np.array(embeddings)
            self.embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        with open(pickle_path + '/filenames.pickle', 'rb') as f:
            list_filenames = pickle.load(f)
            print('list_filenames: ', len(list_filenames), list_filenames[0])
        with open(pickle_path + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f, encoding='bytes')
            # Bring classes from range [1: 102] to [0: 101]
            class_id = np.array(class_id) - 1
            print('Class ids:')
            print(np.unique(class_id))

        return Dataset(images, self.image_shape[0], embeddings,
                       list_filenames, self.workdir, class_id,
                       aug_flag, class_id)

    @property
    def name(self):
        return self._dataset_name
