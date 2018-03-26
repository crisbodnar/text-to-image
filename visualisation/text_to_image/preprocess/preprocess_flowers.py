"""
Parts of this code are taken from: https://github.com/hanzhanggit/StackGAN/blob/master/misc/preprocess_flowers.py
"""
import os
from preprocess.utils import get_image
import scipy.misc
import numpy as np
from sklearn.externals import joblib


# Edit this list to specify which files to be created
IMG_SIZES = [600]
LOAD_SIZE = 600
FLOWER_DIR = './data/flowers'


def load_filenames(data_dir):
    filepath = data_dir + 'filenames.pickle'
    filenames = []
    for filename in joblib.load(filepath):
        filenames.append(filename)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


def save_data_list(inpath, outpath, filenames):

    for size in IMG_SIZES:
        print('Processing images of size %d' % size)

        cnt = 0
        images = np.ndarray(shape=(len(filenames), size, size, 3), dtype=np.uint8)
        for idx, key in enumerate(filenames):
            f_name = '%s/%s.jpg' % (inpath, key)
            img = get_image(f_name, LOAD_SIZE, is_crop=False)
            img = img.astype('uint8')
            img = img.astype('uint8')

            if size != LOAD_SIZE:
                img = scipy.misc.imresize(img, [size, size], 'bicubic')
            images[idx, :, :, :] = np.array(img)

            cnt += 1
            if cnt % 100 == 0:
                print('\rLoad %d......' % cnt, end="", flush=True)

        print('Images processed: %d', len(filenames))

        outfile = outpath + str(size) + 'images.pickle'
        joblib.dump(images, outfile)
        print('save to: ', outfile)


def convert_flowers_dataset_pickle(inpath):
    # For Train data
    train_dir = os.path.join(inpath, 'train/')
    train_filenames = load_filenames(train_dir)
    save_data_list(inpath, train_dir, train_filenames)

    # For Test data
    test_dir = os.path.join(inpath, 'test/')
    test_filenames = load_filenames(test_dir)
    save_data_list(inpath, test_dir, test_filenames)


if __name__ == '__main__':
    convert_flowers_dataset_pickle(FLOWER_DIR)
