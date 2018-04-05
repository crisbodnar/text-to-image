""""Bits of this code are taken from https://github.com/hanzhanggit/StackGAN/blob/master/misc/preprocess_birds.py"""

import numpy as np
import os
import pickle
from preprocess.utils import get_image
import scipy.misc
import pandas as pd
from sklearn.externals import joblib


IMG_SIZES = [360]
LOAD_SIZE = 360
BIRD_DIR = './data/birds'


def load_filenames(data_dir):
    filepath = data_dir + 'filenames.pickle'
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


def load_bbox(data_dir):
    bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
    df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)
    #
    filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
    df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
    filenames = df_filenames[1].tolist()
    print('Total filenames: ', len(filenames), filenames[0])
    #
    filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    numImgs = len(filenames)
    for i in range(0, numImgs):
        # bbox = [x-left, y-top, width, height]
        bbox = df_bounding_boxes.iloc[i][1:].tolist()

        key = filenames[i][:-4]
        filename_bbox[key] = bbox

    return filename_bbox


def save_data_list(inpath, outpath, filenames, filename_bbox):

    for size in IMG_SIZES:
        print('Processing images of size %d' % size)

        cnt = 0
        images = np.ndarray(shape=(len(filenames), size, size, 3), dtype=np.uint8)
        for idx, key in enumerate(filenames):
            bbox = filename_bbox[key]
            f_name = '%s/CUB_200_2011/images/%s.jpg' % (inpath, key)
            img = get_image(f_name, LOAD_SIZE, is_crop=True, bbox=bbox)
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


def convert_birds_dataset_pickle(inpath):
    # Load dictionary between image filename to its bbox

    filename_bbox = load_bbox(inpath)
    # ## For Train data
    train_dir = os.path.join(inpath, 'train/')
    train_filenames = load_filenames(train_dir)
    save_data_list(inpath, train_dir, train_filenames, filename_bbox)

    # ## For Test data
    test_dir = os.path.join(inpath, 'test/')
    test_filenames = load_filenames(test_dir)
    save_data_list(inpath, test_dir, test_filenames, filename_bbox)


if __name__ == '__main__':
    convert_birds_dataset_pickle(BIRD_DIR)
