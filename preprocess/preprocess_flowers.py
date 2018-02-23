"""
Parts of this code are taken from: https://github.com/hanzhanggit/StackGAN/blob/master/misc/preprocess_flowers.py
"""
import os
import pickle
from preprocess.utils import get_image
import scipy.misc

# Edit this list to specify which files to be created
IMG_SIZES = [6, 12, 22, 40]
LOAD_SIZE = IMG_SIZES[-1]
FLOWER_DIR = './data/flowers'


def load_filenames(data_dir):
    filepath = data_dir + 'filenames.pickle'
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


def save_data_list(inpath, outpath, filenames):
    images = [[], [], [], []]
    cnt = 0
    for key in filenames:
        f_name = '%s/%s.jpg' % (inpath, key)
        img = get_image(f_name, LOAD_SIZE, is_crop=False)
        img = img.astype('uint8')

        for idx, size in enumerate(IMG_SIZES):
            if size != LOAD_SIZE:
                img = scipy.misc.imresize(img, [size, size], 'bicubic')
            images[idx].append(img)

        cnt += 1
        if cnt % 100 == 0:
            print('Load %d......' % cnt)

    print(len(images[0]), 'images processed')
    print('Image sizes:', IMG_SIZES)

    for idx, size in enumerate(IMG_SIZES):
        outfile = outpath + str(size) + 'images.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(images[idx], f_out)
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
