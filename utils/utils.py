"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import pprint
import scipy.misc
import numpy as np
import os
import imageio

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()
imageio.plugins.ffmpeg.download()
get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def save_images(images, size, image_path):
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(x,size) x parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)

    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    clip.write_gif(fname, fps=len(images) / duration)


def get_balanced_factorization(x):
    """Gets the factorization x=a*b with a,b being numbers as close as possible to each other"""
    if x <= 0:
        raise ValueError('Argument must be a strictly positive number but it is %d', x)
    a = int(np.sqrt(x))
    if a**2 == x:
        return a, a
    start = a
    for a in range(start, 0, -1):
        if np.mod(x, a) == 0:
            return a, x // a
    raise ValueError('Error finding the balanced factorization of %d' % x)


def save_captions(directory: str, captions):
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = 'captions.txt'
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    with open(filepath, 'w+') as f:
        f.write('Captions of the sampled x:\n')
        for idx, caption in enumerate(captions):
            f.write('{}: {}\n'.format(idx + 1, caption[0]))


def load_inception_data(full_path, alphabetic=False):
    print(full_path)
    if not os.path.exists(full_path):
        raise RuntimeError('Path %s does not exits' % full_path)
    images = []
    for path, subdirs, files in os.walk(full_path):
        if alphabetic:
            files = sorted(files)
        for name in files:
            if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                filename = os.path.join(path, name)
                # print('filename', filename)
                # print('path', path, '\nname', name)
                # print('filename', filename)
                if os.path.isfile(filename):
                    img = scipy.misc.imread(filename)
                    images.append(img)
    print('x', len(images), images[0].shape)
    return images


def prep_incep_img(img):
    # print('img', img.shape, img.max(), img.min())
    # img = Image.fromarray(img, 'RGB')
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = scipy.misc.imresize(img, (299, 299, 3), interp='bilinear')
    img = img.astype(np.float32)
    # [0, 255] --> [0, 1] --> [-1, 1]
    img = img / 127.5 - 1.
    # print('img', img.shape, img.max(), img.min())
    return img


def denormalize_images(images):
    return ((images + 1.0) * 127.5).astype('uint8')


def initialize_uninitialized(sess, verbose=True):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if verbose:
        print('Initializing the following %d variables:\n' % len(not_initialized_vars))
        print_vars(not_initialized_vars)
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def resize_imgs(imgs, size):
    res = []
    for img in imgs:
        res.append(scipy.misc.imresize(img, size, 'bicubic'))
    return res


def print_vars(vars):
    for var in vars:
        print(var.name)

