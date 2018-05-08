import numpy as np
import scipy
from scipy import misc
from PIL import Image, ImageDraw, ImageFont

from preprocess.dataset import TextDataset
from utils.utils import denormalize_images, resize_imgs
import os


def slerp(a, b, miu):
    """Spherical interpolation between a and b. miu is in [0, 1]"""
    if miu < 0 or miu > 1:
        raise ValueError('miu must be in [0, 1] but it is %d' % miu)
    if miu == 0:
        return a
    if miu == 1:
        return b

    omega = np.arccos(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))
    sin_omega = np.sin(omega)
    return np.sin((1.0 - miu) * omega) / sin_omega * a + np.sin(miu * omega) / sin_omega * b


def lerp(a, b, miu):
    """Linear interpolation between a and b. miu is in [0, 1]"""
    if miu < 0 or miu > 1:
        raise ValueError('miu must be in [0, 1] but it is %d' % miu)
    return miu * a + (1. - miu) * b


def get_interpolated_batch(a, b, batch_size=64, method='slerp'):
    """Generates a batch of given size of interpolations between a, b"""

    step_size = 1 / batch_size
    interp = []
    for val in np.arange(1.0, 0.0 + step_size, -step_size):
        if method == 'slerp':
            interp.append(slerp(a, b, val))
        elif method == 'lerp':
            interp.append(lerp(a, b, val))
    if method == 'slerp':
        interp.append(slerp(a, b, 0.0))
    elif method == 'lerp':
        interp.append(lerp(a, b, 0.0))

    return interp


def interp_z(sess, gen_op, cond_sample, z1, z2, z='z:0', cond='cond:0', method='slerp', bs=64):
    """Generates a set of interpolated images in the space of the noise vector z"""

    z_batch = get_interpolated_batch(z1, z2, bs, method=method)
    return sess.run(gen_op, feed_dict={z: z_batch, cond: cond_sample})


def write_caption(img, caption, font_size, vert_pos, split=50):
    """Writes a caption on the top row of the provided image. Blank space should be left on the top row."""
    img_txt = Image.fromarray(img)
    # get a font
    try:
        fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', font_size)  # MacOS
    except OSError:
        try:
            fnt = ImageFont.truetype('arial.ttf', font_size)  # Windows
        except OSError:
            fnt = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size)  # Linux
    # get a drawing context
    d = ImageDraw.Draw(img_txt)

    idx = caption.find(' ', split)
    if idx == -1:
        # Write the caption on one row
        d.text((2, vert_pos), caption, font=fnt, fill=(0, 0, 0, 0))
    else:
        # Write the caption on two rows
        cap1 = caption[:idx]
        cap2 = caption[idx + 1:]
        d.text((2, vert_pos), cap1, font=fnt, fill=(0, 0, 0, 0))
        d.text((2, vert_pos + font_size), cap2, font=fnt, fill=(0, 0, 0, 0))
    return np.array(img_txt)


def preporcess_caption(cap: str):
    cap = cap[:1].upper() + cap[1:]
    if cap[-1] != '.':
        cap += '.'
    return cap


def save_cap_batch(img_batch, caption, path, rows=None, split=50):
    """Creates a super image of generated images with the caption of the images written on a top blank row."""
    img_shape = img_batch[0].shape
    font_size = img_shape[0] // 3 - 2
    super_img = prepare_img_for_captioning(img_batch, bottom=False, rows=rows)
    caption = preporcess_caption(caption)

    super_img = Image.fromarray(write_caption(super_img, caption, font_size, 10, split=split))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    misc.imsave(path, super_img)


def prepare_img_for_captioning(img_batch, bottom, rows=None):
    # Show at most 8 images per row
    batch_size = img_batch.shape[0]
    n = min(8, batch_size)
    img_shape = img_batch[0].shape

    if rows is None:
        rows = batch_size // n

    # Leave top row empty for the caption text
    text_row = np.tile(np.ones(img_shape) * 255, reps=(1, n, 1))
    top_row = text_row

    # Fill in the super image row by row
    img_batch = denormalize_images(img_batch)
    super_img = top_row
    for rown in range(rows):
        if batch_size > rown * n:
            row = []
            for i in range(rown * n, (rown + 1) * n):
                row.append(img_batch[i])
            row = np.concatenate(row, axis=1)
            super_img = np.concatenate([super_img, row], axis=0)

    if bottom:
        super_img = np.concatenate([super_img, text_row], axis=0)

    super_img = super_img.astype(np.uint8)
    return super_img


def save_interp_cap_batch(img_batch, cap1, cap2, path, rows=None):
    """Creates a super image of interpolated captions."""
    """Creates a super image of generated images with the caption of the images written on a top blank row."""
    img_shape = img_batch[0].shape
    font_size = img_shape[0] // 3 - 2
    super_img = prepare_img_for_captioning(img_batch, bottom=True, rows=rows)
    cap1 = preporcess_caption(cap1)
    cap2 = preporcess_caption(cap2)

    super_img = write_caption(super_img, cap1, font_size, 10)
    super_img = write_caption(super_img, cap2, font_size, super_img.shape[0] - img_shape[0] + 10)
    super_img = Image.fromarray(super_img)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    misc.imsave(path, super_img)


def gen_noise_interp_img(sess, gen_op, cond, z_dim, batch_size):
    """Generates a batch of images interpolated in the noise space"""
    z = np.random.standard_normal(size=(2, z_dim))
    sample_z = get_interpolated_batch(z[0], z[1], batch_size=batch_size, method='slerp')
    cond = np.expand_dims(cond, 0)
    cond = np.tile(cond, reps=(batch_size, 1))

    samples = sess.run(gen_op, feed_dict={
        'z:0': sample_z,
        'cond:0': cond,
    })

    return samples


def gen_cond_interp_img(sess, gen_op, cond1, cond2, z_dim, batch_size):
    """Generates a batch of images interpolated in the condition space"""
    sample_z = np.random.standard_normal(size=(batch_size, z_dim))
    cond = get_interpolated_batch(cond1, cond2, batch_size=batch_size, method='lerp')

    samples = sess.run(gen_op, feed_dict={
        'z:0': sample_z,
        'cond:0': cond,
    })

    return samples


def gen_captioned_img(sess, gen_op, cond, z_dim, batch_size):
    """Generates a batch of images with the same caption"""
    sample_z = np.random.standard_normal(size=(batch_size, z_dim))
    cond = np.tile(np.expand_dims(cond, 0), reps=(batch_size, 1))

    samples = sess.run(gen_op, feed_dict={
        'z:0': sample_z,
        'cond:0': cond,
    })

    return samples


def closest_image(fake_img, dataset: TextDataset):
    """Finds the closest image from the dataset of a given image"""
    min_distance = float('inf')
    closest_img = None
    for idx in range(dataset.train.num_examples):
        imgs, _, _, captions = dataset.train.next_batch_test(1, idx, 1)
        real_img = imgs[0]
        # caption = captions[0]

        dist = np.linalg.norm(fake_img - real_img)
        if min_distance > dist:
            min_distance = dist
            closest_img = real_img
    return closest_img


def closest_images_of_batch(fake_imgs, dataset: TextDataset):
    """Finds the closest images of a given batch of images"""
    closest_batch = []
    for img in fake_imgs:
        closest_batch.append(closest_image(img, dataset))
    return np.array(closest_batch)


def gen_closest_neighbour_img(sess, gen_op, cond, z_dim, batch_size, dataset):
    """Generates a batch of images and appends to it their closest neighbours"""
    sample_z = np.random.standard_normal(size=(batch_size, z_dim))

    samples = sess.run(gen_op, feed_dict={
        'z:0': sample_z,
        'cond:0': cond,
    })
    samples = samples[:8]
    samples = np.clip(samples, -1., 1.)

    neighbours = closest_images_of_batch(samples, dataset)
    return samples, neighbours


def gen_multiple_stage_img(sess, gen_ops, cond, z_dim, batch_size, size=128):
    """Generates a batch of images from multiple generator stages"""
    sample_z = np.random.standard_normal(size=(batch_size, z_dim))

    imgs = []
    for gen_op in gen_ops:
        samples = sess.run(gen_op, feed_dict={
            'z:0': sample_z,
            'cond:0': cond,
        })
        samples = samples[:8]
        samples = (samples + 1.0) * 127.5
        samples = resize_imgs(samples, (size, size), interp='nearest')
        samples = np.array(samples) / 127.5 - 1.0
        imgs.extend(samples)

    return np.array(imgs)


def gen_pggan_sample(samples, size=128, interp='bicubic'):
    """Same image at multiple PGGAN scales"""
    stages = len(samples)
    batch_size = len(samples[0])
    new_samples = np.empty(shape=(stages, batch_size, size, size, 3))
    for sidx, stage in enumerate(samples):
        for sam_idx, sample in enumerate(stage):
            sample = np.array(sample)
            sample = (sample + 1.0) * 127.5
            sample = scipy.misc.imresize(sample, size=(size, size), interp='nearest')
            sample = np.array(sample) / 127.5 - 1.0
            new_samples[sidx, sam_idx, :, :, :] = sample

    return new_samples



