import numpy as np
from scipy import misc
from PIL import Image, ImageDraw, ImageFont
from utils.utils import denormalize_images
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
    for val in np.arange(1.0, 0.0, -step_size):
        if method == 'slerp':
            interp.append(slerp(a, b, val))
        elif method == 'lerp':
            interp.append(lerp(a, b, val))
    return interp


def interp_z(sess, gen_op, cond_sample, z1, z2, z='z:0', cond='cond:0', method='slerp', bs=64):
    """Generates a set of interpolated images in the space of the noise vector z"""

    z_batch = get_interpolated_batch(z1, z2, bs, method=method)
    return sess.run(gen_op, feed_dict={z: z_batch, cond: cond_sample})


def write_caption(img, caption, font_size, vert_pos):
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

    idx = caption.find(' ', 50)
    if idx == -1:
        # Write the caption on one row
        d.text((2, vert_pos), caption, font=fnt, fill=(255, 255, 255, 255))
    else:
        # Write the caption on two rows
        cap1 = caption[:idx]
        cap2 = caption[idx + 1:]
        d.text((2, vert_pos), cap1, font=fnt, fill=(255, 255, 255, 255))
        d.text((2, vert_pos + font_size), cap2, font=fnt, fill=(255, 255, 255, 255))
    return np.array(img_txt)


def save_cap_batch(img_batch, caption, path, rows=None):
    """Creates a super image of generated images with the caption of the images written on a top blank row."""
    img_shape = img_batch[0].shape
    font_size = img_shape[0] // 3 - 2
    super_img = prepare_img_for_captioning(img_batch, bottom=False, rows=rows)

    super_img = Image.fromarray(write_caption(super_img, caption, font_size, 10))
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
    text_row = np.tile(np.zeros(img_shape), reps=(1, n, 1))
    top_row = text_row

    # Fill in the super image row by row
    img_batch = denormalize_images(img_batch)
    super_img = top_row
    for rown in range(rows):
        if batch_size >= rown * n:
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

    super_img = write_caption(super_img, cap1, font_size, 10)
    super_img = write_caption(super_img, cap2, font_size, super_img.shape[0] - img_shape[0] + 10)
    super_img = Image.fromarray(super_img)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    misc.imsave(path, super_img)


def gen_noise_interp_img(sess, gen_op, cond, z_dim, batch_size):
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
    sample_z = np.random.standard_normal(size=(batch_size, z_dim))
    cond = get_interpolated_batch(cond1, cond2, batch_size=batch_size, method='lerp')

    samples = sess.run(gen_op, feed_dict={
        'z:0': sample_z,
        'cond:0': cond,
    })

    return samples


def gen_captioned_img(sess, gen_op, cond, z_dim, batch_size):
    sample_z = np.random.standard_normal(size=(batch_size, z_dim))
    cond = np.tile(np.expand_dims(cond, 0), reps=(batch_size, 1))

    samples = sess.run(gen_op, feed_dict={
        'z:0': sample_z,
        'cond:0': cond,
    })

    return samples