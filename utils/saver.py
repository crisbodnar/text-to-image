import os
import re
import tensorflow as tf


def save(saver: tf.train.Saver, sess: tf.Session, checkpoint_dir, step):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, checkpoint_dir, global_step=step, write_meta_graph=False)


def load(saver: tf.train.Saver, sess: tf.Session, checkpoint_dir: str):
    print(" [*] Reading checkpoints from %s..." % checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find checkpoints")
        return False, 0
