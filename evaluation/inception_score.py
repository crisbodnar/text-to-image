# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""

import numpy as np
import math
from utils.utils import prep_incep_img


def get_inception_from_predictions(preds, splits, verbose=True):
    scores = []
    for i in range(splits):
        if verbose:
            print("\rComputing score for slice %d/%d" % (i + 1, splits), end="", flush=True)

        istart = i * preds.shape[0] // splits
        iend = (i + 1) * preds.shape[0] // splits
        part = preds[istart:iend, :]
        kl = (part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0))))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def get_inception_score(images, sess, batch_size, splits, pred_op, verbose=False):
    # assert(type(x) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)

    preds = []
    num_examples = len(images)
    n_batches = int(math.floor(float(num_examples) / float(batch_size)))
    indices = list(np.arange(num_examples))
    np.random.shuffle(indices)
    for i in range(n_batches):
        inp = []
        for j in range(batch_size):
            if (i*batch_size + j) == num_examples:
                break
            img = images[indices[i*batch_size + j]]
            img = prep_incep_img(img)
            inp.append(img)

        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)

        pred = sess.run(pred_op, {'inputs:0': inp})
        preds.append(pred)

    preds = np.concatenate(preds, 0)
    return get_inception_from_predictions(preds, splits)
