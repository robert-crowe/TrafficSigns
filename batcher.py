# -*- coding: utf-8 -*-
""" Batcher

Usage:
    data = Batcher(X, y)
    batch_x, batch_y = data.next_batch(batch_size)

Used to create batches from a dataset for use in training and/or evaluating models.  Maintains state.
"""

import numpy as np
from sklearn import preprocessing


class Batcher(object):
    def __init__(self, examples, labels):
        self._num_examples = examples.shape[0]
        self._examples = examples
        self._lb = preprocessing.LabelBinarizer()
        self._lb.fit(labels)
        self._labels = self._lb.transform(labels)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def examples(self):
        return self._examples

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def get_one_hot(self, labels):
        return self._lb.transform(labels)

    def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._examples = self._examples[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._examples[start:end], self._labels[start:end]
