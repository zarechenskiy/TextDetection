import random
import sys
import cv2

__author__ = 'Mikhail Zarechenskiy'
# __path__ = "dataset.images.natural.text.icdar2013"

import numpy as np
from pylearn2.datasets import dense_design_matrix
from glob import glob
import matplotlib.image as mpimg


def cmp_with_len(x, y):
    return -1 if len(x) < len(y) or x < y else 1


class ICDAR2013(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, groundtruth, one_hot=False, max_count=None, data_path=None,
                 axes=('b', 0, 1, 'c'), start=0):
        self.start = start
        self.img_shape = (32, 32)
        self.img_size = np.prod(self.img_shape)
        self.max_count = max_count if max_count is not None else sys.maxsize
        self.label_names = ['text', 'non-text']
        self.count_classes = len(self.label_names)

        self.label_map = {k: v for k, v in zip(self.label_names, range(self.count_classes))}
        self.label_unmap = {v: k for k, v in zip(self.label_names, range(self.count_classes))}

        if data_path is not None:
            if data_path[-1] != '/':
                data_path += '/'

        files = glob(data_path + which_set + '/*.png')
        files = sorted(files, cmp=lambda x, y: cmp_with_len(x, y))

        assert len(files) > 0, "There are no files in " + data_path + which_set

        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        print("Random start: ", self.start)
        np.random.shuffle(files)

        files = files[self.start:self.start + self.max_count]

        print("Total number of files: ", len(files))

        X = np.array([cv2.imread(f, cv2.CV_LOAD_IMAGE_GRAYSCALE) for f in files])

        m, r, c = X.shape
        assert r == 32
        assert c == 32
        X = X.reshape((m, r, c, 1))


        y = np.zeros((len(files), self.count_classes))
        i = 0
        for f in files:
            y[i] = 0 if "positive" in f else 1

        super(ICDAR2013, self).__init__(y=y, topo_view=dimshuffle(X), axes=axes)