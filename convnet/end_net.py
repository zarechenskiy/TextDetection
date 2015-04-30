from glob import glob

from pylearn2.utils import serial
import theano
import numpy as np
import cv2


def cmp_with_len(x, y):
    return -1 if len(x) < len(y) or x < y else 1


def load_images(path):
    files = glob(path + '/*.png')
    files = sorted(files, cmp=lambda x, y: cmp_with_len(x, y))
    x_test = np.array([cv2.imread(f, cv2.CV_LOAD_IMAGE_GRAYSCALE) for f in files])

    for x in x_test:
        cv2.normalize(x, x, 0, 255, cv2.NORM_MINMAX)
        x = x - cv2.mean(x)[0]

    return x_test


model_path = 'convolutional_network_best.pkl'
# model_path = 'convnet_best_test.pkl'
model = serial.load(model_path)
X = model.get_input_space().make_theano_batch()
Y = model.fprop(X)
# Y = T.argmax(Y, axis=1)
f = theano.function([X], Y)

images = load_images("../datasets/icdar2013/end_net_test")
# images = load_images("../datasets/synthetic/end_net_test")
print("Ready")
pos = 0
neg = 0
for image in images:
    l = np.array([image])
    m, r, c = l.shape
    assert r == 32
    assert c == 32
    l = l.reshape(m, r, c, 1)
    y = f(l)[0]

    if y[0] >= y[1]:
        pos += 1
    else:
        neg += 1
print("pos: ", pos)
print("neg: ", neg)