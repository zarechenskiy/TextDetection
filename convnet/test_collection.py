from glob import glob
import cv2
from pylearn2.utils import serial
import theano
import numpy as np


def load_model(model_path):
    model = serial.load(model_path)
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)

    return theano.function([X], Y), model


def preprocess(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    # img = img - cv2.mean(img)[0]


def has_text(f, img):
    preprocess(img)
    x = np.array([img])
    m, r, c = x.shape
    assert r == 32
    assert c == 32
    x = x.reshape(m, r, c, 1)

    y = f(x)[0]

    return y[0] >= y[1]


def compute_measure(f, dataset_path):
    files = glob(dataset_path + '/*.png')
    tp = tn = fp = fn = 0
    for file_name in files:
        img = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        is_positive = "positive" in file_name
        if has_text(f, img):
            if is_positive:
                tp += 1
            else:
                fp += 1
        else:
            if is_positive:
                fn += 1
            else:
                tn += 1

    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)

    return precision, recall


def compute_f1score(precision, recall):
    return 2.0 * precision * recall / (precision + recall)


model_path = '../convnet/convolutional_network_best.pkl'
dataset = "/Users/endermz/PycharmProjects/TextDetection/datasets/icdar2003_processed/validation"

print("Loading model")
f, model = load_model(model_path)
print("Model has been loaded")

p, r = compute_measure(f, dataset)
print("Precision: ", p)
print("Recall: ", r)
f1 = compute_f1score(p, r)
print("F1 score: ", f1)

