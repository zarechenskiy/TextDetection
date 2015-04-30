from glob import glob
from pylearn2.training_algorithms.sgd import SGD

from pylearn2.utils import serial, safe_zip
from pylearn2.utils.data_specs import DataSpecsMapping
import theano
import numpy as np
import cv2
from scipy.cluster.vq import vq, kmeans, whiten


def preprocess(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    # img = img - cv2.mean(img)[0]


def load_image(path):
    return cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)


def load_original_image(path):
    return cv2.imread(path)


def load_model(model_path):
    model = serial.load(model_path)
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)

    return theano.function([X], Y), model


def get_gradients(model):
    cost = model.get_default_cost()

    data_specs = cost.get_data_specs(model)
    mapping = DataSpecsMapping(data_specs)
    space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
    source_tuple = mapping.flatten(data_specs[1], return_tuple=True)

    theano_args = []
    for space, source in safe_zip(space_tuple, source_tuple):
        name = '%s[%s]' % (SGD.__class__.__name__, source)
        arg = space.make_theano_batch(name=name,
                                      batch_size=model.batch_size)
        theano_args.append(arg)
    theano_args = tuple(theano_args)

    nested_args = mapping.nest(theano_args)

    fixed_var_descr = cost.get_fixed_var_descr(model, nested_args)
    grads, updates = cost.get_gradients(model, nested_args, ** fixed_var_descr.fixed_vars)

    params = list(model.get_params())
    for param in params:
        some = grads[param]
        print("ok")

    return grads


def has_text(f, img):
    preprocess(img)
    x = np.array([img])
    m, r, c = x.shape
    assert r == 32
    assert c == 32
    x = x.reshape(m, r, c, 1)

    y = f(x)[0]

    return y[0] >= y[1]


def draw_if_text(img, hx, wx, h, w, img_with_text, mask):
    cr = img[hx:hx + h, wx:wx + w]
    if cr.shape[0] == h and cr.shape[1] == w:
        if has_text(f, cr):
            cv2.rectangle(img_with_text, (wx, hx), (wx + w, hx + h), (0, 255, 0), 2)
            mask[(hx + h) / 2, (wx + w) / 2] += 1
            return True

    return False


def draw_around(img, hx, wx, h, w, img_with_text, n, p, mask):
    for i in range(n):
        k = i * p
        draw_if_text(img, hx + k, wx, h, w, img_with_text, mask)
        draw_if_text(img, hx, wx + k, h, w, img_with_text, mask)
        draw_if_text(img, hx - k, wx, h, w, img_with_text, mask)
        draw_if_text(img, hx, wx - k, h, w, img_with_text, mask)
        draw_if_text(img, hx + k, wx + k, h, w, img_with_text, mask)
        draw_if_text(img, hx - k, wx - k, h, w, img_with_text, mask)


def slide_and_find_text(f, original_img, img, w, h, exhaust=False):
    img_with_text = original_img
    height, width, _ = original_img.shape
    mask = np.zeros((height, width))
    wx = 0
    while wx < width:
        hx = 0
        while hx < height:
            with_text = draw_if_text(img, hx, wx, h, w, img_with_text, mask)
            # if with_text:
            #     draw_around(img, hx, wx, h, w, img_with_text, 3, 5, mask)
            hx += 16 if exhaust else h
        wx += 16 if exhaust else w

    return img_with_text, mask


# model_path = '../convnet_settings/convolutional_network_bestAll2.pkl'
name = "D158"
model_path = '../convnet_settings/convolutional_network_bestAll2.pkl'
image_path = '../datasets/icdar2013/original/some/' + name + '.JPG'

original_img = load_original_image(image_path)
img = load_image(image_path)
# cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), img, iterations=1)
f, model = load_model(model_path)
# some = get_gradients(model)

print("Loaded")

img_with_text, mask = slide_and_find_text(f, original_img, img, 32, 32, False)

# cv2.imshow("Some", img_with_text)
# cv2.waitKey()
cv2.imwrite("../results/" + name + "_text1.jpg", img_with_text)
# cv2.imwrite("../results/" + name + "_text_mask.jpg", mask)
