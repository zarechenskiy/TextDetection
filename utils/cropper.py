import csv
from os import listdir
import os
from os.path import isfile, join
import cv2
import numpy as np


def get_files(dir_path):
    return [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f[0] != '.']


def crop_using_window(img, w, h, exhaust=False):
    cropped = []

    height, width, depth = img.shape
    wx = 0
    while wx < width:
        hx = 0
        while hx < height:
            cr = img[hx:hx + h, wx:wx + w]
            if cr.shape[0] == h and cr.shape[1] == w:
                cropped.extend([cr])
            hx += 1 if exhaust else h
        wx += 1 if exhaust else w

    return cropped


def crop_area_image(img, area, w, h, exhaust=False):
    x0, y0, x1, y1 = area
    return crop_using_window(img[y0:y1, x0:x1], w, h, exhaust)


def get_groundtruth_area(filename):
    result = []
    if not os.path.exists(filename):
        return None
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            row = [int(float(i)) for i in row]
            (x0, y0, x1, y1) = row
            # result.extend([[y0, x0, y1, x1]])
            result.extend([[x0, y0, x1, y1]])  # This works for ICDAR2013

    return result


def safe_class_label(filename, class_label):
    f = open(filename, "w")
    f.write(str(class_label))
    f.close()


def is_intersect(area1, area2):
    x0, y0, x1, y1 = area1
    v0, w0, v1, w1 = area2
    return (x0 < v0 < x1 and y0 < w0 < y1) or \
           (v0 < x0 < v1 and w0 < y0 < w1) or \
           (v0 < x1 < v1 and w0 < y1 < w1) or \
           (x0 < v1 < x1 and y0 < w1 < y1)


def is_intersect_areas(area1, areas):
    res = False
    for area2 in areas:
        res = res or is_intersect(area1, area2)

    return res


def crop_part_image(cv_image, groundtruth, destination_dir, w, h, tag, postfix, short_name, inc_w, inc_h, is_positive):
    k = 0
    for i in range(len(groundtruth)):
        x0, y0, x1, y1 = groundtruth[i]
        cropped = cv_image[x0:x1, y0:y1]
        height, width, _ = cropped.shape
        wx = 0
        while wx < width:
            hx = 0
            while hx < height:
                if not is_positive:
                    area = (wx, hx, wx + w, hx + h)
                    if is_intersect_areas(area, groundtruth):
                        hx += inc_h
                        continue

                cr = cropped[hx:hx + h, wx:wx + w]
                if cr.shape[0] == h and cr.shape[1] == w:
                    name = tag + "crop" + short_name + str(k)
                    cv2.imwrite(destination_dir + name + postfix + ".png", cr)
                    k += 1
                hx += inc_h
            wx += inc_w

    return k


def crop_negative_part(cv_image, groundtruth, destination_dir, w, h, tag, postfix, short_name, inc_w, inc_h, cnt_pos):
    k = 0
    cropped = cv_image.copy()
    height, width, _ = cropped.shape
    wx = 0
    while wx < width:
        hx = 0
        while hx < height:
            if cnt_pos < k:
                return
            area = (wx, hx, wx + w, hx + h)
            if is_intersect_areas(area, groundtruth):
                hx += inc_h
                continue

            cr = cropped[hx:hx + h, wx:wx + w]
            if cr.shape[0] == h and cr.shape[1] == w:
                name = tag + "crop" + short_name + str(k)
                cv2.imwrite(destination_dir + name + postfix + ".png", cr)
                k += 1
            hx += inc_h
        wx += inc_w


def crop_image_with_groundtruth(cv_image, groundtruth, destination_dir,
                                w, h, tag, postfix_pos, postfix_neg, short_name):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    k = crop_part_image(cv_image, groundtruth, destination_dir, w, h, tag,
                        postfix=postfix_pos, short_name=short_name,
                        inc_h=12, inc_w=12, is_positive=True)

    print("Positive: ", k)

    crop_negative_part(cv_image, groundtruth, destination_dir, w, h, tag,
                       postfix=postfix_neg, short_name=short_name,
                       inc_h=h, inc_w=w, cnt_pos=k)


def make_one_dataset(images_dir, images, groundtruth_dir, w, h, tag, destination_dir, postfix_pos, postfix_neg):
    for image in images:
        cv_image = cv2.imread(images_dir + image)
        just_name = image[0:len(image) - len(".jpg")]
        groundtruth = get_groundtruth_area(groundtruth_dir + just_name + ".txt")
        if groundtruth is None:
            continue
        crop_image_with_groundtruth(cv_image, groundtruth, destination_dir,
                                    w, h, tag, postfix_pos, postfix_neg, just_name)


# Creates three datasets: learn_set, validatation_set, test_set
def make_full_work_dataset(
        images_dir, w, h, learn_size, val_size, groundtruth_dir, tag,
        destination="cropped",
        postfix_pos="positive",
        postfix_neg="negative"):

    images = get_files(images_dir)
    np.random.shuffle(images)

    count_images = len(images)

    learn_real_size = int(count_images * learn_size)
    learn_set = images[0:learn_real_size]
    val_real_size = int(count_images * val_size)
    val_set = images[learn_real_size:learn_real_size + val_real_size]
    test_set = images[learn_real_size + val_real_size:]

    print("Learn")
    make_one_dataset(images_dir, learn_set, groundtruth_dir, w, h,
                     tag, destination + "learn/", postfix_pos, postfix_neg)
    print("Validation")
    make_one_dataset(images_dir, val_set, groundtruth_dir, w, h,
                     tag, destination + "validation/", postfix_pos, postfix_neg)
    print("Test")
    make_one_dataset(images_dir, test_set, groundtruth_dir, w, h,
                     tag, destination + "test/", postfix_pos, postfix_neg)


icdar2013_train_images = "../datasets/icdar2013/original/train_images/"
icdar2013_train_groundtruth = "../datasets/icdar2013/original/train_groundtruth/"

icdar2013_cropped_images = "../datasets/icdar2013/cropped_images/"
icdar2013_cropped_labels = "../datasets/icdar2013/cropped_labels/"

synthetic_english_basic_images = "../datasets/synthetic/english_basic_images/"
synthetic_english_basic_groundtruth = "../datasets/synthetic/english_basic_groundtruth/"

synthetic_english_basic_cropped_images = "../datasets/synthetic/english_basic_cropped_images/"
synthetic_english_basic_cropped_labels = "../datasets/synthetic/english_basic_cropped_labels/"


# make_dataset(
# images_dir=synthetic_english_basic_images,
# groundtruth_dir=synthetic_english_basic_groundtruth,
# destination_img_dir=synthetic_english_basic_cropped_images,
# destination_labels=synthetic_english_basic_cropped_labels,
# w=32,
# h=32,
#     tag="syneng"
# )


# make_dataset(
#     images_dir=icdar2013_train_images,
#     groundtruth_dir=icdar2013_train_groundtruth,
#     destination_img_dir=icdar2013_cropped_images,
#     destination_labels=icdar2013_cropped_labels,
#     w=32,
#     h=32,
#     tag="icdar2013"
# )



# make_dataset(
#     images_dir="../datasets/icdar2013/original/additional/",
#     groundtruth_dir="../datasets/icdar2013/original/additional_groundtruth/",
#     destination_img_dir="../datasets/icdar2013/original/cropped_additional/",
#     destination_labels="../datasets/icdar2013/original/cropped_additional_labels/",
#     w=32,
#     h=32,
#     tag="icdar2013",
#     start_with_postfix=2500
# )


make_full_work_dataset(
    images_dir="/Users/endermz/spbu/coursework/2015/datasets/icdar2013/train/small/",
    w=32,
    h=32,
    learn_size=0.0,
    val_size=0.2,
    groundtruth_dir="/Users/endermz/spbu/coursework/2015/datasets/icdar2013/train/MRRCTrainLocalization/",
    tag="icdar2003",
    destination="/Users/endermz/spbu/coursework/2015/datasets/icdar2013/train/result/",
    postfix_pos="positive",
    postfix_neg="negative"
)