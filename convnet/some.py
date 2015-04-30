import cv2
import numpy as np

# img = cv2.imread("../datasets/icdar2013/end_net_test/icdar2013crop942.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
# img = cv2.imread("../datasets/icdar2013/end_net_test/icdar2013crop942.png")
img = cv2.imread("../datasets/icdar2013/original/some/D148.JPG", cv2.CV_LOAD_IMAGE_GRAYSCALE)
cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
# img = img - cv2.mean(img)[0]
# cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), img, iterations=1)
# cv2.imshow("some", img[300:1000, 0:1000])
cv2.imshow("some", img)
cv2.waitKey()
# print(cv2.mean(img)[0])