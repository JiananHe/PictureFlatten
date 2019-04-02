# Import both skimage and cv
from skimage import transform as tf
from skimage import io
import cv2

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Could use either skimage or cv to read the image
# img = cv2.imread('label.png')
img = cv2.imread('./picture/mask.jpg')
img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(thresh, 0, 200)

cv2.imshow("edges", edges)
cv2.waitKey(0)

# Find largest contour (should be the label)
_, contours, hierarchy = cv2.findContours(edges, 0, 1)
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt = contours[max_index]

# Create a mask of the label
mask = np.zeros(img.shape, np.uint8)
cv2.drawContours(mask, [cnt], 0, 255, -1)

cv2.imshow("mask", mask)
cv2.waitKey(0)


# Find the 4 borders
scale = 1
delta = 0
ddepth = cv2.CV_8U
borderType = cv2.BORDER_DEFAULT
left = cv2.Sobel(mask, ddepth, 1, 0, ksize=1, scale=1, delta=0, borderType=borderType)
right = cv2.Sobel(mask, ddepth, 1, 0, ksize=1, scale=-1, delta=0, borderType=borderType)
top = cv2.Sobel(mask, ddepth, 0, 1, ksize=1, scale=1, delta=0, borderType=borderType)
bottom = cv2.Sobel(mask, ddepth, 0, 1, ksize=1, scale=-1, delta=0, borderType=borderType)

# Remove noise from borders
kernel = np.ones((2, 2), np.uint8)
left_border = cv2.erode(left, kernel, iterations=1)
right_border = cv2.erode(right, kernel, iterations=1)
top_border = cv2.erode(top, kernel, iterations=1)
bottom_border = cv2.erode(bottom, kernel, iterations=1)

cv2.imshow("left_border", left_border)
cv2.imshow("right_border", right_border)
cv2.imshow("top_border", top_border)
cv2.imshow("bottom_border", bottom_border)
cv2.waitKey(0)

# Alternatively, use PiecewiseAffineTransform from SciKit-image to transform the image
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

dst = list()
src = list()
for y, x, z in np.transpose(np.nonzero(top_border)):
    dst.append([x, y])
    src.append([x, topmost[1]])
for y, x, z in np.transpose(np.nonzero(bottom_border)):
    dst.append([x, y])
    src.append([x, bottommost[1]])
for y, x, z in np.transpose(np.nonzero(left_border)):
    dst.append([x, y])
    src.append([leftmost[0], y])
for y, x, z in np.transpose(np.nonzero(right_border)):
    dst.append([x, y])
    src.append([rightmost[0], y])
src = np.array(src)
dst = np.array(dst)

tform3 = tf.PiecewiseAffineTransform()
tform3.estimate(src, dst)
warped = tf.warp(img, tform3, order=2)
warped = warped[85:170, 31:138]

cv2.imshow("warped", warped)
cv2.waitKey(0)