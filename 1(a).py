import cv2 as cv
import numpy as np
original_image = cv.imread('Image/srity.jpg',0)
window_size = (512,512)

cv.namedWindow('Image',cv.WINDOW_NORMAL)
cv.resizeWindow('Image', *window_size)
cv.imshow("Image",original_image)
cv.waitKey(1000)
#display for 1 second
resized_image=original_image.copy()
while resized_image.shape[0] > 10 and resized_image.shape[1] > 10:
   resized_image = cv.resize(resized_image, (resized_image.shape[1] // 2, resized_image.shape[0] // 2))
   resized_display = cv.resize(resized_image, window_size)
   cv.imshow('Image', resized_display)
   cv.waitKey(1000)