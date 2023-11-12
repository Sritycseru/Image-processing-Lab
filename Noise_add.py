import random
import numpy as np
import cv2 as cv

def add_noise(img):
    row, col = img.shape
    pixel = 100
    for i in range(pixel):
        x = random.randint(0,row-1)
        y = random.randint(0,col-1)
        img[x][y] = 255

    pixel = 100
    for i in range(pixel):
        x = random.randint(0,row-1)
        y = random.randint(0,col-1)
        img[x][y] = 0

    return img

img = cv.imread('Image/srity.jpg', 0)
img = cv.resize(img, (512,512))
img = add_noise(img)
cv.imshow('Image', img)
cv.waitKey(0)