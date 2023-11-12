import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image=cv.imread('Image/noisysalterpepper.png',0)
window_size=(512,512)
cv.namedWindow('Image',cv.WINDOW_NORMAL)
cv.resizeWindow('Image', *window_size)
cv.imshow("Image",image)
cv.waitKey(1000)
m,n=image.shape
#mask=np.ones([3,3],dtype=int)
mask=np.ones([5,5],dtype=int)
#mask=np.ones([7,7],dtype=int)

mask=mask/25
#mask=mask/9
image_new=np.zeros([m,n])
for i in range(1, m - 1):
    for j in range(1, n - 1):
        temp = image[i - 1, j - 1] * mask[0, 0] + image[i - 1, j] * mask[0, 1] + image[i - 1, j + 1] * mask[0, 2] + image[
            i, j - 1] * mask[1, 0] + image[i, j] * mask[1, 1] + image[i, j + 1] * mask[1, 2] + image[i + 1, j - 1] * mask[
                   2, 0] + image[i + 1, j] * mask[2, 1] + image[i + 1, j + 1] * mask[2, 2]

        image_new[i, j] = temp

image_new = image_new.astype(np.uint8)



cv.imshow("Image",image_new)
cv.waitKey(1000)
img_noisy = cv.imread('Image/noisysalterpepper.png', 0)
m, n = img_noisy.shape
img_new2 = np.zeros([m, n])
for i in range(1, m - 1):
    for j in range(1, n - 1):
        temp = [img_noisy[i - 1, j -1],
                img_noisy[i - 1, j],
                img_noisy[i - 1, j + 1],
                img_noisy[i, j - 1],
                img_noisy[i, j],
                img_noisy[i, j + 1],
                img_noisy[i + 1, j - 1],
                img_noisy[i + 1, j],
                img_noisy[i + 1, j + 1]]

        temp = sorted(temp)
        img_new2[i, j] = temp[4]


window_size=(512,512)
img_new2 = img_new2.astype(np.uint8)
cv.imshow("new_median_filtered.jpg", img_new2)
cv.waitKey(10000)



