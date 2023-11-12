import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plot
image = cv.imread('Image/srity.jpg',0)
plt.subplot(2,1,1)
plt.imshow(image,cmap='gray')

gray_range = (0, 100)
row,col = image.shape
copy_image=image.copy()

for i in range(row):
    for j in range(col):
        if copy_image[i][j]>=gray_range[0] and copy_image[i][j]<=gray_range[1]:
            copy_image[i][j]+=155
print(copy_image)
plt.subplot(2,1,2)
plt.imshow(copy_image,cmap='gray')
plt.show()