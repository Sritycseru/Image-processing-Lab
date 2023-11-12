import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image=cv.imread('Image/erosion.jpg',0)
image=cv.resize(image,(512,512))
row,column=image.shape
_,binary_image=cv.threshold(image,128,255,cv2.THRESH_BINARY)

erosion_img=np.copy(image)
dilation_img=np.copy(image)

structuring_element=np.ones([3,3])

for i in range(1, row - 1):
    for j in range(1, column - 1):
        erosion_img[i][j] = np.min(image[i - 1:i + 2, j - 1:j + 2] * structuring_element)

for i in range (1,row-1):
    for j in range (1,column-1):
        dilation_img[i][j]=np.max(image[i-1:i+2,j-1:j+2]*structuring_element)
plt.subplot(1,3,1)
plt.imshow(binary_image,cmap='gray')
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(erosion_img,cmap='gray')
plt.title('Erosion Image')

plt.subplot(1,3,3)
plt.imshow(dilation_img,cmap='gray')
plt.title('Dilation Image')

plt.show()