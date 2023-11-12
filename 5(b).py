import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('Image/opening.png', 0)
image = cv.resize(image, (512, 512))
_, binary_image = cv.threshold(image, 128, 255, cv2.THRESH_BINARY)

row, column = image.shape
structuring_element = np.ones([3, 3])

def erosion_operation(image, structuring_element):
    erosion_img = np.copy(binary_image)
    for i in range(1, row - 1):
        for j in range(1, column - 1):
            erosion_img[i][j] = np.min(image[i - 1:i + 2, j - 1:j + 2] * structuring_element)
    return erosion_img

def dilation_operation(image, structuring_element):
    dilation_img = np.copy(binary_image)
    for i in range(1, row - 1):
        for j in range(1, column - 1):
            dilation_img[i][j] = np.max(image[i - 1:i + 2, j - 1:j + 2] * structuring_element)
    return dilation_img

opening_img = dilation_operation(erosion_operation(image, structuring_element), structuring_element)
closing_img = erosion_operation(dilation_operation(image, structuring_element), structuring_element)

plt.subplot(1, 3, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(opening_img, cmap='gray')
plt.title('Opening Image')

plt.subplot(1, 3, 3)
plt.imshow(closing_img, cmap='gray')
plt.title('Closing Image')

plt.show()
