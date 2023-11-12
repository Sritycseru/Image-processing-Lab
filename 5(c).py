import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
image = cv.imread('Image/opening.png',cv.IMREAD_GRAYSCALE)
image = cv.resize(image,(512,512))
_, binary_image = cv.threshold(image, 128, 255, cv.THRESH_BINARY)
row,column = image.shape

structuring_element = np.ones((3,3))
erosion_image = np.copy(image)
def erosion_operation(image, structuring_element):
    for i in range(1, row-1):
        for j in range(1,column-1):
            erosion_image[i,j] = np.min(image[i-1:i+2,j-1:j+2]*structuring_element)

    return erosion_image

boundary_image = image-erosion_operation(image,structuring_element);

plt.subplot(121)
plt.imshow(binary_image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(boundary_image,'gray')
plt.title('Boundary Image')
plt.show()