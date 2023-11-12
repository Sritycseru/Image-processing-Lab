import cv2
import numpy as np
import matplotlib.pyplot as plt

original_image = cv2.imread("Image/srity.jpg", 0)

row, col = original_image.shape

log_image = original_image.copy()
power_image = original_image.copy()

pwr = 5

for i in range(row):
    for j in range(col):
        log_image[i][j] = np.exp(1 + log_image[i][j])
        power_image[i][j] = np.power(power_image[i][j], pwr)

plt.subplot(3,1,1)
plt.imshow(original_image,cmap='gray')

plt.subplot(3,1,2)
plt.imshow(log_image,cmap='gray')

plt.subplot(3,1,3)
plt.imshow(power_image,cmap='gray')

plt.show()
