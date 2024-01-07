import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
original_image = cv.imread('Images/ideal_low_pass.png', cv.IMREAD_GRAYSCALE)
original_image = cv.resize(original_image, (512, 512))

image = np.copy(original_image)

# Add Gaussian noise to the image
noise = np.random.normal(0, 0.5, image.shape).astype(np.uint8)
img = cv.add(image,noise)

D0 = 10

# Calculate the frequency domain representation
fft_image = np.fft.fftshift(np.fft.fft2(image))

rows,columns = img.shape
D = np.zeros((rows, columns))
for u in range(rows):
    for v in range(columns):
        D[u, v] = np.sqrt((u - rows / 2) ** 2 + (v - columns / 2) ** 2)

# Gaussian High-Pass Filter
ghpf = 1 - np.exp(-((D ** 2) / (2 * D0 ** 2)))
ghpf = fft_image * ghpf
ghpf = np.abs(np.fft.ifft2(ghpf))
gaussian_hpf_img = ghpf / 255

idhf = D > D0
idhf = fft_image * idhf
idhf = np.abs(np.fft.ifft2(idhf))
ideal_hpf_img = idhf / 255

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(gaussian_hpf_img, cmap='gray')
plt.title('Noisy Gaussian High Pass Image')

plt.subplot(1, 3, 3)
plt.imshow(ideal_hpf_img, cmap='gray')
plt.title('Noisy Ideal High Pass Image')

plt.show()
