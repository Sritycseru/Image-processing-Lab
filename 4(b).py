import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

original_image = cv.imread('Image/srity.jpg', 0)
image = cv.resize(original_image, (512, 512))

rows, columns = image.shape

mean_of_distribution = 0
standard_deviation = 25
noise = np.random.normal(0, 0.5, image.shape).astype(np.uint8)

noise_image = cv.add(image, noise)

fft_image = np.fft.fftshift(np.fft.fft2(noise_image))

D0 = 2
number_of_image = 15
dimension = 4

n, m = fft_image.shape
lowpass_image = np.zeros((n, m))

for u in range(n):
    for v in range(m):
        lowpass_image[u, v] = np.sqrt((u - n / 2) ** 2 + (v - m / 2) ** 2)

temporary_image = np.copy(lowpass_image)
for i in range(number_of_image):
    lowpass_image = np.copy(temporary_image)
    lowpass_image = lowpass_image <= D0
    lowpass_image = lowpass_image * fft_image

    lowpass_image = np.abs(np.fft.ifft2(lowpass_image))
    lowpass_image = lowpass_image / 255
    temp = int(i + 1)
    plt.subplot(dimension, dimension, temp)
    plt.title(f'Low pass img D0= {D0}')
    plt.imshow(lowpass_image, cmap='gray')
    plt.tight_layout()
    D0 = D0 + 2

plt.tight_layout()
plt.show()
