import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('Image/srity.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))
height, width = image.shape
def addNoise(image):
    noise_image = image.copy()
    noise = 0.02
    height, width = image.shape

    for h in range(height):
        for w in range(width):
            random_val = np.random.rand()
            if random_val < noise / 2:
                noise_image[h, w] = 0
            elif random_val < noise:
                noise_image[h, w] = 255
    return noise_image


noise_image = addNoise(image)
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(noise_image, cmap='gray')


def harmonic_geometric_mean(noise_image, height, width):
    n = 3
    pad_height = n // 2
    pad_width = n // 2
    harmonic = np.zeros_like(noise_image)
    geometric = np.zeros_like(noise_image)
    pad_image = np.pad(noise_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    for h in range(height):
        for w in range(width):
            tmp_window = pad_image[h:h + n, w:w + n]

            harmonic_weight = n * n / np.sum(1.0 / (tmp_window + 1e-3))  # harmonic weight
            harmonic[h, w] = harmonic_weight

            # for geometric mean filter
            geometric_weight = 0
            count_non_zero = 0
            for i in range(n):
                for j in range(n):
                    if tmp_window[i][j] > 0:
                        geometric_weight += np.log(tmp_window[i][j])
                        count_non_zero += 1
            if count_non_zero > 0:
                geometric_weight = np.exp(geometric_weight / count_non_zero)
            else:
                geometric_weight = 0

            # geometric_weight= np.prod(tmp_window) ** (1 /n*n)
            geometric[h, w] = geometric_weight

    return harmonic, geometric


def PSNR(original, noisy):
    mse = np.mean((original - noisy) ** 2)
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


harmonic_image, geometric_image = harmonic_geometric_mean(noise_image, height, width)
har = PSNR(image, harmonic_image)
med_original = PSNR(image, geometric_image)
plt.subplot(2, 2, 3)
plt.imshow(harmonic_image, cmap='gray')
plt.title(f'harmonic - PSNR: {har:.2f} dB')
plt.subplot(2, 2, 4)
plt.imshow(geometric_image, cmap='gray')
plt.title(f'geometric - PSNR: {med_original:.2f} dB')
plt.tight_layout()
plt.show()


