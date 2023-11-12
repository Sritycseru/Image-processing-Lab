import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('Image/srity.jpg',cv.IMREAD_GRAYSCALE)
image = cv.resize(image,(512,512))

row,column = image.shape
mean_of_distribution = 0
standard_deviation = 25
noise = np.random.normal(mean_of_distribution,standard_deviation,image.shape).astype(np.uint8)


noise_image = cv.add(image,noise)

FFT_image = np.fft.fftshift(np.fft.fft2(noise_image)) #2D DFT

D0 = 10 #d0 = cut-off frequency
filter_order = 4

def gaussian_filter(FFT_image):
    m,n = FFT_image.shape
    gaussian_image = np.zeros((m,n))

    for u in range(m):
        for v in range(n):
            D = np.sqrt((u-m/2)**2+(v-n/2)**2)
            gaussian_image[u,v] = np.exp(-((D**2)/(2*D0**2)))

    gaussian_constant = gaussian_image*FFT_image
    gaussian_image = np.abs(np.fft.ifft2(gaussian_constant))
    gaussian_image = gaussian_image/255
    return  gaussian_image

def butterworth_filter(FFT_image):
    m, n = FFT_image.shape
    Butterworth_image = np.zeros((m, n))
    for u in range(m):
        for v in range(n):
            D = np.sqrt((u - m / 2) ** 2 + (v - n / 2) ** 2)
            Butterworth_image[u, v] = 1 / (1 + (D / D0) ** (2 * filter_order))
    Butterworth_image = FFT_image * Butterworth_image
    Butterworth_image = np.abs(np.fft.ifft2(Butterworth_image))
    Butterworth_image = Butterworth_image / 255
    return Butterworth_image

plt.subplot(131)
plt.imshow(image,cmap='gray')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(gaussian_filter(FFT_image),cmap='gray')
plt.title('Gaussian Filter Image')

plt.subplot(133)
plt.imshow(butterworth_filter(FFT_image),cmap='gray')
plt.title('Butterworth Filter Image')

plt.tight_layout()
plt.show()