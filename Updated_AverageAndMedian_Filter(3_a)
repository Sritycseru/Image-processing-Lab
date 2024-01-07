import random

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def add_noise(image):
    noisy_image = image.copy()
    row,col = noisy_image.shape

    number_of_salt_noise = 50000
    for i in range(number_of_salt_noise):
        x = random.randint(0,row-1)
        y = random.randint(0,col-1)
        noisy_image[x,y] = 255

    number_of_peeper_noise = 50000
    for i in range(number_of_peeper_noise):
        x = random.randint(0,row-1)
        y = random.randint(0,col-1)
        noisy_image[x,y] = 0

    return noisy_image

def pad_image(image, karnel):
    row,col = image.shape
    padding_row = (karnel-1)+row
    padding_column = (karnel-1)+col
    padding_image = np.zeros((padding_row,padding_column))

    for i in range(karnel,padding_row-karnel):
        for j in range(karnel,padding_column-karnel):
            padding_image[i, j] = image[i - karnel, j - karnel]

    return padding_image

def averaging_filter(noisy_image,mask_size):
    averaging_filter_image = np.zeros_like(noisy_image)
    padding_image = pad_image(noisy_image,mask_size)
    mask = np.full((mask_size, mask_size), (1 / (mask_size * mask_size)))
    row,col = padding_image.shape
    mask_row,mask_col = mask_size,mask_size
    pad_size = int((mask_row-1)/2)
    start_row = pad_size
    end_row = row-pad_size
    start_col = pad_size
    end_col = col-pad_size

    for i in range(start_row,end_row):
        for j in range(start_col,end_col):
            window = padding_image[i-pad_size:i+pad_size+1,j-pad_size:j+pad_size+1]
            window = window*mask
            sum_window = np.sum(window)
            averaging_filter_image[i-pad_size][j-pad_size] = sum_window

    return averaging_filter_image

def median_filter(noisy_image,mask_size):
    median_filter_image = np.zeros_like(noisy_image)
    padding_image = pad_image(noisy_image, mask_size)
    mask = np.full((mask_size, mask_size), 1)
    row, col = padding_image.shape
    mask_row, mask_col = mask_size, mask_size
    pad_size = int((mask_row - 1) / 2)
    start_row = pad_size
    end_row = row - pad_size
    start_col = pad_size
    end_col = col - pad_size

    for i in range(start_row,end_row):
        for j in range(start_col,end_col):
            window = padding_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            median_value = np.median(window*mask)
            median_filter_image[i-pad_size][j-pad_size] = median_value
    return  median_filter_image

#PSNR = 10log10(Max^2/MSE) MSE-> Mean Squared Error
def PSNR(original_image,filtered_image):
    mse = np.mean((original_image-filtered_image)**2)
    max = 255
    psnr = 10*np.log10(max**2/mse)
    return psnr


image = cv.imread('Images/lena.jpg',0)
image = cv.resize(image,(512,512))
noisy_image = add_noise(image)
Padding_image = pad_image(noisy_image,5)
height,width = Padding_image.shape
average_spatial_filter = averaging_filter(noisy_image,5)
average_spatial_filter_psnr = PSNR(image,average_spatial_filter)
median_spatial_filter = median_filter(noisy_image,5)
median_spatial_filter_psnr = PSNR(image,median_spatial_filter)

plt.subplot(2,2,1)
plt.imshow(image,cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(noisy_image,cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(average_spatial_filter,cmap='gray')
plt.title(f'Average Filterd Image  PSNR: {average_spatial_filter_psnr:.2f} DB')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(median_spatial_filter,cmap='gray')
plt.title(f'Median Filter Image  PSNR: {median_spatial_filter_psnr:.2f} DB')
plt.axis('off')
plt.show()
