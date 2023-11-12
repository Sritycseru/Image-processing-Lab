import cv2
import numpy as np
import matplotlib.pyplot as plt

original_image = cv2.imread("Image/srity.jpg", 0)

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()
    total_pixels = image.size

    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

salt_prob = 0.01
pepper_prob = 0.01

noisy_image = add_salt_and_pepper_noise(original_image, salt_prob, pepper_prob)

plt.subplot(3, 3, 1)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap="gray")

def calculateAverage(i, j, row, mask):
    average = 0

    for x in range(-row, row + 1):
        for y in range(-row, row + 1):
            #if 0 <= i + x < image_height and 0 <= j + y < image_width:
                average += noisy_image[i + x, j + y]

    return average / mask

def calculateMedian(i, j, row):
    result = []

    for x in range(-row, row + 1):
        for y in range(-row, row + 1):
            #if 0 <= i + x < image_height and 0 <= j + y < image_width:
                result.append(noisy_image[i + x, j + y])

    result = sorted(result)
    length = len(result)

    median = 0

    if length % 2 != 0:
        median = (result[length // 2] + result[(length // 2) - 1]) / 2
    else:
        median = result[length // 2]

    return median

mask_size = 5
mask_row = mask_size // 2
image_height, image_width = noisy_image.shape

average_filter = noisy_image.copy()
median_filter = noisy_image.copy()

for i in range(mask_row, image_height - mask_row):
    for j in range(mask_row, image_width - mask_row):
        average_filter[i, j] = calculateAverage(i, j, mask_row, mask_size * mask_size)
        median_filter[i, j] = calculateMedian(i, j, mask_row)

plt.subplot(3, 3, 2)
plt.title("Average Filter (5x5)")
plt.imshow(average_filter, cmap="gray")

plt.subplot(3, 3, 3)
plt.title("Median Filter (5x5)")
plt.imshow(median_filter, cmap="gray")

plt.show()
