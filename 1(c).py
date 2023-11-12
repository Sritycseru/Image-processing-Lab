import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Image/srity.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure the image is 512x512 pixels
image = cv2.resize(image, (512, 512))

# Plot the histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 6))
plt.title("Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(hist)
plt.xlim([0, 256])
#threshold for segmentation
# we use a fixed threshold here.
threshold_value = 128

# Create a binary mask using the threshold
binary_mask = np.where(image >= threshold_value, 255, 0).astype(np.uint8)

# Display the original image and the segmented image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title("Segmented Image (Thresholded)")

plt.show()
