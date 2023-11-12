import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("Image/srity.jpg", cv2.IMREAD_GRAYSCALE)

window_size = (512, 512)

# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Image", *window_size)

# cv2.imshow("Image", image)
# cv2.waitKey(1000)

num_bits = 1
quantization_step = 255 / (2**num_bits - 1)
quantized_images = []
while num_bits <= 8:
    quantized_image = (image / quantization_step).astype(np.uint8) * quantization_step
    quantized_images.append(quantized_image)

    num_bits += 1
    quantization_step = 255 / (2**num_bits - 1)
    # cv2.imshow("Image", cv2.resize(quantized_image, window_size))
    # cv2.waitKey(1000)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('Quantized Images', fontsize=30)

for i, ax in enumerate(axes.flat):
    ax.imshow(quantized_images[i], cmap='gray')
    ax.set_title(f'{i+1} bits')
    ax.axis('off')

#plt.tight_layout()
plt.show()