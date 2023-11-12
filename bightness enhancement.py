import cv2 as cv
import numpy as np

img = cv.imread('Image/srity.jpg', 0)

[row, col] = img.shape


cv.namedWindow('Downsampling',cv.WINDOW_NORMAL)
cv.resizeWindow('Downsampling', 512, 512)
cv.imshow('Downsampling',img)

#down sampling
factor = 2
while True:
    key = cv.waitKey(0)
    if key == ord('q') or key == 27:
        break
    img2 = np.zeros((row // factor, col // factor), dtype=np.uint8)
    for i in range(0, row, factor):
        for j in range(0, col, factor):
            try:
                img2[i//factor][j//factor] = img[i][j]
            except IndexError:
                pass

    cv.imshow('Downsampling', img2)
    factor *= 2

cv.destroyAllWindows()