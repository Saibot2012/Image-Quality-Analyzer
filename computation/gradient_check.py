import cv2
import numpy as np

def blur_direction_hint(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag_x = np.mean(np.abs(gx))
    mag_y = np.mean(np.abs(gy))

    ratio = mag_x / (mag_y + 1e-6)

    if ratio > 1.3 or ratio < 0.75:
        return "Possible motion blur"
    else:
        return "Likely defocus blur or uniform softness"