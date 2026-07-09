import cv2
import numpy as np


def compute_gaussian_noise(gray, kernel_size=5):

    # Low-pass filter
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Residual
    residual = gray.astype(np.float32) - blurred.astype(np.float32)

    # Root Mean Square (RMS)
    noise_rms = np.sqrt(np.mean(residual ** 2))

    return {
        "noise_rms": noise_rms
    }