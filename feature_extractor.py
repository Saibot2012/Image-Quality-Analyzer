import cv2
import numpy as np

from metrics import laplacian_sharpness
from heatmap import compute_patch_sharpness
from fft_analysis import compute_fft_features
from wavelets import compute_wavelets


def extract_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ----- Laplacian -----
    sharpness = laplacian_sharpness(gray)
    norm_sharpness = sharpness / (sharpness + 1000)

    # ----- Heatmap -----
    heatmap = compute_patch_sharpness(gray)

    sharp_ratio = np.sum(heatmap > 500) / heatmap.size
    consistency = np.mean(heatmap) / (np.max(heatmap) + 1e-6)

    # ----- Exposure -----
    brightness = np.mean(gray) / 255.0
    exposure = max(0, 1 - (2 * (brightness - 0.5)) ** 2)

    # ----- FFT -----
    fft_features = compute_fft_features(gray)

    # ----- Wavelets -----
    wavelet_features = compute_wavelets(gray)

    return {
        "laplacian": sharpness,
        "norm_sharpness": norm_sharpness,
        "heatmap": heatmap,
        "sharp_ratio": sharp_ratio,
        "consistency": consistency,
        "exposure": exposure,
        "fft_ratio": fft_features["high_freq_ratio"],
        "wavelet_ratio": wavelet_features["wavelet_ratio"],
    } #dictionary