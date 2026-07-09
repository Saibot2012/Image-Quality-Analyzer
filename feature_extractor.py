import cv2
import numpy as np

from metrics import laplacian_sharpness
from heatmap import compute_patch_sharpness
from fft_analysis import compute_fft_features
from wavelets import compute_wavelets
from ml.noise import compute_gaussian_noise
def extract_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Laplacian
    sharpness = laplacian_sharpness(gray)
    norm_sharpness = sharpness / (sharpness + 1000)

    # FFT
    fft_features = compute_fft_features(gray)
    fft_ratio = fft_features["high_freq_ratio"]

    # Wavelets
    wavelet_features = compute_wavelets(gray)

    # Noise
    gaussian_noise = compute_gaussian_noise(gray)
    noise_rms = gaussian_noise["noise_rms"]

    # Heatmap
    heatmap = compute_patch_sharpness(gray)

    threshold = 500 + (noise_rms * 50)

    sharp_ratio = np.sum(heatmap > threshold) / heatmap.size
    consistency = np.mean(heatmap) / (np.max(heatmap) + 1e-6)

    # Exposure
    brightness = np.mean(gray) / 255.0
    exposure = max(0, 1 - (2 * (brightness - 0.5)) ** 2)

    # Detail quality
    detail_quality = fft_features["high_freq_ratio"] / (noise_rms + 1e-6)


    return {
        "laplacian": sharpness,
        "norm_sharpness": norm_sharpness,
        "heatmap": heatmap,
        "sharp_ratio": sharp_ratio,
        "consistency": consistency,
        "exposure": exposure,
        "fft_ratio": fft_ratio,
        "wavelet_ratio": wavelet_features["wavelet_ratio"],
        "noise": noise_rms,
        "detail_quality": detail_quality,
    } #dictionary