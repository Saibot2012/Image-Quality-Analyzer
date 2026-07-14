import cv2
import numpy as np

from computation.metrics import laplacian_sharpness
from unused.heatmap import compute_patch_sharpness
from computation.fft_analysis import compute_fft_features
from computation.wavelets import compute_wavelets
from ml.noise import compute_gaussian_noise
from visualization.brightness_analysis import analyze_brightness
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

    brightness = np.mean(gray) / 255.0
    exposure = max(0, 1 - (2 * (brightness - 0.5)) ** 2)
    #brightness
    brightness_info = analyze_brightness(gray)

    # Detail quality
    detail_quality = fft_features["high_freq_ratio"] / (noise_rms + 1e-6)

    #Contrast
    contrast = np.std(gray)

    #Saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])

    #color temperature
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    r_mean = np.mean(rgb[:, :, 0])
    b_mean = np.mean(rgb[:, :, 2])

    temperature = r_mean - b_mean

    return {
        "laplacian": sharpness,
        "norm_sharpness": norm_sharpness,
        "heatmap": heatmap,
        "sharp_ratio": sharp_ratio,
        "consistency": consistency,
        "exposure": exposure,
        "brightness": brightness_info["brightness"],
        "shadow_clip": brightness_info["shadow_clip"],
        "highlight_clip": brightness_info["highlight_clip"],
        "fft_ratio": fft_ratio,
        "wavelet_ratio": wavelet_features["wavelet_ratio"],
        "noise": noise_rms,
        "detail_quality": detail_quality,
        "contrast": contrast,
        "saturation": saturation,
        "temperature": temperature,
    } #dictionary