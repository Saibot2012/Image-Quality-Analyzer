import cv2
import numpy as np
import matplotlib.pyplot as plt

from metrics import laplacian_sharpness
from heatmap import compute_patch_sharpness
from overlay import create_overlay
from gradient_check import blur_direction_hint
from fft_analysis import (compute_fft_visualization, compute_fft_features)
from wavelets import compute_wavelets

def show_heatmap(original, heatmap, patch_size=32):
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Sharpness Heatmap")
    plt.imshow(heatmap_resized, cmap="inferno")
    plt.axis("off")

    plt.show()
def draw_report(ax, img, heatmap, title, stats):
    overlay = create_overlay(img, heatmap)

    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(title)
    ax[0].axis("off")

    ax[1].imshow(heatmap, cmap="inferno")
    ax[1].set_title("Heatmap")
    ax[1].axis("off")

    ax[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    ax[3].text(
        0.1, 0.5,
        "\n".join([
            f"Sharpness: {stats['sharpness']:.1f}",
            f"Sharp Ratio: {stats['sharp_ratio']:.2f}",
            f"Consistency: {stats['consistency']:.3f}",
            f"Exposure: {stats['exposure']:.3f}",
            f"Score: {stats['score']:.3f}",
            f"FFT Ratio: {stats['fft_ratio']:.4f}",
            f"Wavelet: {stats['wavelet_ratio']:.6f}",
        ]),
        fontsize=12
    )
    ax[3].axis("off")


def main(image_paths):
    results = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Image Not Found: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        wavelet = compute_wavelets(gray)
        fft_features = compute_fft_features(gray)
        sharpness = laplacian_sharpness(gray)
        norm_sharpness = sharpness / (sharpness + 1000)

        heatmap = compute_patch_sharpness(gray)

        sharp_ratio = np.sum(heatmap > 500) / heatmap.size
        consistency = np.mean(heatmap) / (np.max(heatmap) + 1e-6)

        brightness = np.mean(gray) / 255.0
        exposure = max(0, 1 - (2 * (brightness - 0.5)) ** 2)

        score = (
            0.4 * norm_sharpness +
            0.25 * sharp_ratio +
            0.15 * consistency +
            0.2 * exposure
        )

        results.append({
            "path": path,
            "img": img,
            "heatmap": heatmap,
            "sharpness": sharpness,
            "sharp_ratio": sharp_ratio,
            "consistency": consistency,
            "exposure": exposure,
            "score": score,
            "fft_ratio": fft_features["high_freq_ratio"],
            "wavelet_ratio": wavelet["wavelet_ratio"],
        })

        print(f"{path}")
        print(f"Laplacian : {sharpness:.2f}")
        print(f"FFT Ratio : {fft_features['high_freq_ratio']:.6f}")
        print(f"Wavelet   : {wavelet['wavelet_ratio']:.6f}")

        print()
    # --- UI LAYOUT ---
    fig = plt.figure(figsize=(14, 4 * len(results)))

    for i, r in enumerate(results):
        ax = [
            plt.subplot(len(results), 4, i * 4 + 1),
            plt.subplot(len(results), 4, i * 4 + 2),
            plt.subplot(len(results), 4, i * 4 + 3),
            plt.subplot(len(results), 4, i * 4 + 4),
        ]

        draw_report(ax, r["img"], r["heatmap"], r["path"], r)

    plt.tight_layout()
    plt.show()









if __name__ == "__main__":
    main(["test_images/test (1).jpeg","test_images/test (3).jpeg", "test_images/test (6).jpeg"])