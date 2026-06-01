import cv2
import numpy as np

def create_overlay(img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # normalise to 0–1 first, then scale to 0–255
    heatmap_norm = heatmap_resized / (heatmap_resized.max() + 1e-6)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)

    blended = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    return blended