import cv2
import numpy as np

def compute_patch_sharpness(img, patch_size=32):
    h, w = img.shape[:2]
    heatmap = np.zeros((h // patch_size, w // patch_size))

    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            score = cv2.Laplacian(patch, cv2.CV_64F).var()
            heatmap[i // patch_size, j // patch_size] = score

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    return heatmap_resized  # raw values, normalise separately when visualising