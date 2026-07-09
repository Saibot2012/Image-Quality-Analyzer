import cv2
import numpy as np

def compute_noise(gray, patch_size=32, edge_threshold=300):
    h, w = gray.shape[:2]

    noise_values = []

    for i in range(0, h-patch_size, patch_size):
        for j in range(0, w-patch_size, patch_size):

            patch = gray[i:i+patch_size, j:j+patch_size]

            laplacian = cv2.Laplacian(patch,cv2.CV_64F).var()

            if laplacian < edge_threshold:
                noise_std = np.std(patch)
                noise_values.append(noise_std)
    
    if len(noise_values) == 0:
        mean_noise = 0
    else:
        mean_noise = np.mean(noise_values)

    return{
        "noise_std": mean_noise,
        "num_smooth_patches": len(noise_values)
    }