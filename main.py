import cv2
import numpy as np
import matplotlib.pyplot as plt

from metrics import laplacian_sharpness
from heatmap import compute_patch_sharpness
from overlay import create_overlay
from gradient_check import blur_direction_hint

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

def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image Not Found")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = laplacian_sharpness(gray)
    heatmap = compute_patch_sharpness(gray)
    heatmap_std = np.std(heatmap)
    threshold = np.percentile(heatmap, 70)
    blur_hint = blur_direction_hint(img)
    sharp_ratio = np.sum(heatmap > threshold) / heatmap.size
    "What fraction of the image is considered “sharp"

    overlay = create_overlay(img, heatmap)
    plt.figure(figsize=(15, 5))

    # Original
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Heatmap (if you still have it separately)
    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap, cmap="inferno")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()


    if sharpness > 1000:
        status = "Sharp Image"
    elif sharpness > 500:
        status = "Ok image"
    else:
        status = "Blur image"

    text = f"""
    === Image Quality Report ===
    Sharpness: {sharpness:.2f}
    Status: {status}
    Sharp Ratio: {sharp_ratio:.2f}
    Std: {heatmap_std:.3f}
    Blur Type Hint: {blur_hint}
    === End of Report ===
    """
    print(text)

    # print("=== Image Quality Report ===\n"
    #       "Sharpness score: ", sharpness)
    # print("Status: ", status)



if __name__ == "__main__":
    main("test_images/yn-6.jpg")