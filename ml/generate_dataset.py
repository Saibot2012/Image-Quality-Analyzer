import os
import csv
import cv2
import matplotlib.pyplot as plt

from feature_extractor import extract_features
IMAGE_FOLDER = "dataset/images"
CSV_FILE = "ml/dataset.csv"
sharp_count = 0
blurry_count = 0

header = [
    "filename",
    "laplacian",
    "fft_ratio",
    "wavelet_ratio",
    "sharp_ratio",
    "consistency",
    "exposure",
    "label"
]

# Create CSV and write header
with open(CSV_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)

    # Loop through every image
    for filename in os.listdir(IMAGE_FOLDER):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(IMAGE_FOLDER, filename)

        img = cv2.imread(image_path)

        if img is None:
            print(f"Could not read {filename}")
            continue

        features = extract_features(img)

        # Show image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(filename)
        plt.axis("off")
        plt.show()

        # Ask user for label
        while True:
            label = input("Sharp? (1 = Sharp, 0 = Blurry): ")

            if label in ["0", "1"]:
                break

            print("Please enter either 0 or 1.")
        if label == "1":
            sharp_count += 1
        else:
            blurry_count += 1

        # Write one row
        writer.writerow([
            filename,
            features["sharpness"],
            features["fft_ratio"],
            features["wavelet_ratio"],
            features["sharp_ratio"],
            features["consistency"],
            features["exposure"],
            int(label)
        ])

print(f"\nDataset saved to {CSV_FILE}")

print("\n===== Dataset Summary =====")
print(f"Sharp Images : {sharp_count}")
print(f"Blurry Images: {blurry_count}")
print(f"Total Images : {sharp_count + blurry_count}")