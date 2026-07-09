import cv2
import joblib
import os
import pandas as pd

from feature_extractor import extract_features

IMAGE_FOLDER = "dataset/images"
CSV_FILE = "ml/dataset.csv"

model = joblib.load("ml/model.pkl")

data = pd.read_csv(CSV_FILE)

wrong = 0
total = 0
true_blurry = 0
true_sharp = 0

correct_blurry = 0
correct_sharp = 0

print("\n===== Incorrect Predictions =====")

for _, row in data.iterrows():

    filename = row["filename"]
    actual = row["label"]

    image_path = os.path.join(IMAGE_FOLDER, filename)

    img = cv2.imread(image_path)

    if img is None:
        continue

    features = extract_features(img)

    X = pd.DataFrame([{
        "laplacian": features["laplacian"],
        "fft_ratio": features["fft_ratio"],
        "wavelet_ratio": features["wavelet_ratio"],
        "sharp_ratio": features["sharp_ratio"],
        "consistency": features["consistency"],
        "exposure": features["exposure"],
        "noise": features["noise"],
        "detail_quality": features["detail_quality"],
    }])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    total += 1

    if prediction != actual:

        wrong += 1

        print("\n--------------------------")
        print(filename)

        print("Actual     :", "Sharp" if actual == 1 else "Blurry")
        print("Prediction :", "Sharp" if prediction == 1 else "Blurry")

        print(
            "Confidence :",
            f"{max(probability):.2%}"
        )

        print("Laplacian  :", round(features["laplacian"], 2))
        print("FFT        :", round(features["fft_ratio"], 6))
        print("Wavelet    :", round(features["wavelet_ratio"], 6))
        print("Sharp Ratio:", round(features["sharp_ratio"], 3))
        print("Noise RMS  :", round(features["noise"], 2))
    if actual == 1:
        true_sharp += 1
        if prediction == 1:
            correct_sharp += 1

    else:
        true_blurry += 1
        if prediction == 0:
            correct_blurry += 1


print("\n============================")
print(f"Wrong: {wrong}/{total}")
print(f"Accuracy: {(1-wrong/total):.2%}")

print("\n===== Class Accuracy =====")

print(
    f"Sharp accuracy: "
    f"{correct_sharp}/{true_sharp} "
    f"({correct_sharp/true_sharp:.2%})"
)

print(
    f"Blurry accuracy: "
    f"{correct_blurry}/{true_blurry} "
    f"({correct_blurry/true_blurry:.2%})"
)