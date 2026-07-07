import cv2
import joblib
import os 
import pandas as pd
IMAGE_FOLDER = "test_images"
from feature_extractor import extract_features

log = open("prediction_results.log", "w")
# Load trained model
model = joblib.load("ml/model.pkl")

def log_print(*args, **kwargs):
    print(*args, **kwargs)              # Print to terminal
    print(*args, **kwargs, file=log)    # Print to log file

# Load image
for filename in os.listdir(IMAGE_FOLDER):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png", "webp")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)

    img = cv2.imread(image_path)

    if img is None:
        log_print(f"Could not read {filename}")
        continue

    # Extract features
    features = extract_features(img)

    # Create feature vector
    X = pd.DataFrame([{
        "laplacian": features["laplacian"],
        "fft_ratio": features["fft_ratio"],
        "wavelet_ratio": features["wavelet_ratio"],
        "sharp_ratio": features["sharp_ratio"],
        "consistency": features["consistency"],
        "exposure": features["exposure"]
    }])

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)

    log_print(f"\n===== {filename} =====")
    log_print(f"Laplacian   : {features['laplacian']:.2f}")
    log_print(f"FFT Ratio   : {features['fft_ratio']:.6f}")
    log_print(f"Wavelet     : {features['wavelet_ratio']:.6f}")
    log_print(f"Sharp Ratio : {features['sharp_ratio']:.3f}")
    log_print(f"Consistency : {features['consistency']:.3f}")
    log_print(f"Exposure    : {features['exposure']:.3f}")  



    log_print(f"Prediction : {'Sharp' if prediction == 1 else 'Blurry'}")
    log_print(f"Probability (Blurry): {probabilities[0]:.2%}")
    log_print(f"Probability (Sharp) : {probabilities[1]:.2%}")
    log_print("-" * 40)
log.close()
