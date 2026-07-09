import cv2
import joblib
import os 
import pandas as pd
IMAGE_FOLDER = "testimages2"
from feature_extractor import extract_features
from quality_report import (interpret_exposure, interpret_noise, interpret_sharpness)


log = open("prediction_results_stress_test.log", "w")
# Load trained model
model = joblib.load("ml/model.pkl")

def log_print(*args, **kwargs):
    print(*args, **kwargs)              # Print to terminal
    print(*args, **kwargs, file=log)    # Print to log file

# Load image
for filename in os.listdir(IMAGE_FOLDER):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".ARW")):
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
        "exposure": features["exposure"],
        "noise": features["noise"],
        "detail_quality": features["detail_quality"],

        
        

    }])

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)

    log_print(f"\n===== {filename} =====")
    # log_print(f"Laplacian   : {features['laplacian']:.2f}")
    # log_print(f"FFT Ratio   : {features['fft_ratio']:.6f}")
    # log_print(f"Wavelet     : {features['wavelet_ratio']:.6f}")
    # log_print(f"Sharp Ratio : {features['sharp_ratio']:.3f}")
    # log_print(f"Consistency : {features['consistency']:.3f}")
    # log_print(f"Exposure    : {features['exposure']:.3f}")  
    # log_print(f"Noise Rms   : {features['noise']:.2f}")
    # log_print(f"Detail Quality : {features['detail_quality']:.8f}")

    # log_print(f"Patch min : {features['heatmap'].min():.2f}")
    # log_print(f"Patch mean: {features['heatmap'].mean():.2f}")
    # log_print(f"Patch max : {features['heatmap'].max():.2f}")
    # log_print(f"Confidence : {confidence}")
    # log_print(f"Prediction : {'Sharp' if prediction == 1 else 'Blurry'}")
    # log_print(f"Probability (Blurry): {probabilities[0]:.2%}")
    # log_print(f"Probability (Sharp) : {probabilities[1]:.2%}")
    # log_print("-" * 40)
    sharp_title, sharp_msg = interpret_sharpness(
        features["laplacian"]
    )

    noise_title, noise_msg = interpret_noise(
        features["noise"]
    )

    exposure_title, exposure_msg = interpret_exposure(
            features["exposure"]
        )
    log_print("\n===== Image Assessment =====")

    log_print(f"Sharpness: {sharp_title}")
    log_print(f"{sharp_msg}")

    log_print(f"Noise: {noise_title}")
    log_print(f"{noise_msg}")

    log_print(f"Exposure: {exposure_title}")
    log_print(f"{exposure_msg}")
log.close()
