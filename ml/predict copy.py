import cv2
import os
import joblib
import os 
import pandas as pd
import hashlib

IMAGE_FOLDER = "testimages4"

from analyzer.feature_extractor import extract_features
from detection.face_analysis_mediapipe import detect_faces
from analyzer.quality_report import (brightness_report, interpret_noise, interpret_sharpness, interpret_contrast, interpret_saturation, interpret_temperature)
from detection.eye_analysis import detect_eye_state
from visualization.visualization import draw_face_boxes
from analyzer.score import (
    score_higher_better,
    score_lower_better,
    SHARPNESS_THRESHOLDS,
    NOISE_THRESHOLDS,
    CONTRAST_THRESHOLDS,
    overall_score
)
from analyzer.report_generator import generate_report
log = open("prediction_results_brightness_test.log", "w")
# Load trained model
model = joblib.load("ml/model.pkl")

def log_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    print(*args, **kwargs, file=log, flush=True)

# Load image
for filename in os.listdir(IMAGE_FOLDER):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)

    img = cv2.imread(image_path)

    if img is None:
        log_print(f"Could not read {filename}")
        continue


    # Extract features
    features = extract_features(img)
    scores = {

        "Sharpness": score_higher_better(
            features["laplacian"],
            SHARPNESS_THRESHOLDS
        ),

        "Noise": score_lower_better(
            features["noise"],
            NOISE_THRESHOLDS
        ),

        "Contrast": score_higher_better(
            features["contrast"],
            CONTRAST_THRESHOLDS
        ),

    }
    overall = overall_score(

    *[
        metric["score"]
        for metric in scores.values()
    ]

    )

     # Detect faces
    face_result = detect_faces(img)

    eye_visual_results = []

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

    

    
    overall = overall_score(

    *[metric["score"] for metric in scores.values()]

    )
    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)

    log_print(f"\n===== {filename} =====")
    log_print(filename, img.shape)
    log_print(f"Laplacian   : {features['laplacian']:.2f}")
    log_print(f"FFT Ratio   : {features['fft_ratio']:.6f}")
    log_print(f"Wavelet     : {features['wavelet_ratio']:.6f}")
    log_print(f"Sharp Ratio : {features['sharp_ratio']:.3f}")
    log_print(f"Brightness     : {features['brightness']:.3f}")
    log_print(f"Shadow Clip    : {features['shadow_clip']:.2%}")
    log_print(f"Highlight Clip : {features['highlight_clip']:.2%}")
    log_print(f"Consistency : {features['consistency']:.3f}")
    log_print(f"Noise Rms   : {features['noise']:.2f}")
    log_print(f"Detail Quality : {features['detail_quality']:.8f}")

    log_print(f"Patch min : {features['heatmap'].min():.2f}")
    log_print(f"Patch mean: {features['heatmap'].mean():.2f}")
    log_print(f"Patch max : {features['heatmap'].max():.2f}")
    log_print(f"Confidence : {confidence}")
    log_print(f"Prediction : {'Sharp' if prediction == 1 else 'Blurry'}")
    log_print(f"Probability (Blurry): {probabilities[0]:.2%}")
    log_print(f"Probability (Sharp) : {probabilities[1]:.2%}")
    log_print("-" * 40)
    sharp_title, sharp_msg = interpret_sharpness(
        features["laplacian"]
    )

    noise_title, noise_msg = interpret_noise(
        features["noise"]
    )
    contrast_title, contrast_msg = interpret_contrast(
    features["contrast"]
    )

    lighting, lighting_desc, lighting_tip = brightness_report(
    features["brightness"],
    features["shadow_clip"],
    features["highlight_clip"]
    )
    sat_title, sat_msg = interpret_saturation(
    features["saturation"]
    )

    temp_title, temp_msg = interpret_temperature(
    features["temperature"]
    )
    
    os.makedirs("results", exist_ok=True)

    log_print("\n===== Image Summary =====\n")
    
    
    log_print("[Subject Detection]")

    log_print("\n===== Quality Scores =====")

    for metric, result in scores.items():

        log_print(
            f"{metric}: {result['grade']} ({result['score']}/100)"
        )

    log_print(
        f"\nOverall Score: {overall}/100"
    )

    for msg in face_result["log_messages"]:
        log_print(msg)

    log_print("\n[Eye Detection]")


    if face_result["face_count"] == 0:

        log_print(
            "No faces available for eye analysis"
        )
    else:
        for i, box in enumerate(face_result["face_boxes"]):

            x, y, w, h = box

            padding = 0.25

            x1 = max(0, int(x - w*padding))
            y1 = max(0, int(y - h*padding))

            x2 = min(img.shape[1], int(x+w+w*padding))
            y2 = min(img.shape[0], int(y+h+h*padding))

            face_crop = img[y1:y2, x1:x2]
            base_name = os.path.splitext(filename)[0]

            # Debug save
            

            eye_result = detect_eye_state(face_crop)


            if "eye_results" in eye_result:

                eye = eye_result["eye_results"][0]

                eye_visual_results.append(eye)

                log_print(
                    f"Face {i+1}: {eye['status']} (EAR={eye['ear']:.3f})"
                )   

            else:
                eye_visual_results.append(
                    {
                        "status": eye_result["eye_status"]
                    }
                )
                log_print(f"Face {i+1}: {eye_result     ['eye_status']}"
                        )
        annotated = draw_face_boxes(
            img,
            face_result,
            eye_visual_results
        )
        cv2.imwrite(
        f"results/visual_{filename}",
        annotated
        )
        log_print()
        log_print("\n===== General Report =====")

        for line in report["general"]:
            log_print(f"• {line}")
        # log_print(f"[Sharpness]")
        # log_print(f"Result : {sharp_title}")
        # log_print(f"{sharp_msg}\n")

        # log_print(f"[Contrast]")
        # log_print(f"Value: {features['contrast']}")
        # log_print(f"Result : {contrast_title}")
        # log_print(f"{contrast_msg}\n")

        log_print(f"[Saturation]")
        log_print(f"Value : {features['saturation']:.2f}")
        log_print(f"Result : {sat_title}")
        log_print(f"{sat_msg}\n")

        log_print("[Colour Temperature]")
        log_print(f"Value : {features['temperature']:.2f}")
        log_print(f"Result : {temp_title}")
        log_print(f"{temp_msg}\n")

        log_print(f"[Image Grain]")
        log_print(f"Result : {noise_title}")
        log_print(f"{noise_msg}\n")

        # log_print(f"[Lighting]")
        # log_print(f"Result : {lighting}")
        # log_print(f"{lighting_desc}")
        # log_print(f"Recommendation: {lighting_tip}")
        log_print("\n===== Problems =====")

        if report["problems"]:
            for line in report["problems"]:
                log_print(f"⚠ {line}")
        else:
            log_print("✓ No significant issues detected.")
            log_print("\n===== Suggestions =====")

        if report["suggestions"]:
            for line in report["suggestions"]:
                log_print(f"• {line}")
        else:
            log_print("No recommendations.")
log.close()
