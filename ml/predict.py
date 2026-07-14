# ----- Imports -----

import cv2
import hashlib
import joblib
import os
import pandas as pd

from analyzer.feature_extractor import extract_features
from detection.face_analysis import detect_faces
from detection.eye_analysis import detect_eye_state
from visualization.visualization import draw_face_boxes
from analyzer.report_generator import generate_report, generate_verdict

from analyzer.score import (
    score_higher_better,
    score_lower_better,
    SHARPNESS_THRESHOLDS,
    NOISE_THRESHOLDS,
    CONTRAST_THRESHOLDS,
    overall_score
)

from analyzer.quality_report import (
    brightness_report,
    interpret_saturation,
    interpret_temperature,
)

# ----- Setup -----
IMAGE_FOLDER = "testimages4"

os.makedirs("results", exist_ok=True)

model = joblib.load("ml/model.pkl")

os.makedirs("logs", exist_ok=True)
report_log = open("logs/simplified.log", "w", encoding="utf-8")
analysis_log = open("logs/analysis.log", "w", encoding="utf-8") 



# ----- Loggers -----
def report_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    print(*args, **kwargs, file=report_log, flush=True)

def dev_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    print(*args, **kwargs, file=analysis_log, flush=True)

def common_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=report_log)
    print(*args, **kwargs, file=analysis_log)

def score_ml_sharpness(prediction, confidence):
    """
    prediction:
        1 = Sharp
        0 = Blurry
    """

    if prediction == 1:
        if confidence >= 0.95:
            return {"grade": "Excellent", "score": 100}
        elif confidence >= 0.85:
            return {"grade": "Good", "score": 90}
        elif confidence >= 0.70:
            return {"grade": "Fair", "score": 75}
        else:
            return {"grade": "Poor", "score": 50}

    else:
        if confidence >= 0.85:
            return {"grade": "Very Poor", "score": 20}
        else:
            return {"grade": "Poor", "score": 50}

# ----- Start loop -----
for filename in os.listdir(IMAGE_FOLDER):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)

    img = cv2.imread(image_path)

    if img is None:
        common_print(f"Could not read {filename}")
        continue


# ----- Extract features -----
    features = extract_features(img)

# ----- ML Prediction -----

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



# ----- Subject Detection -----
    face_result = detect_faces(img)

    eye_visual_results = []

    if face_result["face_count"] == 0:

        common_print(
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



            else:
                eye_visual_results.append(
                    {
                        "status": eye_result["eye_status"]
                    }
                )
                common_print(f"Face {i+1}: {eye_result     ['eye_status']}"
                        )


# ----- Quality Scores -----
    scores = {
        "Sharpness": score_ml_sharpness(
        prediction,
        confidence
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


# ----- Report Generation -----
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
    report = generate_report(
    scores,
    face_result,
    eye_visual_results,
    lighting,
    sat_title,
    temp_title
    )
    verdict = generate_verdict(
    overall,
    report["problems"]
    )

#----- Visualization -----
    annotated = draw_face_boxes(
    img,
    face_result,
    eye_visual_results,
    overall,
    scores,
    verdict
)
    cv2.imwrite(
    f"results/visual_{filename}",
    annotated
    )

#----- Logging -----
    common_print(f"\n===== {filename} =====")

    common_print(
        f"\nOverall Score: {overall}/100"
    )
    common_print("\n===== Assessment Report =====\n")

    common_print("General")
    for line in report["general"]:
        common_print(f"• {line}")

    common_print()
    common_print("Problems")

    if report["problems"]:
        for line in report["problems"]:
            common_print(f"⚠ {line}")
    else:
        common_print("✓ No significant issues detected.")
        common_print("\n===== Suggestions =====")

    common_print()
    common_print("Suggestions")

    if report["suggestions"]:
        for line in report["suggestions"]:
            common_print(f"• {line}")
    else:
        common_print("No recommendations.")
    dev_print()
    common_print("========== END OF SIMPLIFIED REPORT ==========")
    dev_print()
    dev_print()

    dev_print("============= Values Analysis =============")

    dev_print(f"Laplacian   : {features['laplacian']:.2f}")
    dev_print(f"FFT Ratio   : {features['fft_ratio']:.6f}")
    dev_print(f"Wavelet     : {features['wavelet_ratio']:.6f}")
    dev_print(f"Sharp Ratio : {features['sharp_ratio']:.3f}")
    dev_print(f"Brightness     : {features['brightness']:.3f}")
    dev_print(f"Shadow Clip    : {features['shadow_clip']:.2%}")
    dev_print(f"Highlight Clip : {features['highlight_clip']:.2%}")
    dev_print(f"Consistency : {features['consistency']:.3f}")
    dev_print(f"Noise Rms   : {features['noise']:.2f}")
    dev_print(f"Detail Quality : {features['detail_quality']:.8f}")

    dev_print(f"Patch min : {features['heatmap'].min():.2f}")
    dev_print(f"Patch mean: {features['heatmap'].mean():.2f}")
    dev_print(f"Patch max : {features['heatmap'].max():.2f}")
    dev_print(f"Confidence : {confidence}")
    dev_print(f"Prediction : {'Sharp' if prediction == 1 else 'Blurry'}")
    dev_print(f"Probability (Blurry): {probabilities[0]:.2%}")
    dev_print(f"Probability (Sharp) : {probabilities[1]:.2%}")
    dev_print()
    dev_print("-" * 40)


    dev_print("\n===== Quality Scores =====")

    for metric, result in scores.items():

        dev_print(
            f"{metric}: {result['grade']} ({result['score']}/100)"
        )



    for msg in face_result["log_messages"]:
        dev_print(msg)

    dev_print("\n[Eye Detection]")

dev_print("\n[Eye Detection]")

for i, eye in enumerate(eye_visual_results):

    if "ear" in eye:
        dev_print(
            f"Face {i+1}: {eye['status']} (EAR={eye['ear']:.3f})"
        )
    else:
        dev_print(
            f"Face {i+1}: {eye['status']}"
        )
    
    dev_print()
    dev_print("\n===== General Report =====")

    for line in report["general"]:
        dev_print(f"• {line}")
    dev_print()
    dev_print()
    dev_print(f"[Saturation]")
    dev_print(f"Value : {features['saturation']:.2f}")
    dev_print(f"Result : {sat_title}")
    dev_print(f"{sat_msg}\n")

    dev_print("[Colour Temperature]")
    dev_print(f"Value : {features['temperature']:.2f}")
    dev_print(f"Result : {temp_title}")
    dev_print(f"{temp_msg}\n")

    dev_print(f"[Lighting]")
    dev_print(f"Result : {lighting}")
    dev_print(f"{lighting_desc}")
    dev_print(f"Recommendation: {lighting_tip}")
    dev_print()
    dev_print("========= END OF COMPREHENSIVE REPORT =========")

    

    
    


