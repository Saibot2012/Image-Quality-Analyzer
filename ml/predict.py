# ----- Imports -----

import csv
import numpy as np
import cv2
import hashlib
import joblib
import os
import pandas as pd
import shutil
from datetime import datetime

from analyzer.feature_extractor import extract_features
from detection.face_analysis_SCRFD import detect_faces
from detection.eye_analysis import detect_eye_state, detect_eye_state_ear_fallback
from visualization.visualization import draw_face_boxes
from analyzer.report_generator import generate_report, generate_verdict, generate_summary

from analyzer.score import (
    score_higher_better,
    score_lower_better,
    SHARPNESS_THRESHOLDS,
    NOISE_THRESHOLDS,
    CONTRAST_THRESHOLDS,
    score_exposure,
    overall_score,
)

from analyzer.quality_report import (
    brightness_report,
    interpret_saturation,
    interpret_temperature,
)
from visualization.display_utils import (
    resize_for_display,
    scale_face_result
)

ranking_results = []
# ----- Setup -----

IMAGE_FOLDER = "testimages4"


model = joblib.load("ml/model.pkl")

os.makedirs("logs", exist_ok=True)
report_log = open("logs/simplified.log", "w", encoding="utf-8")
analysis_log = open("logs/analysis.log", "w", encoding="utf-8") 

CURRENT_OUTPUT = "sorted_images"
CACHE_OUTPUT = "image_cache"
MIN_FACE_WIDTH = 50
MIN_FACE_HEIGHT = 60

VERDICTS = [
    "Excellent",
    "Good",
    "Fair",
    "Poor"
]

if os.path.exists(CURRENT_OUTPUT):
    shutil.rmtree(CURRENT_OUTPUT)

for verdict in VERDICTS:
    os.makedirs(
        os.path.join(CURRENT_OUTPUT, verdict),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(CACHE_OUTPUT, verdict),
        exist_ok=True
    )

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
MAX_WIDTH = 4000
MAX_HEIGHT = 6000

eye_dataset = open(
    "ml/eye_analysis_dataset.csv",
    "w",
    newline="",
    encoding="utf-8"
)

eye_writer = csv.writer(eye_dataset)

eye_writer.writerow([
    "filename",
    "left_ear",
    "right_ear",
    "average_ear",
    "difference",
    "ratio",
    "predicted_status"
])



for filename in os.listdir(IMAGE_FOLDER):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)

    img = cv2.imread(image_path)    

    if img is None:
        common_print(f"Could not read {filename}")
        continue   

    h, w = img.shape[:2]

    scale = min(
        MAX_WIDTH / w,
        MAX_HEIGHT / h
    )

    if scale < 1:
        new_width = int(w * scale)
        new_height = int(h * scale)

        img = cv2.resize(
            img,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )







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
        common_print()
    else:

        for face_idx, box in enumerate(face_result["face_boxes"]):

            x, y, w, h = box


            min_dimension = min(w, h)

            padding = 0.50

            landmarks = face_result["face_landmarks"][face_idx]


            if landmarks is None:
                continue


            dev_print( "[DEBUG]"
                "FACE INDEX:",
                face_idx,
                "LANDMARK 0:",
                landmarks[0]
            )
            dev_print("[DEBUG]"
                        "RAW BOX:",
                        box
                    )
            dev_print( "[DEBUG]"
                        f"FACE {face_idx}: width={w}, height={h}"
                    )
            dev_print( "[DEBUG]"
                        f"FACE {face_idx} DIMENSIONS: dimensions={min_dimension}"
                    )

            if min_dimension >= 160:

                dev_print(
                    f"FACE {face_idx}: using ML"
                )

                eye_result = detect_eye_state(
                    landmarks
                )

                confidence = eye_result["eye_results"][0]["confidence"]

                if confidence < 70:
                    dev_print(
                        f"FACE {face_idx}: Low ML confidence ({confidence:.1f}%), switching to EAR"
                    )

                    eye_result = detect_eye_state_ear_fallback(
                        landmarks
                    )

            else:

                dev_print(
                    f"FACE {face_idx}: using EAR fallback"
                )

                eye_result = detect_eye_state_ear_fallback(
                    landmarks
                )


            if "eye_results" not in eye_result:
                continue


            eye = eye_result["eye_results"][0]

            eye_visual_results.append({
                "face_index": face_idx,
                **eye
            })

            dev_print(
                "FINAL EYE RESULT:",
                eye_visual_results[-1]
            )





# ----- Quality Scores -----
    scores = {
        "Sharpness": score_ml_sharpness(
        prediction,
        confidence,
        ),


        "Noise": score_lower_better(
            features["noise"],
            NOISE_THRESHOLDS
        ),

        "Contrast": score_higher_better(
            features["contrast"],
            CONTRAST_THRESHOLDS
        ),

        "Exposure": score_exposure(
            features["brightness"],
            features["shadow_clip"],
            features["highlight_clip"]
        )

    }
    overall = overall_score(scores)
    


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
    summary = generate_summary(
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

    ranking_results.append({
    "filename": filename,
    "image_path": image_path,
    "verdict": verdict,
    "score": overall
    })

    timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
    )

#----- Visualization -----
    display_img, scale = resize_for_display(img)

    display_face_result = scale_face_result(
        face_result,
        scale
    )

    annotated = draw_face_boxes(
    display_img,
    display_face_result,
    eye_visual_results,
    overall,
    scores,
    verdict
)
    cv2.imwrite(
    f"e/visual_{filename}",
    annotated
    )

#----- Logging -----
    common_print(f"\n===== {filename} =====")
    common_print(f"Image dimensions: {img.shape[:2]}")
    common_print(
        f"\nOverall Score: {overall}/100"
    )
    common_print("\n===== Assessment Report =====\n")

    common_print("General")
    for line in report["general"]:
        common_print(f"• {line}")

    common_print()
    common_print("\nStrengths")

    if report["strengths"]:
        for line in report["strengths"]:
            common_print(f"✓ {line}")

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
    common_print()
    common_print()

    common_print(f"Sorted into {verdict}")
    common_print()
    common_print()
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

    dev_print()

    for msg in face_result["log_messages"]:
        dev_print(msg)

    dev_print("\n[Eye Detection]")

    if not eye_visual_results:
        dev_print("No faces detected. Continuing.....")
    else:
        for i, eye in enumerate(eye_visual_results):
            

            if "ear" in eye:
                dev_print(
                            f"""
                            FACE {i+1}
                            Method: {eye['method']}
                            Status: {eye['status']}
                            Confidence: {eye['confidence']}

                            Probabilities:
                            {eye['probabilities']}
                            """
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
    dev_print(f"\n")

    common_print(f"======== SUMMARY ===========")

    for line in summary:
        common_print(f"• {line}")
    dev_print(f"\n")

    dev_print("================================ END OF COMPREHENSIVE REPORT =====================================")

report_log.close()
analysis_log.close()
eye_dataset.close()

for verdict in VERDICTS:

    images = [
        img for img in ranking_results
        if img["verdict"] == verdict
    ]

    images.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    for rank, img_info in enumerate(images, start=1):

        filename = f"{rank:03d}_{img_info['score']}_{img_info['filename']}"

        shutil.copy2(
            img_info["image_path"],
            os.path.join(
                CURRENT_OUTPUT,
                verdict,
                filename
            )
        )

        shutil.copy2(
            img_info["image_path"],
            os.path.join(
                CACHE_OUTPUT,
                verdict,
                filename
            )
        )




        

    
    


