import numpy as np
import matplotlib.pyplot as plt
import cv2
import joblib
import pandas as pd


eye_model = joblib.load(
    "ml/eye_model_6class_newest.pkl"
)

print(
    "MODEL FEATURES:",
    eye_model.feature_names_in_
)

LEFT_EYE = [35, 36, 37, 39, 41, 42]
RIGHT_EYE = [89, 90, 91, 93, 95, 96]

  #Only to be used for debugging

def show_eye_points(img, landmarks, filename="debug_eye_points.jpg"):
    debug = img.copy()

    for idx in LEFT_EYE:
        x, y = landmarks[idx]
        cv2.circle(debug, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.putText(debug, str(idx), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    for idx in RIGHT_EYE:
        x, y = landmarks[idx]
        cv2.circle(debug, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv2.putText(debug, str(idx), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

    cv2.imwrite(filename, debug)


def calculate_ear(landmarks, eye_points):

    p = []

    for idx in eye_points:
        p.append(np.array(landmarks[idx]))

    vertical1 = np.linalg.norm(p[1]-p[4])
    vertical2 = np.linalg.norm(p[2]-p[5])
    horizontal = np.linalg.norm(p[0]-p[3])

    return (vertical1 + vertical2) / (2 * horizontal)

def extract_eye_features(landmarks):  #ML approach

    left_ear = calculate_ear(
        landmarks,
        LEFT_EYE
    )

    right_ear = calculate_ear(
        landmarks,
        RIGHT_EYE
    )

    avg_ear = (left_ear + right_ear) / 2

    eye_difference   = left_ear - right_ear


    ratio = min(left_ear, right_ear) / max(left_ear, right_ear)


    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "avg_ear": avg_ear,
        "eye_difference": eye_difference,
        "ratio": ratio,
        "eye_difference_sign": np.sign(eye_difference)


    }
def detect_eye_state(landmarks):



    landmarks = np.asarray(landmarks)

    if landmarks.shape != (106,2):
        return {
            "eye_results": [{
                "status": f"Invalid landmarks shape {landmarks.shape}"
            }]
        }
    
    features = extract_eye_features(landmarks)

    X = pd.DataFrame(
        [[
            features["left_ear"],
            features["right_ear"],
            features["avg_ear"],
            features["eye_difference"],
            features["ratio"],
            features["eye_difference_sign"]

        ]],
        columns=[
            "left_ear",
            "right_ear",
            "avg_ear",
            "eye_difference",
            "ratio",
            "eye_difference_sign"
        ]
    )

    prediction = eye_model.predict(X)[0]

    probabilities = eye_model.predict_proba(X)[0]



    confidence = probabilities[prediction] * 100

    labels = {
        0: "Eyes open",
        1: "Eyes closed",
        2: "Left eye closed",
        3: "Right eye closed"
    }

    status = labels[prediction]


    return {
        "eye_results": [{
            "status": status,
            "confidence": confidence,
            "method": "ML",
            "probabilities": {
                labels[class_id]: float(prob)
                for class_id, prob in zip(
                    eye_model.classes_,
                    probabilities
                )
            },

            "left_ear": features["left_ear"],
            "right_ear": features["right_ear"],
            "ear": features["avg_ear"],
            "eye_difference": features["eye_difference"],
            "ratio": features["ratio"],
            "eye_difference_sign": features["eye_difference_sign"]
        }]
    }
        
    


def detect_eye_state_ear_fallback(landmarks):

    left_ear = calculate_ear(
        landmarks,
        LEFT_EYE
    )

    right_ear = calculate_ear(
        landmarks,
        RIGHT_EYE
    )

    avg_ear = (left_ear + right_ear) / 2

    closed_threshold = 0.16
    open_threshold = 0.18

    left_closed = left_ear < closed_threshold
    right_closed = right_ear < closed_threshold

    left_open = left_ear > open_threshold
    right_open = right_ear > open_threshold

    if left_closed and right_closed:
        status = "Eyes closed"

    elif left_closed and right_open:
        status = "Left eye closed"

    elif right_closed and left_open:
        status = "Right eye closed"

    elif left_open and right_open:
        status = "Eyes open"
    else:
        status = "Eyes open"

    return {
        "eye_results":[{
            "status": status,
            "confidence": 100,
            "method": "EAR",
            "probabilities": {},
            "left_ear": left_ear,
            "right_ear": right_ear,
            "ear": avg_ear,
            "eye_difference": left_ear - right_ear,
            "ratio": min(left_ear, right_ear) / max(left_ear, right_ear)
        }]
    }
