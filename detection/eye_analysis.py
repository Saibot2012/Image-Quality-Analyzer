import numpy as np
import matplotlib.pyplot as plt
import cv2
import joblib
import pandas as pd


eye_model = joblib.load(
    "ml/eye_model_6class.pkl"
)

print(
    "MODEL FEATURES:",
    eye_model.feature_names_in_
)

LEFT_EYE = [35, 36, 37, 39, 41, 42]
RIGHT_EYE = [89, 90, 91, 93, 95, 96]

def show_eye_points(img, landmarks):  #Only to be used for debugging

    pts = landmarks
    for idx in LEFT_EYE + RIGHT_EYE:
        x, y = pts[idx]

    plt.imshow(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )

    for idx in LEFT_EYE:
        x,y = pts[idx]
        plt.scatter(x,y,c="red")
        plt.text(x,y,str(idx),color="red")

    for idx in RIGHT_EYE:
        x,y = pts[idx]
        plt.scatter(x,y,c="blue")
        plt.text(x,y,str(idx),color="blue")

    plt.show()


def calculate_ear(landmarks, eye_points):

    p = []

    for idx in eye_points:
        p.append(np.array(landmarks[idx]))

    vertical1 = np.linalg.norm(p[1] - p[5])
    vertical2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])

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
        2: "Right eye closed",
        3: "Left eye closed"
    }

    status = labels[prediction]


    return {
        "eye_results":[{
            "status": status,
            "confidence": confidence,
            "method": "ML",
            "probabilities": dict(
                zip(
                    eye_model.classes_,
                    probabilities
                )
            )
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

    threshold = 0.56

    if avg_ear < threshold:
        status = "Eyes closed"
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
