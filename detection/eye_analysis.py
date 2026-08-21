import numpy as np
import matplotlib.pyplot as plt
import cv2
import joblib
import pandas as pd


eye_model = joblib.load(
    "ml/eye_model_4class_newest.pkl"
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

    left_center  = np.mean([landmarks[i] for i in LEFT_EYE], axis=0)
    right_center = np.mean([landmarks[i] for i in RIGHT_EYE], axis=0)

    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]
    head_roll = np.degrees(np.arctan2(dy, dx))

        # eye_points = [outer_corner, bottom1, bottom2, inner_corner, top1, top2]
    left_outer = np.array(landmarks[LEFT_EYE[0]])
    left_inner = np.array(landmarks[LEFT_EYE[3]])
    left_top1  = np.array(landmarks[LEFT_EYE[4]])
    left_top2  = np.array(landmarks[LEFT_EYE[5]])

    left_chord = left_inner - left_outer
    left_chord_len = np.linalg.norm(left_chord)
    left_chord_unit = left_chord / left_chord_len
    left_perp = np.array([-left_chord_unit[1], left_chord_unit[0]])

    left_top1_height = np.dot(left_top1 - left_outer, left_perp)
    left_top2_height = np.dot(left_top2 - left_outer, left_perp)
    left_bulge = (left_top1_height + left_top2_height) / 2 / left_chord_len

    right_outer = np.array(landmarks[RIGHT_EYE[0]])
    right_inner = np.array(landmarks[RIGHT_EYE[3]])
    right_top1  = np.array(landmarks[RIGHT_EYE[4]])
    right_top2  = np.array(landmarks[RIGHT_EYE[5]])

    right_chord = right_inner - right_outer
    right_chord_len = np.linalg.norm(right_chord)
    right_chord_unit = right_chord / right_chord_len
    right_perp = np.array([-right_chord_unit[1], right_chord_unit[0]])

    right_top1_height = np.dot(right_top1 - right_outer, right_perp)
    right_top2_height = np.dot(right_top2 - right_outer, right_perp)
    right_bulge = (right_top1_height + right_top2_height) / 2 / right_chord_len

    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "avg_ear": avg_ear,
        "eye_difference": eye_difference,
        "ratio": ratio,
        "head_roll": head_roll,
        "left_lid_bulge": left_bulge,
        "right_lid_bulge": right_bulge,

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
            features["head_roll"],
            features["left_lid_bulge"],
            features["right_lid_bulge"]
        ]],
        columns=[
            "left_ear",
            "right_ear",
            "avg_ear",
            "eye_difference",
            "ratio",
            "head_roll",
            "left_lid_bulge",
            "right_lid_bulge"
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
            "head_roll": features["head_roll"],
            "left_lid_bulge": features["left_lid_bulge"],
            "right_lid_bulge": features["right_lid_bulge"]

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

    closed_threshold = 0.176
    open_threshold = 0.19

    left_closed = left_ear < closed_threshold
    right_closed = right_ear < closed_threshold

    left_open = left_ear > open_threshold
    right_open = right_ear > open_threshold

    if left_closed and right_closed:
        status = "Eyes closed"
        confidence = 100


    elif left_closed and right_open:
        status = "Left eye closed"
        confidence = 100


    elif right_closed and left_open:
        status = "Right eye closed"
        confidence = 100

    elif left_open and right_open:
        status = "Eyes open"
        confidence = 100
    else:
        status = "Undecided"
        confidence = 0

    return {
        "eye_results":[{
            "status": status,
            "confidence": confidence,
            "method": "EAR",
            "probabilities": {},
            "left_ear": left_ear,
            "right_ear": right_ear,
            "ear": avg_ear,
            "eye_difference": left_ear - right_ear,
            "ratio": min(left_ear, right_ear) / max(left_ear, right_ear)
        }]
    }
