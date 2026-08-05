import cv2
import os
import pandas as pd
import numpy as np

from detection.face_analysis_SCRFD import detect_faces



LEFT_EYE = [35,36,37,39,41,42]
RIGHT_EYE = [89,90,91,93,95,96]

def calculate_ear(landmarks, eye_points):
    p = []

    for idx in eye_points:
        p.append(np.array(landmarks[idx]))


    vertical1 = np.linalg.norm(p[1]-p[4])
    vertical2 = np.linalg.norm(p[2]-p[5])
    horizontal = np.linalg.norm(p[0]-p[3])

    return (vertical1 + vertical2)/(2*horizontal)

def extract_features(img):

    result = detect_faces(img)

    print("RESULT:", result.keys())

    if not result["face_detected"]:
        print("NO FACE")
        return None
    
    landmarks = result["face_landmarks"][0]



    if landmarks is None:
        return None

    if len(landmarks) < 106:
        return None

    left = calculate_ear(
        landmarks,
        LEFT_EYE
    )

    right = calculate_ear(
        landmarks,
        RIGHT_EYE
    )
    difference = left - right

    ratio = min(left, right) / max(left, right)

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
        "left_ear": left,
        "right_ear": right,
        "avg_ear": (left + right) / 2,
        "eye_difference": difference,
        "ratio": ratio,
        "head_roll": head_roll,
        "left_lid_bulge": left_bulge,
        "right_lid_bulge": right_bulge,
        }

dataset = []


folders = {
    "open":0,
    "closed":1,
    "left_closed":2,
    "right_closed":3
}

for folder,label in folders.items():
    path=f"dataset/eyes_dataset/{folder}"

    for file in os.listdir(path):

        img = cv2.imread(
            os.path.join(path,file)
        )
        
        try:
            features = extract_features(img)

        except Exception as e:
            print("FAILED:", file, e)
            continue
        
        if features:
            features["filename"] = file
            features["label"] = label
            features["class"] = folder

            dataset.append(features)

            print(file, features)



df=pd.DataFrame(dataset)

df.to_csv(
    "eye_dataset.csv",
    index=False
)


print("\nCLASS DISTRIBUTION")
print(df["class"].value_counts())

print(df.groupby("class")["avg_ear"].describe())

print(
    df.groupby("class")[["left_ear","right_ear","eye_difference"]].mean()
)

print(
    df.groupby("class")[["left_ear","right_ear"]].describe()
)