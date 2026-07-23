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


    vertical1 = np.linalg.norm(p[1]-p[5])
    vertical2 = np.linalg.norm(p[2]-p[4])
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

    return {
        "left_ear": left,
        "right_ear": right,
        "avg_ear": (left + right) / 2,
        "eye_difference": difference,
        "ratio": ratio,
        "eye_difference_sign": np.sign(difference)
    }

dataset = []


folders = {
    "open":0,
    "closed":1,
    "left_closed":2,
    "right_closed":3
}

for folder,label in folders.items():
    path=f"eyes/{folder}"

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