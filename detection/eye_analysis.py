import mediapipe as mp
import cv2 
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

#Eye landmark points (Mediapipe Face Mesh)
LEFT_EYE = [
    33, 160, 158, 133, 154, 144
]

RIGHT_EYE = [
    362, 385, 387, 263, 373, 380
]

def calculate_ear(landmarks, eye_points):
    p = []

    for idx in eye_points:
        p.append(
            np.array([
                landmarks[idx].x,
                landmarks[idx].y
            ])
        )

    #Vertical distances
    vertical1 = np.linalg.norm(p[1] - p[5])
    vertical2 = np.linalg.norm(p[2]-p[4])

    #Horizontal distance
    horizontal = np.linalg.norm(p[0]-p[3])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear

def detect_eye_state(img):
    
    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    with mp_face_mesh.FaceMesh(
        static_image_mode = True,
        max_num_faces = 5,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
    ) as mesh:
        result = mesh.process(rgb)

    if not result.multi_face_landmarks:
        return{
            "eye_status": "Face Mesh failed"
            "Reason: Faces too small or unclear"
        }
    
    results = []

    for face_landmarks in result.multi_face_landmarks:

        left_ear = calculate_ear(
            face_landmarks.landmark,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            face_landmarks.landmark,
            RIGHT_EYE
        )

        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < 0.18:
            status = "Eyes closed"

        elif avg_ear > 0.55:
            status = "Eyes unclear"

        else:
            status = "Eyes open"

        results.append({
            "ear": avg_ear,
            "status": status,
            })
        
        return{
            "eye_results" : results
        }