import cv2
import numpy as np
import os
import onnxruntime as ort

MODEL_PATH = "models\open_closed_eye.onnx"

MODEL_PATH = os.path.abspath(MODEL_PATH)
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

def preprocess(face_crop):

    h, w = face_crop.shape[:2]

    # take upper face region
    eye_region = face_crop[
        0:int(h*0.55),
        :
    ]

    img = cv2.cvtColor(
        eye_region,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (32,32)
    )

    img = img.astype(
        np.float32
    ) / 255.0

    img = np.transpose(
        img,
        (2,0,1)
    )

    img = np.expand_dims(
        img,
        axis=0
    )

    return img

def detect_eye_state(face_crop):
    input_tensor = preprocess(face_crop)


    output = session.run(
        None,
        {input_name: input_tensor}
    )

    scores = output[0][0, :, 0, 0]

    closed_prob = float(scores[0])
    open_prob = float(scores[1])

    if open_prob > closed_prob:
        return {
            "eye_status": "Eyes open",
            "confidence": open_prob
        }

    return {
        "eye_status": "Eyes closed",
        "confidence": closed_prob
    }