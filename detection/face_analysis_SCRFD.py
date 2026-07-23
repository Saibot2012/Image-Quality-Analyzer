import cv2
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import numpy as np

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
    rcond = None
)

print(app.models.keys())

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


def show_landmarks(img, face):

    pts = face.landmark_2d_106

    if pts is None:
        print("106 landmarks failed")
        return

    print("106 landmarks shape:", pts.shape)

    x = pts[:, 0]
    y = pts[:, 1]

    plt.figure(figsize=(8, 8))

    plt.imshow(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )

    plt.scatter(
        x,
        y,
        s=5
    )

    for i, (px, py) in enumerate(pts):
        plt.text(
            px,
            py,
            str(i),
            fontsize=6
        )

    plt.axis("off")
    plt.show()



def detect_faces(img):

    faces = app.get(img)

    if len(faces) == 0:
        return {
            "face_count": 0,
            "face_detected": False,
            "face_boxes": [],
            "face_landmarks": [],
            "log_messages": [
                "No faces detected."
            ]
        }


    face_results = []
    landmarks_list = []


    for i, face in enumerate(faces):

        print("\nFACE", i)

        print(
            "bbox:",
            face.bbox,
            "det score:",
            face.det_score
        )

        print(
            "106 landmarks:",
            face.landmark_2d_106 is not None
        )


        if face.landmark_2d_106 is not None:

            print(
                "shape:",
                face.landmark_2d_106.shape
            )

            landmarks_list.append(
                np.asarray(face.landmark_2d_106)
            )

        else:
            landmarks_list.append(None)


        x1, y1, x2, y2 = face.bbox.astype(int)

        face_results.append(
            (
                x1,
                y1,
                x2-x1,
                y2-y1
            )
        )




    return {
        "face_count": len(face_results),
        "face_detected": True,
        "face_boxes": face_results,
        "face_landmarks": landmarks_list,
        "log_messages": [
            f"SCRFD detected {len(face_results)} faces."
        ]
    }