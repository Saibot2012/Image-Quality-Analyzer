import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

def calculate_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa1 = x1
    ya1 = y1
    xa2 = x1 + w1
    ya2 = y1 + h1

    xb1 = x2
    yb1 = y2
    xb2 = x2 + w2
    yb2 = y2 + h2

    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)

    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)

    intersection = max(0, xi2-xi1) * max(0, yi2-yi1)

    area1 = w1*h1
    area2 = w2*h2

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union



def remove_duplicates(faces, threshold=0.4):

    unique = []

    for face in faces:

        duplicate = False

        for existing in unique:

            if calculate_iou(face, existing) > threshold:
                duplicate = True
                break

        if not duplicate:
            unique.append(face)

    return unique



def detect_on_crop(crop, offset_x, offset_y, messages):

    faces = []
    confidences = []

    rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    h, w, _ = crop.shape

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.3
    ) as detector:

        results = detector.process(rgb)

        if results.detections:

            for detection in results.detections:

                box = detection.location_data.relative_bounding_box

                x = int(box.xmin * w)
                y = int(box.ymin * h)

                bw = int(box.width * w)
                bh = int(box.height * h)

                x += offset_x
                y += offset_y

                faces.append(
                    (x, y, bw, bh)
                )



    messages.append(
        f"Faces detected in tile: {len(faces)}"
    )

    return faces, confidences




def detect_faces(img):
    
    messages = ["=== RESULTS FROM THE TILE FACE DETECTOR ==="]

    height, width, _ = img.shape


    all_faces = []
    all_confidences = []


    overlap = 0.2


    tiles = []


    # Full image
    tiles.append(
        (img, 0, 0)
    )


    # Tile size
    tile_w = int(width / 2)
    tile_h = int(height / 2)


    positions = [
        (0,0),
        (tile_w-int(tile_w*overlap), 0),
        (0, tile_h-int(tile_h*overlap)),
        (tile_w-int(tile_w*overlap),
         tile_h-int(tile_h*overlap))
    ]


    for x, y in positions:

        crop = img[
            y:min(y+tile_h, height),
            x:min(x+tile_w, width)
        ]

        tiles.append(
            (crop, x, y)
        )


    for crop, offset_x, offset_y in tiles:

        faces, confidences = detect_on_crop(
            crop,
            offset_x,
            offset_y,
            messages, 
        )

        all_faces.extend(faces)
        all_confidences.extend(confidences)








    final_faces = remove_duplicates(
        all_faces
    )


    return {

        "face_count": len(final_faces),

        "face_detected": len(final_faces) > 0,

        "face_boxes": final_faces,

        "log_messages": messages

    }