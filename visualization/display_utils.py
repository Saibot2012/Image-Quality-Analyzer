import cv2

def resize_for_display(img, max_width=1728):

    _, w = img.shape[:2]
    
    if w <= max_width:
        return img, 1.0
    
    scale = max_width / w

    resized = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    return resized, scale

def scale_face_result(face_result, scale):

    scaled = face_result.copy()

    scaled["face_boxes"] = []

    for box in face_result["face_boxes"]:
        x,y,w,h = box
        scaled["face_boxes"].append(
            (
                int(x * scale),
                int(y * scale),
                int(w * scale),
                int(h * scale)
            )
        )
    return scaled