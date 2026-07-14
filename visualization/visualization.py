import cv2
GRADE_NAMES = {"Excellent", "Good", "Fair", "Poor","Very Poor"}
def draw_corner_box(img, x, y, w, h, colour, thickness=3):

    line = int(min(w, h) * 0.25)

    # Top-left
    cv2.line(img, (x, y), (x+line, y), colour, thickness)
    cv2.line(img, (x, y), (x, y+line), colour, thickness)

    # Top-right
    cv2.line(img, (x+w, y), (x+w-line, y), colour, thickness)
    cv2.line(img, (x+w, y), (x+w, y+line), colour, thickness)

    # Bottom-left
    cv2.line(img, (x, y+h), (x+line, y+h), colour, thickness)
    cv2.line(img, (x, y+h), (x, y+h-line), colour, thickness)

    # Bottom-right
    cv2.line(img, (x+w, y+h), (x+w-line, y+h), colour, thickness)
    cv2.line(img, (x+w, y+h), (x+w, y+h-line), colour, thickness)


def draw_info_row(img, label, value, label_x, value_x, y, font_scale, thickness):
    cv2.putText(
    img, 
    label, #Draws label
    (label_x, y), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    font_scale, 
    (0,0,0), 
    thickness
    )
    if value in GRADE_NAMES:
        draw_rounded_badge(
            img,
            value,
            value_x,
            y,
            font_scale,
            thickness
        )
    else:

        cv2.putText(
            img, 
            str(value), #Draws value
            (value_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0,0,0),
            thickness
        )

def draw_rounded_badge(img, grade, x, y, font_scale, thickness):
    
    colours = {
        "Excellent": (120, 180, 120),   # metallic green
        "Good": (160, 170, 180),        # silver
        "Fair": (170, 170, 120),        # bronze
        "Poor": (120, 140, 190),        # copper
        "Very Poor": (100, 100, 180)
        }
    colour = colours.get(grade, (180, 180, 180))

    (tw, th), _ = cv2.getTextSize(
        grade,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    padding_x = 12
    padding_y = 6

    width = tw + padding_x*2
    height = th + padding_y*2

    radius = height//2

    # middle rectangle
    cv2.rectangle(
        img,
        (x+radius, y-height),
        (x+width-radius, y),
        colour,
        -1
    )

    # circles
    cv2.circle(
        img,
        (x+radius, y-radius),
        radius,
        colour,
        -1
    )

    cv2.circle(
        img,
        (x+width-radius, y-radius),
        radius,
        colour,
        -1
    )

    cv2.putText(
        img,
        grade,
        (x+padding_x, y-padding_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255,255,255),
        thickness
    )

def draw_face_boxes(img, face_result, eye_results, overall_score, scores, verdict):

    output = img.copy()

    for i, box in enumerate(face_result["face_boxes"]):

        x, y, w, h = box

        status = eye_results[i]["status"]

        if "open" in status.lower():
            colour = (0,255,0) #green
        elif "closed" in status.lower():
            colour = (0,0,255) #red
        else:
            colour = (0,255,255) #yellow

    # Draw rectangle
        draw_corner_box(
            output,
            x,
            y,
            w,
            h,
            colour
        )
        (text_w, text_h), _ = cv2.getTextSize(
            status.upper(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            2
        )
        #bacground for text
        padding = 8

        cv2.rectangle(
            output,
            (x, y-text_h-2*padding),
            (x+text_w+2*padding, y),
            colour,
            -1
        )

        # Text
        cv2.putText(
            output,
            status.upper(),
            (x+5, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0,0,0),
            2
        )
    output = draw_dashboard(
        output,
        face_result,
        eye_results,
        overall_score,
        scores,
        verdict, 

    )

    return output

def draw_section_title(img, title, x, y, font_scale, thickness):

    cv2.putText(
        img,
        title.upper(),
        (x,y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.8,
        (120, 120, 120),
        thickness,
    )
def draw_dashboard(output, face_result, eye_results, overall_score, scores, verdict):
    open_count = 0
    closed_count = 0
    unknown_count = 0

    for eye in eye_results:
        status = eye["status"].lower()

        if "open" in status:
            open_count += 1

        elif "closed" in status:
            closed_count += 1

        else:
            unknown_count += 1


    h, w = output.shape[:2]


    ui = {
        "margin": int(min(w, h) * 0.02),
        "padding": int(min(w, h) * 0.01),
        "font_scale": max(min(w, h) / 1800, 0.5),
        "thickness": max(1, int(min(w, h) / 700)),
        "width": int(w * 0.30),
    }
    x = ui["margin"]
    y = ui["margin"]

    font_scale = ui["font_scale"]
    thickness = ui["thickness"]

    title_offset_width = int(20 * font_scale)
    title_offset_height = int(35 * font_scale)
    label_x = x + title_offset_width
    value_x = x + int(ui["width"] * 0.70)
    current_y = y + title_offset_height

    line_height = int(32 * font_scale)
    rows = 11
    ui["height"] = int(
        title_offset_height + (rows + 4)* line_height
    )

    thickness = max(1, int(font_scale * 2))



    #white background
    cv2.rectangle(
        output,
        (x, y),
        (x+ui["width"], y+ui["height"]),
        (255,255,255),
        -1
    )

    # Title
    cv2.putText(
        output,
        "IMAGE ASSISTANT",
        (label_x, current_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0,0,0),
        thickness,
    )

    # Move down one line
    current_y += line_height 
    current_y += line_height 

    draw_section_title(
    output,
    "Subject",
    label_x,
    current_y,
    font_scale,
    thickness
    )
    current_y += line_height

    # Faces detected
    draw_info_row(
        output,
        "Faces",
        face_result["face_count"],
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness
    )

    current_y += line_height

    draw_info_row(
        output,
        "Eyes Open",
        f"{open_count}/{face_result['face_count']}",
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness
    )

    current_y += line_height

    draw_info_row(
        output,
        "Eyes closed",
        f"{closed_count}/{face_result['face_count']}",
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness
    )

    current_y += line_height
    current_y += line_height

    draw_section_title(
    output,
    "Quality",
    label_x,
    current_y,
    font_scale,
    thickness
    )   
    current_y += line_height

    draw_info_row(
        output,
        "Score",
        f"{overall_score}/100",
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness
    )

    current_y += line_height

    draw_info_row(
        output,
        "Verdict",
        verdict.split()[0],
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness
    )
    current_y += line_height
    current_y += line_height

    draw_section_title(
        output,
        "Image Quality",
        label_x,
        current_y,
        font_scale,
        thickness
    )

    current_y += line_height    

    draw_info_row(
        output,
        "Sharpness",
        scores["Sharpness"]["grade"],
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness
    )

    current_y += line_height

    draw_info_row(
        output,
        "Noise",
        scores["Noise"]["grade"],
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness
    )

    current_y += line_height

    draw_info_row(
        output,
        "Contrast",
        scores["Contrast"]["grade"],
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness
    )
    
    

    return output