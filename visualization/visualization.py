import cv2
GRADE_NAMES = {"Excellent", "Good", "Fair", "Poor","Very Poor"}

MAX_DISPLAY_WIDTH = 1920

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


def draw_info_row(img, label, value, label_x, value_x, y, font_scale, thickness, ui):
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
            thickness,
            ui
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
def draw_hero_badge(img, grade, x, y, ui, hero_font, thickness):
    colours = {
        "Excellent": (70, 165, 95),      # emerald
        "Good":      (226, 43, 138),     # Purple
        "Fair":      (110, 170, 205),    # steel blue
        "Poor":      (80, 140, 205),     # bronze/copper
        "Very Poor": (60, 60, 185),      # deep red

        }
    colour = colours.get(grade, (180, 180, 180))

    (tw, th), _ = cv2.getTextSize(
        grade,
        cv2.FONT_HERSHEY_SIMPLEX,
        hero_font,
        thickness,
    )

    padding_x = 50
    padding_y = 6
    badge_width = tw + padding_x * 2

    badge_x = x + (ui["width"] - badge_width) // 2
    text_y = y - 15

    width = tw + padding_x*2
    height = th + padding_y*4

    radius = height//2

    # middle rectangle
    cv2.rectangle(
        img,
        (badge_x + radius, y - height),
        (badge_x + width - radius, y),
        colour,
        -1
    )

    cv2.circle(
        img,
        (badge_x + radius, y - radius),
        radius,
        colour,
        -1
    )

    cv2.circle(
        img,
        (badge_x + width - radius, y - radius),
        radius,
        colour,
        -1
    )

    cv2.putText(
        img,
        grade,
        (badge_x+padding_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        hero_font,
        (255,255,255),
        thickness
    )
def draw_rounded_badge(img, grade, x, y, font_scale, thickness, ui):
    
    colours = {
        "Excellent": (70, 165, 95),      # emerald
        "Good":      (226, 43, 138),    # purple
        "Fair":      (110, 170, 205),    # steel blue
        "Poor":      (80, 140, 205),     # bronze/copper
        "Very Poor": (60, 60, 185),      # deep red

        }
    colour = colours.get(grade, (180, 180, 180))

    (tw, th), _ = cv2.getTextSize(
        grade,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    padding_x = 12
    padding_y = 5
    badge_width = tw + padding_x * 2
    badge_area_width = int(ui["width"] * 0.30)   # roughly the right column width

    badge_x = x + (badge_area_width - badge_width) // 2
    text_y = y - 4

    width = tw + padding_x*2
    height = th + padding_y*2

    radius = height//2

    # middle rectangle
    cv2.rectangle(
        img,
        (badge_x + radius, y - height),
        (badge_x + width - radius, y),
        colour,
        -1
    )

    cv2.circle(
        img,
        (badge_x + radius, y - radius),
        radius,
        colour,
        -1
    )

    cv2.circle(
        img,
        (badge_x + width - radius, y - radius),
        radius,
        colour,
        -1
    )

    cv2.putText(
        img,
        grade,
        (badge_x+padding_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255,255,255),
        thickness
    )

def draw_face_boxes(img, face_result, eye_results, overall_score, scores, verdict):

    output = img.copy()

    for i, box in enumerate(face_result["face_boxes"]):

        x, y, w, h = box

        if i < len(eye_results):
            status = eye_results[i]["status"]
            confidence = eye_results[i]["confidence"]
        else:
            status = "Skipped"
            confidence = 0

        if "eyes open" in status.lower():
            colour = (0,255,0) #green
        elif "one eye closed" in status.lower():
            colour = (0,165,255) #red
        elif status == "eyes closed":
            colour = (0,0,255)
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
    both_open_count = 0
    both_closed_count = 0
    one_closed_count = 0
    unknown_count = 0

    _, w = output.shape[:2]

    for eye in eye_results:
        status = eye["status"].lower()

        if "eyes open" in status:
            both_open_count += 1

        elif "eyes closed" in status:
            both_closed_count += 1

        elif "one eye closed" in status:
            one_closed_count += 1

        else:
            unknown_count += 1


    h, w = output.shape[:2]


    ui = {
        "margin": int(min(w, h) * 0.02),
        "padding": int(min(w, h) * 0.01),
        "hero_font": max(min(w, h) / 1300, 0.5),
        "font_scale": max(min(w, h) / 1800, 0.5),
        "thickness": max(1, int(min(w, h) / 700)),
        "width": int(w * 0.28),
        }
    x = ui["margin"]
    y = ui["margin"]

    font_scale = ui["font_scale"]
    hero_font = ui["hero_font"]
    thickness = ui["thickness"]

    title_offset_width = int(20 * font_scale)
    title_offset_height = int(35 * font_scale)

    label_x = x + title_offset_width
    value_x = x + int(ui["width"] * 0.70)

    current_y = y + title_offset_height

    line_height = int(32 * font_scale)
    rows = 16
    ui["height"] = int(
        title_offset_height + (rows + 7)* line_height
    )

    thickness = max(1, int(font_scale * 2))

    score_text = f"{overall_score}/100"


    #white background
    cv2.rectangle(
        output,
        (x, y),
        (x+ui["width"], y+ui["height"]),
        (255,255,255),
        -1
    )

    (text_w, text_h), _ = cv2.getTextSize(
    score_text,
    cv2.FONT_HERSHEY_SIMPLEX,
    font_scale * 1.6,
    thickness + 1
    ) #For centre alignment

    score_x = x + (ui["width"] - text_w) // 2

    (title_w, _), _ = cv2.getTextSize(
    "IMAGE ASSISTANT",
    cv2.FONT_HERSHEY_SIMPLEX,
    font_scale,
    thickness
)

    title_x = x + (ui["width"] - title_w) // 2
    # Title
    cv2.putText(
        output,
        "IMAGE ASSISTANT",
        (title_x, current_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0,0,0),
        thickness,
    )
    current_y += int(line_height * 1.6)


    cv2.putText(
        output,
        score_text,
        (score_x, current_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 1.6,
        (0, 0, 0),
        thickness + 1
    ) 
    
    if h < 700:
        current_y += int(line_height * 3.2)
    else:
        current_y += int(line_height * 2.6)

    draw_hero_badge(
        output,
        verdict,
        x,
        current_y,
        ui,
        hero_font,
        thickness
    )

    current_y += int(line_height * 1.23)


    # Move down one line
    cv2.line(
    output,
    (x + 10, current_y),
    (x + ui["width"] - 10, current_y),
    (200, 200, 200),
    1
    ) 


    current_y += line_height 


    draw_section_title(
    output,
    "Subjects",
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
        thickness,
        ui
    )

    current_y += line_height

    draw_info_row(
        output,
        "Both eyes Open",
        f"{both_open_count}/{face_result['face_count']}",
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness,
        ui
    )
    current_y += int(line_height * 1.2) 

    draw_info_row(
        output,
        "One eye closed",
        f"{one_closed_count}/{face_result['face_count']}",
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness,
        ui
    )
    current_y += int(line_height * 1.2) 
    draw_info_row(
        output,
        "Both eyes closed",
        f"{both_closed_count}/{face_result['face_count']}",
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness,
        ui
    )
    current_y += int(line_height * 1.2) 

    cv2.line(
        output,
        (x + 10, current_y),
        (x + ui["width"] - 10, current_y),
        (200, 200, 200),
        1
    )

    current_y += int(line_height * 1.2)

    draw_section_title(
        output,
        "Image Quality",
        label_x,
        current_y,
        font_scale,
        thickness
    )

    current_y += int(line_height * 1.2)  

    draw_info_row(
        output,
        "Sharpness",
        scores["Sharpness"]["grade"],
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness,
        ui

    )

    if w > 1920:
        current_y += int(line_height * 2.0)
    else:
        current_y += int(line_height * 1.6)

    draw_info_row(
        output,
        "Noise",
        scores["Noise"]["grade"],
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness,
        ui
    )

    if w > 1920:
        current_y += int(line_height * 2.0)
    else:
        current_y += int(line_height * 1.6)

    draw_info_row(
        output,
        "Contrast",
        scores["Contrast"]["grade"],
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness,
        ui
    )

    if w > 1920:
        current_y += int(line_height * 2.0)
    else:
        current_y += int(line_height * 1.6)

    draw_info_row(
        output,
        "Exposure",
        scores["Exposure"]["grade"],
        label_x,
        value_x,
        current_y,
        font_scale,
        thickness,
        ui
    )
    
    

    return output