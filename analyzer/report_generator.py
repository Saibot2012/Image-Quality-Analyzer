VERDICT_LEVELS = {
    "Excellent": 4,
    "Good": 3,
    "Fair": 2,
    "Poor": 1,
}


def generate_report(
    scores,
    face_result,
    eye_results,
    lighting,
    saturation,
    temperature
):

    report = {
        "general": [],
        "problems": [],
        "strengths": [],
        "suggestions": []
    }
    
    report["general"].append(
        f"Faces detected: {face_result['face_count']}."
    )
    both_open_count = sum(
        1
        for eye in eye_results
        if eye["status"] == "Eyes open"
    )

    one_closed_count = sum(
        1
        for eye in eye_results
        if eye["status"] == "One eye closed"
    )

    both_closed_count = sum(
        1
        for eye in eye_results
        if eye["status"] == "Eyes closed"
    )

    report["general"].append(
        f"Both eyes open: {both_open_count} / {len(eye_results)}."
    )
    if one_closed_count > 0:
        report["general"].append(
            f"One eye closed: {one_closed_count}/{len(eye_results)}."
        )

    if both_closed_count > 0:
        report["general"].append(
            f"Both eyes closed: {both_closed_count}/{len(eye_results)}."
        )

    report["general"].append(
        f"Sharpness: {scores['Sharpness']['grade']}."
    )

    report["general"].append(
        f"Contrast: {scores['Contrast']['grade']}."
    )

    report["general"].append(
        f"Lighting: {lighting}."
    )

    report["general"].append(
        f"Colours: {saturation.lower()}."
    )

    report["general"].append(
        f"White Balance: {temperature.lower()}."
        
    )


# PROBLEMS AND SUGGESTIONS
    if scores["Noise"]["grade"] in ["Poor", "Very Poor"]:

        report["problems"].append(
            "High image noise detected."
        )
        report["suggestions"].append(
        "Use a lower ISO or increase available light."
        )
    if both_closed_count > 0:

        report["problems"].append(
            f"{both_closed_count} subject(s) have both closed eyes."
        )
        report["suggestions"].append(
        "Capture another frame with all subjects' eyes open"
        )
    if one_closed_count > 0:
        report["problems"].append(
            f"{one_closed_count}/{len(eye_results)} subjects have 1 eye closed."
        )
        report["suggestions"].append(
        "Capture another frame with all subjects' eyes open"
        )

#STRENGTHS

    if scores["Sharpness"]["grade"] == "Excellent":
        report["strengths"].append("Excellent sharpness.")
    if scores["Contrast"]["grade"] == "Excellent":
        report["strengths"].append("Excellent contrast.")
    elif scores["Contrast"]["grade"] == "Good":
        report["strengths"].append("Good contrast.")
    if lighting == "Balanced":
        report["strengths"].append("Balanced lighting.")
    if saturation == "Natural":
        report["strengths"].append("Natural colours.")
    if face_result["face_count"] > 0 and both_closed_count == 0 and one_closed_count == 0:
        report["strengths"].append("All subjects have eyes open.")
    return report
    

def generate_verdict(score, problems):
    if score >= 90:
        verdict = "Excellent"

    elif score >= 75:
        verdict = "Good"
    
    elif score >= 60:
        verdict = "Fair"
    else:
        verdict = "Poor"

    severity = 0

    for problem in problems:

        if "noise" in problem.lower():
            severity += 1
        if "closed_eyes" in problem.lower():
            severity += 1
        if "blur" in problem.lower():
            severity += 2

    levels = [
        "Poor",
        "Fair",
        "Good",
        "Excellent"
    ]
    
    index = levels.index(verdict)

    index -= severity

    index = max(index, 0)

    return levels[index]


def generate_summary(
        scores,
        face_result,
        eye_results,
        lighting,
        saturation,
        temperature
):
    summary = []

    sharpness = scores["Sharpness"]["grade"]
    noise = scores["Noise"]["grade"]
    contrast = scores["Contrast"]["grade"]
    exposure = scores["Exposure"]["grade"]

    face_count = face_result["face_count"]

    open_count = sum(
        1
        for eye in eye_results
        if "open" in eye["status"].lower()
    )

    closed_count = face_count - open_count

    good_metrics = 0

    for metric in [
        sharpness,
        noise,
        contrast,
        exposure
    ]:
        if metric in ["Excellent", "Good"]:
            good_metrics += 1
    
    if good_metrics == 4:
        summary.append(
            "The image is technically strong."
        )
    elif good_metrics >= 3:
        summary.append(
            "The image has good overall technical quality."
        )
    elif good_metrics >= 2:
        summary.append(
            "The image has acceptable technical quality."
        )

    else:
        summary.append(
            "The image has several technical quality issues."
        )

    if sharpness == "Excellent":
        summary.append(
            "Details are crisp and well defined."
        )

    elif sharpness == "Good":
        summary.append(
            "The image appears sharp."
        )

    elif sharpness == "Fair":
        summary.append(
            "Sharpness is acceptable."
        )

    else:
        summary.append(
            "The image lacks sufficient sharpness."
        )
    if exposure == "Excellent":
        summary.append(
            "Exposure is well balanced."
        )

    elif exposure == "Good":
        summary.append(
            "Exposure is generally well balanced."
        )

    elif exposure == "Fair":
        summary.append(
            "Exposure is usable but could be improved."
        )

    else:
        summary.append(
            "Exposure requires correction."
        )

    if noise in ["Excellent", "Good"]:
        summary.append(
            "Noise is well controlled."
            )
    elif noise == "Fair":
        summary.append(
            "Some visible image noise is present."
            )
    else:
        summary.append(
            "Noticeable image noise reduces image quality."
            )



    if face_count == 0:

        summary.append(
            "No people were detected in the scene."
        )

    elif closed_count == 0:

        summary.append(
            "All detected subjects have their eyes open."
        )

    else:

        summary.append(
            f"{closed_count} subject(s) have closed eyes/ eyes' cannot be detected. "
        )

    return summary