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
        "suggestions": []
    }
    report["general"].append(
        f"Faces detected: {face_result['face_count']}."
    )
    open_count = sum(
    1
    for eye in eye_results
    if "open" in eye["status"].lower()
    )

    closed_count = len(eye_results) - open_count

    report["general"].append(
        f"Eyes open: {open_count} / {len(eye_results)}."
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

    if scores["Noise"]["grade"] in ["Poor", "Very Poor"]:

        report["problems"].append(
            "High image noise detected."
        )
        report["suggestions"].append(
        "Use a lower ISO or increase available light."
        )
    if closed_count > 0:

        report["problems"].append(
            f"{closed_count} subject(s) have closed eyes."
        )
        report["suggestions"].append(
        "Capture another frame with all subjects' eyes open"
        )
    return report

def generate_verdict(overall_score, problems):
    if overall_score >= 90 and len(problems) == 0:
        return "Excellent Photo"

    elif overall_score >= 75:
        return "Good Photo - Minor Issues"

    elif overall_score >= 60:
        return "Acceptable - Consider Improvements"

    else:
        return "Retake Recommended"



