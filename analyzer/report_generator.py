from datetime import datetime

VERDICT_LEVELS = {
    "Excellent": 4,
    "Good": 3,
    "Fair": 2,
    "Poor": 1,
}


def generate_report_data(
    filename,
    img_path,
    image_shape,
    features,
    scores,
    overall_score,
    face_result,
    eye_results,
    brightness_info,
    saturation,
    temperature
):

    report = {
        "metadata":{
            "version": "2.0",
            "generated_at": datetime.now().isoformat()
        },

        "image_info": {},

        "quality":{
        "overall_score": overall_score,
        "verdict": None,
        "quality_grades": {},
        "quality_scores": {},
        },

        "technical_details": {},
        "face_analysis": {},
        "colour_analysis": {},
        "lighting_analysis": {},

        "report": {
            "general": [],
            "problems": [],
            "strengths": [],
            "suggestions": [],
            "summary": []
        },

        "model_info": {
            "face_detector": "SCRFD",
            "eye_model": "RandomForest 5-class",
            "eye_method": "ML + EAR fallback"
        }
    }

    report["quality"]["quality_scores"] = {
        "sharpness": scores["Sharpness"]["score"],
        "noise": scores["Noise"]["score"],
        "contrast": scores["Contrast"]["score"],
        "exposure": scores["Exposure"]["score"]
    }
    report["quality"]["quality_grades"] = {
        "sharpness": scores["Sharpness"]["grade"],
        "noise": scores["Noise"]["grade"],
        "contrast": scores["Contrast"]["grade"],
        "exposure": scores["Exposure"]["grade"]
    }
    report["technical_details"] = {
        "laplacian": float(features["laplacian"]),
        "fft_ratio": float(features["fft_ratio"]),
        "wavelet": float(features["wavelet_ratio"]),
        "noise_rms": float(features["noise"]),
        "brightness": float(features["brightness"]),
        "shadow_clip": float(features["shadow_clip"]),
        "highlight_clip": float(features["highlight_clip"]),
        "consistency": float(features["consistency"]),
        "detail_quality": float(features["detail_quality"])
    }
    report["lighting_analysis"] = {
        "result": brightness_info["status"],
        "description": brightness_info["description"],
        "recommendation": brightness_info["tip"]
    }
    report["colour_analysis"] = {
        "saturation": {
            "result": saturation
        },

        "white_balance": {
            "result": temperature
        }
    }
    report["image_info"] = {
        "filename": filename,
        "annotated_filename": f"visual_{filename}",
        "path": img_path,
        "width": int(image_shape[1]),
        "height": int(image_shape[0]),
        "faces_detected":face_result["face_count"]
    }

    report["report"]["general"].append(
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

    report["report"]["general"].append(
        f"Both eyes open: {both_open_count} / {len(eye_results)}."
    )
    if one_closed_count > 0:
        report["report"]["general"].append(
            f"One eye closed: {one_closed_count}/{len(eye_results)}."
        )

    if both_closed_count > 0:
        report["report"]["general"].append(
            f"Both eyes closed: {both_closed_count}/{len(eye_results)}."
        )

    report["report"]["general"].append(
        f"Sharpness: {scores['Sharpness']['grade']}."
    )

    report["report"]["general"].append(
        f"Contrast: {scores['Contrast']['grade']}."
    )

    report["report"]["general"].append(
        f"Lighting: {brightness_info['status']}."
    )

    report["report"]["general"].append(
        f"Colours: {saturation.lower()}."
    )

    report["report"]["general"].append(
        f"White Balance: {temperature.lower()}."
        
    )


# PROBLEMS AND SUGGESTIONS
    if scores["Noise"]["grade"] in ["Poor", "Very Poor"]:

        report["report"]["problems"].append(
            "High image noise detected."
        )
        report["report"]["suggestions"].append(
        "Use a lower ISO or increase available light."
        )
    if scores["Exposure"]["grade"] in ["Fair", "Poor", "Very Poor"]:
        report["report"]["problems"].append(
            "Low brightness detected."
        )
        report["report"]["suggestions"].append(
        "Use a higher aperture or increase available light."
        )
    if both_closed_count > 0:

        report["report"]["problems"].append(
            f"{both_closed_count} subject(s) have both closed eyes."
        )
        report["report"]["suggestions"].append(
        "Capture another frame with all subjects' eyes open"
        )
    if one_closed_count > 0:
        report["report"]["problems"].append(
            f"{one_closed_count}/{len(eye_results)} subjects have 1 eye closed."
        )
        report["report"]["suggestions"].append(
        "Capture another frame with all subjects' eyes open"
        )
    report["face_analysis"] = {
        "total_faces": face_result["face_count"],
        "eyes_open": both_open_count,
        "one_eye_closed": one_closed_count,
        "eyes_closed": both_closed_count,
        "subjects": eye_results
        }

    verdict = generate_verdict(
        overall_score,
        report["report"]["problems"]
        )

    report["quality"]["verdict"] = verdict
        

#STRENGTHS

    if scores["Sharpness"]["grade"] == "Excellent":
        report["report"]["strengths"].append("Excellent sharpness.")
    if scores["Contrast"]["grade"] == "Excellent":
        report["report"]["strengths"].append("Excellent contrast.")
    elif scores["Contrast"]["grade"] == "Good":
        report["report"]["strengths"].append("Good contrast.")
    if brightness_info["status"] == "Balanced":
        report["report"]["strengths"].append("Balanced lighting.")
    if saturation == "Natural":
        report["report"]["strengths"].append("Natural colours.")
    if face_result["face_count"] > 0 and both_closed_count == 0 and one_closed_count == 0:
        report["report"]["strengths"].append("All subjects have eyes open.")

    report["report"]["summary"] = generate_summary(
    scores,
    face_result,
    eye_results,
    brightness_info,
    saturation,
    temperature
)

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
        if "closed eye" in problem.lower():
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
        brightness_info,
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
            f"{closed_count} subject(s) have closed eyes/eyes that cannot be detected/eye analysis skipped."
        )



    return summary

def print_report(report):

    text = ""

    text += "\n===== Assessment Report =====\n\n"

    text += "General\n"
    for item in report["report"]["general"]:
        text += f"• {item}\n"

    text += "\nStrengths\n"
    if len(report["report"]["strengths"]) == 0:
        text += "No major strengths detected. \n"
    else:
        for item in report["report"]["strengths"]:
            text += f"✓ {item}\n"


    text += "\nProblems\n"
    if len(report["report"]["problems"]) == 0:
        text += "No major problems detected.\n"
    else:
        for item in report["report"]["problems"]:
            text += f"⚠ {item}\n"


    text += "\nSuggestions\n"
    if len(report["report"]["suggestions"]) == 0:
        text += "No recommendations.\n"
    else:
        for item in report["report"]["suggestions"]:
            text += f"• {item}\n"


    text += f"\nVerdict: {report['quality']['verdict']}\n"

    return text