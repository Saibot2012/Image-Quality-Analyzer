
SHARPNESS_THRESHOLDS = {
    "excellent": 300,
    "good": 200,
    "fair": 120,
    "poor": 60,
}

NOISE_THRESHOLDS = {
    "excellent": 2,
    "good": 5,
    "fair": 8,
    "poor": 12,
}

CONTRAST_THRESHOLDS = {
    "excellent": 70,
    "good": 60,
    "fair": 45,
    "poor": 30,
}

EXPOSURE_THRESHOLDS = {
     "brightness_min": 0.25,
     "brightness_max": 0.75,

     "shadow_warning": 0.10,
     "highlight_warning": 0.10,

}
GRADE_INFO = {
    "excellent": {
        "grade": "Excellent",
        "score": 100
    },

    "good": {
        "grade": "Good",
        "score": 90
    },

    "fair": {
        "grade": "Fair",
        "score": 75
    },

    "poor": {
        "grade": "Poor",
        "score": 50
    },

    "very_poor": {
        "grade": "Very Poor",
        "score": 20
    }
}


def score_higher_better(value, thresholds):

    if value >= thresholds["excellent"]:
        return GRADE_INFO["excellent"]

    if value >= thresholds["good"]:
        return GRADE_INFO["good"]

    if value >= thresholds["fair"]:
        return GRADE_INFO["fair"]

    if value >= thresholds["poor"]:
        return GRADE_INFO["poor"]

    return GRADE_INFO["very_poor"]


def score_lower_better(value, thresholds):

    if value <= thresholds["excellent"]:
        return GRADE_INFO["excellent"]

    if value <= thresholds["good"]:
        return GRADE_INFO["good"]

    if value <= thresholds["fair"]:
        return GRADE_INFO["fair"]

    if value <= thresholds["poor"]:
        return GRADE_INFO["poor"]

    return GRADE_INFO["very_poor"]

def score_exposure(brightness, shadow_clip, highlight_clip):
    
    score = 100
    score = 100

    # Brightness
    if brightness < 0.15:
        score -= 50

    elif brightness < 0.25:
        score -= 30

    elif brightness > 0.85:
        score -= 50

    elif brightness > 0.75:
        score -= 30


    # Shadow clipping
    if shadow_clip > 0.15:
        score -= 25

    elif shadow_clip > 0.08:
        score -= 10


    # Highlight clipping
    if highlight_clip > 0.15:
        score -= 25

    elif highlight_clip > 0.08:
        score -= 10


    
    #prevent negative score
    score = max(score, 0)

    if score >= 90:
        grade = "Excellent"

    elif score >= 75:
        grade = "Good"

    elif score >= 60:
        grade = "Fair"

    elif score >= 40:
        grade = "Poor"

    else:
        grade = "Very Poor"


    return {
        "grade": grade,
        "score": score
    }
    
def overall_score(scores):
    weights = {
        "Sharpness": 0.40,
        "Noise": 0.25,
        "Contrast": 0.20,
        "Subject": 0.15
    }
    total = 0
    used_weight = 0

    for metric, weight in weights.items():
        if metric in scores:
                    total += scores[metric]["score"] * weight
                    used_weight += weight

    return round(total / used_weight)

