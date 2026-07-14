
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

def overall_score(*scores):
    return round(sum(scores) / len(scores))

