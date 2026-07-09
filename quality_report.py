def interpret_sharpness(laplacian):

    if laplacian < 50:
        return{
            "Very Blurry",
            "Very little fine detail is preserved"
        }
    
    elif 51 < laplacian < 150:
        return{
            "Soft",
            "Image appears slightly blurred"
        }
    elif 151 < laplacian < 500:
        return{
            "Acceptable",
            "Most fine details are preserved"
        }
    elif 501 < laplacian < 1000:
        return{
            "Sharp",
            "Most fine details are preserved"
        }
    else:
        return{
            "Excellent",
            "Excellent detail and edge definition"
        }
    
def interpret_noise(noise):

    if noise < 2:
        return (
            "Very Clean",
            "Noise is barely visible."
        )

    elif 2.1 < noise < 4:
        return (
            "Low",
            "Minor grain is present."
        )

    elif 4.1 < noise < 7:
        return (
            "Moderate",
            "Visible grain appears in smooth areas."
        )

    elif 7.1 < noise < 10:
        return (
            "High",
            "Noise is clearly noticeable."
        )

    else:
        return (
            "Very High",
            "Strong grain reduces image quality."
        )


def interpret_exposure(exposure):

    if exposure < 0.25:
        return (
            "Underexposed",
            "Image is darker than ideal."
        )

    elif exposure < 0.85:
        return (
            "Well Exposed",
            "Brightness is balanced."
        )

    else:
        return (
            "Bright",
            "Image is very bright and may contain clipped highlights."
        )