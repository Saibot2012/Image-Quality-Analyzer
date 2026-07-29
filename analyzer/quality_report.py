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




def brightness_report(brightness, shadow_clip, highlight_clip):

    if highlight_clip > 0.15:
        return {
            "status": "Too Bright",
            "description": "Some bright areas have lost detail.",
            "tip": "Reduce exposure slightly when taking similar photos."
        }

    if shadow_clip > 0.15 and brightness < 0.35:
        return {
            "status": "Too Dark",
            "description": "Dark areas contain little visible detail.",
            "tip": "Increase exposure or brighten the image."
        }

    if highlight_clip > 0.05:
        return {
            "status": "Balanced",
            "description": "Overall brightness looks natural, but some highlights are clipped.",
            "tip": "No major correction needed."
        }

    if shadow_clip > 0.10 and brightness < 0.40:
        return {
            "status": "Balanced",
            "description": "Overall brightness is good, but some shadows are very dark.",
            "tip": "Consider lifting the shadows slightly."
        }

    if brightness < 0.40:
        return {
            "status": "Too Dark",
            "description": "Overall brightness is slightly low.",
            "tip": "Consider increasing brightness."
        }
    else:
        return {
            "status": "Balanced",
            "description": "Overall brightness looks natural.",
            "tip": "No correction needed."
        }
def interpret_contrast(contrast):

    if contrast < 30:
        return (
            "Low",
            "The image appears flat and lacks tonal separation."
        )

    elif contrast < 60:
        return (
            "Balanced",
            "The image has a natural level of contrast."
        )

    else:
        return (
            "High",
            "The image has strong contrast with distinct light and dark areas."
        )
    
def interpret_saturation(saturation):

    if saturation < 40:
        return (
            "Low",
            "Colours appear muted and less vibrant."
        )

    elif saturation < 120:
        return (
            "Natural",
            "Colours appear natural."
        )

    else:
        return (
            "Vivid",
            "Colours are vivid and highly saturated."
        )

def interpret_temperature(temperature):

    if temperature > 20:
        return (
            "Warm",
            "The image has a warmer colour balance with a yellow/orange cast."
        )

    elif temperature < -20:
        return (
            "Cool",
            "The image has a cool colour balance with a noticeable blue cast."
        )

    else:
        return (
            "Natural",
            "The colour balance appears natural."
        )