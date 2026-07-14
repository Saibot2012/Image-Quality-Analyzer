import cv2
import numpy as np

def analyze_brightness(gray):
    brightness = np.mean(gray) / 255.0

    #Histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist = hist.flatten()

    total_pixels = gray.size

    #Percentage of nearly black pixels
    shadow_clip = np.sum(hist[:10]) / total_pixels

    #Percentage of nearly white pixels
    highlight_clip = np.sum(hist[246:]) / total_pixels

    return{
        "brightness": brightness,
        "shadow_clip": shadow_clip,
        "highlight_clip": highlight_clip,
    }