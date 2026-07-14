import numpy as np
import cv2 as cv

def laplacian_sharpness(gray_img):
    return cv.Laplacian(gray_img, cv.CV_64F).var()