import cv2
import numpy as np

def compute_motion_directionality(gray):
    """
    Returnshow strongly edge directions are concentrated.
    0 -> edges spread in many directions
    1 -> edges mostly point in one direction
    """

    #sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    #edge strength(how strong each edge is)
    magnitude = np.sqrt(gx**2 + gy**2)

    #ignore tiny gradients (Otherwise tiny noisy pixels become 'edges')
    mask = magnitude > 20

    if np.sum(mask) == 0:
        return{
            "motion_directionality": 0
        }
    
    #edge angle
    angles = np.arctan2(gy[mask], gx[mask])

    #Map angles into [0, 180]
    angles = (angles + 180) % 180

    #histogram
    hist, _ = np.histogram(
        angles,
        bins=18,
        range=(0,180)
    )

    directionality = hist.max() / hist.sum()

    return{
        "motion_directionality": float(directionality)
    }
