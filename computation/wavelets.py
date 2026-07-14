import cv2 
import numpy as np 
import pywt 

def compute_wavelets(img):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    energy_lh = np.sum(LH ** 2)
    energy_hl = np.sum(HL ** 2)
    energy_hh = np.sum(HH ** 2)

    detail_energy = (
    energy_lh +
    energy_hl +
    energy_hh
    )
    total_energy = detail_energy + np.sum(LL ** 2)
    wavelet_ratio = detail_energy / (total_energy + 1e-12)

    return {
    "wavelet_ratio": wavelet_ratio
    }
