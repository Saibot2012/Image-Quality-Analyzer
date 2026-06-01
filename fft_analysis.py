import numpy as np

def compute_fft(img):
    fft = np.fft.fft2(img)  #converts pixels to frequencies
    fft_shifted = np.fft.fftshift(fft) #moves low frequencies to center
    magnitude = np.abs(fft_shifted) #only keeps strength
    log_magnitude = np.log1p(magnitude) #make visualization look better
    fft_norm = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min()) #normalization
    return fft_norm
