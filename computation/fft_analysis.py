import numpy as np

def compute_fft_visualization(img):
    fft = np.fft.fft2(img)  #converts pixels to frequencies
    fft_shifted = np.fft.fftshift(fft) #moves low frequencies to center
    magnitude = np.abs(fft_shifted) #only keeps strength
    log_magnitude = np.log1p(magnitude) #make visualization look better
    fft_norm = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min()) #normalization
    return fft_norm


def compute_fft_features(img):
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    power_spectrum = magnitude ** 2 #magnitude spectrum to power spectrum. Amplitude = strength. Energy is proportional to amplitude^2.

    rows, cols = power_spectrum.shape #returns image shape
    crow = rows // 2
    ccol = cols // 2

    radius = min(rows, cols) * 0.1

    Y, X = np.ogrid[:rows, :cols] #creates coordinate grids. Compute distance from center.
    distance = np.sqrt((X - ccol)** 2 + (Y - crow) ** 2) 
    high_freq_mask = distance > radius #boolean: false inside the center circle(low freq), true outside the circle (high freq).

    high_energy = np.sum(power_spectrum[high_freq_mask])
    total_energy = np.sum(power_spectrum)

    high_freq_ratio = high_energy / (total_energy + 1e-12)

    return{
        "high_freq_ratio": high_freq_ratio
    }