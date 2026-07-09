import pandas as pd

data = pd.read_csv("ml/dataset.csv")

features = [
    "laplacian",
    "fft_ratio",
    "wavelet_ratio",
    "sharp_ratio",
    "consistency",
    "exposure",
    "noise",
    "detail_quality",
]

print("\n===== Dataset Statistics =====")

sharp = data[data["label"] == 1]
blurry = data[data["label"] == 0]
for feature in features:

    print(f"\n===== {feature} =====")

    print(f"Sharp Mean  : {sharp[feature].mean():.6f}")
    print(f"Blurry Mean : {blurry[feature].mean():.6f}")

    print(f"Sharp Min   : {sharp[feature].min():.6f}")
    print(f"Sharp Max   : {sharp[feature].max():.6f}")

    print(f"Blur Min    : {blurry[feature].min():.6f}")
    print(f"Blur Max    : {blurry[feature].max():.6f}")

