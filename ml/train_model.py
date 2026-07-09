import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("ml/dataset.csv")

# Features (X)
X = data[
    [
        "laplacian",
        "fft_ratio",
        "wavelet_ratio",
        "sharp_ratio",
        "consistency",
        "exposure",
        "noise",
        "detail_quality",

    ]
]

# Labels (y)
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2%}")

feature_names = X.columns

print("\nFeature Importance")
print("------------------")

for name, importance in zip(feature_names, model.feature_importances_):
    print(f"{name:15}: {importance:.3f}")

# Save model
joblib.dump(model, "ml/model.pkl")

print("Model saved to ml/model.pkl")

