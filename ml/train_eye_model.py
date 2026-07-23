import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Load dataset
df = pd.read_csv(
    "eye_dataset.csv"
)

print(df.head())
print("\nDataset size:", df.shape)


# Features
X = df[
    [
    "left_ear",
    "right_ear",
    "avg_ear",
    "eye_difference",
    "ratio",
    "eye_difference_sign"
    ]
]


# Labels
y = df["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


# Predict
predictions = model.predict(
    X_test
)


# Evaluation
accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", accuracy)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Open",
            "Closed",
            "Right Closed",
            "Left Closed"
        ]
    )
)




print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# Feature importance
print("\nFeature Importance:")

for name, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(
        name,
        ":",
        round(importance, 3)
    )


# Save model
joblib.dump(
    model,
    "eye_model_6class.pkl"
)


print("\nModel saved as eye_model_6class.pkl")