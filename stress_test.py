import pandas as pd
import numpy as np
import joblib
import sys

# Load model
model = joblib.load("eye_model_4class_newest.pkl")

# Load dataset
df = pd.read_csv("eye_dataset_copy.csv")

# Debug: show actual columns read from CSV
print("CSV columns:", list(df.columns))

# Common expected feature names (from your dataset header)
expected_features = [
    "left_ear","right_ear","avg_ear","eye_difference","ratio",
    "head_roll","left_lid_bulge","right_lid_bulge"
]

# If the model exposes feature names used during training, prefer them
model_features = None
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    print("Model expects features:", model_features)

# Decide which columns to use for prediction
if model_features:
    feature_cols = [c for c in model_features if c in df.columns]
else:
    # fallback: use intersection of expected_features and CSV columns
    feature_cols = [c for c in expected_features if c in df.columns]

if not feature_cols:
    print("Error: No matching feature columns found between CSV and model.")
    print("CSV columns:", list(df.columns))
    print("Expected (sample):", expected_features)
    sys.exit(1)

print("Using feature columns:", feature_cols)

# Helper to create a DataFrame with the exact columns the model expects
def make_input_df(sample_dict_or_series):
    # If model has feature_names_in_, use that order and fill missing with NaN
    if model_features:
        cols = model_features
    else:
        cols = feature_cols
    # Build a dict with all cols
    row = {c: np.nan for c in cols}
    # update with provided values
    for k, v in sample_dict_or_series.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row], columns=cols)

# Stress test 1: Extreme values
# Use first numeric row available (safe selection)
first_row = df.iloc[0]
# If first_row doesn't contain feature columns as labels, extract from df[feature_cols].iloc[0]
if set(feature_cols).issubset(df.columns):
    base = df.loc[0, feature_cols].to_dict()
else:
    # fallback: take numeric columns by position (first len(expected_features) numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= len(expected_features):
        base = df.loc[0, numeric_cols[:len(expected_features)]].to_dict()
    else:
        base = {}
# modify extremes
base.update({"left_ear": 10.0, "right_ear": -5.0})

extreme_df = make_input_df(base)
print("Extreme input (columns and values):")
print(extreme_df.to_dict(orient="records")[0])

try:
    pred_extreme = model.predict(extreme_df)
    print("Extreme prediction:", pred_extreme)
except Exception as e:
    print("Prediction failed for extreme sample:", e)

# Stress test 2: Add noise to entire dataset (only on feature_cols)
noisy_features = df[feature_cols].copy()
noise = np.random.normal(0, 0.5, noisy_features.shape)
noisy_features = noisy_features + noise
# Align columns to model if needed
if model_features:
    # ensure order and presence
    noisy_input = pd.DataFrame(np.nan, index=range(len(noisy_features)), columns=model_features)
    for c in noisy_features.columns:
        if c in noisy_input.columns:
            noisy_input[c] = noisy_features[c].values
else:
    noisy_input = noisy_features

try:
    preds_noisy = model.predict(noisy_input)
    print("Noisy predictions sample (first 10):", preds_noisy[:10])
except Exception as e:
    print("Prediction failed for noisy dataset:", e)

# Stress test 3: Duplicate consistency
dups = df[df['filename'].isin(['open28.webp','open45.webp'])] if 'filename' in df.columns else pd.DataFrame()
if not dups.empty:
    dup_input = dups[feature_cols].copy()
    # align to model features
    if model_features:
        dup_input = pd.DataFrame({c: dup_input[c] if c in dup_input.columns else np.nan for c in model_features})
    try:
        print("Duplicate filenames:", list(dups['filename']))
        print("Duplicate predictions:", model.predict(dup_input))
    except Exception as e:
        print("Prediction failed for duplicates:", e)
else:
    print("No duplicate filenames found or 'filename' column missing.")
