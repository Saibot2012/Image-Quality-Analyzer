import pickle
import numpy as np
import joblib

import pandas as pd
from sklearn.model_selection import cross_val_score

# Load your existing model and data
with open('eye_model_4class_newest.pkl', 'rb') as f:
    rf = joblib.load("ml/eye_model_4class_newest.pkl")

df = pd.read_csv('eye_dataset.csv')

# Check if dropping ratio and head_roll even matters
# Cross-validate with current features vs reduced features
features_full = ['left_ear', 'right_ear', 'avg_ear', 'eye_difference', 'ratio', 'head_roll', "left_lid_bulge", "right_lid_bulge"]
features_reduced = ['left_ear', 'right_ear', 'avg_ear', 'eye_difference']

X_full = df[features_full]
X_reduced = df[features_reduced]
y = df['label']

from sklearn.ensemble import RandomForestClassifier
rf_test = RandomForestClassifier(n_estimators=100, random_state=42)

score_full = cross_val_score(rf_test, X_full, y, cv=5, scoring='accuracy').mean()
score_reduced = cross_val_score(rf_test, X_reduced, y, cv=5, scoring='accuracy').mean()

print(f"Full features CV accuracy:    {score_full:.4f}")
print(f"Reduced features CV accuracy: {score_reduced:.4f}")
print(f"Difference: {(score_reduced - score_full):.4f}")

df['min_ear'] = df[['left_ear', 'right_ear']].min(axis=1)
df['max_ear'] = df[['left_ear', 'right_ear']].max(axis=1)
df['ear_asymmetry_ratio'] = df['min_ear'] / df['max_ear']

features_new = ['left_ear', 'right_ear', 'avg_ear', 'eye_difference', 'head_roll']
X_new = df[features_new]

score_new = cross_val_score(rf_test, X_new, y, cv=5, scoring='accuracy').mean()
print(f"New features CV accuracy: {score_new:.4f}")