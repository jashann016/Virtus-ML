import os
import sys
import pickle
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_engineering import FEATURE_COLUMNS

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "lightgbm_training_pairs_v2.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "lightgbm_baseline_v2.pkl"
)

SCORED_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "scored_candidate_pairs.csv"
)

BORDERLINE_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "borderline_pairs.csv"
)

print("Loading candidate pairs...")

df = pd.read_csv(INPUT_PATH)

print("Pairs:", len(df))

print("Loading LightGBM model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Scoring pairs...")

df["probability"] = model.predict_proba(
    df[FEATURE_COLUMNS]
)[:, 1]

def classify(probability):
    if probability >= 0.88:
        return "clear_match"
    elif probability < 0.35:
        return "reject"
    else:
        return "borderline"

df["decision"] = df["probability"].apply(classify)

df.to_csv(
    SCORED_PATH,
    index=False
)

borderline = df[
    df["decision"] == "borderline"
].copy()

borderline.to_csv(
    BORDERLINE_PATH,
    index=False
)

print("\n========== PREDICTION RESULT ==========")
print("Total pairs:", len(df))
print("Clear matches:", (df["decision"] == "clear_match").sum())
print("Reject:", (df["decision"] == "reject").sum())
print("Borderline:", (df["decision"] == "borderline").sum())
print("========================================")

print("\nSaved:")
print(SCORED_PATH)
print(BORDERLINE_PATH)
