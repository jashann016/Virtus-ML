import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from metrics import compute_macro_f05
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

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "threshold_results_v2.csv"
)

print("Loading data...")
df = pd.read_csv(INPUT_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

all_s1_ids = df["source1_entity_id"].unique()

train_s1, valid_s1 = train_test_split(
    all_s1_ids,
    test_size=0.20,
    random_state=42
)

valid_s1_set = set(valid_s1)

valid_df = df[
    df["source1_entity_id"].isin(valid_s1_set)
].copy()

print("Validation S1:", len(valid_s1))
print("Validation rows:", len(valid_df))

valid_df["probability"] = model.predict_proba(
    valid_df[FEATURE_COLUMNS]
)[:, 1]

# Ground truth from the validation pairs
ground_truth = {}

for s1_id in valid_s1:
    ids = set(
        valid_df.loc[
            (valid_df["source1_entity_id"] == s1_id) &
            (valid_df["label"] == 1),
            "candidate_entity_id"
        ]
    )
    ground_truth[s1_id] = ids

thresholds = [
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.88,
    0.90
]

results = []

print("\n========== THRESHOLD RESULTS ==========")

for threshold in thresholds:

    predictions = {}

    for s1_id in valid_s1:

        group = valid_df[
            valid_df["source1_entity_id"] == s1_id
        ]

        predicted = set(
            group.loc[
                group["probability"] >= threshold,
                "candidate_entity_id"
            ]
        )

        predictions[s1_id] = predicted

    score = compute_macro_f05(
        ground_truth,
        predictions
    )

    predicted_links = sum(
        len(v) for v in predictions.values()
    )

    s1_with_predictions = sum(
        bool(v) for v in predictions.values()
    )

    results.append({
        "threshold": threshold,
        "macro_f05": score,
        "predicted_links": predicted_links,
        "s1_with_predictions": s1_with_predictions
    })

    print(
        f"Threshold={threshold:.2f} | "
        f"Macro-F0.5={score:.4f} | "
        f"Predicted links={predicted_links} | "
        f"S1 with predictions={s1_with_predictions}"
    )

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)
