import os
import sys
import pickle

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_engineering import FEATURE_COLUMNS

V2_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "lightgbm_training_pairs_v2.csv"
)

HARD_NEG_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "hard_negatives_v3.csv"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lightgbm_v3.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 80)
print("LIGHTGBM V3 - HARD NEGATIVE TRAINING")
print("=" * 80)

# ---------------------------------------------------------
# 1. Load existing V2 pairs
# ---------------------------------------------------------

print("\nLoading V2 training pairs...")

df = pd.read_csv(V2_PATH)

print("V2 rows:", len(df))
print("V2 positives:", int((df["label"] == 1).sum()))
print("V2 negatives:", int((df["label"] == 0).sum()))
print("Unique S1:", df["source1_entity_id"].nunique())

# ---------------------------------------------------------
# 2. SAME S1-LEVEL SPLIT AS V2
# ---------------------------------------------------------

all_s1_ids = df["source1_entity_id"].unique()

train_s1, valid_s1 = train_test_split(
    all_s1_ids,
    test_size=0.20,
    random_state=42
)

train_s1 = set(train_s1)
valid_s1 = set(valid_s1)

train_mask = df["source1_entity_id"].isin(train_s1)
valid_mask = df["source1_entity_id"].isin(valid_s1)

v2_train = df.loc[train_mask].copy()
v2_valid = df.loc[valid_mask].copy()

print("\n========== V2 SPLIT ==========")
print("Train S1:", len(train_s1))
print("Validation S1:", len(valid_s1))
print("V2 train rows:", len(v2_train))
print("V2 validation rows:", len(v2_valid))
print("V2 train positives:", int(v2_train["label"].sum()))
print("V2 train negatives:", int((v2_train["label"] == 0).sum()))
print("==============================")

# ---------------------------------------------------------
# 3. Load hard negatives
# ---------------------------------------------------------

print("\nLoading hard negatives...")

hard = pd.read_csv(HARD_NEG_PATH)

print("Hard negatives:", len(hard))
print("Hard negative labels:", hard["label"].value_counts().to_dict())

# Safety checks
hard = hard[hard["label"] == 0].copy()

# Only use hard negatives belonging to TRAIN S1s.
hard_train = hard[
    hard["source1_entity_id"].isin(train_s1)
].copy()

print("Hard negatives used for training:", len(hard_train))

# ---------------------------------------------------------
# 4. Remove accidental duplicate pairs
# ---------------------------------------------------------

pair_cols = [
    "source1_entity_id",
    "candidate_entity_id"
]

existing_pairs = set(
    zip(
        v2_train["source1_entity_id"],
        v2_train["candidate_entity_id"]
    )
)

hard_train = hard_train[
    ~hard_train.apply(
        lambda r: (
            r["source1_entity_id"],
            r["candidate_entity_id"]
        ) in existing_pairs,
        axis=1
    )
].copy()

print("Hard negatives after duplicate removal:", len(hard_train))

# ---------------------------------------------------------
# 5. Build V3 training set
# ---------------------------------------------------------

keep_columns = [
    "source1_entity_id",
    "candidate_entity_id",
    "label"
] + FEATURE_COLUMNS

v2_train = v2_train[keep_columns]
hard_train = hard_train[keep_columns]

v3_train = pd.concat(
    [v2_train, hard_train],
    ignore_index=True
)

print("\n========== V3 TRAINING DATA ==========")
print("V2 training rows:", len(v2_train))
print("Hard negatives added:", len(hard_train))
print("V3 total rows:", len(v3_train))
print("V3 positives:", int((v3_train["label"] == 1).sum()))
print("V3 negatives:", int((v3_train["label"] == 0).sum()))
print("======================================")

# ---------------------------------------------------------
# 6. Prepare features
# ---------------------------------------------------------

X_train = v3_train[FEATURE_COLUMNS]
y_train = v3_train["label"]

X_valid = v2_valid[FEATURE_COLUMNS]
y_valid = v2_valid["label"]

# ---------------------------------------------------------
# 7. Train LightGBM V3
# ---------------------------------------------------------

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=2
)

print("\nTraining LightGBM V3...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[
        lgb.early_stopping(40),
        lgb.log_evaluation(50)
    ]
)

# ---------------------------------------------------------
# 8. Validation
# ---------------------------------------------------------

probabilities = model.predict_proba(X_valid)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)

print("\n========== V3 VALIDATION ==========")

print(
    classification_report(
        y_valid,
        predictions,
        digits=4
    )
)

try:
    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                y_valid,
                probabilities
            ),
            4
        )
    )
except Exception:
    pass

# ---------------------------------------------------------
# 9. Save model
# ---------------------------------------------------------

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\nModel saved:")
print(MODEL_PATH)

print("\n========== V3 COMPLETE ==========")