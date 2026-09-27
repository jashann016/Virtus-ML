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

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "lightgbm_training_pairs_v2.csv"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lightgbm_baseline_v2.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading training pairs...")
df = pd.read_csv(INPUT_PATH)

print("Rows:", len(df))
print("Positive:", int((df["label"] == 1).sum()))
print("Negative:", int((df["label"] == 0).sum()))
print("Unique S1:", df["source1_entity_id"].nunique())

all_s1_ids = df["source1_entity_id"].unique()

train_s1, valid_s1 = train_test_split(
    all_s1_ids,
    test_size=0.20,
    random_state=42
)

train_mask = df["source1_entity_id"].isin(train_s1)
valid_mask = df["source1_entity_id"].isin(valid_s1)

X_train = df.loc[train_mask, FEATURE_COLUMNS]
y_train = df.loc[train_mask, "label"]

X_valid = df.loc[valid_mask, FEATURE_COLUMNS]
y_valid = df.loc[valid_mask, "label"]

print("\n========== SPLIT ==========")
print("Train S1:", len(train_s1))
print("Validation S1:", len(valid_s1))
print("Train rows:", len(X_train))
print("Validation rows:", len(X_valid))
print("Train positives:", int(y_train.sum()))
print("Validation positives:", int(y_valid.sum()))
print("============================")

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("\nTraining LightGBM...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[
        lgb.early_stopping(40),
        lgb.log_evaluation(50)
    ]
)

probabilities = model.predict_proba(X_valid)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("\n========== VALIDATION ==========")
print(classification_report(y_valid, predictions, digits=4))

try:
    print(
        "ROC-AUC:",
        round(roc_auc_score(y_valid, probabilities), 4)
    )
except Exception:
    pass

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\nModel saved:")
print(MODEL_PATH)
