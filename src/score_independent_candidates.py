import pandas as pd
import joblib
from pathlib import Path

FEATURE_FILE = "output/independent_features.parquet"
MODEL_FILE = "models/lightgbm_baseline_v2.pkl"
OUTPUT_FILE = "output/independent_scored_candidates.parquet"

FEATURE_COLUMNS = [
    "levenshtein_ratio",
    "jaro_winkler",
    "token_sort_ratio",
    "token_set_ratio",
    "exact_name_match",
    "domain_match_flag",
    "number_overlap_ratio",
    "exact_number_match",
    "number_conflict_flag",
    "address_token_jaccard",
    "address_token_sort_ratio",
    "missing_address_flag",
    "name_length_diff",
    "token_count_ratio",
]

print("=" * 70)
print("INDEPENDENT LIGHTGBM SCORING")
print("=" * 70)

if not Path(FEATURE_FILE).exists():
    raise FileNotFoundError(FEATURE_FILE)

if not Path(MODEL_FILE).exists():
    raise FileNotFoundError(MODEL_FILE)

print("Loading LightGBM model...")
model = joblib.load(MODEL_FILE)

print("Loading independent features...")
df = pd.read_parquet(FEATURE_FILE)

print(f"Rows loaded: {len(df):,}")

missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

X = df[FEATURE_COLUMNS]

print("Generating probabilities...")
df["probability"] = model.predict_proba(X)[:, 1]

df["decision"] = "borderline"
df.loc[df["probability"] >= 0.88, "decision"] = "clear_match"
df.loc[df["probability"] < 0.35, "decision"] = "reject"

df.to_parquet(OUTPUT_FILE, index=False)

print()
print("Decision counts:")
print(df["decision"].value_counts().to_string())

print()
print(f"Output: {OUTPUT_FILE}")
print("=" * 70)
print("STATUS: PASS - independent candidates scored.")
print("=" * 70)
