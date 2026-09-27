import pandas as pd
from pathlib import Path

BASE = Path(".")
OUT = BASE / "output"

INPUT = OUT / "lightgbm_training_pairs_v2.csv"
SCORED = OUT / "scored_candidate_pairs.csv"

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

PAIR_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
]

print("Loading v2 training pairs...")
df = pd.read_csv(INPUT)

required = PAIR_COLUMNS + ["label"] + FEATURE_COLUMNS
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

print(f"Rows: {len(df):,}")
print(f"Features: {len(FEATURE_COLUMNS)}")

# ---------------------------------------------------------
# 1. Candidate pairs
# ---------------------------------------------------------
candidate_pairs = df[PAIR_COLUMNS].copy()

candidate_pairs.to_parquet(
    OUT / "candidate_pairs.parquet",
    index=False,
)

# ---------------------------------------------------------
# 2. Features
# Keep IDs + label + exactly the 14 feature columns
# ---------------------------------------------------------
features = df[PAIR_COLUMNS + ["label"] + FEATURE_COLUMNS].copy()

features.to_parquet(
    OUT / "features.parquet",
    index=False,
)

# ---------------------------------------------------------
# 3. Scored candidates
# Existing prediction output is the source of truth
# ---------------------------------------------------------
if not SCORED.exists():
    raise FileNotFoundError(
        "scored_candidate_pairs.csv does not exist."
    )

scored = pd.read_csv(SCORED)

scored.to_parquet(
    OUT / "scored_candidates.parquet",
    index=False,
)

# ---------------------------------------------------------
# 4. Borderline cases
# Existing borderline output -> required filename
# ---------------------------------------------------------
borderline_source = OUT / "borderline_pairs.csv"

if not borderline_source.exists():
    raise FileNotFoundError(
        "borderline_pairs.csv does not exist."
    )

borderline = pd.read_csv(borderline_source)

borderline.to_csv(
    OUT / "borderline_cases.csv",
    index=False,
)

print()
print("=" * 60)
print("FINAL ARTIFACTS CREATED")
print("=" * 60)

for filename in [
    "candidate_pairs.parquet",
    "features.parquet",
    "scored_candidates.parquet",
    "borderline_cases.csv",
]:
    path = OUT / filename
    print(f"{filename:30} {path.stat().st_size:,} bytes")

print("=" * 60)
