import pandas as pd
from pathlib import Path

INPUT = Path("output/independent_scored_candidates.parquet")
OUTPUT = Path("output/person2_final_predictions_095.csv")

THRESHOLD = 0.95

df = pd.read_parquet(INPUT)

print("Rows:", len(df))
print("Columns:", list(df.columns))

# Detect ID columns
s1_candidates = ["source1_entity_id"]
candidate_candidates = ["candidate_entity_id"]

if not s1_candidates or not candidate_candidates:
    raise ValueError(
        f"Could not detect ID columns.\nAvailable columns: {list(df.columns)}"
    )

S1_COL = s1_candidates[0]
CANDIDATE_COL = candidate_candidates[0]

print("S1 column:", S1_COL)
print("Candidate column:", CANDIDATE_COL)

# Keep only high-confidence matches
matches = df[df["probability"] >= THRESHOLD].copy()

# Group candidate IDs by Source-1 entity
predictions = (
    matches.groupby(S1_COL)[CANDIDATE_COL]
    .apply(lambda x: sorted(set(x.astype(str))))
    .to_dict()
)

# Create one row per S1 appearing in the benchmark
all_s1 = sorted(df[S1_COL].astype(str).unique())

rows = []
for s1 in all_s1:
    rows.append({
        "s1_id": s1,
        "predicted_matches": predictions.get(s1, [])
    })

result = pd.DataFrame(rows)

# CSV-friendly representation
result["predicted_matches"] = result["predicted_matches"].apply(
    lambda x: ",".join(x)
)

result.to_csv(OUTPUT, index=False)

print()
print("=" * 60)
print("PERSON 2 FINAL PREDICTION")
print("=" * 60)
print("Threshold:", THRESHOLD)
print("S1 entities:", len(result))
print("Entities with predictions:", (result["predicted_matches"] != "").sum())
print("Entities with empty predictions:", (result["predicted_matches"] == "").sum())
print("Total predicted links:", matches.shape[0])
print("Output:", OUTPUT)
print("=" * 60)
