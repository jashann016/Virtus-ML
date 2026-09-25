import pandas as pd
from src.metrics import compute_macro_f05

SCORED_FILE = "output/independent_scored_candidates.parquet"
GT_FILE = "train_ground_truth.tsv"
S1_FILE = "train_source1.tsv"

THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90]

print("=" * 70)
print("INDEPENDENT MACRO-F0.5 THRESHOLD EVALUATION")
print("=" * 70)

# Exact held-out S1 IDs
s1_ids = pd.read_csv(
    S1_FILE,
    sep="\t",
    skiprows=range(1, 20001),
    nrows=1000,
    usecols=["entity_id"],
    dtype=str
)["entity_id"].tolist()

s1_set = set(s1_ids)

# Ground truth
gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

gt = gt[gt["source1_entity_id"].isin(s1_set)]

ground_truth = {}

for _, row in gt.iterrows():
    value = row["matched_entity_ids"]

    if pd.isna(value) or not str(value).strip():
        matches = []
    else:
        matches = [
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        ]

    ground_truth[row["source1_entity_id"]] = matches

for s1_id in s1_ids:
    ground_truth.setdefault(s1_id, [])

print(f"Held-out S1 entities : {len(s1_ids):,}")
print(
    "True match links     :",
    sum(len(v) for v in ground_truth.values())
)

# Read only columns needed for evaluation
df = pd.read_parquet(
    SCORED_FILE,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "probability"
    ]
)

print(f"Scored candidate rows: {len(df):,}")

results = []

for threshold in THRESHOLDS:

    selected = df[df["probability"] >= threshold]

    predictions = {
        s1_id: []
        for s1_id in s1_ids
    }

    for s1_id, group in selected.groupby("source1_entity_id"):
        predictions[s1_id] = group["candidate_entity_id"].tolist()

    score = compute_macro_f05(
        ground_truth,
        predictions
    )

    total_predictions = len(selected)
    entities_with_predictions = sum(
        1 for v in predictions.values() if v
    )

    results.append({
        "threshold": threshold,
        "macro_f05": score,
        "predicted_links": total_predictions,
        "entities_with_predictions": entities_with_predictions
    })

    print(
        f"Threshold {threshold:.2f} | "
        f"Macro-F0.5 {score:.6f} | "
        f"Predicted links {total_predictions:,} | "
        f"S1 with predictions {entities_with_predictions:,}"
    )

result_df = pd.DataFrame(results)

result_df.to_csv(
    "output/independent_threshold_results.csv",
    index=False
)

print()
print("Saved: output/independent_threshold_results.csv")
print("=" * 70)
