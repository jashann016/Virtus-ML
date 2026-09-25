import pandas as pd
from src.metrics import compute_macro_f05

FEATURES = "output/independent_features.parquet"
SCORED = "output/independent_scored_candidates.parquet"

# Held-out S1 IDs
s1_ids = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    skiprows=range(1, 20001),
    nrows=1000,
    usecols=["entity_id"],
    dtype=str
)["entity_id"].tolist()

# Ground truth
gt = pd.read_csv(
    "train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

gt = gt[gt["source1_entity_id"].isin(set(s1_ids))]

ground_truth = {}

for _, row in gt.iterrows():
    value = row["matched_entity_ids"]

    ground_truth[row["source1_entity_id"]] = (
        []
        if pd.isna(value) or not str(value).strip()
        else [x.strip() for x in str(value).split(",") if x.strip()]
    )

for s1_id in s1_ids:
    ground_truth.setdefault(s1_id, [])

# Load scores + required features
scores = pd.read_parquet(
    SCORED,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "probability"
    ]
)

features = pd.read_parquet(
    FEATURES,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "number_conflict_flag",
        "number_overlap_ratio",
        "exact_number_match"
    ]
)

df = scores.merge(
    features,
    on=["source1_entity_id", "candidate_entity_id"],
    how="left"
)

def evaluate(name, mask):
    selected = df[mask]

    predictions = {s1_id: [] for s1_id in s1_ids}

    for s1_id, group in selected.groupby("source1_entity_id"):
        predictions[s1_id] = group["candidate_entity_id"].tolist()

    score = compute_macro_f05(
        ground_truth,
        predictions
    )

    print(
        f"{name:<65} "
        f"Macro-F0.5={score:.6f} | "
        f"Predicted={len(selected):,}"
    )

print("=" * 100)
print("NUMBER-CONFLICT POST-MODEL EXPERIMENT")
print("=" * 100)

base = df["probability"] >= 0.99

evaluate(
    "Baseline: probability >= 0.99",
    base
)

evaluate(
    "Reject all number conflicts",
    base & (df["number_conflict_flag"] == 0)
)

evaluate(
    "Reject conflict + overlap < 1",
    base & ~(
        (df["number_conflict_flag"] == 1) &
        (df["number_overlap_ratio"] < 1)
    )
)

evaluate(
    "Reject conflict + exact_number_match = 0",
    base & ~(
        (df["number_conflict_flag"] == 1) &
        (df["exact_number_match"] == 0)
    )
)
