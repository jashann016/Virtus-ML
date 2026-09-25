import pandas as pd
from src.metrics import compute_macro_f05

GROUND_TRUTH = "train_ground_truth.tsv"
CANDIDATES = "output/independent_candidate_pairs.csv"

# Held-out S1 = rows 20,001–21,000
s1_ids = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    skiprows=range(1, 20001),
    nrows=1000,
    usecols=["entity_id"],
    dtype=str
)["entity_id"].tolist()

print(f"Held-out S1 entities: {len(s1_ids)}")

# Load ground truth only for held-out S1
gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str
)

gt = gt[gt["source1_entity_id"].isin(s1_ids)]

ground_truth = {}
for _, row in gt.iterrows():
    value = row["matched_entity_ids"]
    if pd.isna(value) or not str(value).strip():
        ground_truth[row["source1_entity_id"]] = []
    else:
        ground_truth[row["source1_entity_id"]] = [
            x.strip() for x in str(value).split(",") if x.strip()
        ]

for s1_id in s1_ids:
    ground_truth.setdefault(s1_id, [])

print(f"Ground-truth S1 loaded: {len(ground_truth)}")
print(
    "Ground-truth positive links:",
    sum(len(v) for v in ground_truth.values())
)

print("\nEvaluation script ready.")
print("Waiting for independent scored candidates...")
