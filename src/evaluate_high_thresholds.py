import pandas as pd
from src.metrics import compute_macro_f05

SCORED = "output/independent_scored_candidates.parquet"

s1_ids = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    skiprows=range(1, 20001),
    nrows=1000,
    usecols=["entity_id"],
    dtype=str
)["entity_id"].tolist()

s1_set = set(s1_ids)

gt = pd.read_csv(
    "train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

gt = gt[gt["source1_entity_id"].isin(s1_set)]

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

df = pd.read_parquet(
    SCORED,
    columns=["source1_entity_id", "candidate_entity_id", "probability"]
)

for threshold in [0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:

    selected = df[df["probability"] >= threshold]

    predictions = {s1_id: [] for s1_id in s1_ids}

    for s1_id, group in selected.groupby("source1_entity_id"):
        predictions[s1_id] = group["candidate_entity_id"].tolist()

    score = compute_macro_f05(ground_truth, predictions)

    print(
        f"Threshold {threshold:.2f} | "
        f"Macro-F0.5 {score:.6f} | "
        f"Predicted links {len(selected):,}"
    )
