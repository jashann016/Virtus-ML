import pandas as pd

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

s1_set = set(s1_ids)

# Ground truth
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

# Load high-confidence predictions
df = pd.read_parquet(
    SCORED,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "probability"
    ]
)

df = df[df["probability"] >= 0.99].copy()

print("High-confidence predictions:", len(df))

# Mark true/false
df["is_true_match"] = [
    candidate in ground_truth.get(source1, [])
    for source1, candidate
    in zip(df["source1_entity_id"], df["candidate_entity_id"])
]

fp = df[~df["is_true_match"]].copy()
tp = df[df["is_true_match"]].copy()

print("True positives :", len(tp))
print("False positives:", len(fp))

print()
print("FALSE POSITIVE PROBABILITY SUMMARY")
print(fp["probability"].describe())

print()
print("TOP 30 FALSE POSITIVES")
print(
    fp.sort_values("probability", ascending=False)
      .head(30)
      .to_string(index=False)
)

fp.to_csv(
    "output/independent_false_positives_099.csv",
    index=False
)

print()
print("Saved: output/independent_false_positives_099.csv")
