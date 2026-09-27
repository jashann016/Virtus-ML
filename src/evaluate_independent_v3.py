import pandas as pd
from metrics import compute_macro_f05

SCORED_FILE = "output/independent_scored_candidates_v3.parquet"
GROUND_TRUTH = "train_ground_truth.tsv"
SOURCE1_FILE = "train_source1.tsv"

THRESHOLD = 0.50


print("=" * 70)
print("INDEPENDENT LIGHTGBM V3 EVALUATION")
print("=" * 70)


# --------------------------------------------------
# 1. Load held-out Source 1 IDs
# --------------------------------------------------

print("\nLoading held-out Source 1 entities...")

s1_ids = pd.read_csv(
    SOURCE1_FILE,
    sep="\t",
    skiprows=range(1, 20001),
    nrows=1000,
    usecols=["entity_id"],
    dtype=str
)["entity_id"].tolist()

print(f"Held-out S1 entities: {len(s1_ids):,}")


# --------------------------------------------------
# 2. Load ground truth
# --------------------------------------------------

print("\nLoading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str
)

gt = gt[
    gt["source1_entity_id"].isin(s1_ids)
]

ground_truth = {}

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]
    value = row["matched_entity_ids"]

    if pd.isna(value) or not str(value).strip():

        ground_truth[s1_id] = []

    else:

        ground_truth[s1_id] = [
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        ]


# Make sure every held-out S1 is present
for s1_id in s1_ids:
    ground_truth.setdefault(s1_id, [])


print(
    f"Ground-truth S1 loaded: "
    f"{len(ground_truth):,}"
)

print(
    "Ground-truth positive links:",
    sum(len(v) for v in ground_truth.values())
)


# --------------------------------------------------
# 3. Load V3 scored candidates
# --------------------------------------------------

print("\nLoading V3 scored candidates...")

df = pd.read_parquet(
    SCORED_FILE,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "probability"
    ]
)

print(
    f"Scored candidates loaded: "
    f"{len(df):,}"
)


# --------------------------------------------------
# 4. Keep only held-out S1
# --------------------------------------------------

df = df[
    df["source1_entity_id"].isin(s1_ids)
].copy()

print(
    f"Held-out candidate rows: "
    f"{len(df):,}"
)


# --------------------------------------------------
# 5. Create predictions at threshold
# --------------------------------------------------

pred_df = df[
    df["probability"] >= THRESHOLD
].copy()

print(
    f"Predicted links @ {THRESHOLD:.2f}: "
    f"{len(pred_df):,}"
)


# --------------------------------------------------
# 6. Build prediction dictionary
# --------------------------------------------------

predictions = {}

for s1_id, group in pred_df.groupby("source1_entity_id"):

    predictions[s1_id] = (
        group["candidate_entity_id"]
        .astype(str)
        .tolist()
    )


# Make sure every S1 exists
for s1_id in s1_ids:
    predictions.setdefault(s1_id, [])


# --------------------------------------------------
# 7. Calculate Macro-F0.5
# --------------------------------------------------

macro_f05 = compute_macro_f05(
    ground_truth,
    predictions
)


# --------------------------------------------------
# 8. Additional statistics
# --------------------------------------------------

predicted_s1 = sum(
    1 for values in predictions.values()
    if len(values) > 0
)

singleton_count = sum(
    1 for values in ground_truth.values()
    if len(values) == 0
)

non_singleton_count = len(ground_truth) - singleton_count


print()
print("=" * 70)
print("INDEPENDENT V3 RESULTS")
print("=" * 70)

print(
    f"Held-out S1 entities       : {len(ground_truth):,}"
)

print(
    f"Ground-truth links         : "
    f"{sum(len(v) for v in ground_truth.values()):,}"
)

print(
    f"Predicted links            : "
    f"{len(pred_df):,}"
)

print(
    f"S1 entities with matches   : "
    f"{predicted_s1:,}"
)

print(
    f"Singleton S1 entities      : "
    f"{singleton_count:,}"
)

print(
    f"Non-singleton S1 entities  : "
    f"{non_singleton_count:,}"
)

print(
    f"Threshold                  : "
    f"{THRESHOLD:.2f}"
)

print()
print(
    f"Macro-F0.5                 : "
    f"{macro_f05:.6f}"
)

print("=" * 70)

print()
print("STATUS: PASS - Independent V3 evaluation completed.")