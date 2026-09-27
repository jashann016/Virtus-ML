import pandas as pd

SCORED_FILE = "output/independent_scored_candidates_v3.parquet"
GROUND_TRUTH = "train_ground_truth.tsv"
SOURCE1_FILE = "train_source1.tsv"

THRESHOLD = 0.50

print("=" * 70)
print("V3 INDEPENDENT ERROR ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# 1. Held-out S1 IDs
# --------------------------------------------------

s1_ids = pd.read_csv(
    SOURCE1_FILE,
    sep="\t",
    skiprows=range(1, 20001),
    nrows=1000,
    usecols=["entity_id"],
    dtype=str
)["entity_id"].tolist()


# --------------------------------------------------
# 2. Ground truth
# --------------------------------------------------

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

        ground_truth[s1_id] = set()

    else:

        ground_truth[s1_id] = set(
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        )

for s1_id in s1_ids:
    ground_truth.setdefault(s1_id, set())


# --------------------------------------------------
# 3. Load scored candidates
# --------------------------------------------------

df = pd.read_parquet(
    SCORED_FILE
)

df = df[
    df["source1_entity_id"].isin(s1_ids)
].copy()


# --------------------------------------------------
# 4. Evaluate every candidate
# --------------------------------------------------

def classify(row):

    s1 = row["source1_entity_id"]
    candidate = row["candidate_entity_id"]

    is_true = candidate in ground_truth.get(s1, set())
    is_pred = row["probability"] >= THRESHOLD

    if is_pred and is_true:
        return "TP"

    if is_pred and not is_true:
        return "FP"

    if not is_pred and is_true:
        return "FN"

    return "TN"


print("\nClassifying candidates...")

df["error_type"] = df.apply(
    classify,
    axis=1
)


# --------------------------------------------------
# 5. Summary
# --------------------------------------------------

counts = df["error_type"].value_counts()

print()
print("=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

for label in ["TP", "FP", "FN", "TN"]:

    print(
        f"{label}: "
        f"{int(counts.get(label, 0)):,}"
    )


# --------------------------------------------------
# 6. Precision / Recall
# --------------------------------------------------

tp = int(counts.get("TP", 0))
fp = int(counts.get("FP", 0))
fn = int(counts.get("FN", 0))

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

print()
print("Precision:", round(precision, 6))
print("Recall   :", round(recall, 6))


# --------------------------------------------------
# 7. Probability distributions
# --------------------------------------------------

print()
print("=" * 70)
print("PROBABILITY DISTRIBUTION")
print("=" * 70)

for label in ["TP", "FP", "FN"]:

    subset = df[
        df["error_type"] == label
    ]["probability"]

    if len(subset) == 0:
        continue

    print()
    print(label)

    print("Count :", len(subset))
    print("Mean  :", round(subset.mean(), 6))
    print("Median:", round(subset.median(), 6))
    print("Min   :", round(subset.min(), 6))
    print("Max   :", round(subset.max(), 6))


# --------------------------------------------------
# 8. Top false positives
# --------------------------------------------------

fp_df = df[
    df["error_type"] == "FP"
].sort_values(
    "probability",
    ascending=False
)

print()
print("=" * 70)
print("TOP 50 FALSE POSITIVES")
print("=" * 70)

print(
    fp_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "probability",
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
            "token_count_ratio"
        ]
    ]
    .head(50)
    .to_string(index=False)
)


# --------------------------------------------------
# 9. Top false negatives
# --------------------------------------------------

fn_df = df[
    df["error_type"] == "FN"
].sort_values(
    "probability",
    ascending=False
)

print()
print("=" * 70)
print("FALSE NEGATIVE SUMMARY")
print("=" * 70)

print(
    fn_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "probability"
        ]
    ]
    .head(50)
    .to_string(index=False)
)


# --------------------------------------------------
# 10. Save error analysis
# --------------------------------------------------

OUTPUT = "output/independent_v3_error_analysis.csv"

df[
    df["error_type"].isin(["TP", "FP", "FN"])
].to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved:", OUTPUT)

print()
print("=" * 70)
print("STATUS: ERROR ANALYSIS COMPLETE")
print("=" * 70)