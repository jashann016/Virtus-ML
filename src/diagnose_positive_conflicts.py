import csv
from collections import Counter

from evaluate_blocking_recall import load_ground_truth_map
from feature_engineering import compute_pair_features

GT_PATH = "train_ground_truth.tsv"
S1_PATH = "train_source1.tsv"

POSITIVE_FILES = [
    "output/positive_source2.tsv",
    "output/positive_source3.tsv",
]

S1_LIMIT = 20_000

print("Loading ground truth...")
gt = load_ground_truth_map(GT_PATH)

selected_s1_ids = set(list(gt.keys())[:S1_LIMIT])

print("Loading S1...")
s1_records = []

with open(S1_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        if row["entity_id"] in selected_s1_ids:
            s1_records.append(row)

        if len(s1_records) == S1_LIMIT:
            break

print("S1 records:", len(s1_records))

print("Loading positive records...")

candidate_records = []

for path in POSITIVE_FILES:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        candidate_records.extend(reader)

candidate_map = {
    row["entity_id"]: row
    for row in candidate_records
}

print("Candidate records:", len(candidate_records))

counts = Counter()

total_positive = 0

for i, s1 in enumerate(s1_records, start=1):

    s1_id = s1["entity_id"]
    true_ids = gt.get(s1_id, set())

    for pid in true_ids:

        if pid not in candidate_map:
            continue

        features = compute_pair_features(
            s1,
            candidate_map[pid]
        )

        total_positive += 1

        if features["number_conflict_flag"] == 1:
            counts["number_conflict"] += 1

        if features["exact_number_match"] == 1:
            counts["exact_number_match"] += 1

        if features["address_token_jaccard"] >= 0.5:
            counts["high_jaccard"] += 1

        if features["jaro_winkler"] >= 0.9:
            counts["high_name"] += 1

        if (
            features["number_conflict_flag"] == 1
            and features["address_token_jaccard"] >= 0.5
        ):
            counts["conflict_high_jaccard"] += 1

        if (
            features["number_conflict_flag"] == 1
            and features["jaro_winkler"] >= 0.9
        ):
            counts["conflict_high_name"] += 1

    if i % 2000 == 0:
        print(f"Processed {i}/{len(s1_records)}")

print("\n========== RESULT ==========")

print("Positive pairs inspected:", total_positive)

print(
    "Number conflict:",
    counts["number_conflict"],
    f"({counts['number_conflict']/total_positive:.4%})"
)

print(
    "Exact number match:",
    counts["exact_number_match"],
    f"({counts['exact_number_match']/total_positive:.4%})"
)

print(
    "Address Jaccard >= 0.5:",
    counts["high_jaccard"],
    f"({counts['high_jaccard']/total_positive:.4%})"
)

print(
    "Name Jaro-Winkler >= 0.9:",
    counts["high_name"],
    f"({counts['high_name']/total_positive:.4%})"
)

print(
    "Conflict + high Jaccard:",
    counts["conflict_high_jaccard"],
    f"({counts['conflict_high_jaccard']/total_positive:.4%})"
)

print(
    "Conflict + high name:",
    counts["conflict_high_name"],
    f"({counts['conflict_high_name']/total_positive:.4%})"
)

print("============================")
