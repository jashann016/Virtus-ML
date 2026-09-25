import os
import csv
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluate_blocking_recall import load_ground_truth_map
from blocking import InvertedIndexBlocker
from feature_engineering import compute_pair_features, FEATURE_COLUMNS

GT_PATH = os.path.join(PROJECT_ROOT, "train_ground_truth.tsv")
S1_PATH = os.path.join(PROJECT_ROOT, "train_source1.tsv")
S2_POS_PATH = os.path.join(PROJECT_ROOT, "output", "positive_source2.tsv")
S3_POS_PATH = os.path.join(PROJECT_ROOT, "output", "positive_source3.tsv")

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "output", "lightgbm_training_pairs_v2.csv"
)

S1_LIMIT = 20000
MAX_CANDIDATES = 20
NEGATIVES_PER_S1 = 5

print("Loading ground truth...")
gt = load_ground_truth_map(GT_PATH)
selected_s1_ids = set(list(gt.keys())[:S1_LIMIT])

print("Selected S1:", len(selected_s1_ids))

print("\nLoading selected S1 records...")
s1_records = []

with open(S1_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        if row["entity_id"] in selected_s1_ids:
            s1_records.append(row)

        if len(s1_records) == S1_LIMIT:
            break

print("S1 records loaded:", len(s1_records))

print("\nLoading positive candidate records...")
candidate_records = []

for path in [S2_POS_PATH, S3_POS_PATH]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            candidate_records.append(row)

candidate_map = {
    row["entity_id"]: row
    for row in candidate_records
}

print("Candidate records loaded:", len(candidate_records))

print("\nBuilding blocker...")
blocker = InvertedIndexBlocker()
blocker.index_candidates(candidate_records)

print("Blocker ready.")

rows = []
positive_count = 0
negative_count = 0
s1_with_positive = 0

for i, s1 in enumerate(s1_records, start=1):

    s1_id = s1["entity_id"]
    true_ids = gt.get(s1_id, set())

    positive_ids = [
        pid for pid in true_ids
        if pid in candidate_map
    ]

    if positive_ids:
        s1_with_positive += 1

        for pid in positive_ids:
            features = compute_pair_features(
                s1,
                candidate_map[pid]
            )

            row = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": pid,
                "label": 1,
            }

            row.update(features)
            rows.append(row)
            positive_count += 1

    candidate_ids = blocker.retrieve_candidates(
        s1,
        max_candidates=MAX_CANDIDATES
    )

    negative_ids = [
        cid for cid in candidate_ids
        if cid not in true_ids
    ]

    negative_ids = negative_ids[:NEGATIVES_PER_S1]

    for cid in negative_ids:

        features = compute_pair_features(
            s1,
            candidate_map[cid]
        )

        row = {
            "source1_entity_id": s1_id,
            "candidate_entity_id": cid,
            "label": 0,
        }

        row.update(features)
        rows.append(row)
        negative_count += 1

    if i % 1000 == 0:
        print(
            f"Processed {i}/{len(s1_records)} | "
            f"positives={positive_count} | "
            f"negatives={negative_count}"
        )

print("\nSaving training pairs...")

fieldnames = [
    "source1_entity_id",
    "candidate_entity_id",
    "label",
] + FEATURE_COLUMNS

with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n========== RESULT ==========")
print("S1 processed:", len(s1_records))
print("S1 with positive:", s1_with_positive)
print("Positive pairs:", positive_count)
print("Negative pairs:", negative_count)
print("Total pairs:", len(rows))
print("Output:", OUTPUT_PATH)
print("============================")
