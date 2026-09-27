import csv
from collections import Counter

from evaluate_blocking_recall import load_ground_truth_map
from blocking import InvertedIndexBlocker
from feature_engineering import compute_pair_features


GT_PATH = "train_ground_truth.tsv"
S1_PATH = "train_source1.tsv"

POSITIVE_FILES = [
    "output/positive_source2.tsv",
    "output/positive_source3.tsv",
]

S1_LIMIT = 20_000
MAX_CANDIDATES = 100


print("Loading ground truth...")
gt = load_ground_truth_map(GT_PATH)

selected_s1_ids = set(list(gt.keys())[:S1_LIMIT])

print("Selected S1:", len(selected_s1_ids))


print("\nLoading S1...")
s1_records = []

with open(S1_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        if row["entity_id"] in selected_s1_ids:
            s1_records.append(row)

        if len(s1_records) == S1_LIMIT:
            break

print("S1 records:", len(s1_records))


print("\nLoading candidate records...")
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


print("\nBuilding blocker...")
blocker = InvertedIndexBlocker()
blocker.index_candidates(candidate_records)

print("Blocker ready.")


counts = Counter()
total_negative_candidates = 0
examples = []


print("\nScanning development candidates...")

for i, s1 in enumerate(s1_records, start=1):

    s1_id = s1["entity_id"]
    true_ids = gt.get(s1_id, set())

    candidate_ids = blocker.retrieve_candidates(
        s1,
        max_candidates=MAX_CANDIDATES
    )

    negative_ids = [
        cid for cid in candidate_ids
        if cid not in true_ids
    ]

    for cid in negative_ids:

        features = compute_pair_features(
            s1,
            candidate_map[cid]
        )

        total_negative_candidates += 1

        conflict = features["number_conflict_flag"] == 1
        high_jaccard = features["address_token_jaccard"] >= 0.5
        high_name = features["jaro_winkler"] >= 0.9

        if conflict:
            counts["number_conflict"] += 1

        if conflict and high_jaccard:
            counts["conflict_high_jaccard"] += 1

        if conflict and high_name:
            counts["conflict_high_name"] += 1

        if conflict and high_jaccard and high_name:
            counts["extreme"] += 1

            if len(examples) < 10:
                examples.append({
                    "s1": s1_id,
                    "candidate": cid,
                    "jaro_winkler": round(
                        features["jaro_winkler"], 4
                    ),
                    "address_jaccard": round(
                        features["address_token_jaccard"], 4
                    ),
                    "address_sort": round(
                        features["address_token_sort_ratio"], 4
                    ),
                    "number_overlap": round(
                        features["number_overlap_ratio"], 4
                    ),
                })

    if i % 2000 == 0:
        print(f"Processed {i}/{len(s1_records)}")


print("\n========== RESULT ==========")

print("Development S1:", len(s1_records))
print(
    "Negative candidates inspected:",
    total_negative_candidates
)

print(
    "Number conflict:",
    counts["number_conflict"]
)

print(
    "Conflict + address Jaccard >= 0.5:",
    counts["conflict_high_jaccard"]
)

print(
    "Conflict + name Jaro-Winkler >= 0.9:",
    counts["conflict_high_name"]
)

print(
    "Extreme hard negatives:",
    counts["extreme"]
)

print("\nExamples:")

for example in examples:
    print(example)

print("============================")
