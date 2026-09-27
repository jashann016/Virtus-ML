from src.blocking import load_tsv_records
from src.evaluate_blocking_recall import load_ground_truth_map
from src.feature_engineering import compute_pair_features, FEATURE_COLUMNS

s1 = load_tsv_records("train_source1.tsv", 5000)
s1_ids = {r["entity_id"] for r in s1}
gt = load_ground_truth_map("train_ground_truth.tsv", s1_ids)

s2 = load_tsv_records("train_source2.tsv", 50000)
s3 = load_tsv_records("train_source3.tsv", 50000)

pool = s2 + s3
pool_map = {r["entity_id"]: r for r in pool}

found = False

for s1_record in s1:
    true_targets = gt.get(s1_record["entity_id"], set())

    for target_id in true_targets:
        if target_id in pool_map:
            target = pool_map[target_id]

            features = compute_pair_features(s1_record, target)

            print("S1:", s1_record["entity_id"])
            print("Target:", target_id)
            print("Name 1:", s1_record["business_name"])
            print("Name 2:", target["business_name"])
            print("Feature count:", len(features))
            print("Expected:", len(FEATURE_COLUMNS))
            print("Features:", features)

            found = True
            break

    if found:
        break

if not found:
    print("No positive pair found in the loaded 100,000 candidate records.")
