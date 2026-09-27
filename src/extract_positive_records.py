import os
import csv
from evaluate_blocking_recall import load_ground_truth_map

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GT_PATH = os.path.join(PROJECT_ROOT, "train_ground_truth.tsv")
S2_PATH = os.path.join(PROJECT_ROOT, "train_source2.tsv")
S3_PATH = os.path.join(PROJECT_ROOT, "train_source3.tsv")

OUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

S2_OUT = os.path.join(OUT_DIR, "positive_source2.tsv")
S3_OUT = os.path.join(OUT_DIR, "positive_source3.tsv")

S1_LIMIT = 20000

print("Loading ground truth...")
gt = load_ground_truth_map(GT_PATH)

selected_s1 = set(list(gt.keys())[:S1_LIMIT])

positive_ids = set()
for s1_id in selected_s1:
    positive_ids.update(gt[s1_id])

print("Selected S1:", len(selected_s1))
print("Unique positive IDs:", len(positive_ids))

def extract(source_path, output_path):
    found = 0

    print(f"\nScanning: {os.path.basename(source_path)}")

    with open(source_path, "r", encoding="utf-8", newline="") as fin, \
         open(output_path, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(
            fout,
            fieldnames=reader.fieldnames,
            delimiter="\t"
        )

        writer.writeheader()

        for row in reader:
            if row["entity_id"] in positive_ids:
                writer.writerow(row)
                found += 1

                if found % 10000 == 0:
                    print("Found:", found)

    print("Extracted:", found)
    return found

s2_found = extract(S2_PATH, S2_OUT)
s3_found = extract(S3_PATH, S3_OUT)

print("\n========== RESULT ==========")
print("S2 positives:", s2_found)
print("S3 positives:", s3_found)
print("Total extracted:", s2_found + s3_found)
print("Expected:", len(positive_ids))
print("============================")
