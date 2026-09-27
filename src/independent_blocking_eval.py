import os
import sys
import csv
import time
import gc
import heapq
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.blocking import InvertedIndexBlocker, load_tsv_records


# ============================================================
# CONFIGURATION
# ============================================================

S1_START = 20000
S1_COUNT = 1000

CANDIDATE_CHUNK_SIZE = 100000
MAX_CANDIDATES_PER_CHUNK = 35

S1_PATH = os.path.join(PROJECT_ROOT, "train_source1.tsv")
S2_PATH = os.path.join(PROJECT_ROOT, "train_source2.tsv")
S3_PATH = os.path.join(PROJECT_ROOT, "train_source3.tsv")
GT_PATH = os.path.join(PROJECT_ROOT, "train_ground_truth.tsv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

RECALL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "independent_blocking_recall.csv"
)

CANDIDATE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "independent_candidate_pairs.csv"
)


# ============================================================
# LOAD S1 SLICE
# ============================================================

def load_s1_slice(path, start, count):
    records = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for idx, row in enumerate(reader):
            if idx < start:
                continue

            records.append({
                "entity_id": row["entity_id"],
                "business_name": row["business_name"],
                "business_address": row["business_address"],
                "country": row["country"],
            })

            if len(records) >= count:
                break

    return records


# ============================================================
# LOAD GROUND TRUTH ONLY FOR FINAL EVALUATION
# ============================================================

def load_ground_truth(path, s1_ids):
    gt = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = row["source1_entity_id"]

            if s1_id not in s1_ids:
                continue

            raw = row["matched_entity_ids"].strip()

            if raw:
                gt[s1_id] = {
                    x.strip()
                    for x in raw.split(",")
                    if x.strip()
                }
            else:
                gt[s1_id] = set()

            if len(gt) == len(s1_ids):
                break

    for s1_id in s1_ids:
        gt.setdefault(s1_id, set())

    return gt


# ============================================================
# CHUNK LOADER
# ============================================================

def iter_chunks(path, chunk_size):
    with open(path, "r", encoding="utf-8", newline="") as f:

        header = f.readline()

        records = []

        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 4:
                continue

            records.append({
                "entity_id": parts[0],
                "business_name": parts[1],
                "business_address": parts[2],
                "country": parts[3],
            })

            if len(records) >= chunk_size:
                yield records
                records = []

        if records:
            yield records


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("INDEPENDENT BLOCKING + CANDIDATE UNIVERSE EVALUATION")
    print("=" * 70)

    print()
    print(f"Held-out S1 start : {S1_START:,}")
    print(f"Held-out S1 count : {S1_COUNT:,}")
    print(f"Candidate chunk   : {CANDIDATE_CHUNK_SIZE:,}")
    print(f"Top candidates/chunk : {MAX_CANDIDATES_PER_CHUNK}")

    # --------------------------------------------------------
    # STEP 1: HOLD-OUT S1
    # --------------------------------------------------------

    print("\n[1/5] Loading held-out Source 1 records...")

    s1_records = load_s1_slice(
        S1_PATH,
        S1_START,
        S1_COUNT
    )

    if not s1_records:
        raise RuntimeError("No S1 records loaded.")

    s1_ids = {
        r["entity_id"]
        for r in s1_records
    }

    print(f"Held-out S1 loaded: {len(s1_records):,}")

    # --------------------------------------------------------
    # STEP 2: INITIALIZE GLOBAL CANDIDATE STORAGE
    # --------------------------------------------------------

    # We retain the union of top candidates returned from every
    # independent candidate chunk.
    candidate_map = {
        s1_id: {}
        for s1_id in s1_ids
    }

    total_candidate_records = 0
    chunk_number = 0

    start_time = time.time()

    # --------------------------------------------------------
    # STEP 3: STREAM SOURCE 2
    # --------------------------------------------------------

    print("\n[2/5] Streaming Source 2...")

    for chunk in iter_chunks(
        S2_PATH,
        CANDIDATE_CHUNK_SIZE
    ):

        chunk_number += 1
        total_candidate_records += len(chunk)

        blocker = InvertedIndexBlocker()
        blocker.index_candidates(chunk)

        for s1 in s1_records:

            retrieved = blocker.retrieve_candidates(
                s1,
                max_candidates=MAX_CANDIDATES_PER_CHUNK
            )

            store = candidate_map[s1["entity_id"]]

            for candidate_id in retrieved:
                store[candidate_id] = True

        del blocker
        del chunk
        gc.collect()

        print(
            f"  S2 chunk {chunk_number:03d} processed "
            f"| records={total_candidate_records:,}"
        )

    # --------------------------------------------------------
    # STEP 4: STREAM SOURCE 3
    # --------------------------------------------------------

    print("\n[3/5] Streaming Source 3...")

    source3_records = 0
    chunk_number = 0

    for chunk in iter_chunks(
        S3_PATH,
        CANDIDATE_CHUNK_SIZE
    ):

        chunk_number += 1
        source3_records += len(chunk)

        blocker = InvertedIndexBlocker()
        blocker.index_candidates(chunk)

        for s1 in s1_records:

            retrieved = blocker.retrieve_candidates(
                s1,
                max_candidates=MAX_CANDIDATES_PER_CHUNK
            )

            store = candidate_map[s1["entity_id"]]

            for candidate_id in retrieved:
                store[candidate_id] = True

        del blocker
        del chunk
        gc.collect()

        print(
            f"  S3 chunk {chunk_number:03d} processed "
            f"| records={source3_records:,}"
        )

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # STEP 5: SAVE CANDIDATE UNIVERSE
    # --------------------------------------------------------

    print("\n[4/5] Saving independent candidate universe...")

    total_pairs = 0

    with open(
        CANDIDATE_OUTPUT,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "source1_entity_id",
            "candidate_entity_id"
        ])

        for s1_id, candidates in candidate_map.items():

            for candidate_id in candidates:

                writer.writerow([
                    s1_id,
                    candidate_id
                ])

                total_pairs += 1

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    print("\n[5/5] Evaluating recall against ground truth...")

    gt = load_ground_truth(
        GT_PATH,
        s1_ids
    )

    total_true_links = 0
    captured_true_links = 0

    entities_with_truth = 0
    entities_with_capture = 0

    candidate_counts = []

    for s1_id in s1_ids:

        true_targets = gt.get(
            s1_id,
            set()
        )

        candidates = set(
            candidate_map[s1_id].keys()
        )

        candidate_counts.append(
            len(candidates)
        )

        if true_targets:

            entities_with_truth += 1

            total_true_links += len(
                true_targets
            )

            captured = (
                true_targets
                .intersection(candidates)
            )

            captured_true_links += len(
                captured
            )

            if captured:
                entities_with_capture += 1

    link_recall = (
        captured_true_links / total_true_links
        if total_true_links
        else 0.0
    )

    entity_recall = (
        entities_with_capture / entities_with_truth
        if entities_with_truth
        else 0.0
    )

    avg_candidates = (
        sum(candidate_counts) /
        len(candidate_counts)
        if candidate_counts
        else 0.0
    )

    print("\n" + "=" * 70)
    print("INDEPENDENT BLOCKING RESULTS")
    print("=" * 70)

    print(
        f"Held-out S1 entities          : "
        f"{len(s1_records):,}"
    )

    print(
        f"Entities with ≥1 true match   : "
        f"{entities_with_truth:,}"
    )

    print(
        f"Total true match links         : "
        f"{total_true_links:,}"
    )

    print(
        f"Captured true links            : "
        f"{captured_true_links:,}"
    )

    print(
        f"LINK-LEVEL RECALL              : "
        f"{link_recall * 100:.2f}%"
    )

    print(
        f"ENTITY-LEVEL RECALL            : "
        f"{entity_recall * 100:.2f}%"
    )

    print(
        f"Independent candidate pairs   : "
        f"{total_pairs:,}"
    )

    print(
        f"Average candidates / S1       : "
        f"{avg_candidates:.1f}"
    )

    print(
        f"Candidate processing time     : "
        f"{elapsed / 60:.2f} minutes"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    with open(
        RECALL_OUTPUT,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "heldout_s1",
            "entities_with_truth",
            "true_links",
            "captured_links",
            "link_recall",
            "entity_recall",
            "candidate_pairs",
            "avg_candidates_per_s1",
            "processing_minutes"
        ])

        writer.writerow([
            len(s1_records),
            entities_with_truth,
            total_true_links,
            captured_true_links,
            link_recall,
            entity_recall,
            total_pairs,
            avg_candidates,
            elapsed / 60
        ])

    print("\nSaved:")
    print(RECALL_OUTPUT)
    print(CANDIDATE_OUTPUT)


if __name__ == "__main__":
    main()
