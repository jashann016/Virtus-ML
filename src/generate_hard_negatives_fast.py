import os
import csv
import heapq
from collections import defaultdict

from src.text_preprocessing import (
    normalize_business_name,
    normalize_address,
    extract_numbers,
)

from src.blocking import (
    extract_address_keys,
    get_char_trigrams,
    STOP_WORDS,
    COMMON_CITY_TOKENS,
)

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "."

S1_FILE = "train_source1.tsv"
S2_FILE = "train_source2.tsv"
S3_FILE = "train_source3.tsv"
GT_FILE = "train_ground_truth.tsv"

OUTPUT_FILE = "output/hard_negatives_v3.tsv"

# Development S1 only.
# Independent benchmark starts later and remains untouched.
S1_START = 0
S1_COUNT = 20000

# Keep only strongest blocking candidates per S1.
MAX_CANDIDATES_PER_S1 = 80

# Final hard negatives per S1.
MAX_HARD_NEGATIVES_PER_S1 = 25


# ============================================================
# TSV HELPERS
# ============================================================

def stream_tsv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield {
                "entity_id": row["entity_id"].strip(),
                "business_name": row["business_name"].strip(),
                "business_address": row["business_address"].strip(),
                "country": row["country"].strip(),
            }


def load_s1_subset():
    records = []

    for idx, row in enumerate(stream_tsv(S1_FILE)):
        if idx < S1_START:
            continue

        if idx >= S1_START + S1_COUNT:
            break

        records.append(row)

    return records


def load_ground_truth():
    gt = {}

    with open(GT_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            raw = row["matched_entity_ids"].strip()

            if raw:
                ids = {
                    x.strip()
                    for x in raw.split(",")
                    if x.strip()
                }
            else:
                ids = set()

            gt[s1_id] = ids

    return gt


# ============================================================
# LIGHTWEIGHT S1 INDEX
# ============================================================

def build_s1_index(s1_records):
    """
    Reverse index built ONLY for 2,000 S1 records.
    Source 2/3 are streamed and never loaded into RAM.
    """

    index = defaultdict(set)

    for s1 in s1_records:
        sid = s1["entity_id"]
        country = s1["country"]

        name = normalize_business_name(s1["business_name"])
        addr = normalize_address(s1["business_address"])

        # Exact normalized name
        if name:
            index[(country, "name_exact", name)].add(sid)

        words = [
            w for w in name.split()
            if len(w) >= 3
            and w not in STOP_WORDS
            and w not in COMMON_CITY_TOKENS
        ]

        # First word
        if words:
            index[(country, "name_first", words[0])].add(sid)

        # Important name tokens
        for w in words[:5]:
            index[(country, "name_token", w)].add(sid)

            # Character trigrams for typo / spelling variation
            if len(w) >= 5:
                for tri in get_char_trigrams(w):
                    index[(country, "name_tri", tri)].add(sid)

        # Address keys
        for key in extract_address_keys(
            addr,
            raw_address=s1["business_address"]
        ):
            index[(country, "addr", key)].add(sid)

    return index


# ============================================================
# CANDIDATE KEY EXTRACTION
# ============================================================

def get_candidate_keys(row):
    country = row["country"]

    name = normalize_business_name(row["business_name"])
    addr = normalize_address(row["business_address"])

    keys = []

    if name:
        keys.append((country, "name_exact", name))

    words = [
        w for w in name.split()
        if len(w) >= 3
        and w not in STOP_WORDS
        and w not in COMMON_CITY_TOKENS
    ]

    if words:
        keys.append((country, "name_first", words[0]))

    for w in words[:5]:
        keys.append((country, "name_token", w))

        if len(w) >= 5:
            for tri in get_char_trigrams(w):
                keys.append((country, "name_tri", tri))

    for key in extract_address_keys(
        addr,
        raw_address=row["business_address"]
    ):
        keys.append((country, "addr", key))

    return keys


# ============================================================
# BLOCKING SCORE
# ============================================================

def blocking_score(s1, candidate):
    score = 0

    s1_name = normalize_business_name(s1["business_name"])
    c_name = normalize_business_name(candidate["business_name"])

    s1_addr = normalize_address(s1["business_address"])
    c_addr = normalize_address(candidate["business_address"])

    # Exact name
    if s1_name and s1_name == c_name:
        score += 50

    # Shared name tokens
    s1_words = set(
        w for w in s1_name.split()
        if len(w) >= 3 and w not in STOP_WORDS
    )

    c_words = set(
        w for w in c_name.split()
        if len(w) >= 3 and w not in STOP_WORDS
    )

    shared_name = len(s1_words & c_words)

    score += min(shared_name * 8, 32)

    # Shared address keys
    s1_keys = set(
        extract_address_keys(
            s1_addr,
            raw_address=s1["business_address"]
        )
    )

    c_keys = set(
        extract_address_keys(
            c_addr,
            raw_address=candidate["business_address"]
        )
    )

    shared_addr = len(s1_keys & c_keys)

    score += min(shared_addr * 10, 40)

    # Number overlap
    s1_numbers = set(extract_numbers(s1_addr))
    c_numbers = set(extract_numbers(c_addr))

    if s1_numbers and c_numbers:
        overlap = len(s1_numbers & c_numbers)

        if overlap:
            score += 20

        # Important: conflicting numbers are still valuable
        # hard negatives when name/address are highly similar.
        if s1_numbers.isdisjoint(c_numbers):
            score += 5

    return score


# ============================================================
# STREAMING MINER
# ============================================================

def mine_source(path, s1_records, s1_by_id, s1_index, gt):
    """
    Stream one entire source file.

    Only top MAX_CANDIDATES_PER_S1 candidates are retained.
    """

    heaps = defaultdict(list)

    processed = 0
    matched_keys = 0

    for row in stream_tsv(path):
        processed += 1

        if processed % 500000 == 0:
            print(
                f"Processed {processed:,} records from "
                f"{os.path.basename(path)} | "
                f"RAM-safe streaming active"
            )

        candidate_id = row["entity_id"]

        candidate_s1_ids = set()

        for key in get_candidate_keys(row):
            candidate_s1_ids.update(
                s1_index.get(key, ())
            )

        if not candidate_s1_ids:
            continue

        matched_keys += 1

        for sid in candidate_s1_ids:

            # Never use true positive as negative.
            if candidate_id in gt.get(sid, set()):
                continue

            s1 = s1_by_id[sid]

            score = blocking_score(s1, row)

            item = (score, candidate_id)

            heap = heaps[sid]

            if len(heap) < MAX_CANDIDATES_PER_S1:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    print(
        f"Finished {os.path.basename(path)}: "
        f"{processed:,} records, "
        f"{matched_keys:,} records shared at least one blocking key"
    )

    return heaps


# ============================================================
# MERGE + HARD NEGATIVE SELECTION
# ============================================================

def merge_heaps(target, source):
    for sid, items in source.items():

        heap = target[sid]

        for item in items:

            if len(heap) < MAX_CANDIDATES_PER_S1:
                heapq.heappush(heap, item)

            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)


def main():

    print("=" * 70)
    print("MEMORY-SAFE HARD NEGATIVE MINING V3")
    print("=" * 70)

    print("\nLoading ground truth...")
    gt = load_ground_truth()

    print("Ground truth loaded:", len(gt))

    print(f"\nLoading development S1 subset: {S1_COUNT:,}")

    s1_records = load_s1_subset()

    print("S1 loaded:", len(s1_records))

    s1_by_id = {
        row["entity_id"]: row
        for row in s1_records
    }

    print("\nBuilding lightweight S1 index...")

    s1_index = build_s1_index(s1_records)

    print("Index keys:", len(s1_index))

    print("\nStreaming Source 2...")

    heaps = mine_source(
        S2_FILE,
        s1_records,
        s1_by_id,
        s1_index,
        gt
    )

    print("\nStreaming Source 3...")

    heaps3 = mine_source(
        S3_FILE,
        s1_records,
        s1_by_id,
        s1_index,
        gt
    )

    print("\nMerging Source 2 + Source 3 candidates...")

    merge_heaps(heaps, heaps3)

    os.makedirs("output", exist_ok=True)

    total_written = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f, delimiter="\t")

        writer.writerow([
            "source1_entity_id",
            "candidate_entity_id",
            "blocking_score"
        ])

        for sid in s1_by_id:

            candidates = sorted(
                heaps[sid],
                reverse=True
            )

            # Keep strongest hard candidates.
            candidates = candidates[
                :MAX_HARD_NEGATIVES_PER_S1
            ]

            for score, cid in candidates:

                writer.writerow([
                    sid,
                    cid,
                    score
                ])

                total_written += 1

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("Development S1:", len(s1_records))
    print("Hard negatives:", f"{total_written:,}")
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
