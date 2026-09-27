import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from evaluate_blocking_recall import load_ground_truth_map
from blocking import InvertedIndexBlocker
from feature_engineering import compute_pair_features


# ============================================================
# PATHS
# ============================================================

GT_PATH = os.path.join(
    PROJECT_ROOT,
    "train_ground_truth.tsv"
)

S1_PATH = os.path.join(
    PROJECT_ROOT,
    "train_source1.tsv"
)

S2_PATH = os.path.join(
    PROJECT_ROOT,
    "train_source2.tsv"
)

S3_PATH = os.path.join(
    PROJECT_ROOT,
    "train_source3.tsv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "hard_negatives_v3.csv"
)


# ============================================================
# SETTINGS
# ============================================================

S1_LIMIT = 20_000
MAX_CANDIDATES = 100


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = load_ground_truth_map(
    GT_PATH
)

selected_s1_ids = set(
    list(gt.keys())[:S1_LIMIT]
)

print(
    "Selected development S1:",
    len(selected_s1_ids)
)


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("\nLoading S1...")

s1_records = []

with open(
    S1_PATH,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    for row in reader:

        if row["entity_id"] in selected_s1_ids:
            s1_records.append(row)

        if len(s1_records) == S1_LIMIT:
            break


print(
    "S1 records loaded:",
    len(s1_records)
)


# ============================================================
# LOAD SOURCE 2
# ============================================================

print("\nLoading Source 2...")

s2_records = []

with open(
    S2_PATH,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    s2_records = list(reader)


print(
    "Source 2 records:",
    len(s2_records)
)


# ============================================================
# LOAD SOURCE 3
# ============================================================

print("\nLoading Source 3...")

s3_records = []

with open(
    S3_PATH,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    s3_records = list(reader)


print(
    "Source 3 records:",
    len(s3_records)
)


# ============================================================
# COMBINE SOURCE 2 + SOURCE 3
# ============================================================

candidate_records = (
    s2_records +
    s3_records
)

candidate_map = {
    row["entity_id"]: row
    for row in candidate_records
}

print(
    "Total candidate records:",
    len(candidate_records)
)


# ============================================================
# BUILD BLOCKER
# ============================================================

print("\nBuilding blocker...")

blocker = InvertedIndexBlocker()

blocker.index_candidates(
    candidate_records
)

print("Blocker ready.")


# ============================================================
# INITIALIZE MINING
# ============================================================

hard_negatives = []

seen_pairs = set()

normal_negative_count = 0
hard_negative_count = 0

total_candidates = 0
true_match_count = 0

# Additional counters to understand
# which hard-negative rule is generating pairs.

number_address_trap_count = 0
name_address_hard_count = 0
conflict_hard_count = 0


print("\nMining hard negatives...")


# ============================================================
# PROCESS S1 RECORDS
# ============================================================

for i, s1 in enumerate(
    s1_records,
    start=1
):

    s1_id = s1["entity_id"]

    true_ids = gt.get(
        s1_id,
        set()
    )


    # --------------------------------------------------------
    # GET BLOCKED CANDIDATES
    # --------------------------------------------------------

    candidate_ids = blocker.retrieve_candidates(
        s1,
        max_candidates=MAX_CANDIDATES
    )

    total_candidates += len(
        candidate_ids
    )


    # --------------------------------------------------------
    # PROCESS EACH CANDIDATE
    # --------------------------------------------------------

    for cid in candidate_ids:


        # ----------------------------------------------------
        # SKIP TRUE MATCHES
        # ----------------------------------------------------

        if cid in true_ids:

            true_match_count += 1

            continue


        # ----------------------------------------------------
        # REMOVE DUPLICATE PAIRS
        # ----------------------------------------------------

        pair_key = (
            s1_id,
            cid
        )

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(
            pair_key
        )


        # ----------------------------------------------------
        # GET CANDIDATE RECORD
        # ----------------------------------------------------

        candidate = candidate_map[cid]


        # ----------------------------------------------------
        # COMPUTE ALL FEATURES
        # ----------------------------------------------------

        features = compute_pair_features(
            s1,
            candidate
        )


        # ====================================================
        # EXTRACT FEATURES
        # ====================================================

        jaro = features[
            "jaro_winkler"
        ]

        address_jaccard = features[
            "address_token_jaccard"
        ]

        address_sort = features[
            "address_token_sort_ratio"
        ]

        number_overlap = features[
            "number_overlap_ratio"
        ]

        exact_number = features[
            "exact_number_match"
        ]

        number_conflict = features[
            "number_conflict_flag"
        ]


        # ====================================================
        # HARD NEGATIVE TYPE 1
        #
        # NUMBER + ADDRESS TRAP
        #
        # Same/overlapping number
        # +
        # Highly similar address
        # +
        # Weak business-name similarity
        #
        # This targets the major FP pattern found in
        # the independent benchmark.
        # ====================================================

        is_number_address_trap = (

            (
                exact_number == 1
                or
                number_overlap >= 0.50
            )

            and

            (
                address_jaccard >= 0.70
                or
                address_sort >= 0.85
            )

            and

            jaro <= 0.70
        )


        # ====================================================
        # HARD NEGATIVE TYPE 2
        #
        # STRONG NAME + ADDRESS LOOK-ALIKE
        #
        # Preserve the original hard-negative rule.
        # ====================================================

        is_name_address_hard = (

            (
                jaro >= 0.90
                and
                address_jaccard >= 0.40
            )

            or

            (
                jaro >= 0.90
                and
                address_sort >= 0.70
            )
        )


        # ====================================================
        # HARD NEGATIVE TYPE 3
        #
        # NUMBER CONFLICT
        #
        # Preserve the original conflict-based rule.
        # ====================================================

        is_conflict_hard = (

            number_conflict == 1

            and

            jaro >= 0.85

            and

            address_jaccard >= 0.30
        )


        # ====================================================
        # FINAL HARD NEGATIVE DECISION
        # ====================================================

        is_hard = (

            is_number_address_trap

            or

            is_name_address_hard

            or

            is_conflict_hard
        )


        # ====================================================
        # SAVE HARD NEGATIVE
        # ====================================================

        if is_hard:

            row = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "label": 0,
            }

            row.update(
                features
            )

            hard_negatives.append(
                row
            )

            hard_negative_count += 1


            # Track rule type

            if is_number_address_trap:
                number_address_trap_count += 1

            if is_name_address_hard:
                name_address_hard_count += 1

            if is_conflict_hard:
                conflict_hard_count += 1


        else:

            normal_negative_count += 1


    # ========================================================
    # PROGRESS
    # ========================================================

    if i % 1000 == 0:

        print(
            f"Processed {i}/{len(s1_records)} | "
            f"candidates={total_candidates} | "
            f"hard_negatives={hard_negative_count}"
        )


# ============================================================
# SAVE HARD NEGATIVES
# ============================================================

print(
    "\nSaving hard negatives..."
)


fieldnames = [
    "source1_entity_id",
    "candidate_entity_id",
    "label",

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
    "token_count_ratio",
]


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        hard_negatives
    )


# ============================================================
# FINAL RESULT
# ============================================================

print(
    "\n========== RESULT =========="
)

print(
    "Development S1:",
    len(s1_records)
)

print(
    "Candidate records:",
    len(candidate_records)
)

print(
    "Candidate pairs inspected:",
    total_candidates
)

print(
    "True matches skipped:",
    true_match_count
)

print(
    "Normal negatives:",
    normal_negative_count
)

print(
    "Hard negatives:",
    hard_negative_count
)

print(
    "Number + address trap:",
    number_address_trap_count
)

print(
    "Name + address hard:",
    name_address_hard_count
)

print(
    "Number conflict hard:",
    conflict_hard_count
)

print(
    "Output:",
    OUTPUT_PATH
)

print(
    "============================"
)
