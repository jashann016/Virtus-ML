import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics import compute_macro_f05


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "lightgbm_baseline.pkl"
)

TRAINING_PAIRS_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "lightgbm_training_pairs.csv"
)

GT_PATH = os.path.join(
    PROJECT_ROOT,
    "train_ground_truth.tsv"
)


FEATURE_COLUMNS = [
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


def load_ground_truth(path):
    gt = {}

    with open(path, "r", encoding="utf-8") as f:
        f.readline()

        for line in f:
            parts = line.rstrip("\n").split("\t")

            if not parts:
                continue

            s1_id = parts[0].strip()

            if len(parts) > 1 and parts[1].strip():
                matches = {
                    x.strip()
                    for x in parts[1].split(",")
                    if x.strip()
                }
            else:
                matches = set()

            gt[s1_id] = matches

    return gt


def main():

    print("=" * 70)
    print("HELD-OUT S1 MACRO F0.5 THRESHOLD SWEEP")
    print("=" * 70)

    print("Loading model...")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Loading candidate pairs...")

    df = pd.read_csv(
        TRAINING_PAIRS_PATH
    )

    print(f"Candidate pairs: {len(df):,}")

    # Recreate exactly the same S1-level validation split
    all_s1_ids = df["source1_entity_id"].unique()

    train_s1_ids, valid_s1_ids = train_test_split(
        all_s1_ids,
        test_size=0.2,
        random_state=42
    )

    valid_s1_ids = set(valid_s1_ids)

    print(f"Validation S1 entities: {len(valid_s1_ids):,}")

    # Keep only candidate pairs belonging to held-out S1 entities
    valid_df = df[
        df["source1_entity_id"].isin(valid_s1_ids)
    ].copy()

    print(
        f"Validation candidate pairs: "
        f"{len(valid_df):,}"
    )

    print("Calculating probabilities...")

    X_valid = valid_df[FEATURE_COLUMNS]

    valid_df["probability"] = (
        model.predict_proba(X_valid)[:, 1]
    )

    print("Loading ground truth...")

    gt = load_ground_truth(GT_PATH)

    # Every held-out S1 must be evaluated,
    # including S1 entities with zero candidate rows.
    gt_eval = {
        s1_id: gt.get(s1_id, set())
        for s1_id in valid_s1_ids
    }

    print(
        f"Ground-truth S1 entities evaluated: "
        f"{len(gt_eval):,}"
    )

    results = []

    thresholds = [
        0.35,
        0.50,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.88,
        0.90
    ]

    for threshold in thresholds:

        predictions = {
            s1_id: set()
            for s1_id in valid_s1_ids
        }

        for s1_id, group in valid_df.groupby(
            "source1_entity_id"
        ):

            matched = group.loc[
                group["probability"] >= threshold,
                "candidate_entity_id"
            ]

            predictions[s1_id] = set(
                matched.tolist()
            )

        score = compute_macro_f05(
            gt_eval,
            predictions
        )

        predicted_links = sum(
            len(v)
            for v in predictions.values()
        )

        nonempty_predictions = sum(
            1
            for v in predictions.values()
            if v
        )

        results.append({
            "threshold": threshold,
            "macro_f05": score,
            "predicted_links": predicted_links,
            "s1_with_predictions": nonempty_predictions
        })

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("HELD-OUT THRESHOLD RESULTS")
    print("=" * 70)

    print(
        result_df.to_string(
            index=False,
            formatters={
                "macro_f05": "{:.4f}".format
            }
        )
    )

    output_path = os.path.join(
        PROJECT_ROOT,
        "output",
        "heldout_threshold_results.csv"
    )

    result_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()