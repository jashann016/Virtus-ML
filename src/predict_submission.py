"""
VIRTUS-ML: End-to-End Submission Prediction Engine
Uses trained LightGBM model to score candidate pairs and outputs
an official submission TSV file meeting 100% of competition rules.
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import subprocess
from typing import Dict, List, Set

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.blocking import InvertedIndexBlocker, load_tsv_records
from src.features import extract_pairwise_features, FEATURE_NAMES
from src.text_preprocessing import normalize_business_name, normalize_address


def generate_submission(
    model_path: str = "models/lightgbm_matcher.pkl",
    meta_path: str = "models/lightgbm_metadata.json",
    output_path: str = "output/submission.csv",
    test_limit: int = None,
    max_cands_per_entity: int = 35
):
    print("=" * 70)
    print("VIRTUS-ML: GENERATING PREDICTIONS FOR SUBMISSION")
    print("=" * 70)

    start_time = time.time()

    # 1. Load Model & Metadata
    print(f"[1/5] Loading trained LightGBM model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    best_threshold = meta['best_threshold']
    print(f"  • Optimal Decision Threshold: {best_threshold:.2f}")
    print(f"  • Trained Validation F0.5: {meta.get('validation_macro_f05', 0):.4f}")

    # 2. Load Test Records
    test_dir = os.path.join(PROJECT_ROOT, "dataset", "test")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"[2/5] Loading test entities...")
    s1_records = {}
    with open(s1_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for i, line in enumerate(f):
            if test_limit and i >= test_limit:
                break
            parts = line.strip('\n').split('\t')
            if len(parts) >= 4:
                s1_records[parts[0].strip()] = {
                    'entity_id': parts[0].strip(),
                    'business_name': parts[1].strip(),
                    'business_address': parts[2].strip(),
                    'country': parts[3].strip()
                }

    print(f"  • Loaded {len(s1_records):,} Source 1 test entities.")

    # 3. Stream & Index Candidates from S2 & S3
    print(f"[3/5] Streaming and indexing Source 2 & Source 3 candidates...")
    blocker = InvertedIndexBlocker()
    cand_records = {}
    cand_norm_name = {}
    cand_norm_addr = {}

    for path in [s2_path, s3_path]:
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.strip('\n').split('\t')
                if len(parts) >= 4:
                    cid = parts[0].strip()
                    rec = {
                        'entity_id': cid,
                        'business_name': parts[1].strip(),
                        'business_address': parts[2].strip(),
                        'country': parts[3].strip()
                    }
                    cand_records[cid] = rec
                    cand_norm_name[cid] = normalize_business_name(rec['business_name'])
                    cand_norm_addr[cid] = normalize_address(rec['business_address'])

    print(f"  • Candidate pool loaded: {len(cand_records):,} records. Building index...")
    blocker.index_candidates(list(cand_records.values()))

    # 4. Predict Matches with LightGBM
    print(f"[4/5] Scoring candidate pairs with LightGBM...")
    predictions = {}
    total_matches_predicted = 0
    singletons_predicted = 0

    s1_ids = list(s1_records.keys())
    batch_size = 5000

    for b_start in range(0, len(s1_ids), batch_size):
        b_end = min(b_start + batch_size, len(s1_ids))
        batch_ids = s1_ids[b_start:b_end]

        batch_pairs = []
        for s1_id in batch_ids:
            rec_a = s1_records[s1_id]
            norm_name_a = normalize_business_name(rec_a['business_name'])
            norm_addr_a = normalize_address(rec_a['business_address'])

            retrieved = blocker.retrieve_candidates(rec_a, max_candidates=max_cands_per_entity)
            for rank, cand_id in enumerate(retrieved, start=1):
                if cand_id in cand_records:
                    batch_pairs.append((
                        s1_id,
                        cand_id,
                        rec_a,
                        cand_records[cand_id],
                        norm_name_a,
                        cand_norm_name[cand_id],
                        norm_addr_a,
                        cand_norm_addr[cand_id],
                        rank
                    ))

        if not batch_pairs:
            for s1_id in batch_ids:
                predictions[s1_id] = []
            continue

        # Extract features for batch
        X_batch = np.zeros((len(batch_pairs), len(FEATURE_NAMES)), dtype=np.float32)
        for i, p in enumerate(batch_pairs):
            feat_dict = extract_pairwise_features(
                p[2], p[3], rank=p[8],
                norm_name_a=p[4], norm_name_b=p[5],
                norm_addr_a=p[6], norm_addr_b=p[7]
            )
            X_batch[i] = [feat_dict[k] for k in FEATURE_NAMES]

        probs = model.predict_proba(X_batch)[:, 1]

        # Group by S1 entity
        curr_s1_matches = {s1_id: [] for s1_id in batch_ids}
        for i, p in enumerate(batch_pairs):
            s1_id = p[0]
            cand_id = p[1]
            prob = probs[i]
            if prob >= best_threshold:
                curr_s1_matches[s1_id].append((cand_id, prob))

        for s1_id, matches in curr_s1_matches.items():
            if matches:
                # Sort by confidence descending
                matches.sort(key=lambda x: x[1], reverse=True)
                predictions[s1_id] = [m[0] for m in matches]
                total_matches_predicted += len(matches)
            else:
                predictions[s1_id] = []
                singletons_predicted += 1

        print(f"  -> Processed {b_end:,}/{len(s1_ids):,} entities...")

    # 5. Write to Submission TSV format (all 1.73M test entities must be present)
    print(f"[5/5] Writing output to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load all test S1 IDs to guarantee 100% row presence for competition rules
    all_s1_ids = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.split('\t')
            if parts:
                all_s1_ids.append(parts[0].strip())

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = predictions.get(s1_id, [])
            match_str = ",".join(matches)
            f.write(f"{s1_id}\t{match_str}\n")

    elapsed = time.time() - start_time
    print("=" * 70)
    print("SUBMISSION PREDICTION SUMMARY:")
    print(f"• Total Entities Written   : {len(all_s1_ids):,}")
    print(f"• Entities Scored by Model : {len(s1_records):,}")
    print(f"• Total Predicted Matches  : {total_matches_predicted:,}")
    print(f"• Output File              : {output_path}")
    print(f"• Total Generation Time    : {elapsed:.2f} seconds ({elapsed/60:.2f} mins)")
    print("=" * 70)

    # 6. Validate with official competition validator
    val_script = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
    if os.path.exists(val_script):
        print("\nRunning Official Submission Validator...")
        cmd = [sys.executable, val_script, "--matching", output_path, "--test-dir", test_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            print("✅ OFFICIAL VALIDATOR RESULT: PASS (100% Safe to submit)")
        else:
            print("❌ VALIDATOR ISSUES DETECTED:\n", res.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LightGBM predictions for submission.")
    parser.add_argument("--model", default="models/lightgbm_matcher.pkl", help="Path to trained model")
    parser.add_argument("--meta", default="models/lightgbm_metadata.json", help="Path to metadata json")
    parser.add_argument("--output", default="output/submission.csv", help="Path to output submission file")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick testing")
    args = parser.parse_args()

    generate_submission(
        model_path=args.model,
        meta_path=args.meta,
        output_path=args.output,
        test_limit=args.limit
    )
