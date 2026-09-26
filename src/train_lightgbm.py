"""
VIRTUS-ML: LightGBM Training & Macro-F0.5 Optimization Pipeline
Trains a tabular gradient-boosted decision tree on candidate pairs
and sweeps probability thresholds to maximize the competition Macro-F0.5 score.
"""

import os
import sys
import json
import time
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.blocking import InvertedIndexBlocker
from src.features import extract_pairwise_features, FEATURE_NAMES
from src.text_preprocessing import normalize_business_name, normalize_address
from src.metrics import compute_macro_f05, compute_entity_f_beta


def load_training_dataset(
    s1_limit: int = 25000,
    cand_limit: int = 200000,
    max_cands_per_entity: int = 35
):
    """Loads a slice of training entities and ground truth, blocks candidates, and creates labeled pairs."""
    data_dir = os.path.join(PROJECT_ROOT, "dataset", "train")
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")

    print("=" * 70)
    print(f"LOADING DATA FOR LIGHTGBM TRAINING (S1 Limit: {s1_limit:,})")
    print("=" * 70)

    # 1. Load S1 entities
    print(f"[1/4] Loading {s1_limit:,} Source 1 reference entities...")
    s1_records = {}
    with open(s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            if len(s1_records) >= s1_limit:
                break
            parts = line.strip('\n').split('\t')
            if len(parts) >= 4:
                s1_records[parts[0].strip()] = {
                    'entity_id': parts[0].strip(),
                    'business_name': parts[1].strip(),
                    'business_address': parts[2].strip(),
                    'country': parts[3].strip()
                }

    # 2. Load Ground Truth
    print("[2/4] Loading Ground Truth match labels...")
    gt_map = {}
    needed_targets = set()
    with open(gt_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            s1_id = parts[0].strip()
            if s1_id in s1_records:
                matches = set()
                if len(parts) > 1 and parts[1].strip():
                    matches = {m.strip() for m in parts[1].split(',') if m.strip()}
                gt_map[s1_id] = matches
                needed_targets.update(matches)

    # 3. Load Candidate Records (True matches + pool)
    print(f"[3/4] Streaming S2 & S3 candidates (capturing true matches + pool up to {cand_limit:,})...")
    cand_records = {}
    with open(s2_path, 'r', encoding='utf-8') as f2, open(s3_path, 'r', encoding='utf-8') as f3:
        for f in [f2, f3]:
            f.readline()
            for line in f:
                parts = line.strip('\n').split('\t')
                if len(parts) >= 4:
                    cid = parts[0].strip()
                    if cid in needed_targets or len(cand_records) < cand_limit:
                        cand_records[cid] = {
                            'entity_id': cid,
                            'business_name': parts[1].strip(),
                            'business_address': parts[2].strip(),
                            'country': parts[3].strip()
                        }

    # 4. Block and build labeled pairs
    print("[4/4] Indexing candidates and generating labeled pairs...")
    blocker = InvertedIndexBlocker()
    blocker.index_candidates(list(cand_records.values()))

    # Precompute normalized names and addresses for speed
    cand_norm_name = {}
    cand_norm_addr = {}
    for cid, cand in cand_records.items():
        cand_norm_name[cid] = normalize_business_name(cand['business_name'])
        cand_norm_addr[cid] = normalize_address(cand['business_address'])

    # Assemble pairs
    pairs_data = []
    entity_ids = list(s1_records.keys())
    
    # Shuffle entities
    random.seed(42)
    random.shuffle(entity_ids)

    pos_count = 0
    neg_count = 0

    for s1_id in entity_ids:
        rec_a = s1_records[s1_id]
        norm_name_a = normalize_business_name(rec_a['business_name'])
        norm_addr_a = normalize_address(rec_a['business_address'])
        true_set = gt_map.get(s1_id, set())

        retrieved = blocker.retrieve_candidates(rec_a, max_candidates=max_cands_per_entity)
        
        # Ensure true matches in pool are included so model sees them
        for target_id in true_set:
            if target_id in cand_records and target_id not in retrieved:
                retrieved.append(target_id)

        for rank, cand_id in enumerate(retrieved, start=1):
            if cand_id not in cand_records:
                continue
            rec_b = cand_records[cand_id]
            label = 1 if cand_id in true_set else 0

            if label == 1:
                pos_count += 1
            else:
                neg_count += 1

            pairs_data.append({
                's1_id': s1_id,
                'cand_id': cand_id,
                'rec_a': rec_a,
                'rec_b': rec_b,
                'norm_name_a': norm_name_a,
                'norm_name_b': cand_norm_name[cand_id],
                'norm_addr_a': norm_addr_a,
                'norm_addr_b': cand_norm_addr[cand_id],
                'rank': rank,
                'label': label
            })

    print(f"Total labeled pairs generated: {len(pairs_data):,}")
    print(f"  • True Matches (Label 1): {pos_count:,}")
    print(f"  • Hard Look-Alikes (Label 0): {neg_count:,}")
    return s1_records, gt_map, pairs_data


def train_and_optimize_lightgbm(
    s1_limit: int = 25000,
    n_estimators: int = 300,
    learning_rate: float = 0.05
):
    start_time = time.time()
    s1_records, gt_map, pairs_data = load_training_dataset(s1_limit=s1_limit)

    # 1. Train / Validation Split by S1 Entity (prevent data leakage)
    all_s1_ids = list(s1_records.keys())
    random.seed(42)
    random.shuffle(all_s1_ids)

    split_idx = int(len(all_s1_ids) * 0.8)
    train_entities = set(all_s1_ids[:split_idx])
    val_entities = set(all_s1_ids[split_idx:])

    print(f"\nEntity Split: {len(train_entities):,} Train entities | {len(val_entities):,} Validation entities")

    train_rows = [p for p in pairs_data if p['s1_id'] in train_entities]
    val_rows = [p for p in pairs_data if p['s1_id'] in val_entities]

    print(f"Pair Split  : {len(train_rows):,} Train pairs | {len(val_rows):,} Validation pairs")

    # 2. Extract Features
    print("\nExtracting 17 features for Train pairs...")
    X_train = np.zeros((len(train_rows), len(FEATURE_NAMES)), dtype=np.float32)
    y_train = np.zeros(len(train_rows), dtype=np.int32)
    for i, row in enumerate(train_rows):
        feat_dict = extract_pairwise_features(
            row['rec_a'], row['rec_b'], rank=row['rank'],
            norm_name_a=row['norm_name_a'], norm_name_b=row['norm_name_b'],
            norm_addr_a=row['norm_addr_a'], norm_addr_b=row['norm_addr_b']
        )
        X_train[i] = [feat_dict[k] for k in FEATURE_NAMES]
        y_train[i] = row['label']

    print("Extracting 17 features for Validation pairs...")
    X_val = np.zeros((len(val_rows), len(FEATURE_NAMES)), dtype=np.float32)
    y_val = np.zeros(len(val_rows), dtype=np.int32)
    for i, row in enumerate(val_rows):
        feat_dict = extract_pairwise_features(
            row['rec_a'], row['rec_b'], rank=row['rank'],
            norm_name_a=row['norm_name_a'], norm_name_b=row['norm_name_b'],
            norm_addr_a=row['norm_addr_a'], norm_addr_b=row['norm_addr_b']
        )
        X_val[i] = [feat_dict[k] for k in FEATURE_NAMES]
        y_val[i] = row['label']

    # 3. Train LightGBM Classifier
    print("\n" + "=" * 70)
    print("TRAINING LIGHTGBM CLASSIFIER")
    print("=" * 70)
    
    # Calculate scale_pos_weight
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos = neg_count / max(pos_count, 1)
    print(f"Class Distribution: {pos_count:,} positive, {neg_count:,} negative (ratio 1:{scale_pos:.1f})")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=31,
        max_depth=7,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.5,  # Moderately boost positive recall while preserving precision
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
    )

    # 4. Standard Classification Metrics
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    val_ap = average_precision_score(y_val, val_probs)
    print(f"\nLightGBM Model Validation Performance:")
    print(f"  • ROC AUC: {val_auc:.4f}")
    print(f"  • Average Precision (PR-AUC): {val_ap:.4f}")

    # Feature Importances
    print("\nTop Feature Importances:")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx[:8]:
        print(f"  • {FEATURE_NAMES[idx]:26s}: {importances[idx]:4d}")

    # 5. Sweep Thresholds specifically for Macro-F0.5
    print("\n" + "=" * 70)
    print("MACRO-F0.5 THRESHOLD OPTIMIZATION (SWEEP)")
    print("=" * 70)

    val_gt_dict = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_entities}
    
    # Map predictions back to validation entities
    entity_cand_probs = {}
    for i, row in enumerate(val_rows):
        s1_id = row['s1_id']
        cid = row['cand_id']
        prob = val_probs[i]
        if s1_id not in entity_cand_probs:
            entity_cand_probs[s1_id] = []
        entity_cand_probs[s1_id].append((cid, prob))

    best_threshold = 0.70
    best_margin = 0.15
    best_f05 = 0.0

    print(f"{'Threshold':>10} | {'Margin':>8} | {'Macro-F0.5':>10} | {'Singletons':>12} | {'Matched':>10}")
    print("-" * 62)

    for thresh in np.arange(0.68, 0.86, 0.02):
        for margin in [0.10, 0.14, 0.18, 0.25, 1.0]:  # 1.0 means no relative margin
            pred_dict = {}
            for s1_id in val_entities:
                cands = entity_cand_probs.get(s1_id, [])
                if not cands:
                    pred_dict[s1_id] = []
                    continue
                sorted_cands = sorted(cands, key=lambda x: x[1], reverse=True)
                top_prob = sorted_cands[0][1]
                matches = [cid for cid, p in sorted_cands if p >= thresh and (top_prob - p) <= margin]
                pred_dict[s1_id] = matches

            score = compute_macro_f05(val_gt_dict, pred_dict)

            if score > best_f05:
                best_f05 = score
                best_threshold = thresh
                best_margin = margin
                
                # Compute singletons and matched breakdown for the new best
                sing_scores = [compute_entity_f_beta(val_gt_dict[s1_id], set(pred_dict[s1_id]), beta=0.5) 
                               for s1_id in val_entities if len(val_gt_dict[s1_id]) == 0]
                matched_scores = [compute_entity_f_beta(val_gt_dict[s1_id], set(pred_dict[s1_id]), beta=0.5) 
                                  for s1_id in val_entities if len(val_gt_dict[s1_id]) > 0]
                mean_sing = np.mean(sing_scores) if sing_scores else 1.0
                mean_match = np.mean(matched_scores) if matched_scores else 0.0

                print(f"{thresh:>10.2f} | {margin:>8.2f} | {score:>10.4f} | {mean_sing:>12.4f} | {mean_match:>10.4f} ◄ NEW PEAK")

    print("=" * 70)
    print(f"OPTIMAL DECISION PARAMETERS: Threshold = {best_threshold:.2f}, Relative Margin = {best_margin:.2f}")
    print(f"FINAL PEAK MACRO-F0.5 SCORE = {best_f05:.4f}")
    print("=" * 70)

    # 6. Save Model & Metadata
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    model_path = os.path.join(PROJECT_ROOT, "models", "lightgbm_matcher.pkl")
    meta_path = os.path.join(PROJECT_ROOT, "models", "lightgbm_metadata.json")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    meta = {
        "best_threshold": float(best_threshold),
        "best_margin": float(best_margin),
        "validation_macro_f05": float(best_f05),
        "validation_auc": float(val_auc),
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Model saved successfully to: {model_path}")
    print(f"Metadata saved successfully to: {meta_path}")
    print(f"Total Pipeline Runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} mins)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM Classifier for Entity Resolution.")
    parser.add_argument("--s1-limit", type=int, default=25000, help="Number of S1 reference entities to train on")
    parser.add_argument("--estimators", type=int, default=300, help="Number of gradient boosting trees")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    args = parser.parse_args()

    train_and_optimize_lightgbm(
        s1_limit=args.s1_limit,
        n_estimators=args.estimators,
        learning_rate=args.lr
    )
