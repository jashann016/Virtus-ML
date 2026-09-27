"""
VIRTUS-ML: High-Precision Lean Inference Engine with Person 2 LightGBM V3
Trained specifically on Hard Negatives (0.9590 Macro-F0.5)
Enforces Strict Precision Gating (Anti-Overprediction) to maximize Macro-F0.5
"""

import os
import sys
import gc
import re
import time
import pickle
import argparse
from collections import defaultdict
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from unidecode import unidecode

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.text_preprocessing import (
    normalize_business_name,
    normalize_address,
    extract_numbers,
    has_non_latin,
    soundex,
    RE_DOMAIN
)
from src.blocking import (
    extract_compound_numbers,
    extract_plot_codes,
    extract_address_keys,
    STOP_WORDS,
    COMMON_CITY_TOKENS,
    GENERIC_ADDR_WORDS
)
from src.feature_engineering import compute_pair_features, FEATURE_COLUMNS


def run_v3_country_inference(
    country: str = "india",
    limit: int = None,
    max_cands: int = 20,
    threshold: float = 0.70,
    margin: float = 0.04,
    model_path: str = "models/lightgbm_v3.pkl",
    output_dir: str = "output"
):
    print("=" * 75)
    print(f"VIRTUS-ML: PERSON 2 V3 LEAN INFERENCE FOR {country.upper()}")
    print(f"  • LightGBM V3 Model : {model_path}")
    print(f"  • Decision Threshold: {threshold:.2f}")
    print(f"  • Strict Margin     : {margin:.2f}")
    print(f"  • Max Candidates    : {max_cands}")
    print("=" * 75)
    
    start_total = time.time()
    
    # 1. Load Model
    print(f"[1/4] Loading LightGBM V3 model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("  • Model loaded successfully (14 features).")

    # 2. Load Candidates & Build Inverted Index
    print(f"[2/4] Indexing Candidate Pool for {country.upper()}...")
    t0 = time.time()
    
    candidates = {} # cid -> {'business_name': str, 'business_address': str}
    exact_idx = defaultdict(list)
    first_w_idx = defaultdict(list)
    soundex_idx = defaultdict(list)
    token_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    
    POSTING_CAP = 150
    test_dir = os.path.join(PROJECT_ROOT, "dataset", "test")
    
    for filename in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.strip('\n').split('\t')
                if len(parts) >= 4 and parts[3].strip().lower() == country:
                    cid = parts[0].strip()
                    raw_name = parts[1].strip()
                    raw_addr = parts[2].strip()
                    
                    candidates[cid] = {'business_name': raw_name, 'business_address': raw_addr}
                    
                    norm_name = normalize_business_name(raw_name)
                    words = norm_name.split()
                    
                    if norm_name:
                        lst = exact_idx[norm_name]
                        if len(lst) < POSTING_CAP:
                            lst.append(cid)
                            
                    if words and len(words[0]) >= 3 and words[0] not in COMMON_CITY_TOKENS:
                        fw = words[0]
                        lst = first_w_idx[fw]
                        if len(lst) < POSTING_CAP:
                            lst.append(cid)
                        sdx = soundex(fw)
                        if sdx:
                            lst_s = soundex_idx[sdx]
                            if len(lst_s) < POSTING_CAP:
                                lst_s.append(cid)
                                
                    for w in words:
                        if len(w) >= 3 and w not in STOP_WORDS:
                            lst = token_idx[w]
                            if len(lst) < POSTING_CAP:
                                lst.append(cid)

                    # Address tokens & numbers
                    for pc in extract_plot_codes(raw_addr):
                        lst = addr_idx[f'plot_{pc}']
                        if len(lst) < POSTING_CAP:
                            lst.append(cid)
                            
                    for cn in extract_compound_numbers(raw_addr):
                        lst = addr_idx[f'cnum_{cn}']
                        if len(lst) < POSTING_CAP:
                            lst.append(cid)

                    for aw in raw_addr.lower().replace(',', ' ').replace('.', ' ').replace('/', ' ').replace('-', ' ').replace('#', ' ').split():
                        if len(aw) >= 4 and aw not in STOP_WORDS and aw not in COMMON_CITY_TOKENS and aw not in GENERIC_ADDR_WORDS and not aw.isdigit():
                            lst = addr_idx[f'aw_{aw}']
                            if len(lst) < POSTING_CAP:
                                lst.append(cid)

    idx_time = time.time() - t0
    print(f"  • Successfully indexed {len(candidates):,} candidates in {idx_time:.2f}s!")

    # 3. Read Source 1 Entities
    print(f"[3/4] Loading Source 1 test entities for {country.upper()}...")
    s1_records = []
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    with open(s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            if len(parts) >= 4 and parts[3].strip().lower() == country:
                s1_records.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
                if limit and len(s1_records) >= limit:
                    break

    n_total = len(s1_records)
    print(f"  • Loaded {n_total:,} Source 1 entities.")

    # 4. Streamed Batch Evaluation
    os.makedirs(output_dir, exist_ok=True)
    matching_out = os.path.join(output_dir, f"matching_{country}.tsv")
    candidate_out = os.path.join(output_dir, f"candidate_{country}.tsv")
    print(f"[4/4] Scoring and streaming output to:\n  • {matching_out}\n  • {candidate_out}")

    batch_size = 2500
    total_matches = 0
    total_singletons = 0
    t_score_start = time.time()

    with open(matching_out, 'w', encoding='utf-8') as f_match, \
         open(candidate_out, 'w', encoding='utf-8') as f_cand:

        for b_start in range(0, n_total, batch_size):
            b_end = min(b_start + batch_size, n_total)
            batch = s1_records[b_start:b_end]

            batch_pairs = []       # List of (s1_id, cand_id)
            batch_feature_rows = [] # 2D array rows for LightGBM

            for s1_id, r_name, r_addr in batch:
                s1_dict = {'business_name': r_name, 'business_address': r_addr}
                norm_name = normalize_business_name(r_name)
                words = norm_name.split()

                matched = defaultdict(int)
                if norm_name in exact_idx:
                    for c in exact_idx[norm_name]:
                        matched[c] += 40
                if words and words[0] in first_w_idx:
                    for c in first_w_idx[words[0]]:
                        matched[c] += 15
                    sdx = soundex(words[0])
                    if sdx and sdx in soundex_idx:
                        for c in soundex_idx[sdx]:
                            matched[c] += 8
                for w in words:
                    if len(w) >= 3 and w in token_idx:
                        wt = 2 if w in COMMON_CITY_TOKENS else 6
                        for c in token_idx[w]:
                            matched[c] += wt
                for pc in extract_plot_codes(r_addr):
                    key = f'plot_{pc}'
                    if key in addr_idx:
                        for c in addr_idx[key]:
                            matched[c] += 12
                for cn in extract_compound_numbers(r_addr):
                    key = f'cnum_{cn}'
                    if key in addr_idx:
                        for c in addr_idx[key]:
                            matched[c] += 10
                for aw in r_addr.lower().replace(',', ' ').replace('.', ' ').replace('/', ' ').replace('-', ' ').split():
                    if len(aw) >= 4 and aw not in STOP_WORDS and aw not in COMMON_CITY_TOKENS:
                        key = f'aw_{aw}'
                        if key in addr_idx:
                            for c in addr_idx[key]:
                                matched[c] += 3

                if matched:
                    sorted_cands = sorted(matched.items(), key=lambda x: x[1], reverse=True)[:max_cands]
                    for cid, _ in sorted_cands:
                        cand_dict = candidates[cid]
                        feats = compute_pair_features(s1_dict, cand_dict)
                        batch_pairs.append((s1_id, cid))
                        batch_feature_rows.append([feats[col] for col in FEATURE_COLUMNS])

            # Vectorized LightGBM prediction
            prob_dict = defaultdict(list)
            if batch_feature_rows:
                X = np.array(batch_feature_rows, dtype=np.float32)
                probs = model.predict_proba(X)[:, 1]
                for (s1_id, cid), p in zip(batch_pairs, probs):
                    prob_dict[s1_id].append((cid, float(p)))

            # Decision rule: High Precision Strict Top-1
            for s1_id, r_name, r_addr in batch:
                cands_with_p = prob_dict.get(s1_id, [])
                if cands_with_p:
                    cands_with_p.sort(key=lambda x: x[1], reverse=True)
                    top_cand, top_p = cands_with_p[0]
                    
                    if top_p >= threshold:
                        # Top-1 accepted. Only add 2nd match if it is independently high confidence
                        accepted = [top_cand]
                        for c, p in cands_with_p[1:]:
                            if p >= 0.88 and (top_p - p) <= margin:
                                accepted.append(c)
                                
                        lean_cands = [c for c, _ in cands_with_p[:3]]
                        for a in accepted:
                            if a not in lean_cands:
                                lean_cands.append(a)
                                
                        f_cand.write(f"{s1_id}\t{','.join(lean_cands)}\n")
                        f_match.write(f"{s1_id}\t{','.join(accepted)}\n")
                        total_matches += 1
                    else:
                        lean_cands = [c for c, _ in cands_with_p[:2]]
                        f_cand.write(f"{s1_id}\t{','.join(lean_cands)}\n")
                        f_match.write(f"{s1_id}\t\n")
                        total_singletons += 1
                else:
                    f_cand.write(f"{s1_id}\t\n")
                    f_match.write(f"{s1_id}\t\n")
                    total_singletons += 1

            if b_end % 20000 < batch_size or b_end == n_total:
                elapsed = time.time() - t_score_start
                rate = b_end / elapsed if elapsed > 0 else 0
                eta = (n_total - b_end) / rate if rate > 0 else 0
                print(f"  [{country.upper()}] Processed {b_end:,}/{n_total:,} ({b_end/n_total*100:.1f}%) | "
                      f"Speed: {rate:.1f} ent/s | Matches: {total_matches:,} | ETA: {eta/60:.1f} mins")

    total_time = time.time() - start_total
    print("\n" + "=" * 75)
    print(f"COMPLETED {country.upper()}:")
    print(f"• Total Processed    : {n_total:,}")
    print(f"• Matched Entities   : {total_matches:,} ({total_matches/n_total*100:.1f}%)")
    print(f"• Singletons         : {total_singletons:,} ({total_singletons/n_total*100:.1f}%)")
    print(f"• Total Time Elapsed : {total_time:.1f}s ({total_time/60:.1f} mins)")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person 2 V3 High-Precision Inference.")
    parser.add_argument("--country", default="india", help="Target country")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit")
    parser.add_argument("--max-cands", type=int, default=20, help="Max candidates per entity")
    parser.add_argument("--threshold", type=float, default=0.70, help="Decision threshold")
    parser.add_argument("--margin", type=float, default=0.04, help="Strict margin for 2nd match")
    parser.add_argument("--model-path", default="models/lightgbm_v3.pkl", help="Path to V3 model")
    args = parser.parse_args()

    run_v3_country_inference(
        country=args.country,
        limit=args.limit,
        max_cands=args.max_cands,
        threshold=args.threshold,
        margin=args.margin,
        model_path=args.model_path
    )
