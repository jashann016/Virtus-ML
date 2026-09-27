"""
VIRTUS-ML: Extract Borderline Test Pairs for Qwen 2.5 7B Arbiter
Filters the top ambiguous test pairs where LightGBM was uncertain,
and formats them for fast batch inference with Lakshay's Qwen ER Judge.
"""

import os
import sys
import json
import gzip
import time
import argparse
from typing import Dict, List
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def extract_borderline_pairs(
    matching_file: str = "output/matching_results.tsv",
    candidate_file: str = "output/candidate_pairs.tsv",
    output_file: str = "/kaggle/working/borderline_cases_for_qwen.jsonl.gz",
    max_pairs: int = 15000
):
    print("=" * 70)
    print("VIRTUS-ML: EXTRACTING BORDERLINE TEST PAIRS FOR QWEN 2.5 7B")
    print(f"  • Matching File   : {matching_file}")
    print(f"  • Candidate File  : {candidate_file}")
    print(f"  • Target Pairs Cap: {max_pairs:,}")
    print(f"  • Output JSONL    : {output_file}")
    print("=" * 70)
    t0 = time.time()

    # 1. Identify ambiguous S1 entities and their top candidate
    print("[1/4] Scanning matching and candidate outputs...")
    ambiguous_pairs = [] # (s1_id, cand_id, case_type)

    with open(matching_file, 'r', encoding='utf-8') as f_m, \
         open(candidate_file, 'r', encoding='utf-8') as f_c:
        f_m.readline()
        f_c.readline()

        for l_m, l_c in zip(f_m, f_c):
            p_m = l_m.strip('\n').split('\t')
            p_c = l_c.strip('\n').split('\t')
            s1 = p_m[0].strip()

            matches = [x.strip() for x in p_m[1].split(',') if x.strip()] if len(p_m) > 1 else []
            cands = [x.strip() for x in p_c[1].split(',') if x.strip()] if len(p_c) > 1 else []

            if not matches and cands:
                # Borderline singleton: Blocker found candidates, but LightGBM rejected
                ambiguous_pairs.append((s1, cands[0], 'rejected_top1'))
            elif matches and len(cands) > len(matches):
                # Borderline multi-match: had a 2nd strong candidate that was filtered out
                for c in cands:
                    if c not in matches:
                        ambiguous_pairs.append((s1, c, 'secondary_candidate'))
                        break

    print(f"  • Found {len(ambiguous_pairs):,} total ambiguous candidate pairs.")

    # 2. Load needed S1 and Candidate records from test files
    needed_s1 = {s1 for s1, _, _ in ambiguous_pairs}
    needed_cands = {c for _, c, _ in ambiguous_pairs}

    print(f"[2/4] Reading raw records for {len(needed_s1):,} S1s and {len(needed_cands):,} candidates...")
    test_dir = os.path.join(PROJECT_ROOT, "dataset", "test")
    
    s1_data = {}
    with open(os.path.join(test_dir, "test_source1.tsv"), 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            if parts and parts[0].strip() in needed_s1:
                s1_data[parts[0].strip()] = {
                    'entity_id': parts[0].strip(),
                    'business_name': parts[1].strip() if len(parts) > 1 else '',
                    'business_address': parts[2].strip() if len(parts) > 2 else '',
                    'country': parts[3].strip() if len(parts) > 3 else ''
                }

    cand_data = {}
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                f.readline()
                for line in f:
                    parts = line.strip('\n').split('\t')
                    if parts and parts[0].strip() in needed_cands:
                        cand_data[parts[0].strip()] = {
                            'entity_id': parts[0].strip(),
                            'business_name': parts[1].strip() if len(parts) > 1 else '',
                            'business_address': parts[2].strip() if len(parts) > 2 else '',
                            'country': parts[3].strip() if len(parts) > 3 else ''
                        }

    print(f"  • Loaded {len(s1_data):,} S1 records and {len(cand_data):,} candidate records.")

    # 3. Score string similarity to rank the hardest, highest-potential pairs
    print("[3/4] Ranking ambiguous pairs by semantic similarity...")
    scored_pairs = []

    for s1_id, c_id, ctype in ambiguous_pairs:
        if s1_id in s1_data and c_id in cand_data:
            rec_a = s1_data[s1_id]
            rec_b = cand_data[c_id]
            
            # Hybrid similarity
            jw = JaroWinkler.similarity(rec_a['business_name'], rec_b['business_name'])
            ts = fuzz.token_sort_ratio(rec_a['business_name'], rec_b['business_name']) / 100.0
            score = (jw * 0.5) + (ts * 0.5)
            
            # Boost rejected top1 singletons where name is very close (these are high-yield!)
            if ctype == 'rejected_top1':
                score += 0.15

            scored_pairs.append((score, s1_id, c_id, rec_a, rec_b, ctype))

    # Sort descending and take top N
    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    selected = scored_pairs[:max_pairs]
    print(f"  • Selected top {len(selected):,} highest-yield borderline pairs for Qwen.")

    # 4. Write to JSONL & JSONL.GZ
    print(f"[4/4] Writing formatted pairs to {output_file}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    uncompressed_out = output_file.replace('.gz', '')
    
    with open(uncompressed_out, 'w', encoding='utf-8') as f_raw, \
         gzip.open(output_file, 'wt', encoding='utf-8') as f_gz:
        
        for sim, s1_id, c_id, rec_a, rec_b, ctype in selected:
            item = {
                "pair_id": f"{s1_id}::{c_id}",
                "source1_id": s1_id,
                "candidate_id": c_id,
                "case_type": ctype,
                "similarity_score": round(sim, 4),
                "record_a": rec_a,
                "record_b": rec_b
            }
            line = json.dumps(item) + "\n"
            f_raw.write(line)
            f_gz.write(line)

    raw_mb = os.path.getsize(uncompressed_out) / (1024 * 1024)
    gz_mb = os.path.getsize(output_file) / (1024 * 1024)
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE:")
    print(f"• Total Pairs Selected : {len(selected):,}")
    print(f"• Output (.jsonl)      : {uncompressed_out} ({raw_mb:.2f} MB)")
    print(f"• Output (.jsonl.gz)   : {output_file} ({gz_mb:.2f} MB)")
    print(f"• Time Elapsed         : {time.time()-t0:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract borderline test pairs for Qwen 2.5 7B.")
    parser.add_argument("--matching", default="output/matching_results.tsv", help="Matching TSV path")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Candidate TSV path")
    parser.add_argument("--output", default="/kaggle/working/borderline_cases_for_qwen.jsonl.gz", help="Output path")
    parser.add_argument("--max-pairs", type=int, default=15000, help="Max borderline pairs")
    args = parser.parse_args()

    extract_borderline_pairs(
        matching_file=args.matching,
        candidate_file=args.candidate,
        output_file=args.output,
        max_pairs=args.max_pairs
    )
