"""
VIRTUS-ML: Golden 50,000 Dataset Miner for Qwen 2.5 7B QLoRA Fine-Tuning
Enforces the exact 5-category composition for maximum Macro-F0.5 precision
and zero-shot France generalization.

Golden Composition (50,000 Total Pairs):
1. Hard Positives (Address & Name Shift): 15,000
2. Hard Positives (Website & Domain Links): 5,000
3. Hard Negatives (Same Name, Diff City): 15,000
4. Hard Negatives (Look-Alike / Same Street): 10,000
5. Cross-Lingual & Language-Agnostic Pairs: 5,000 (2,500 Match + 2,500 Non-Match)
"""

import os
import sys
import json
import time
import random
import argparse
from typing import Dict, List, Set, Tuple
from rapidfuzz import fuzz

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.blocking import InvertedIndexBlocker
from src.text_preprocessing import (
    normalize_business_name,
    normalize_address,
    has_non_latin,
    extract_plot_codes,
    RE_DOMAIN
)

SYSTEM_PROMPT = (
    "You are an international business entity resolution expert fluent in multiple languages including English, Hindi, and French. "
    "Analyze Record A and Record B and determine if they refer to the exact same real-world business entity despite differences in language, script, address formatting, abbreviations, or legal entity suffixes. "
    "Return a valid JSON object with keys: 'is_match' (boolean), 'confidence' (float 0.0-1.0), and 'rationale' (string)."
)


def format_qwen_chatml_example(
    rec_a: dict, 
    rec_b: dict, 
    is_match: bool, 
    confidence: float, 
    rationale: str
) -> dict:
    """Formats a pair into standard Qwen2.5 ChatML conversation format."""
    user_msg = (
        f"Record A:\n"
        f"Name: {rec_a.get('business_name', '')}\n"
        f"Address: {rec_a.get('business_address', '')}\n"
        f"Country: {rec_a.get('country', '')}\n\n"
        f"Record B:\n"
        f"Name: {rec_b.get('business_name', '')}\n"
        f"Address: {rec_b.get('business_address', '')}\n"
        f"Country: {rec_b.get('country', '')}"
    )
    
    assistant_msg = json.dumps({
        "is_match": is_match,
        "confidence": round(confidence, 2),
        "rationale": rationale
    }, ensure_ascii=False)
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }


def is_website_or_domain(name_a: str, name_b: str) -> bool:
    """Detects website or domain pattern in either entity name."""
    web_indicators = ['.com', '.in', '.org', '.net', '.co', '.io', 'www.', 'http']
    name_a_low = name_a.lower()
    name_b_low = name_b.lower()
    return any(ind in name_a_low or ind in name_b_low for ind in web_indicators) or bool(RE_DOMAIN.match(name_a.strip())) or bool(RE_DOMAIN.match(name_b.strip()))


def is_cross_lingual_or_agnostic(rec_a: dict, rec_b: dict) -> bool:
    """Detects non-Latin scripts (Devanagari, Odia, etc.) or alphanumeric unit/plot codes."""
    if has_non_latin(rec_a.get('business_name', '')) or has_non_latin(rec_b.get('business_name', '')):
        return True
    if has_non_latin(rec_a.get('business_address', '')) or has_non_latin(rec_b.get('business_address', '')):
        return True
    plots_a = extract_plot_codes(rec_a.get('business_address', ''))
    plots_b = extract_plot_codes(rec_b.get('business_address', ''))
    if plots_a and plots_b and set(plots_a).intersection(plots_b):
        return True
    return False


def build_golden_50k_dataset(
    output_filename: str = "train_pairs_50k.jsonl",
    target_cat1: int = 15000,  # Hard Positives (Name & Address Shift)
    target_cat2: int = 5000,   # Hard Positives (Website & Domain Links)
    target_cat3: int = 15000,  # Hard Negatives (Same Name, Diff City)
    target_cat4: int = 10000,  # Hard Negatives (Look-Alike / Same Street)
    target_cat5: int = 5000,   # Cross-Lingual & Language-Agnostic (2500 pos + 2500 neg)
):
    print("=" * 70)
    print("VIRTUS-ML: MINING THE GOLDEN 50,000 DATASET FOR QWEN 2.5 7B QLoRA")
    print("=" * 70)
    print(f"Target Recipe:")
    print(f"  • Cat 1: Hard Positives (Name & Address Shift)    : {target_cat1:,}")
    print(f"  • Cat 2: Hard Positives (Website & Domain Links)  : {target_cat2:,}")
    print(f"  • Cat 3: Hard Negatives (Same Name, Diff City)    : {target_cat3:,}")
    print(f"  • Cat 4: Hard Negatives (Same Street, Diff Name)  : {target_cat4:,}")
    print(f"  • Cat 5: Cross-Lingual & Language-Agnostic        : {target_cat5:,} (50% match / 50% non-match)")
    total_target = target_cat1 + target_cat2 + target_cat3 + target_cat4 + target_cat5
    print(f"  • TOTAL PAIRS                                     : {total_target:,}")
    print("=" * 70)

    data_dir = os.path.join(PROJECT_ROOT, "dataset", "train")
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")

    out_dir = os.path.join(PROJECT_ROOT, "aws", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)

    start_time = time.time()

    # 1. Stream S1 reference entities
    s1_limit = max(100000, int(target_cat3 * 7))
    print(f"\n[1/5] Loading {s1_limit:,} Source 1 reference entities...")
    s1_records = {}
    with open(s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
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

    # 2. Load Ground Truth mappings for these S1 records
    print(f"[2/5] Loading Ground Truth match labels...")
    gt_map = {}
    needed_positive_ids = set()
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
                needed_positive_ids.update(matches)

    print(f"  -> Loaded ground truth for {len(s1_records):,} entities ({len(needed_positive_ids):,} known matches).")

    # 3. Stream S2 & S3: capture needed true matches + pool for hard negative mining
    cand_limit = max(350000, int(target_cat4 * 35))
    print(f"[3/5] Streaming Source 2 & Source 3 (capturing matches + pool up to {cand_limit:,})...")
    cand_records = {}
    with open(s2_path, 'r', encoding='utf-8') as f2, open(s3_path, 'r', encoding='utf-8') as f3:
        for f in [f2, f3]:
            f.readline()
            for line in f:
                parts = line.strip('\n').split('\t')
                if len(parts) >= 4:
                    cid = parts[0].strip()
                    if cid in needed_positive_ids or len(cand_records) < cand_limit:
                        cand_records[cid] = {
                            'entity_id': cid,
                            'business_name': parts[1].strip(),
                            'business_address': parts[2].strip(),
                            'country': parts[3].strip()
                        }

    print(f"  -> Candidate pool ready: {len(cand_records):,} records.")

    # 4. Build Multi-Angle Inverted Index Blocker
    print(f"[4/5] Indexing candidates with InvertedIndexBlocker for hard negative mining...")
    blocker = InvertedIndexBlocker()
    blocker.index_candidates(list(cand_records.values()))

    # 5. Mine 5 Distinct Categories
    print(f"[5/5] Mining exact 5-category composition...")
    cat1_pairs = []
    cat2_pairs = []
    cat3_pairs = []
    cat4_pairs = []
    cat5_pos_pairs = []
    cat5_neg_pairs = []

    cat5_pos_target = target_cat5 // 2
    cat5_neg_target = target_cat5 - cat5_pos_target

    # Pre-cache normalized names and addresses for speed
    print("  -> Pre-computing normalized representations for candidates...")
    cand_norm_name = {}
    cand_norm_addr = {}
    for cid, cand in cand_records.items():
        cand_norm_name[cid] = normalize_business_name(cand['business_name'])
        cand_norm_addr[cid] = normalize_address(cand['business_address'])

    print("  -> Iterating S1 entities to assemble golden dataset...")
    for s1_id, rec_a in s1_records.items():
        norm_name_a = normalize_business_name(rec_a['business_name'])
        norm_addr_a = normalize_address(rec_a['business_address'])

        # --- A. POSITIVE MINING ---
        true_targets = gt_map.get(s1_id, set()).intersection(cand_records.keys())
        for target_id in true_targets:
            rec_b = cand_records[target_id]
            norm_name_b = cand_norm_name[target_id]
            norm_addr_b = cand_norm_addr[target_id]

            # Check Category 5 Positives (Cross-Lingual / Alphanumeric Plot)
            if len(cat5_pos_pairs) < cat5_pos_target and is_cross_lingual_or_agnostic(rec_a, rec_b):
                conf = round(random.uniform(0.92, 0.99), 2)
                rationale = "Cross-script/structural entity match: Indic native script and Latin transliteration represent the identical enterprise with verified structural address."
                cat5_pos_pairs.append(format_qwen_chatml_example(rec_a, rec_b, True, conf, rationale))
                continue

            # Check Category 2 (Website / Domain Links)
            if len(cat2_pairs) < target_cat2 and is_website_or_domain(rec_a['business_name'], rec_b['business_name']):
                conf = round(random.uniform(0.93, 0.99), 2)
                rationale = "Verified entity match: secondary record utilizes official website domain / URL corresponding directly to reference business identity."
                cat2_pairs.append(format_qwen_chatml_example(rec_a, rec_b, True, conf, rationale))
                continue

            # Check Category 1 (Name & Address Shift)
            if len(cat1_pairs) < target_cat1:
                conf = round(random.uniform(0.91, 0.98), 2)
                rationale = "Verified entity match: core business name and address match with minor typographical, legal suffix (Inc/LLC/Ltd), or street abbreviation (St/Rd/Ave) variation."
                cat1_pairs.append(format_qwen_chatml_example(rec_a, rec_b, True, conf, rationale))

        # --- B. NEGATIVE MINING ---
        # Retrieve blocker candidates
        retrieved_cands = blocker.retrieve_candidates(rec_a, max_candidates=35)
        for cand_id in retrieved_cands:
            if cand_id in gt_map.get(s1_id, set()) or cand_id not in cand_records:
                continue

            rec_b = cand_records[cand_id]
            norm_name_b = cand_norm_name[cand_id]
            norm_addr_b = cand_norm_addr[cand_id]

            name_sim = fuzz.token_sort_ratio(norm_name_a, norm_name_b)
            addr_sim = fuzz.token_set_ratio(norm_addr_a, norm_addr_b)

            # Check Category 5 Negatives (Cross-Lingual or Plot Code mismatch)
            if len(cat5_neg_pairs) < cat5_neg_target and (has_non_latin(rec_b['business_name']) or has_non_latin(rec_b['business_address'])):
                conf = round(random.uniform(0.01, 0.12), 2)
                rationale = "Structural non-match: distinct entities differentiated by non-matching local script and commercial identifiers."
                cat5_neg_pairs.append(format_qwen_chatml_example(rec_a, rec_b, False, conf, rationale))
                continue

            # Check Category 3 (Same Name, Diff City / Location)
            # High name similarity (>= 80), but low address similarity (< 45)
            if len(cat3_pairs) < target_cat3 and name_sim >= 80 and addr_sim < 45:
                conf = round(random.uniform(0.02, 0.15), 2)
                rationale = "Hard negative non-match: identical or franchise business branding operating in distinctly different geographical municipalities/postal codes."
                cat3_pairs.append(format_qwen_chatml_example(rec_a, rec_b, False, conf, rationale))
                continue

            # Check Category 4 (Look-Alike / Same Street, Diff Name)
            # High address similarity (>= 65), but low name similarity (< 50)
            if len(cat4_pairs) < target_cat4 and addr_sim >= 65 and name_sim < 50:
                conf = round(random.uniform(0.01, 0.10), 2)
                rationale = "Hard negative non-match: distinct commercial businesses located in proximate street vicinity or shared retail/office complex."
                cat4_pairs.append(format_qwen_chatml_example(rec_a, rec_b, False, conf, rationale))

        # Check if all quotas reached
        if (len(cat1_pairs) >= target_cat1 and
            len(cat2_pairs) >= target_cat2 and
            len(cat3_pairs) >= target_cat3 and
            len(cat4_pairs) >= target_cat4 and
            len(cat5_pos_pairs) >= cat5_pos_target and
            len(cat5_neg_pairs) >= cat5_neg_target):
            break

    # Combine all 5 categories
    cat5_pairs = cat5_pos_pairs + cat5_neg_pairs
    all_examples = cat1_pairs + cat2_pairs + cat3_pairs + cat4_pairs + cat5_pairs
    
    # Shuffle thoroughly with fixed seed
    random.seed(42)
    random.shuffle(all_examples)

    # Write to target JSONL file
    print(f"\nWriting {len(all_examples):,} curated ChatML examples to {out_path}...")
    with open(out_path, 'w', encoding='utf-8') as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)

    print("=" * 70)
    print("🎉 GOLDEN 50,000 DATASET GENERATION SUMMARY")
    print("=" * 70)
    print(f"• Category 1 (Hard Positives - Name/Address Shift): {len(cat1_pairs):,} pairs")
    print(f"• Category 2 (Hard Positives - Website/Domain Links): {len(cat2_pairs):,} pairs")
    print(f"• Category 3 (Hard Negatives - Same Name, Diff City): {len(cat3_pairs):,} pairs")
    print(f"• Category 4 (Hard Negatives - Same Street/Complex) : {len(cat4_pairs):,} pairs")
    print(f"• Category 5 (Cross-Lingual & Language-Agnostic)    : {len(cat5_pairs):,} pairs ({len(cat5_pos_pairs):,} pos / {len(cat5_neg_pairs):,} neg)")
    print("-" * 70)
    print(f"• TOTAL GENERATED PAIRS: {len(all_examples):,}")
    print(f"• Output File: {out_path} ({file_size_mb:.2f} MB)")
    print(f"• Total Generation Time: {elapsed:.2f} seconds ({elapsed/60:.2f} mins)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine the Golden 50k Dataset for Qwen 2.5 7B QLoRA.")
    parser.add_argument("--cat1", type=int, default=15000, help="Category 1: Hard Positives Name/Address Shift")
    parser.add_argument("--cat2", type=int, default=5000, help="Category 2: Hard Positives Website/Domain Links")
    parser.add_argument("--cat3", type=int, default=15000, help="Category 3: Hard Negatives Same Name, Diff City")
    parser.add_argument("--cat4", type=int, default=10000, help="Category 4: Hard Negatives Same Street, Diff Name")
    parser.add_argument("--cat5", type=int, default=5000, help="Category 5: Cross-Lingual & Language-Agnostic")
    parser.add_argument("--output", type=str, default="train_pairs_50k.jsonl", help="Output file inside aws/data/")
    args = parser.parse_args()

    build_golden_50k_dataset(
        output_filename=args.output,
        target_cat1=args.cat1,
        target_cat2=args.cat2,
        target_cat3=args.cat3,
        target_cat4=args.cat4,
        target_cat5=args.cat5
    )
