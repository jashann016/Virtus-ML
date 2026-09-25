import os
import sys
import json
import time
import random
from typing import Dict, List, Set

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.blocking import InvertedIndexBlocker, load_tsv_records
from src.text_preprocessing import normalize_business_name, normalize_address, has_non_latin


def format_qwen_chatml_example(
    rec_a: dict, 
    rec_b: dict, 
    is_match: bool, 
    confidence: float, 
    rationale: str
) -> dict:
    """Formats a pair into standard Qwen2.5 ChatML conversation format."""
    system_msg = (
        "You are an expert Entity Resolution auditor. "
        "Analyze Record A and Record B and determine if they refer to the exact same real-world business entity. "
        "Return a valid JSON object with keys: 'is_match' (boolean), 'confidence' (float 0.0-1.0), and 'rationale' (string)."
    )
    
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
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }


def generate_rationale(rec_a: dict, rec_b: dict, is_match: bool) -> str:
    """Generates an intuitive rationale for the training example."""
    name_a = rec_a.get('business_name', '').lower()
    name_b = rec_b.get('business_name', '').lower()
    addr_a = rec_a.get('business_address', '').lower()
    addr_b = rec_b.get('business_address', '').lower()
    
    if is_match:
        if has_non_latin(rec_b.get('business_name', '')):
            return "Cross-script entity match: Latin and native Indic script represent the same business; address verified."
        elif not addr_b:
            return "Matching business name with omitted/empty address field in secondary source."
        elif '.com' in name_b or '.in' in name_b:
            return "Entity match where secondary record uses official website domain name."
        elif any(w in name_b for w in name_a.split()[:2] if len(w) > 3):
            return "Core business name and address components match with minor format/legal suffix variations."
        else:
            return "Verified match: shared street address and municipal unit under DBA/trade name variation."
    else:
        if addr_a and addr_b and any(w in addr_b for w in addr_a.split()[:3] if len(w) > 4):
            return "Hard negative: businesses share locality/street vicinity but operate as distinct independent entities."
        else:
            return "Non-match: distinct business names and conflicting addresses."


def build_hard_pairs_dataset(
    target_positives: int = 25000,
    target_negatives: int = 25000,
    output_filename: str = "train_pairs_50k.jsonl"
):
    print("=" * 60)
    print("PREPARING HARD-CASE DATASET FOR QWEN 2.5 7B QLoRA")
    print("=" * 60)
    
    data_dir = os.path.join(PROJECT_ROOT, "dataset", "train")
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    
    out_dir = os.path.join(PROJECT_ROOT, "aws", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    
    # 1. Load S1 entities
    print("Loading Source 1 reference entities...")
    s1_records = {}
    with open(s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            if len(s1_records) >= 15000:
                break
            parts = line.strip('\n').split('\t')
            if len(parts) >= 4:
                s1_records[parts[0].strip()] = {
                    'entity_id': parts[0].strip(),
                    'business_name': parts[1].strip(),
                    'business_address': parts[2].strip(),
                    'country': parts[3].strip()
                }
                
    # 2. Load ground truth mappings
    print("Loading Ground Truth pairs...")
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
                
    print(f"Loaded {len(s1_records):,} S1 entities with {len(needed_positive_ids):,} known true matches.")
    
    # 3. Stream S2 and S3: gather positive records + pool for hard negative mining
    print("Streaming Source 2 & Source 3 records...")
    cand_records = {}
    with open(s2_path, 'r', encoding='utf-8') as f2, open(s3_path, 'r', encoding='utf-8') as f3:
        for f in [f2, f3]:
            f.readline()
            for line in f:
                parts = line.strip('\n').split('\t')
                if len(parts) >= 4:
                    cid = parts[0].strip()
                    if cid in needed_positive_ids or len(cand_records) < 75000:
                        cand_records[cid] = {
                            'entity_id': cid,
                            'business_name': parts[1].strip(),
                            'business_address': parts[2].strip(),
                            'country': parts[3].strip()
                        }
                        
    print(f"Candidate pool loaded: {len(cand_records):,} records.")
    
    # 4. Build Blocker for Hard Negative Mining
    print("Indexing candidate pool for hard negative mining...")
    blocker = InvertedIndexBlocker()
    blocker.index_candidates(list(cand_records.values()))
    
    # 5. Assemble Positive and Hard Negative Pairs
    print("Mining hard pairs...")
    jsonl_records = []
    pos_count = 0
    neg_count = 0
    
    for s1_id, rec_a in s1_records.items():
        true_targets = gt_map.get(s1_id, set()).intersection(cand_records.keys())
        
        # Mine Positives
        for target_id in true_targets:
            if pos_count >= target_positives:
                break
            rec_b = cand_records[target_id]
            rationale = generate_rationale(rec_a, rec_b, is_match=True)
            conf = random.uniform(0.92, 0.99)
            example = format_qwen_chatml_example(rec_a, rec_b, is_match=True, confidence=conf, rationale=rationale)
            jsonl_records.append(example)
            pos_count += 1
            
        # Mine Hard Negatives (Candidates returned by blocker that are NOT true matches)
        retrieved_cands = blocker.retrieve_candidates(rec_a, max_candidates=25)
        for cand_id in retrieved_cands:
            if neg_count >= target_negatives:
                break
            if cand_id not in gt_map.get(s1_id, set()) and cand_id in cand_records:
                rec_b = cand_records[cand_id]
                rationale = generate_rationale(rec_a, rec_b, is_match=False)
                conf = random.uniform(0.01, 0.15)
                example = format_qwen_chatml_example(rec_a, rec_b, is_match=False, confidence=conf, rationale=rationale)
                jsonl_records.append(example)
                neg_count += 1
                
        if pos_count >= target_positives and neg_count >= target_negatives:
            break
            
    # Shuffle for uniform training distribution
    random.seed(42)
    random.shuffle(jsonl_records)
    
    # Write to JSONL
    print(f"Writing {len(jsonl_records):,} examples to {out_path}...")
    with open(out_path, 'w', encoding='utf-8') as f:
        for ex in jsonl_records:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print("=" * 60)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY:")
    print(f"• Total Training Examples: {len(jsonl_records):,}")
    print(f"• True Match Pairs (Positives): {pos_count:,}")
    print(f"• Deceptive Look-Alikes (Hard Negatives): {neg_count:,}")
    print(f"• Output File: {out_path} ({file_size_mb:.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    # Test generation with 5,000 curated pairs
    build_hard_pairs_dataset(target_positives=2500, target_negatives=2500, output_filename="train_pairs_sample.jsonl")
