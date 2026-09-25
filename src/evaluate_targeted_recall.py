import os
import sys
import time
from typing import Dict, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.blocking import InvertedIndexBlocker, load_tsv_records


def run_targeted_recall_benchmark(num_s1: int = 1000, num_distractors: int = 50000):
    print("=" * 60)
    print("TARGETED CANDIDATE BLOCKING RECALL BENCHMARK")
    print("=" * 60)
    
    data_dir = os.path.join(PROJECT_ROOT, "dataset", "train")
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    
    # 1. Load S1 entities with known matches
    s1_records = []
    s1_ids = set()
    with open(s1_path, 'r', encoding='utf-8') as f:
        header = f.readline().split('\t')
        for i, line in enumerate(f):
            if len(s1_records) >= num_s1:
                break
            parts = line.strip('\n').split('\t')
            if len(parts) >= 4:
                s1_records.append({
                    'entity_id': parts[0].strip(),
                    'business_name': parts[1].strip(),
                    'business_address': parts[2].strip(),
                    'country': parts[3].strip()
                })
                s1_ids.add(parts[0].strip())
                
    # 2. Load ground truth for these S1
    gt_map = {}
    target_ids_needed = set()
    with open(gt_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            if parts[0].strip() in s1_ids:
                matches = set()
                if len(parts) > 1 and parts[1].strip():
                    matches = {m.strip() for m in parts[1].split(',') if m.strip()}
                gt_map[parts[0].strip()] = matches
                target_ids_needed.update(matches)
                
    print(f"Loaded {len(s1_records)} S1 entities. Target matches needed: {len(target_ids_needed)}")
    
    # 3. Stream S2 and S3: collect all needed target records + distractors
    candidate_records = []
    found_targets = set()
    distractor_count = 0
    
    for path in [s2_path, s3_path]:
        print(f"Scanning {os.path.basename(path)} for target matches and distractors...")
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.strip('\n').split('\t')
                if len(parts) < 4:
                    continue
                cid = parts[0].strip()
                if cid in target_ids_needed:
                    candidate_records.append({
                        'entity_id': cid,
                        'business_name': parts[1].strip(),
                        'business_address': parts[2].strip(),
                        'country': parts[3].strip()
                    })
                    found_targets.add(cid)
                elif distractor_count < num_distractors:
                    candidate_records.append({
                        'entity_id': cid,
                        'business_name': parts[1].strip(),
                        'business_address': parts[2].strip(),
                        'country': parts[3].strip()
                    })
                    distractor_count += 1
                    
    print(f"Total candidate pool built: {len(candidate_records):,} records (Found {len(found_targets)}/{len(target_ids_needed)} true targets)")
    
    # 4. Build Index and Measure Recall
    print("Indexing candidate pool...")
    t0 = time.time()
    blocker = InvertedIndexBlocker()
    blocker.index_candidates(candidate_records)
    print(f"Indexed in {time.time() - t0:.2f}s")
    
    print("Retrieving candidates for each S1 entity...")
    t0 = time.time()
    total_evaluable_links = 0
    captured_links = 0
    
    for r in s1_records:
        s1_id = r['entity_id']
        true_targets = gt_map.get(s1_id, set()).intersection(found_targets)
        if not true_targets:
            continue
            
        retrieved = set(blocker.retrieve_candidates(r, max_candidates=90))
        total_evaluable_links += len(true_targets)
        captured_links += len(true_targets.intersection(retrieved))
        
    elapsed = time.time() - t0
    recall = (captured_links / total_evaluable_links) * 100 if total_evaluable_links > 0 else 0
    
    print("\n" + "=" * 60)
    print("TARGETED CANDIDATE RECALL RESULTS:")
    print(f"• Total Ground Truth Links Tested: {total_evaluable_links:,}")
    print(f"• True Links Captured in Top-90 Candidates: {captured_links:,}")
    print(f"• TRUE CANDIDATE RECALL CEILING: {recall:.2f}%")
    print(f"• Retrieval Speed: {len(s1_records) / elapsed:.1f} entities/second")
    print("=" * 60)


if __name__ == "__main__":
    run_targeted_recall_benchmark(num_s1=1000, num_distractors=50000)
