import os
import sys
import time
from typing import Dict, Set

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.blocking import InvertedIndexBlocker, load_tsv_records


def load_ground_truth_map(gt_path: str, filter_s1_ids: Set[str] = None) -> Dict[str, Set[str]]:
    """Loads ground truth mappings for evaluation."""
    gt_map = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip('\n').split('\t')
            if not parts:
                continue
            s1_id = parts[0].strip()
            if filter_s1_ids and s1_id not in filter_s1_ids:
                continue
            
            matches = set()
            if len(parts) > 1 and parts[1].strip():
                matches = {m.strip() for m in parts[1].split(',') if m.strip()}
            gt_map[s1_id] = matches
    return gt_map


def evaluate_blocking(
    s1_sample_size: int = 5000,
    cand_sample_size: int = 50000,
    max_candidates: int = 35
):
    print("=" * 60)
    print("STAGE 1 CANDIDATE BLOCKING RECALL EVALUATION")
    print("=" * 60)
    
    data_dir = os.path.join(PROJECT_ROOT, "dataset", "train")
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    
    print(f"Loading {s1_sample_size} Source 1 records...")
    t0 = time.time()
    s1_records = load_tsv_records(s1_path, max_rows=s1_sample_size)
    s1_ids = {r['entity_id'] for r in s1_records}
    print(f"Loaded {len(s1_records)} S1 records in {time.time() - t0:.2f}s")
    
    print("Loading Ground Truth labels...")
    gt_map = load_ground_truth_map(gt_path, filter_s1_ids=s1_ids)
    
    # Identify which S2 and S3 IDs are true matches for these S1 records
    all_true_target_ids = set()
    for targets in gt_map.values():
        all_true_target_ids.update(targets)
    print(f"Found {len(all_true_target_ids)} true matching target IDs for these {len(s1_records)} S1 entities.")
    
    print(f"Loading candidate records from Source 2 and Source 3...")
    t0 = time.time()
    s2_records = load_tsv_records(s2_path, max_rows=cand_sample_size)
    s3_records = load_tsv_records(s3_path, max_rows=cand_sample_size)
    candidate_records = s2_records + s3_records
    print(f"Loaded {len(candidate_records)} candidate pool records in {time.time() - t0:.2f}s")
    
    # Build Inverted Index Blocker
    print("\nIndexing Candidate Pool...")
    t0 = time.time()
    blocker = InvertedIndexBlocker()
    blocker.index_candidates(candidate_records)
    print(f"Index built in {time.time() - t0:.2f}s")
    
    # Retrieve Candidates and Compute Recall
    print(f"\nRetrieving top {max_candidates} candidates for each S1 entity...")
    t0 = time.time()
    
    total_true_links = 0
    captured_true_links = 0
    candidate_counts = []
    singletons_count = 0
    
    for r in s1_records:
        s1_id = r['entity_id']
        true_targets = gt_map.get(s1_id, set())
        
        # Only evaluate on targets that were actually loaded in our candidate pool slice
        evaluable_targets = true_targets.intersection({c['entity_id'] for c in candidate_records})
        
        if len(true_targets) == 0:
            singletons_count += 1
            
        retrieved = blocker.retrieve_candidates(r, max_candidates=max_candidates)
        candidate_counts.append(len(retrieved))
        
        if evaluable_targets:
            total_true_links += len(evaluable_targets)
            captured = evaluable_targets.intersection(set(retrieved))
            captured_true_links += len(captured)
            
    elapsed = time.time() - t0
    
    # Performance Metrics
    recall = (captured_true_links / total_true_links) if total_true_links > 0 else 0.0
    avg_cands = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
    speed = len(s1_records) / elapsed if elapsed > 0 else 0
    
    print("-" * 60)
    print("EVALUATION RESULTS:")
    print(f"• Total S1 Records Evaluated: {len(s1_records):,}")
    print(f"• Singletons (Entities with 0 true matches): {singletons_count:,} ({singletons_count/len(s1_records)*100:.1f}%)")
    print(f"• True Match Links Tested: {total_true_links:,}")
    print(f"• True Links Captured in Top-{max_candidates}: {captured_true_links:,}")
    print(f"• CANDIDATE RECALL CEILING: {recall * 100:.2f}%")
    print(f"• Average Candidates Generated per Entity: {avg_cands:.1f}")
    print(f"• Processing Speed: {speed:.1f} entities/second")
    print("-" * 60)


if __name__ == "__main__":
    evaluate_blocking(s1_sample_size=3000, cand_sample_size=30000, max_candidates=35)
