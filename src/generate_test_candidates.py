import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.blocking import InvertedIndexBlocker, load_tsv_records, write_candidate_pairs_file


def generate_and_validate_demo():
    print("=" * 60)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR DEMO")
    print("=" * 60)
    
    test_dir = os.path.join(PROJECT_ROOT, "dataset", "test")
    s1_test_path = os.path.join(test_dir, "test_source1.tsv")
    s2_test_path = os.path.join(test_dir, "test_source2.tsv")
    s3_test_path = os.path.join(test_dir, "test_source3.tsv")
    
    out_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    candidate_out = os.path.join(out_dir, "candidate_pairs.tsv")
    matching_out = os.path.join(out_dir, "matching_results.tsv")
    
    # Load all test S1 records (to meet the rule: every S1 must appear!)
    print("Loading test_source1.tsv (All entity IDs)...")
    s1_ids = []
    with open(s1_test_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.split('\t')
            if parts:
                s1_ids.append(parts[0].strip())
    print(f"Total test S1 entities: {len(s1_ids):,}")
    
    # Write a baseline compliant matching_results.tsv and candidate_pairs.tsv
    print("Writing candidate_pairs.tsv and matching_results.tsv...")
    with open(candidate_out, 'w', encoding='utf-8') as f_cand, \
         open(matching_out, 'w', encoding='utf-8') as f_match:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        
        # Every test entity is written; empty candidates / matches default
        for s1_id in s1_ids:
            f_cand.write(f"{s1_id}\t\n")
            f_match.write(f"{s1_id}\t\n")
            
    print("Output files generated successfully.")
    
    # Run official validator
    validator_path = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
    cmd = [
        sys.executable,
        validator_path,
        "--matching", matching_out,
        "--candidate", candidate_out,
        "--test-dir", test_dir
    ]
    print("\nRunning official validator script:")
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\nValidator STDOUT:")
    print(res.stdout)
    if res.stderr:
        print("Validator STDERR:", res.stderr)
    print(f"Validator Exit Code: {res.returncode} (0 = PASS)")
    assert res.returncode == 0, "Validator failed"


if __name__ == "__main__":
    generate_and_validate_demo()
