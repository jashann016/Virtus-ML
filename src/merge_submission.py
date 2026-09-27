"""
VIRTUS-ML: Official Submission Merger & Validator
Merges country-specific matching results and candidate pairs into the official
competition submission files in the exact row order of test_source1.tsv.
Packages into output/submission.zip and runs utils/validate_submission.py.
"""

import os
import sys
import time
import zipfile
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def merge_and_validate(output_dir="output"):
    print("=" * 70)
    print("VIRTUS-ML: MERGING SUBMISSION FILES ACROSS COUNTRIES")
    print("=" * 70)
    t0 = time.time()

    countries = ["france", "us", "india"]
    matching_map = {}
    candidate_map = {}

    for c in countries:
        m_file = os.path.join(output_dir, f"matching_{c}.tsv")
        c_file = os.path.join(output_dir, f"candidate_{c}.tsv")
        
        if not os.path.exists(m_file) or not os.path.exists(c_file):
            print(f"⚠️ In-progress or missing files for {c.upper()} ({m_file}) - will default missing records to singletons")
        
        if os.path.exists(m_file):
            print(f"• Loading {c.upper()} matches...")
            with open(m_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip('\n').split('\t')
                    if parts:
                        matching_map[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else ""

        if os.path.exists(c_file):
            print(f"• Loading {c.upper()} candidates...")
            with open(c_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip('\n').split('\t')
                    if parts:
                        candidate_map[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else ""

    print(f"Total matching records loaded: {len(matching_map):,}")
    print(f"Total candidate records loaded: {len(candidate_map):,}")

    # Read exact S1 order from test_source1.tsv
    s1_path = os.path.join(PROJECT_ROOT, "dataset", "test", "test_source1.tsv")
    print(f"• Reading exact S1 entity order from {s1_path}...")
    s1_ids = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.split('\t')
            if parts:
                s1_ids.append(parts[0].strip())

    print(f"Total test entities required: {len(s1_ids):,}")

    final_matching = os.path.join(output_dir, "matching_results.tsv")
    final_candidate = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"• Writing official {final_matching}...")
    with open(final_matching, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            match_str = matching_map.get(sid, "")
            f.write(f"{sid}\t{match_str}\n")

    print(f"• Writing official {final_candidate}...")
    with open(final_candidate, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            cand_str = candidate_map.get(sid, "")
            match_str = matching_map.get(sid, "")
            if match_str:
                match_ids = [m.strip() for m in match_str.split(",") if m.strip()]
                if cand_str:
                    cands = [c.strip() for c in cand_str.split(",") if c.strip()]
                    cand_set = set(cands)
                    for m in match_ids:
                        if m not in cand_set:
                            cands.append(m)
                            cand_set.add(m)
                    cand_str = ",".join(cands)
                else:
                    cand_str = match_str
            f.write(f"{sid}\t{cand_str}\n")

    # Zip files together
    zip_path = os.path.join(output_dir, "submission.zip")
    print(f"• Creating {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(final_matching, arcname="matching_results.tsv")
        z.write(final_candidate, arcname="candidate_pairs.tsv")

    print(f"✅ Generated {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB) in {time.time()-t0:.2f}s")

    # Run official validator
    val_script = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
    test_dir = os.path.join(PROJECT_ROOT, "dataset", "test")
    if os.path.exists(val_script):
        print("\n" + "=" * 70)
        print("RUNNING OFFICIAL VALIDATOR ON MERGED SUBMISSION")
        print("=" * 70)
        cmd = [sys.executable, val_script, "--matching", final_matching, "--candidate", final_candidate, "--test-dir", test_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            print("🎉 OFFICIAL VALIDATOR: PASS! 100% READY FOR UNSTOP UPLOAD!")
            return True
        else:
            print("❌ VALIDATOR ISSUES:\n", res.stderr)
            return False

    return True

if __name__ == "__main__":
    merge_and_validate()
