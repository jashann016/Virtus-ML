"""
VIRTUS-ML: Surgical Patching of LightGBM V3 Predictions with Qwen 2.5 7B Verdicts
Updates only the ambiguous borderline entities with fine-tuned LLM decisions.
Produces the final competition submission package.
"""

import os
import sys
import json
import zipfile
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def patch_predictions(
    matching_file: str = "output/matching_results.tsv",
    candidate_file: str = "output/candidate_pairs.tsv",
    qwen_results_file: str = "qwen_decisions.json",
    output_dir: str = "output",
    output_zip: str = "/kaggle/working/submission_master_v5.zip"
):
    print("=" * 70)
    print("VIRTUS-ML: SURGICAL PATCHING WITH QWEN 2.5 7B ARBITER")
    print(f"  • Base Matches : {matching_file}")
    print(f"  • Qwen Verdicts: {qwen_results_file}")
    print("=" * 70)

    # 1. Load Qwen Decisions
    print("[1/3] Loading Qwen 2.5 7B decisions...")
    qwen_updates = {} # s1_id -> candidate_id (accepted) or None (rejected)

    if qwen_results_file.endswith('.json'):
        with open(qwen_results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # format: list of {"source1_id": ..., "candidate_id": ..., "is_match": bool, "confidence": float}
            for item in data:
                s1 = item.get('source1_id')
                cid = item.get('candidate_id')
                is_match = item.get('is_match', False)
                conf = item.get('confidence', 0.5)
                if is_match and conf >= 0.70:
                    qwen_updates[s1] = cid
                elif not is_match:
                    qwen_updates[s1] = None
    elif qwen_results_file.endswith(('.jsonl', '.jsonl.gz')):
        import gzip
        opener = gzip.open if qwen_results_file.endswith('.gz') else open
        with opener(qwen_results_file, 'rt', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                s1 = item.get('source1_id')
                cid = item.get('candidate_id')
                is_match = item.get('is_match', False)
                conf = item.get('confidence', 0.5)
                if is_match and conf >= 0.70:
                    qwen_updates[s1] = cid
                elif not is_match:
                    qwen_updates[s1] = None

    print(f"  • Loaded {len(qwen_updates):,} Qwen decisions.")

    # 2. Patch matching_results.tsv in-place
    print("[2/3] Applying surgical patch to matching results...")
    patched_matching = os.path.join(output_dir, "matching_results_patched.tsv")
    
    patched_count = 0
    with open(matching_file, 'r', encoding='utf-8') as f_in, \
         open(patched_matching, 'w', encoding='utf-8') as f_out:
        
        header = f_in.readline()
        f_out.write(header)

        for line in f_in:
            parts = line.strip('\n').split('\t')
            s1 = parts[0].strip()
            current_matches = parts[1].strip() if len(parts) > 1 else ""

            if s1 in qwen_updates:
                decision = qwen_updates[s1]
                if decision: # Qwen confirmed match
                    f_out.write(f"{s1}\t{decision}\n")
                    patched_count += 1
                else: # Qwen confirmed false positive / reject
                    f_out.write(f"{s1}\t\n")
                    patched_count += 1
            else:
                f_out.write(line)

    print(f"  • Successfully patched {patched_count:,} borderline entities!")

    # 3. Create Final Submission Zip
    print(f"[3/3] Packaging final patched submission into {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(patched_matching, arcname="matching_results.tsv")
        z.write(candidate_file, arcname="candidate_pairs.tsv")

    mb = os.path.getsize(output_zip) / (1024 * 1024)
    print("\n" + "=" * 70)
    print("SURGICAL PATCHING COMPLETE:")
    print(f"• Final Zip Generated : {output_zip} ({mb:.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch matching results with Qwen verdicts.")
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--qwen", default="qwen_decisions.json")
    parser.add_argument("--output-zip", default="/kaggle/working/submission_master_v5.zip")
    args = parser.parse_args()

    patch_predictions(
        matching_file=args.matching,
        candidate_file=args.candidate,
        qwen_results_file=args.qwen,
        output_zip=args.output_zip
    )
