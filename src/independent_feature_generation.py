import os
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.feature_engineering import compute_pair_features, FEATURE_COLUMNS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANDIDATE_FILE = os.path.join(ROOT, "output", "independent_candidate_pairs.csv")
S1_FILE = os.path.join(ROOT, "train_source1.tsv")
S2_FILE = os.path.join(ROOT, "train_source2.tsv")
S3_FILE = os.path.join(ROOT, "train_source3.tsv")
OUTPUT_FILE = os.path.join(ROOT, "output", "independent_features.parquet")

CHUNK_SIZE = 50_000

print("=" * 70)
print("INDEPENDENT FEATURE GENERATION")
print("=" * 70)
print(f"Expected candidate pairs : 3,637,728")
print(f"Features                 : {len(FEATURE_COLUMNS)}")
print(f"Chunk size               : {CHUNK_SIZE}")
print("=" * 70)

# ------------------------------------------------------------
# 1. Exact held-out S1 IDs: rows 20,001-21,000
# ------------------------------------------------------------
print("\n[1/4] Loading exact 1,000 held-out Source-1 records...")

s1_holdout = pd.read_csv(
    S1_FILE,
    sep="\t",
    skiprows=range(1, 20001),
    nrows=1000,
    dtype=str
)

s1_ids = set(s1_holdout["entity_id"])

s1_lookup = {}

for row in s1_holdout.itertuples(index=False):
    s1_lookup[row.entity_id] = {
        "business_name": row.business_name,
        "business_address": row.business_address,
        "country": row.country,
    }

del s1_holdout
gc.collect()

print(f"S1 records loaded: {len(s1_lookup):,}")

# ------------------------------------------------------------
# 2. Collect candidate IDs
# ------------------------------------------------------------
print("\n[2/4] Collecting unique candidate IDs...")

candidate_ids = set()

for chunk in pd.read_csv(
    CANDIDATE_FILE,
    usecols=["source1_entity_id", "candidate_entity_id"],
    dtype=str,
    chunksize=CHUNK_SIZE
):
    candidate_ids.update(chunk["candidate_entity_id"].dropna().tolist())

print(f"Unique candidate IDs: {len(candidate_ids):,}")

# ------------------------------------------------------------
# 3. Load required S2/S3 records
# ------------------------------------------------------------
print("\n[3/4] Loading required Source-2/Source-3 records...")

candidate_lookup = {}

for source_file, source_name in [
    (S2_FILE, "S2"),
    (S3_FILE, "S3")
]:
    found = 0

    for chunk in pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        chunksize=100_000
    ):
        mask = chunk["entity_id"].isin(candidate_ids)

        selected = chunk.loc[
            mask,
            ["entity_id", "business_name", "business_address", "country"]
        ]

        for row in selected.itertuples(index=False):
            candidate_lookup[row.entity_id] = {
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country,
            }

        found += len(selected)

    print(f"{source_name} records found: {found:,}")

print(f"Total candidate records loaded: {len(candidate_lookup):,}")

# ------------------------------------------------------------
# 4. Feature generation + streaming Parquet writer
# ------------------------------------------------------------
print("\n[4/4] Generating 14 features...")

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

writer = None
total_rows = 0
total_candidates = 0
skipped = 0

try:
    for chunk_no, pairs in enumerate(
        pd.read_csv(
            CANDIDATE_FILE,
            dtype=str,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):
        rows = []

        for pair in pairs.itertuples(index=False):

            total_candidates += 1

            s1 = s1_lookup.get(pair.source1_entity_id)
            cand = candidate_lookup.get(pair.candidate_entity_id)

            if s1 is None or cand is None:
                skipped += 1
                continue

            features = compute_pair_features(s1, cand)

            row = {
                "source1_entity_id": pair.source1_entity_id,
                "candidate_entity_id": pair.candidate_entity_id,
            }

            row.update(features)
            rows.append(row)

        if rows:
            out = pd.DataFrame(rows)

            table = pa.Table.from_pandas(
                out,
                preserve_index=False
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_FILE,
                    table.schema,
                    compression="snappy"
                )

            writer.write_table(table)

            total_rows += len(out)

            del out
            del table

        del rows
        del pairs

        if chunk_no % 10 == 0:
            print(
                f"Chunk {chunk_no:03d} | "
                f"candidates processed: {total_candidates:,} | "
                f"features written: {total_rows:,} | "
                f"skipped: {skipped:,}"
            )

        gc.collect()

finally:
    if writer is not None:
        writer.close()

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("FEATURE GENERATION COMPLETE")
print("=" * 70)
print(f"Candidates processed : {total_candidates:,}")
print(f"Features written     : {total_rows:,}")
print(f"Skipped              : {skipped:,}")
print(f"Expected             : 3,637,728")
print(f"Output               : {OUTPUT_FILE}")
print("=" * 70)

if total_rows != 3_637_728:
    print("WARNING: output row count differs from expected candidate count.")
else:
    print("STATUS: PASS - all candidate pairs converted to features.")

