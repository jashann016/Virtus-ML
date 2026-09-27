import time
import pandas as pd

from src.feature_engineering import compute_pair_features

DATA = "output/lightgbm_training_pairs_v2.csv"

print("Loading benchmark data...")
df = pd.read_csv(DATA, nrows=10000)

# Keep only columns needed to reconstruct records
print("Loading source records...")

s1_df = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    nrows=20000
)

s2_df = pd.read_csv(
    "output/positive_source2.tsv",
    sep="\t"
)

s3_df = pd.read_csv(
    "output/positive_source3.tsv",
    sep="\t"
)

candidate_df = pd.concat(
    [s2_df, s3_df],
    ignore_index=True
)

s1_map = s1_df.set_index("entity_id").to_dict("index")
candidate_map = candidate_df.set_index("entity_id").to_dict("index")

pairs = []

for row in df.itertuples(index=False):
    s1 = s1_map.get(row.source1_entity_id)
    candidate = candidate_map.get(row.candidate_entity_id)

    if s1 is not None and candidate is not None:
        pairs.append((s1, candidate))

print("Benchmark pairs:", len(pairs))

if not pairs:
    raise RuntimeError("No valid pairs found.")

# Warm-up
for s1, candidate in pairs[:10]:
    compute_pair_features(s1, candidate)

start = time.perf_counter()

for s1, candidate in pairs:
    compute_pair_features(s1, candidate)

elapsed = time.perf_counter() - start

pairs_per_sec = len(pairs) / elapsed

print()
print("=" * 60)
print("FEATURE SPEED BENCHMARK")
print("=" * 60)
print(f"Pairs tested: {len(pairs):,}")
print(f"Time: {elapsed:.4f} seconds")
print(f"Pairs/sec: {pairs_per_sec:,.2f}")
print("=" * 60)

if pairs_per_sec >= 5000:
    print("STATUS: PASS (> 5,000 pairs/sec)")
else:
    print("STATUS: BELOW TARGET (< 5,000 pairs/sec)")
