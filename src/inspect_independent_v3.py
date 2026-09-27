import pandas as pd

FILE = "output/independent_scored_candidates_v3.parquet"

print("=" * 70)
print("INDEPENDENT V3 PROBABILITY ANALYSIS")
print("=" * 70)

df = pd.read_parquet(
    FILE,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "probability"
    ]
)

p = df["probability"]

print("\n========== BASIC STATS ==========")

print("Total candidates:", len(df))

print("Min probability :", round(p.min(), 6))
print("Max probability :", round(p.max(), 6))
print("Mean probability:", round(p.mean(), 6))
print("Median          :", round(p.median(), 6))

print("\n========== THRESHOLD COUNTS ==========")

thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    0.995,
    0.999
]

for t in thresholds:

    count = int((p >= t).sum())

    print(
        f"Threshold {t:.3f} -> "
        f"{count:,} predictions"
    )

print("\n========== PROBABILITY BINS ==========")

bins = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    0.999,
    1.0
]

print(
    pd.cut(
        p,
        bins=bins,
        include_lowest=True
    ).value_counts(
        sort=False
    ).to_string()
)

print("\n========== TOP 20 ==========")

print(
    df.nlargest(
        20,
        "probability"
    ).to_string(index=False)
)

print("\nSTATUS: Probability analysis complete.")