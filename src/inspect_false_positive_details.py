import pandas as pd

FP_FILE = "output/independent_false_positives_099.csv"

fp = pd.read_csv(FP_FILE, dtype=str)

# Keep the 30 highest-confidence false positives
fp = fp.sort_values("probability", ascending=False).head(30)

# Load source records
s1 = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s2 = pd.read_csv(
    "train_source2.tsv",
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s3 = pd.read_csv(
    "train_source3.tsv",
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s1 = s1.rename(columns={
    "entity_id": "source1_entity_id",
    "business_name": "s1_name",
    "business_address": "s1_address",
    "country": "s1_country"
})

s2 = s2.rename(columns={
    "entity_id": "candidate_entity_id",
    "business_name": "candidate_name",
    "business_address": "candidate_address",
    "country": "candidate_country"
})

s3 = s3.rename(columns={
    "entity_id": "candidate_entity_id",
    "business_name": "candidate_name",
    "business_address": "candidate_address",
    "country": "candidate_country"
})

s2 = s2[s2["candidate_entity_id"].isin(
    fp["candidate_entity_id"]
)]

s3 = s3[s3["candidate_entity_id"].isin(
    fp["candidate_entity_id"]
)]

candidates = pd.concat([s2, s3], ignore_index=True)

result = fp.merge(
    s1,
    on="source1_entity_id",
    how="left"
)

result = result.merge(
    candidates,
    on="candidate_entity_id",
    how="left"
)

columns = [
    "source1_entity_id",
    "candidate_entity_id",
    "probability",
    "s1_name",
    "candidate_name",
    "s1_address",
    "candidate_address",
    "s1_country",
    "candidate_country"
]

result = result[columns]

print("=" * 100)
print("TOP 30 HIGH-CONFIDENCE FALSE POSITIVES")
print("=" * 100)

for i, row in result.iterrows():

    print()
    print(f"CASE {i + 1}")
    print("-" * 100)

    print("S1 ID       :", row["source1_entity_id"])
    print("Candidate   :", row["candidate_entity_id"])
    print("Probability :", row["probability"])

    print()
    print("SOURCE 1")
    print("Name       :", row["s1_name"])
    print("Address    :", row["s1_address"])
    print("Country    :", row["s1_country"])

    print()
    print("CANDIDATE")
    print("Name       :", row["candidate_name"])
    print("Address    :", row["candidate_address"])
    print("Country    :", row["candidate_country"])

result.to_csv(
    "output/top_30_false_positive_details.csv",
    index=False
)

print()
print("Saved: output/top_30_false_positive_details.csv")
