import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

csv_files = list(DATA_DIR.glob("*.csv"))

print("=" * 80)
print("AI-NIDS - CROSS-FILE LABEL CONSISTENCY ANALYSIS")
print("=" * 80)

feature_label_map = {}

total_rows = 0
conflicting_rows = 0

for file in csv_files:

    print(f"\nProcessing: {file.name}")

    for chunk in pd.read_csv(
        file,
        chunksize=100000,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        labels = chunk["Label"].astype(str).str.strip()

        features = chunk.drop(
            columns=["Label"],
            errors="ignore"
        )

        # Hash feature vectors
        hashes = pd.util.hash_pandas_object(
            features,
            index=False
        )

        for row_hash, label in zip(hashes, labels):

            total_rows += 1

            if row_hash in feature_label_map:

                if feature_label_map[row_hash] != label:

                    conflicting_rows += 1

            else:

                feature_label_map[row_hash] = label


print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print(f"Unique feature vectors : {len(feature_label_map):,}")
print(f"Total rows processed   : {total_rows:,}")
print(f"Label conflicts found  : {conflicting_rows:,}")

print("=" * 80)