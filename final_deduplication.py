import pandas as pd
from pathlib import Path
import hashlib

# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")
FINAL_DIR = Path("data/final")

FINAL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = FINAL_DIR / "final_dataset.csv"

# ============================================================
# IMPORTANT COLUMNS
# ============================================================

TARGET_COLUMNS = [
    "Label",
    "Binary_Label",
    "Attack_Family"
]

# ============================================================
# STEP 1: FIND PROCESSED FILES
# ============================================================

files = list(PROCESSED_DIR.glob("cleaned_*.csv"))

print("=" * 80)
print("AI-NIDS - FINAL DEDUPLICATION")
print("=" * 80)

print(f"\nProcessed files found: {len(files)}")

for file in files:
    print(f"  {file.name}")

# ============================================================
# STEP 2: FIRST PASS
# Create mapping:
#
# feature_hash -> label
#
# If same feature vector appears with different labels,
# mark it as conflicting.
# ============================================================

feature_labels = {}
conflicting_hashes = set()

total_rows = 0

print("\n" + "=" * 80)
print("PASS 1 - FINDING DUPLICATES AND LABEL CONFLICTS")
print("=" * 80)

for file in files:

    print(f"\nProcessing: {file.name}")

    for chunk in pd.read_csv(
        file,
        chunksize=100000,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        feature_columns = [
            column
            for column in chunk.columns
            if column not in TARGET_COLUMNS
        ]

        features = chunk[feature_columns]

        hashes = pd.util.hash_pandas_object(
            features,
            index=False
        )

        labels = chunk["Label"].astype(str).str.strip()

        for row_hash, label in zip(hashes, labels):

            total_rows += 1

            if row_hash in conflicting_hashes:
                continue

            if row_hash in feature_labels:

                if feature_labels[row_hash] != label:

                    conflicting_hashes.add(row_hash)

                    # Remove from normal mapping
                    feature_labels.pop(
                        row_hash,
                        None
                    )

            else:

                feature_labels[row_hash] = label

print("\n" + "=" * 80)
print("PASS 1 COMPLETE")
print("=" * 80)

print(f"Total rows processed       : {total_rows:,}")
print(f"Unique non-conflicting rows: {len(feature_labels):,}")
print(f"Conflicting feature hashes : {len(conflicting_hashes):,}")

# ============================================================
# STEP 3: SECOND PASS
# Keep only:
#
# 1. First occurrence of a non-conflicting feature vector
# 2. Remove all conflicting feature vectors
# ============================================================

print("\n" + "=" * 80)
print("PASS 2 - CREATING FINAL DATASET")
print("=" * 80)

# Clear existing output
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

saved_hashes = set()

total_second_pass = 0
saved_rows = 0
removed_duplicates = 0
removed_conflicts = 0

first_write = True

for file in files:

    print(f"\nWriting: {file.name}")

    for chunk in pd.read_csv(
        file,
        chunksize=100000,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        feature_columns = [
            column
            for column in chunk.columns
            if column not in TARGET_COLUMNS
        ]

        features = chunk[feature_columns]

        hashes = pd.util.hash_pandas_object(
            features,
            index=False
        )

        keep_indices = []

        for index, row_hash in zip(
            chunk.index,
            hashes
        ):

            total_second_pass += 1

            # Remove conflicting feature vectors
            if row_hash in conflicting_hashes:

                removed_conflicts += 1
                continue

            # Remove duplicate feature vectors
            if row_hash in saved_hashes:

                removed_duplicates += 1
                continue

            saved_hashes.add(row_hash)

            keep_indices.append(index)

        cleaned_chunk = chunk.loc[keep_indices]

        if len(cleaned_chunk) > 0:

            cleaned_chunk.to_csv(
                OUTPUT_FILE,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )

            first_write = False

            saved_rows += len(cleaned_chunk)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATASET SUMMARY")
print("=" * 80)

print(f"Original processed rows : {total_rows:,}")
print(f"Final rows              : {saved_rows:,}")
print(f"Duplicates removed      : {removed_duplicates:,}")
print(f"Conflict rows removed   : {removed_conflicts:,}")

print("\n" + "=" * 80)
print("FINAL DATASET LABEL DISTRIBUTION")
print("=" * 80)

# Read only Label columns for final verification
label_counts = {}

for chunk in pd.read_csv(
    OUTPUT_FILE,
    usecols=["Label"],
    chunksize=100000
):

    counts = chunk["Label"].value_counts()

    for label, count in counts.items():

        label_counts[label] = (
            label_counts.get(label, 0) + count
        )

for label, count in sorted(
    label_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    percentage = count / saved_rows * 100

    print(
        f"{label:<35}"
        f"{count:>12,}"
        f"{percentage:>8.2f}%"
    )

print("\n" + "=" * 80)
print("FINAL DATASET CREATED")
print("=" * 80)

print(f"\nSaved to:")
print(OUTPUT_FILE.resolve())