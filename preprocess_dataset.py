import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FEATURES TO REMOVE
# ============================================================

# Constant / near-useless features identified during analysis
CONSTANT_FEATURES = [
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

# Exact redundant features identified during correlation analysis
REDUNDANT_FEATURES = [
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
]

FEATURES_TO_REMOVE = set(
    CONSTANT_FEATURES + REDUNDANT_FEATURES
)

# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):

    label = str(label).strip()

    # Normalize corrupted Web Attack labels
    replacements = {
        "Web Attack � Brute Force": "Web Attack - Brute Force",
        "Web Attack � XSS": "Web Attack - XSS",
        "Web Attack � Sql Injection": "Web Attack - SQL Injection",
    }

    return replacements.get(label, label)


# ============================================================
# ATTACK FAMILY
# ============================================================

def get_attack_family(label):

    if label == "BENIGN":
        return "BENIGN"

    if label in [
        "DDoS",
        "DoS Hulk",
        "DoS GoldenEye",
        "DoS slowloris",
        "DoS Slowhttptest",
    ]:
        return "DoS_DDoS"

    if label == "PortScan":
        return "PortScan"

    if label in [
        "FTP-Patator",
        "SSH-Patator",
    ]:
        return "BruteForce"

    if label in [
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - SQL Injection",
    ]:
        return "WebAttack"

    if label == "Bot":
        return "Bot"

    if label == "Infiltration":
        return "Infiltration"

    if label == "Heartbleed":
        return "Other"

    return "Other"


# ============================================================
# PROCESS FILE
# ============================================================

csv_files = list(RAW_DIR.glob("*.csv"))

print("=" * 80)
print("AI-NIDS - DATASET PREPROCESSING")
print("=" * 80)

print(f"\nFiles found: {len(csv_files)}")

for file in csv_files:
    print(f"  {file.name}")

print("\n" + "=" * 80)

# Statistics
total_rows = 0
removed_rows = 0
saved_rows = 0

global_label_counts = {}
global_family_counts = {}

for file in csv_files:

    print(f"\nProcessing: {file.name}")

    output_file = PROCESSED_DIR / f"cleaned_{file.stem}.csv"

    first_chunk = True

    file_rows = 0
    file_saved = 0

    for chunk in pd.read_csv(
        file,
        chunksize=100000,
        low_memory=False
    ):

        # ----------------------------------------------------
        # Clean column names
        # ----------------------------------------------------

        chunk.columns = chunk.columns.str.strip()

        # ----------------------------------------------------
        # Normalize labels
        # ----------------------------------------------------

        chunk["Label"] = (
            chunk["Label"]
            .astype(str)
            .map(normalize_label)
        )

        # ----------------------------------------------------
        # Remove unwanted features
        # ----------------------------------------------------

        columns_to_drop = [
            column
            for column in chunk.columns
            if column in FEATURES_TO_REMOVE
        ]

        chunk = chunk.drop(
            columns=columns_to_drop,
            errors="ignore"
        )

        # ----------------------------------------------------
        # Replace infinity values
        # ----------------------------------------------------

        numeric_columns = chunk.select_dtypes(
            include=np.number
        ).columns

        chunk[numeric_columns] = (
            chunk[numeric_columns]
            .replace([np.inf, -np.inf], np.nan)
        )

        # ----------------------------------------------------
        # Remove rows with missing numeric values
        # ----------------------------------------------------
        #
        # We currently expect missing values mainly in
        # Flow Bytes/s.
        #
        # We will revisit this strategy if the number of
        # removed rows becomes significant.
        # ----------------------------------------------------

        before = len(chunk)

        chunk = chunk.dropna()

        removed = before - len(chunk)

        removed_rows += removed

        # ----------------------------------------------------
        # Binary target
        # ----------------------------------------------------

        chunk["Binary_Label"] = (
            chunk["Label"] != "BENIGN"
        ).astype(int)

        # ----------------------------------------------------
        # Attack family
        # ----------------------------------------------------

        chunk["Attack_Family"] = (
            chunk["Label"]
            .map(get_attack_family)
        )

        # ----------------------------------------------------
        # Update statistics
        # ----------------------------------------------------

        file_rows += before
        file_saved += len(chunk)

        total_rows += before
        saved_rows += len(chunk)

        # Label counts
        for label, count in chunk["Label"].value_counts().items():

            global_label_counts[label] = (
                global_label_counts.get(label, 0)
                + count
            )

        # Family counts
        for family, count in chunk["Attack_Family"].value_counts().items():

            global_family_counts[family] = (
                global_family_counts.get(family, 0)
                + count
            )

        # ----------------------------------------------------
        # Save processed chunk
        # ----------------------------------------------------

        chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

    print(f"Rows read    : {file_rows:,}")
    print(f"Rows saved   : {file_saved:,}")
    print(f"Rows removed : {file_rows - file_saved:,}")

print("\n" + "=" * 80)
print("PREPROCESSING SUMMARY")
print("=" * 80)

print(f"Total rows processed : {total_rows:,}")
print(f"Rows removed         : {removed_rows:,}")
print(f"Rows saved           : {saved_rows:,}")

print("\n" + "=" * 80)
print("FINAL LABEL DISTRIBUTION")
print("=" * 80)

for label, count in sorted(
    global_label_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{label:<35} {count:>12,}")

print("\n" + "=" * 80)
print("ATTACK FAMILY DISTRIBUTION")
print("=" * 80)

for family, count in sorted(
    global_family_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{family:<35} {count:>12,}")

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)

print(f"\nProcessed files saved in:")
print(PROCESSED_DIR.resolve())