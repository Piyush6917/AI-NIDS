import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw")

csv_files = list(DATA_DIR.glob("*.csv"))

print("=" * 75)
print("AI-NIDS - FULL DATASET QUALITY ANALYSIS")
print("=" * 75)

overall_rows = 0
overall_duplicates = 0

for file in csv_files:

    print("\n" + "=" * 75)
    print(f"FILE: {file.name}")
    print("=" * 75)

    total_rows = 0
    missing_counts = {}
    infinite_count = 0
    duplicate_count = 0

    # Read in chunks
    for chunk in pd.read_csv(
        file,
        chunksize=100000,
        low_memory=False
    ):

        # Clean column names
        chunk.columns = chunk.columns.str.strip()

        total_rows += len(chunk)

        # --------------------------------------------------
        # Missing values
        # --------------------------------------------------
        missing = chunk.isnull().sum()

        for column, count in missing.items():
            if count > 0:
                missing_counts[column] = (
                    missing_counts.get(column, 0) + count
                )

        # --------------------------------------------------
        # Infinite values
        # --------------------------------------------------
        numeric_columns = chunk.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_columns) > 0:
            infinite_count += np.isinf(
                chunk[numeric_columns]
            ).sum().sum()

        # --------------------------------------------------
        # Duplicate rows within chunks
        # --------------------------------------------------
        duplicate_count += chunk.duplicated().sum()

    overall_rows += total_rows
    overall_duplicates += duplicate_count

    print(f"\nTotal rows              : {total_rows:,}")
    print(f"Duplicate rows          : {duplicate_count:,}")
    print(f"Infinite values         : {infinite_count:,}")

    print("\nMissing values:")

    if missing_counts:
        for column, count in sorted(
            missing_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"{column:<40} {count:,}")
    else:
        print("No missing values found.")

print("\n" + "=" * 75)
print("OVERALL SUMMARY")
print("=" * 75)

print(f"Total rows              : {overall_rows:,}")
print(f"Duplicate rows*         : {overall_duplicates:,}")

print("\n* Duplicate count here is based on duplicates detected")
print("  within individual chunks and is only an initial check.")

print("\n" + "=" * 75)
print("QUALITY ANALYSIS COMPLETE")
print("=" * 75)